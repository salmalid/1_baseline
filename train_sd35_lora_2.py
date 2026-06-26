import argparse
import copy
import io
import json
import logging
import math
import os
import re
import shutil
import zipfile
from pathlib import Path

import accelerate
import torch
import torch.utils._device
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from contextlib import nullcontext
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from peft import LoraConfig
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import PretrainedConfig
from torch.utils.data import Dataset

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def gather_params_ctx(model, accelerator):
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        import deepspeed
        return deepspeed.zero.GatheredParameters(model.parameters(), modifier_rank=0)
    return nullcontext()

class PathologyDataset(Dataset):
    def __init__(self, data_dir, resolution=512, latent_suffix=None, require_cached_latents=False, latent_dir=None, embed_dir=None):
        self.data_dir = Path(data_dir) if data_dir else None
        self.resolution = resolution
        self.latent_suffix = latent_suffix or f"_latents_{resolution}.pt"
        self.require_cached_latents = require_cached_latents
        self.latent_dir = Path(latent_dir) if latent_dir else None
        self.embed_dir = Path(embed_dir) if embed_dir else None
        self.image_paths = []
        self.embed_paths = []
        self.latent_paths = []

        scan_dir = self.embed_dir if self.embed_dir else self.data_dir

        for path in scan_dir.rglob("*"):
            if self.embed_dir:

                if not path.name.endswith("_embeds_v2.pt"):
                    continue
                embed_path = path
                rel = path.relative_to(self.embed_dir)
                stem = rel.stem.replace("_embeds_v2", "")  # view1_frontal
                rel_img = rel.with_name(stem + path.suffix.replace(".pt", ".png"))
                if self.latent_dir:
                    latent_path = self.latent_dir / rel.with_name(stem + self.latent_suffix)
                else:
                    latent_path = embed_path.with_name(stem + self.latent_suffix)
                image_path = self.data_dir / rel_img if self.data_dir else None
            else:
                if path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp'):
                    continue
                image_path = path
                embed_path = path.with_name(path.stem + "_embeds_v2.pt")
                rel = path.relative_to(self.data_dir)
                if self.latent_dir:
                    latent_path = self.latent_dir / rel.with_name(rel.stem + self.latent_suffix)
                else:
                    latent_path = path.with_name(path.stem + self.latent_suffix)
                if not embed_path.exists():
                    continue

            if not self.require_cached_latents or latent_path.exists():
                self.image_paths.append(image_path)
                self.embed_paths.append(embed_path)
                self.latent_paths.append(latent_path if latent_path.exists() else None)




        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((self.resolution, self.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [0,1] → [-1,1]
            ]
        )
        if self.require_cached_latents:
            logger.info(
                f"Loaded {len(self.image_paths)} images with precomputed embeddings and cached latents ({self.latent_suffix})."
            )
        else:
            logger.info(f"Loaded {len(self.image_paths)} images with precomputed embeddings.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        embeds = torch.load(self.embed_paths[idx], map_location="cpu", weights_only=True)
        sample = {
            "prompt_embeds": embeds["prompt_embeds"].squeeze(0),
            "pooled_prompt_embeds": embeds["pooled_prompt_embeds"].squeeze(0),
        }

        if self.latent_paths[idx] is not None:
            latent_data = torch.load(self.latent_paths[idx], map_location="cpu", weights_only=True)
            sample["model_input"] = latent_data["model_input"].squeeze(0)
            return sample

        image = Image.open(self.image_paths[idx]).convert("RGB")
        sample["pixel_values"] = self.image_transforms(image)
        return sample

def collate_fn(examples):
    batch = {
        "prompt_embeds": torch.stack([example["prompt_embeds"].clone().detach() for example in examples]),
        "pooled_prompt_embeds": torch.stack([example["pooled_prompt_embeds"].clone().detach() for example in examples]),
    }

    if "model_input" in examples[0]:
        batch["model_input"] = torch.stack([example["model_input"].clone().detach() for example in examples])
    else:
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        batch["pixel_values"] = pixel_values.to(memory_format=torch.contiguous_format).float()

    return batch

def extract_pathology(text):
    """Pull the pathology part out of a prompt or caption, e.g. 'showing Lung Opacity' → 'Lung Opacity'."""
    idx = text.lower().find("showing ")
    if idx >= 0:
        return text[idx + 8:].strip().rstrip(".")
    return text.strip()


_ZIP_RE = re.compile(r"^(.*\.zip)[/\\](.*)", re.IGNORECASE)


def _parse_path(path_str: str):
    """Split 'd:/data/chunk.zip/PNG/train/...' into (zip_file_str, internal_path).
    Returns (None, path_str) for plain file paths."""
    m = _ZIP_RE.match(str(path_str))
    if m:
        return m.group(1), m.group(2).replace("\\", "/")
    return None, str(path_str)


def _open_image_from_path(path_str) -> "Image.Image | None":
    """Open a PIL Image from a plain path or from inside a ZIP archive."""
    if path_str is None:
        return None
    try:
        zip_file, internal = _parse_path(path_str)
        if zip_file:
            if not Path(zip_file).exists():
                return None
            with zipfile.ZipFile(zip_file, "r") as zf:
                with zf.open(internal) as fh:
                    img = Image.open(io.BytesIO(fh.read()))
                    img.load()
                    return img.convert("RGB")
        else:
            p = Path(path_str)
            if not p.exists():
                return None
            return Image.open(p).convert("RGB")
    except Exception:
        return None



def build_reference_images_from_json(captions_file, validation_prompts):
    """Find the first real image path for each validation prompt from the captions JSON."""
    if not captions_file or not Path(captions_file).exists():
        return [None] * len(validation_prompts)
    prompt_to_path = {}
    with open(captions_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            cap = entry.get("caption", "")
            if cap not in prompt_to_path:
                prompt_to_path[cap] = entry["path_to_image"]
    refs = []
    for prompt in validation_prompts:
        path_str = prompt_to_path.get(prompt)
        zip_file, _ = _parse_path(path_str) if path_str else (None, None)
        accessible = Path(zip_file if zip_file else path_str).exists() if path_str else False
        refs.append(path_str if accessible else None)
    return refs


def log_validation(pipeline, args, accelerator, weight_dtype, step, is_final_validation=False, reference_images=None):
    logger.info("Running validation... ")

    pipeline.set_progress_bar_config(disable=True)

    if args.get("seed") is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args["seed"])

    validation_embeds_paths = args.get("validation_embeds", [])
    if not validation_embeds_paths:
        logger.warning("No validation_embeds paths found in args. Skipping validation.")
        return []

    if is_final_validation:
        generate_height = args.get("generate_height", 1024)
        generate_width = args.get("generate_width", 1024)
    else:
        generate_height = 512
        generate_width = 512

    pipeline.enable_attention_slicing()

    neg_embed_path = args.get("negative_validation_embed", "validation_embeds/negative_embeds.pt")
    neg_embeds = torch.load(neg_embed_path, map_location=accelerator.device, weights_only=True)
    neg_prompt_embeds = neg_embeds["prompt_embeds"].to(dtype=weight_dtype)
    neg_pooled_prompt_embeds = neg_embeds["pooled_prompt_embeds"].to(dtype=weight_dtype)

    # Final validation: all pathologies. During training: 6 per call, cycling through batches.
    n_total = len(validation_embeds_paths)
    if is_final_validation:
        selected_indices = list(range(n_total))
    else:
        per_step = 6
        num_batches = math.ceil(n_total / per_step)
        batch_idx = (step // args.get("validation_steps", 250)) % num_batches
        start = batch_idx * per_step
        selected_indices = list(range(start, min(start + per_step, n_total)))

    image_logs = []
    for idx in selected_indices:
        embeds = torch.load(validation_embeds_paths[idx], map_location=accelerator.device, weights_only=True)
        prompt_embeds = embeds["prompt_embeds"].to(dtype=weight_dtype)
        pooled_prompt_embeds = embeds["pooled_prompt_embeds"].to(dtype=weight_dtype)
        prompt = embeds.get("prompt", "precomputed_validation")

        with torch.autocast("cuda", dtype=weight_dtype):
            image = pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=neg_prompt_embeds,
                negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                num_inference_steps=args.get("num_inference_steps", 28),
                guidance_scale=args.get("guidance_scale", 3.0),
                generator=generator,
                height=generate_height,
                width=generate_width,
            ).images[0]

        ref_path = reference_images[idx] if reference_images and idx < len(reference_images) else None
        image_logs.append({"prompt": prompt, "generated": image, "ref_path": ref_path})

    if wandb.run is not None:
        log_dict = {}
        for log in image_logs:
            pathology = log["prompt"].split("showing ")[-1]
            real_pil = _open_image_from_path(log["ref_path"])
            placeholder = Image.new("RGB", (512, 512), (60, 60, 60))
            real_img = (real_pil or placeholder).resize((512, 512), Image.BILINEAR)
            gen_img  = log["generated"].resize((512, 512), Image.BILINEAR)
            log_dict[f"validation/{pathology}"] = [
                wandb.Image(real_img, caption="Real"),
                wandb.Image(gen_img,  caption="Generated"),
            ]
        wandb.log(log_dict, step=step)

    return image_logs


def main(args):
    logging_dir = Path(args["output_dir"], args.get("logging_dir", "logs"))

    accelerator_project_config = ProjectConfiguration(project_dir=args["output_dir"], logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.get("gradient_accumulation_steps", 1),
        mixed_precision=args.get("mixed_precision", "fp16"),
        log_with=args.get("report_to", "tensorboard"),
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.get("report_to", "") == "wandb" and not is_wandb_available():
        raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.get("seed") is not None:
        set_seed(args["seed"])

    if accelerator.is_main_process:
        os.makedirs(args["output_dir"], exist_ok=True)

    model_path = args["pretrained_model_name_or_path"]

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    use_cached_latents = args.get("use_cached_latents", False)

    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    transformer = SD3Transformer2DModel.from_pretrained(model_path, subfolder="transformer")

    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    # Setup LoRA
    transformer_lora_config = LoraConfig(
        r=args.get("rank", 32),
        lora_alpha=args.get("lora_alpha", args.get("rank", 32)),
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)
    print("LoRA adapter applied to MMDiT ...............")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if use_cached_latents:
        vae.to("cpu")
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    transformer.enable_gradient_checkpointing()

    if args.get("allow_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.get("use_8bit_adam", False):
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = optimizer_class(
        lora_params,
        lr=args.get("learning_rate", 1e-4),
        betas=(args.get("adam_beta1", 0.9), args.get("adam_beta2", 0.999)),
        weight_decay=args.get("adam_weight_decay", 1e-2),
        eps=args.get("adam_epsilon", 1e-8),
    )

    dataset = PathologyDataset(
        args.get("train_data_dir"),
        resolution=args.get("resolution", 512),
        latent_suffix=args.get("latent_cache_suffix"),
        require_cached_latents=use_cached_latents,
        latent_dir=args.get("latent_cache_dir"),
        embed_dir=args.get("embed_dir"),
    )

    print("Chest Dataset have been loaded ........")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Did you run the precomputing embeddings script?")

    validation_prompts = args.get("validation_prompt", [])
    if isinstance(validation_prompts, str):
        validation_prompts = [validation_prompts]
    if accelerator.is_main_process and validation_prompts:
        reference_images = build_reference_images_from_json(args.get("captions_file", ""), validation_prompts)
    else:
        reference_images = []

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.get("train_batch_size", 1),
        num_workers=0,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.get("gradient_accumulation_steps", 1))
    max_train_steps = args.get("num_train_epochs", 1) * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.get("lr_scheduler", "constant"),
        optimizer=optimizer,
        num_warmup_steps=args.get("lr_warmup_steps", 0) * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=1,
        power=1.0,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    print('Learning components loaded by accelerate ..............')

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.get("report_to", "") == "wandb":
            os.environ["WANDB_DISABLE_STATS"] = "true"
            wb_init = {}
            if args.get("wandb_run_name"):
                wb_init["name"] = args["wandb_run_name"]
            init_kwargs["wandb"] = wb_init
        accelerator.init_trackers(args.get("tracker_project_name", "train_lora"), config=args, init_kwargs=init_kwargs)

    logger.info("***** Running training *****")
    global_step = 0

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    progress_bar = tqdm(range(0, max_train_steps), desc="Steps", disable=not accelerator.is_local_main_process)

    # --- Step-0 WandB: log one real ground-truth image per validation pathology ---
    if accelerator.is_main_process and wandb.run is not None:
        sample_images = []
        for prompt, ref_path in zip(validation_prompts, reference_images):
            real_pil = _open_image_from_path(ref_path)
            if real_pil is not None:
                pathology = prompt.split("showing ")[-1]
                sample_images.append(wandb.Image(
                    real_pil.resize((512, 512), Image.BILINEAR),
                    caption=f"Ground Truth | {pathology}",
                ))
        if sample_images:
            wandb.log({"ground_truth_samples": sample_images}, step=0)

    # --- Step-0 baseline inference (base model, before any LoRA training) ---
    if accelerator.is_main_process and args.get("validation_embeds"):
        logger.info("Running step-0 baseline inference with base model...")
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            transformer=accelerator.unwrap_model(transformer),
            text_encoder=None, text_encoder_2=None, text_encoder_3=None,
            tokenizer=None, tokenizer_2=None, tokenizer_3=None,
            torch_dtype=weight_dtype,
        )
        pipeline.enable_model_cpu_offload(gpu_id=accelerator.local_process_index)
        log_validation(pipeline, args, accelerator, weight_dtype, step=0, is_final_validation=False, reference_images=reference_images)
        del pipeline
        free_memory()

    for epoch in range(args.get("num_train_epochs", 1)):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                if "model_input" in batch:
                    model_input = batch["model_input"].to(dtype=weight_dtype, device=accelerator.device)
                else:
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype, device=accelerator.device)
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                    model_input = model_input.to(dtype=weight_dtype)

                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.get("weighting_scheme", "logit_normal"),
                    batch_size=bsz,
                    logit_mean=args.get("logit_mean", 0.0),
                    logit_std=args.get("logit_std", 1.0),
                    mode_scale=args.get("mode_scale", 1.29),
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                prompt_embeds = batch["prompt_embeds"].to(dtype=weight_dtype, device=accelerator.device)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(dtype=weight_dtype, device=accelerator.device)

                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                if args.get("precondition_outputs", 1):
                    model_pred = model_pred * (-sigmas) + noisy_model_input
                    target = model_input
                else:
                    target = noise - model_input

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.get("weighting_scheme", "logit_normal"), sigmas=sigmas)

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_params, args.get("max_grad_norm", 1.0))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                do_save = args.get("save_intermediate_checkpoints", True) and global_step % args.get("checkpointing_steps", 500) == 0
                do_val = global_step % args.get("validation_steps", 250) == 0 and args.get("validation_prompt", None) is not None

                if do_save or do_val:
                    transformer.eval()
                    with gather_params_ctx(transformer, accelerator):
                        if accelerator.is_main_process:
                            if do_save:
                                save_path = os.path.join(args["output_dir"], f"checkpoint-{global_step}")
                                accelerator.unwrap_model(transformer).save_pretrained(save_path)
                                logger.info(f"Saved state to {save_path}")

                            if do_val:
                                pipeline = StableDiffusion3Pipeline.from_pretrained(
                                    model_path,
                                    transformer=accelerator.unwrap_model(transformer),
                                    text_encoder=None, text_encoder_2=None, text_encoder_3=None,
                                    tokenizer=None, tokenizer_2=None, tokenizer_3=None,
                                    torch_dtype=weight_dtype,
                                )
                                pipeline.enable_model_cpu_offload(gpu_id=accelerator.local_process_index)
                                log_validation(pipeline, args, accelerator, weight_dtype, global_step, reference_images=reference_images)
                                del pipeline
                                free_memory()
                    transformer.train()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= max_train_steps:
                break
        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    with gather_params_ctx(transformer, accelerator):
        if accelerator.is_main_process:
            accelerator.unwrap_model(transformer).save_pretrained(args["output_dir"])

    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    print(parser)
    script_args = parser.parse_args()

    with open(script_args.config, "r") as f:
        args_dict = json.load(f)

    main(args_dict)