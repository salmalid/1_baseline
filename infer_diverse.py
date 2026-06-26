import argparse
import json
import os
from pathlib import Path

import torch
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from peft import LoraConfig
from safetensors.torch import load_file
from tqdm.auto import tqdm

PATHOLOGIES = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture",
]

DEFAULT_GUIDANCE = [3.0, 5.0, 7.0, 9.0]
DEFAULT_STEPS = 28
SEED = 42


def load_pipeline(model_path, lora_path, dtype):
    transformer = SD3Transformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype
    )
    lora_config = LoraConfig(
        r=32, lora_alpha=32,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(lora_config)

    lora_pt = Path(lora_path) / "pytorch_lora_weights.safetensors"
    if lora_pt.exists():
        state = torch.load(lora_pt, map_location="cpu", weights_only=True)
    else:
        candidates = list(Path(lora_path).glob("*.safetensors"))
        state = load_file(candidates[0]) if candidates else None

    if state:
        missing, unexpected = transformer.load_state_dict(state, strict=False)
        print(f"  LoRA: {len(state)} tensors | missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("  WARNING: no safetensors found in", lora_path)

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_path, transformer=transformer,
        text_encoder=None, text_encoder_2=None, text_encoder_3=None,
        tokenizer=None, tokenizer_2=None, tokenizer_3=None,
        torch_dtype=dtype,
    )
    pipeline.enable_model_cpu_offload(gpu_id=0)
    pipeline.enable_attention_slicing()
    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="config.json")
    parser.add_argument("--checkpoint",   type=int, default=None,
                        help="Checkpoint step (e.g. 7500). Omit for final weights.")
    parser.add_argument("--guidance",     type=float, nargs="+", default=None,
                        help="One or more CFG scales. Defaults to DEFAULT_GUIDANCE list.")
    parser.add_argument("--steps",        type=int,   default=None)
    parser.add_argument("--height",       type=int,   default=None)
    parser.add_argument("--width",        type=int,   default=None)
    parser.add_argument("--val-embeds",   default=None,
                        help="Directory of precomputed val embeds. "
                             "Defaults to validation_embeds_dir from config.")
    parser.add_argument("--out-dir",      default="cfg_ablation")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    model_path = config["pretrained_model_name_or_path"]
    output_dir = config["output_dir"]
    cfg_scales = args.guidance if args.guidance else DEFAULT_GUIDANCE
    steps      = args.steps  or config.get("num_inference_steps", DEFAULT_STEPS)
    height     = args.height or config.get("generate_height", 512)
    width      = args.width  or config.get("generate_width",  512)
    seed       = config.get("seed", SEED)

    lora_path = (os.path.join(output_dir, f"checkpoint-{args.checkpoint}")
                 if args.checkpoint else output_dir)
    print(f"LoRA path  : {lora_path}")
    print(f"CFG scales : {cfg_scales}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16

    # PHASE 1 — load precomputed val embeds, pick one per pathology
    val_embeds_dir = args.val_embeds or config.get("validation_embeds_dir")
    if not val_embeds_dir:
        raise ValueError("Provide --val-embeds or set validation_embeds_dir in config.json")

    val_dir = Path(val_embeds_dir)
    print(f"\nLoading precomputed embeddings from {val_dir} ...")

    neg_data = torch.load(val_dir / "negative_embeds.pt", map_location="cpu", weights_only=True)
    neg_pe   = neg_data["prompt_embeds"]
    neg_pp   = neg_data["pooled_prompt_embeds"]

    pathology_embeds = {}
    for pt_file in sorted(val_dir.glob("val_*_embeds.pt")):
        data   = torch.load(pt_file, map_location="cpu", weights_only=True)
        prompt = data["prompt"]
        if "showing " not in prompt:
            continue
        showing = prompt.split("showing ", 1)[1].split(". ", 1)[0].rstrip(".")
        for p in PATHOLOGIES:
            if p not in pathology_embeds and p in showing:
                pathology_embeds[p] = (data["prompt_embeds"], data["pooled_prompt_embeds"])

    missing = [p for p in PATHOLOGIES if p not in pathology_embeds]
    if missing:
        print(f"  WARNING: no val embed found for: {missing}")
    print(f"  Found embeddings for {len(pathology_embeds)}/{len(PATHOLOGIES)} pathologies")

    # PHASE 2 — load pipeline + LoRA, generate 1 image per pathology per CFG
    print("\nLoading pipeline + LoRA...")
    pipeline = load_pipeline(model_path, lora_path, dtype)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    ckpt_tag = f"ckpt{args.checkpoint}" if args.checkpoint else "final"

    total = len(pathology_embeds) * len(cfg_scales)
    print(f"\nGenerating {len(pathology_embeds)} pathologies × {len(cfg_scales)} CFG scales = {total} images → {out_dir}/\n")

    with tqdm(total=total, desc="Generating") as pbar:
        i = 0
        for pathology, (pe, pp) in pathology_embeds.items():
            slug = pathology.replace(" ", "-")
            for cfg in cfg_scales:
                fname     = out_dir / f"{ckpt_tag}_cfg{cfg}_{slug}.png"
                generator = torch.Generator(device=device).manual_seed(seed + i)

                with torch.autocast("cuda", dtype=dtype):
                    image = pipeline(
                        prompt_embeds=pe.to(device=device, dtype=dtype),
                        pooled_prompt_embeds=pp.to(device=device, dtype=dtype),
                        negative_prompt_embeds=neg_pe.to(device=device, dtype=dtype),
                        negative_pooled_prompt_embeds=neg_pp.to(device=device, dtype=dtype),
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        generator=generator,
                        height=height,
                        width=width,
                    ).images[0]

                image.save(fname)
                pbar.set_postfix(pathology=pathology[:20], cfg=cfg)
                pbar.update(1)
                i += 1

    print(f"\nDone. {total} images saved to ./{args.out_dir}/")


if __name__ == "__main__":
    main()