import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from peft import LoraConfig
from safetensors.torch import load_file
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

PATHOLOGIES = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture",
]

DEFAULT_GUIDANCE = 3.0
DEFAULT_STEPS = 28
SEED = 42


def _import_encoder_class(model_path, subfolder="text_encoder"):
    cfg = PretrainedConfig.from_pretrained(model_path, subfolder=subfolder)
    cls = cfg.architectures[0]
    if cls == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    if cls == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    raise ValueError(cls)


def _encode_clip(encoder, tokenizer, prompt, device):
    inputs = tokenizer(prompt, padding="max_length", max_length=77,
                       truncation=True, return_tensors="pt")
    out = encoder(inputs.input_ids.to(device), output_hidden_states=True)
    return out.hidden_states[-2], out[0]


def _encode_t5(encoder, tokenizer, prompt, max_seq_len, device):
    inputs = tokenizer(prompt, padding="max_length", max_length=max_seq_len,
                       truncation=True, add_special_tokens=True, return_tensors="pt")
    return encoder(inputs.input_ids.to(device))[0]


def encode_prompt(encoders, tokenizers, prompt, max_seq_len, device):
    with torch.no_grad():
        clip1_emb, pool1 = _encode_clip(encoders[0], tokenizers[0], prompt, device)
        clip2_emb, pool2 = _encode_clip(encoders[1], tokenizers[1], prompt, device)
        t5_emb = _encode_t5(encoders[2], tokenizers[2], prompt, max_seq_len, device)

        clip_emb = torch.cat([clip1_emb, clip2_emb], dim=-1)
        pooled   = torch.cat([pool1, pool2], dim=-1)
        clip_emb = torch.nn.functional.pad(
            clip_emb, (0, t5_emb.shape[-1] - clip_emb.shape[-1])
        )
        prompt_embeds = torch.cat([clip_emb, t5_emb], dim=-2)

    return prompt_embeds, pooled


def sample_prompts(captions_file, n_singles, n_pairs, n_triples, n_quads, rng):
    by_combo = defaultdict(list)
    with open(captions_file, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cap = r["caption"]
            if "showing " not in cap:
                continue
            showing = cap.split("showing ", 1)[1].split(". ", 1)[0].rstrip(".")
            active = tuple(p for p in PATHOLOGIES if p in showing)
            if active:
                by_combo[active].append(cap)

    def pick(n, size):
        combos = [c for c in by_combo if len(c) == size]
        if not combos:
            return []
        rng.shuffle(combos)
        chosen = []
        idx = 0
        while len(chosen) < n:
            combo = combos[idx % len(combos)]
            chosen.append(rng.choice(by_combo[combo]))
            idx += 1
        return chosen

    return pick(n_singles, 1) + pick(n_pairs, 2) + pick(n_triples, 3) + pick(n_quads, 4)


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
    parser.add_argument("--guidance",     type=float, default=None)
    parser.add_argument("--steps",        type=int,   default=None)
    parser.add_argument("--height",       type=int,   default=None)
    parser.add_argument("--width",        type=int,   default=None)
    parser.add_argument("--n-singles",    type=int,   default=2400,
                        help="Number of single-condition images to generate.")
    parser.add_argument("--n-pairs",      type=int,   default=0,
                        help="Number of 2-condition images to generate.")
    parser.add_argument("--n-triples",    type=int,   default=0,
                        help="Number of 3-condition images to generate.")
    parser.add_argument("--n-quads",      type=int,   default=0,
                        help="Number of 4-condition images to generate.")
    parser.add_argument("--prompts-file", default=None,
                        help="JSON list of prompt strings; skips random sampling.")
    parser.add_argument("--val-embeds",   default=None,
                        help="Directory of precomputed val embeds. "
                             "Cycles through them to reach the requested count.")
    parser.add_argument("--out-dir",      default="infer_eval")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    model_path  = config["pretrained_model_name_or_path"]
    output_dir  = config["output_dir"]
    max_seq_len = int(config.get("max_sequence_length", 256))
    guidance    = args.guidance if args.guidance is not None else config.get("guidance_scale", DEFAULT_GUIDANCE)
    steps       = args.steps    if args.steps    is not None else config.get("num_inference_steps", DEFAULT_STEPS)
    height      = args.height   or config.get("generate_height", 512)
    width       = args.width    or config.get("generate_width",  512)
    neg_text    = config.get("negative_prompt", "")
    seed        = config.get("seed", SEED)

    lora_path = (os.path.join(output_dir, f"checkpoint-{args.checkpoint}")
                 if args.checkpoint else output_dir)
    print(f"LoRA path : {lora_path}")

    device    = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_t5 = "cuda:1" if torch.cuda.device_count() > 1 else device
    dtype     = torch.bfloat16

    n_total = args.n_singles + args.n_pairs + args.n_triples + args.n_quads

    # -----------------------------------------------------------------------
    # PHASE 1 — obtain prompt embeddings
    # -----------------------------------------------------------------------
    val_embeds_dir = args.val_embeds or config.get("validation_embeds_dir")

    if val_embeds_dir:
        val_dir = Path(val_embeds_dir)
        print(f"\nLoading precomputed val embeddings from {val_dir} ...")
        neg_data = torch.load(val_dir / "negative_embeds.pt", map_location="cpu", weights_only=True)
        neg_pe = neg_data["prompt_embeds"]
        neg_pp = neg_data["pooled_prompt_embeds"]

        base_prompts, base_embeds = [], []
        for pt_file in sorted(val_dir.glob("val_*_embeds.pt")):
            data = torch.load(pt_file, map_location="cpu", weights_only=True)
            base_prompts.append(data["prompt"])
            base_embeds.append((data["prompt_embeds"], data["pooled_prompt_embeds"]))
        print(f"Loaded {len(base_prompts)} val embeddings")

        # Cycle through embeddings to reach requested count
        prompts    = [base_prompts[i % len(base_prompts)] for i in range(n_total)]
        all_embeds = [base_embeds[i  % len(base_embeds)]  for i in range(n_total)]
        print(f"Cycling to {n_total} images (each prompt repeated ~{n_total // len(base_prompts)}x with different seeds)")

    else:
        if args.prompts_file:
            with open(args.prompts_file, encoding="utf-8") as f:
                prompts = json.load(f)
            print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
        else:
            rng = random.Random(seed)
            captions_file = config.get("captions_file", "dataset/train_captions_filtered.json")
            prompts = sample_prompts(
                captions_file,
                args.n_singles, args.n_pairs, args.n_triples, args.n_quads,
                rng,
            )
            print(f"Sampled {len(prompts)} prompts  "
                  f"({args.n_singles} singles / {args.n_pairs} pairs / "
                  f"{args.n_triples} triples / {args.n_quads} quads)")

        print("\nLoading text encoders...")
        tok1 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        tok2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")
        tok3 = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_3")

        enc1 = _import_encoder_class(model_path, "text_encoder").from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=dtype).to(device)
        enc2 = _import_encoder_class(model_path, "text_encoder_2").from_pretrained(
            model_path, subfolder="text_encoder_2", torch_dtype=dtype).to(device)
        enc3 = _import_encoder_class(model_path, "text_encoder_3").from_pretrained(
            model_path, subfolder="text_encoder_3", torch_dtype=dtype).to(device_t5)

        encoders   = [enc1, enc2, enc3]
        tokenizers = [tok1, tok2, tok3]

        print("Encoding prompts...")
        neg_pe, neg_pp = encode_prompt(encoders, tokenizers, neg_text, max_seq_len, device)
        neg_pe, neg_pp = neg_pe.cpu(), neg_pp.cpu()

        all_embeds = []
        for prompt in tqdm(prompts, desc="Encoding"):
            pe, pp = encode_prompt(encoders, tokenizers, prompt, max_seq_len, device)
            all_embeds.append((pe.cpu(), pp.cpu()))

        del enc1, enc2, enc3, encoders
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # PHASE 2 — load pipeline + LoRA, generate
    # -----------------------------------------------------------------------
    print("\nLoading pipeline + LoRA...")
    pipeline = load_pipeline(model_path, lora_path, dtype)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    ckpt_tag = f"ckpt{args.checkpoint}" if args.checkpoint else "final"

    print(f"\nGenerating {len(prompts)} images → {out_dir}/\n")
    for i, (prompt, (pe, pp)) in enumerate(
        tqdm(zip(prompts, all_embeds), total=len(prompts), desc="Generating")
    ):
        showing = prompt.split("showing ", 1)[1].split(". ", 1)[0].rstrip(".") if "showing " in prompt else prompt[:60]
        slug    = showing.replace(", ", "_").replace(" ", "-").replace("/", "-")
        slug    = "".join(c for c in slug if c not in r':*?"<>|\\')[:60]
        fname   = out_dir / f"{ckpt_tag}_gs{guidance}_s{steps}_{i:04d}_{slug}.png"

        generator = torch.Generator(device=device).manual_seed(seed + i)

        with torch.autocast("cuda", dtype=dtype):
            image = pipeline(
                prompt_embeds=pe.to(device=device, dtype=dtype),
                pooled_prompt_embeds=pp.to(device=device, dtype=dtype),
                negative_prompt_embeds=neg_pe.to(device=device, dtype=dtype),
                negative_pooled_prompt_embeds=neg_pp.to(device=device, dtype=dtype),
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                height=height,
                width=width,
            ).images[0]

        image.save(fname)
        print(f"  [{i:04d}] {showing[:72]}")

    print(f"\nDone. {len(prompts)} images saved to ./{args.out_dir}/")


if __name__ == "__main__":
    main()
