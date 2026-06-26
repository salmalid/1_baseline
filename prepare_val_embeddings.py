import argparse
import gc
import json
from pathlib import Path
import torch
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds


def _encode_prompt_with_clip(text_encoder, tokenizer, prompt: str, device=None, num_images_per_prompt: int = 1):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(text_encoders, tokenizers, prompt: str, max_sequence_length, device=None, num_images_per_prompt: int = 1):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []

    with torch.no_grad():
        for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
            prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device if device is not None else text_encoder.device,
                num_images_per_prompt=num_images_per_prompt,
            )
            clip_prompt_embeds_list.append(prompt_embeds)
            clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

        clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

        t5_prompt_embed = _encode_prompt_with_t5(
            text_encoders[-1],
            tokenizers[-1],
            max_sequence_length,
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=text_encoders[-1].device,
        )

        t5_prompt_embed = t5_prompt_embed.to(clip_prompt_embeds.device)
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds,
            (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str,
    revision: str | None,
    subfolder: str = "text_encoder",
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    if model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    raise ValueError(f"{model_class} is not supported.")


def load_text_encoders(class_one, class_two, class_three, pretrained_model_name_or_path):
    text_encoder_one = class_one.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
    )
    text_encoder_two = class_two.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.bfloat16,
    )
    text_encoder_three = class_three.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder_3",
        torch_dtype=torch.bfloat16,
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


def build_validation_prompts(data_dir: Path, per_pathology: int) -> list[str]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Training data directory not found: {data_dir}")

    prompts = []
    pathology_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if not pathology_dirs:
        pathology_dirs = [data_dir]

    for pathology_dir in pathology_dirs:
        text_paths = sorted(pathology_dir.glob("*.txt"))
        pathology_prompts = []
        for text_path in text_paths:
            prompt = text_path.read_text(encoding="utf-8", errors="replace").strip()
            if prompt:
                pathology_prompts.append(prompt)
            if len(pathology_prompts) >= per_pathology:
                break

        if len(pathology_prompts) < per_pathology:
            raise ValueError(
                f"Pathology '{pathology_dir.name}' only has {len(pathology_prompts)} caption prompts, need {per_pathology}."
            )

        prompts.extend(pathology_prompts)

    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--per-pathology", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    validation_prompts = config.get("validation_prompt", [])
    if isinstance(validation_prompts, str):
        validation_prompts = [validation_prompts]
    if not validation_prompts:
        raise ValueError("No 'validation_prompt' entries found in config.json. Add prompts there first.")

    model_path = config.get("pretrained_model_name_or_path", "stabilityai/stable-diffusion-3.5-medium")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_t5 = "cuda:1" if torch.cuda.device_count() > 1 else device
    max_sequence_length = int(config.get("max_sequence_length", 256))

    print("Loading text encoders and tokenizers...")
    tokenizer_one = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")
    tokenizer_three = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_3")

    text_encoder_cls_one = import_model_class_from_model_name_or_path(model_path, None)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(model_path, None, subfolder="text_encoder_2")
    text_encoder_cls_three = import_model_class_from_model_name_or_path(model_path, None, subfolder="text_encoder_3")

    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one,
        text_encoder_cls_two,
        text_encoder_cls_three,
        model_path,
    )

    text_encoder_one.to(device)
    text_encoder_two.to(device)
    text_encoder_three.to(device_t5)

    print(f"Precomputing embeddings for {len(validation_prompts)} prompts with max_sequence_length={max_sequence_length}...")

    validation_dir = Path(args.output_dir) if args.output_dir else Path(config.get("validation_embeds_dir", "validation_embeds"))
    validation_dir.mkdir(exist_ok=True)
    embed_paths = []

    for idx, prompt in enumerate(tqdm(validation_prompts, desc="Validation Prompts")):
        embeds_path = validation_dir / f"val_{idx}_embeds.pt"
        embed_paths.append(embeds_path.as_posix())

        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            [text_encoder_one, text_encoder_two, text_encoder_three],
            [tokenizer_one, tokenizer_two, tokenizer_three],
            prompt,
            max_sequence_length,
            device=device,
        )

        torch.save(
            {
                "prompt_embeds": prompt_embeds.cpu().clone(),
                "pooled_prompt_embeds": pooled_prompt_embeds.cpu().clone(),
                "prompt": prompt,
            },
            embeds_path,
        )

    print("Computing negative embeddings...")
    neg_embeds_path = validation_dir / "negative_embeds.pt"
    neg_prompt_embeds, neg_pooled_prompt_embeds = encode_prompt(
        [text_encoder_one, text_encoder_two, text_encoder_three],
        [tokenizer_one, tokenizer_two, tokenizer_three],
        "",
        max_sequence_length,
        device=device,
    )
    torch.save(
        {
            "prompt_embeds": neg_prompt_embeds.cpu().clone(),
            "pooled_prompt_embeds": neg_pooled_prompt_embeds.cpu().clone(),
            "prompt": "",
        },
        neg_embeds_path,
    )

    config["validation_embeds"] = embed_paths
    config["negative_validation_embed"] = neg_embeds_path.as_posix()
    config["validation_embeds_dir"] = validation_dir.as_posix()
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4)

    print("Validation embeddings ready and config.json updated.")

    del text_encoder_one, text_encoder_two, text_encoder_three
    del tokenizer_one, tokenizer_two, tokenizer_three
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()