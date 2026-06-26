import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast


def _encode_prompt_with_t5(text_encoder, tokenizer, max_sequence_length, prompt, num_images_per_prompt=1, device=None):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(len(prompt) * num_images_per_prompt, seq_len, -1)
    return prompt_embeds


def _encode_prompt_with_clip(text_encoder, tokenizer, prompt, device=None, num_images_per_prompt=1):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(len(prompt) * num_images_per_prompt, seq_len, -1)
    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length, device=None, num_images_per_prompt=1):
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
                device=device,
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


def _path_key(abs_path: Path) -> Path:
    """Extract the train/patientXXX/studyY/filename portion from an absolute data path."""
    parts = abs_path.parts
    for i, part in enumerate(parts):
        if part == "train":
            return Path(*parts[i:])
    return Path(abs_path.name)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision, subfolder="text_encoder"):
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
    text_encoder_one = class_one.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    text_encoder_two = class_two.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
    text_encoder_three = class_three.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_3", torch_dtype=torch.bfloat16)
    return text_encoder_one, text_encoder_two, text_encoder_three


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--captions", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--suffix", type=str, default="_embeds_v2.pt")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_path = config.get("pretrained_model_name_or_path", "stabilityai/stable-diffusion-3.5-medium")
    embed_dir = Path(config.get("embed_dir", "filtered_dataset/hiddenstate"))
    captions_file = args.captions or config.get("captions_file", "filtered_dataset/captions_filtered.json")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_t5 = "cuda:1" if torch.cuda.device_count() > 1 else device
    max_sequence_length = int(config.get("max_sequence_length", 256))

    entries = []
    with open(captions_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    print(f"Found {len(entries)} caption entries in {captions_file}.")
    print(f"Embeddings will be saved under: {embed_dir}")

    print("Loading text encoders and tokenizers...")
    tokenizer_one = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")
    tokenizer_three = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_3")

    text_encoder_cls_one = import_model_class_from_model_name_or_path(model_path, None)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(model_path, None, subfolder="text_encoder_2")
    text_encoder_cls_three = import_model_class_from_model_name_or_path(model_path, None, subfolder="text_encoder_3")

    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three, model_path,
    )

    text_encoder_one.to(device)
    text_encoder_two.to(device)
    text_encoder_three.to(device_t5)

    print(f"Computing embeddings with max_sequence_length={max_sequence_length}...")

    for entry in tqdm(entries, desc="Computing Embeddings"):
        img_path = Path(entry["path_to_image"])
        rel = _path_key(img_path)
        embeds_path = embed_dir / rel.parent / (rel.stem + args.suffix)

        if embeds_path.exists() and not args.overwrite:
            continue

        embeds_path.parent.mkdir(parents=True, exist_ok=True)
        if "caption" not in entry:
            print(f"Warning: no caption for {entry.get('path_to_image', '?')}, skipping.")
            continue
        prompt = entry["caption"]

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

    print("Done computing embeddings.")


if __name__ == "__main__":
    main()