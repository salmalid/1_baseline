import argparse
import io
import json
import os
import re
import zipfile
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

_ZIP_RE = re.compile(r"^(.*\.zip)[/\\](.*)", re.IGNORECASE)


def _parse_path(path_str: str):
    m = _ZIP_RE.match(path_str)
    if m:
        return m.group(1), m.group(2).replace("\\", "/")
    return None, path_str


def _source_exists(path_str: str) -> bool:
    zip_file, _ = _parse_path(path_str)
    return Path(zip_file if zip_file else path_str).exists()


def _path_key_from_str(path_str: str) -> Path:
    """Extract train/patientXXX/studyY/filename.png from any path string."""
    parts = [p for p in path_str.replace("\\", "/").split("/") if p and p != "."]
    for i, part in enumerate(parts):
        if part == "train":
            return Path(*parts[i:])
    return Path(parts[-1] if parts else path_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--captions", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    model_path = config.get("pretrained_model_name_or_path", "stabilityai/stable-diffusion-3.5-medium")
    resolution = int(config.get("resolution", 512))
    suffix = config.get("latent_cache_suffix", f"_latents_{resolution}.pt")
    latent_dir = Path(config["latent_cache_dir"]) if config.get("latent_cache_dir") else None
    if latent_dir is not None:
        latent_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    captions_file = args.captions or config.get("captions_file", "")

    if captions_file:
        seen = set()
        path_strs = []
        with open(captions_file, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ps = json.loads(line)["path_to_image"]
                if ps in seen:
                    continue
                seen.add(ps)
                if _source_exists(ps):
                    path_strs.append(ps)
                else:
                    zip_file, _ = _parse_path(ps)
                    print(f"Warning: not found: {zip_file or ps}")
        path_strs = sorted(path_strs)
        print(f"Found {len(path_strs)} images from captions file. Encoding with VAE...")
    else:
        data_dir = Path(config.get("train_data_dir", "."))
        path_strs = [
            str(Path(root) / file)
            for root, _, files in os.walk(data_dir)
            for file in files
            if file.lower().endswith(".png")
        ]
        path_strs = sorted(path_strs)
        print(f"Found {len(path_strs)} images. Encoding with VAE...")

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.bfloat16)
    vae.to(device)
    vae.eval()

    image_transforms = transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Keep ZIPs open across iterations to avoid repeated open/close overhead
    open_zips: dict[str, zipfile.ZipFile] = {}

    try:
        with torch.no_grad():
            for path_str in tqdm(path_strs, desc="Caching latents"):
                rel = _path_key_from_str(path_str)
                if latent_dir is not None:
                    latent_path = latent_dir / rel.with_name(rel.stem + suffix)
                    latent_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    latent_path = Path(rel.stem + suffix)
                if latent_path.exists() and not args.overwrite:
                    continue

                try:
                    zip_file, internal = _parse_path(path_str)
                    if zip_file:
                        if zip_file not in open_zips:
                            open_zips[zip_file] = zipfile.ZipFile(zip_file, "r")
                        with open_zips[zip_file].open(internal) as fh:
                            img = Image.open(io.BytesIO(fh.read()))
                            img.load()
                            img = img.convert("RGB")
                    else:
                        img = Image.open(path_str).convert("RGB")
                        img.load()
                except Exception as e:
                    print(f"Warning: skipping {path_str}: {e}")
                    continue

                pixel_values = image_transforms(img).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
                latent = vae.encode(pixel_values).latent_dist.sample()
                latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
                torch.save({"model_input": latent.cpu()}, latent_path)
    finally:
        for zf in open_zips.values():
            zf.close()

    dest = latent_dir if latent_dir else "current directory"
    print(f"Done. Latents saved with suffix '{suffix}' in '{dest}'.")


if __name__ == "__main__":
    main()