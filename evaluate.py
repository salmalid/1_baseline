import argparse
import json
import re
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchmetrics.image.fid  import FrechetInceptionDistance
from tqdm.auto import tqdm

PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax",
]

_ZIP_RE = re.compile(r"^(.*\.zip)[/\\](.*)", re.IGNORECASE)
_BATCH  = 32


# ─────────────────────────────────────────────────────────────────────────────
# Image I/O
# ─────────────────────────────────────────────────────────────────────────────

def _open_image(path_str: str, size: int) -> np.ndarray | None:
    m = _ZIP_RE.match(path_str)
    try:
        if m:
            with zipfile.ZipFile(m.group(1)) as zf:
                with zf.open(m.group(2).replace("\\", "/")) as fh:
                    img = Image.open(fh).convert("RGB")
        else:
            img = Image.open(path_str).convert("RGB")
        return np.array(img.resize((size, size), Image.LANCZOS))
    except Exception as e:
        print(f"  [skip] {Path(path_str).name}: {e}")
        return None


def _resolve_local(path_str: str, images_dir: Path) -> Path | None:
    """Map a ZIP-embedded path to the local images directory by finding the patientXXX component."""
    parts = Path(path_str.replace("\\", "/")).parts
    for i, part in enumerate(parts):
        if part.startswith("patient"):
            candidate = images_dir.joinpath(*parts[i:])
            return candidate if candidate.exists() else None
    return None


def _slug_to_pathology(slug: str) -> str:
    return slug.replace("-", " ")


def _parse_gen_dir(gen_dir: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(gen_dir.glob("*.png")):
        parts = p.stem.split("_")
        if len(parts) >= 5:
            pathology = _slug_to_pathology("_".join(parts[4:]))
            if pathology in PATHOLOGIES:
                out[pathology].append(p)
    return out


def _load_real(
    captions_file: str,
    n: int,
    size: int,
    images_dir: Path | None = None,
) -> dict[str, list[np.ndarray]]:
    paths_by: dict[str, list[str]] = defaultdict(list)
    with open(captions_file, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            caption = r.get("caption", "")
            if "showing " not in caption:
                continue
            pathology = caption.split("showing ", 1)[1].strip()
            if pathology in PATHOLOGIES and len(paths_by[pathology]) < n:
                paths_by[pathology].append(r["path_to_image"])

    def _load_one(p: str) -> np.ndarray | None:
        if images_dir is not None:
            local = _resolve_local(p, images_dir)
            if local is not None:
                return _open_image(str(local), size)
        return _open_image(p, size)

    result: dict[str, list[np.ndarray]] = {}
    for pathology, paths in paths_by.items():
        imgs = [x for x in (_load_one(p) for p in tqdm(paths, desc=pathology, leave=False)) if x is not None]
        result[pathology] = imgs
        print(f"  {pathology}: {len(imgs)} real images")
    return result


# Tensor helpers

def _to_float(arr: np.ndarray) -> torch.Tensor:
    """HWC uint8 → 1CHW float32 [0, 1]  (for FID/sFID)."""
    return torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0).unsqueeze(0)


def _to_uint8(arr: np.ndarray) -> torch.Tensor:
    """HWC uint8 → 1CHW uint8  (for Medical FID with custom extractor)."""
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


# Medical FID — DenseNet121 custom feature extractor

class _MedExtractor(nn.Module):

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.mean.device).float() / 255.0
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        f = self.backbone.features(x)
        f = F.relu(f, inplace=True)
        return F.adaptive_avg_pool2d(f, 1).flatten(1)   # [B, 1024]


class _MedExtractorXRV(nn.Module):

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("_dev", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # uint8 [B, 3, H, W] → grayscale float [B, 1, H, W] in [-1024, 1024]
        x = x.to(self._dev.device).float() / 255.0
        x = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = x * 2048.0 - 1024.0
        f = self.model.features(x)
        f = F.relu(f, inplace=True)
        return F.adaptive_avg_pool2d(f, 1).flatten(1)   # [B, 1024]


def _build_med_extractor(backbone_path: str | None, device: str) -> nn.Module:
    import torchvision.models as tvm

    if backbone_path and backbone_path.startswith("xrv:"):
        try:
            import torchxrayvision as xrv  
        except ImportError:
            raise ImportError("pip install torchxrayvision")
        weights_key = backbone_path[4:]
        print(f"  Loading TorchXRayVision DenseNet121 ({weights_key}) ...")
        model = xrv.models.DenseNet(weights=weights_key)
        print(f"  Medical backbone: chest-X-ray DenseNet121 [{weights_key}]")
        return _MedExtractorXRV(model.eval()).to(device)

    # ── local .pth checkpoint ─────────────────────────────────────────────────
    if backbone_path and Path(backbone_path).exists():
        model = tvm.densenet121(weights=None)
        state = torch.load(backbone_path, map_location="cpu", weights_only=True)
        for key in ("model", "state_dict", "net"):
            if key in state:
                state = state[key]
                break
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Medical backbone loaded: missing={len(missing)}, unexpected={len(unexpected)}")
        return _MedExtractor(model.eval()).to(device)

    # ── fallback: ImageNet DenseNet121 ────────────────────────────────────────
    if backbone_path:
        warnings.warn(f"Not found: {backbone_path!r}; using ImageNet DenseNet121.")
    else:
        print("  No --medical-backbone provided → using ImageNet-pretrained DenseNet121.")
        print("  For a clinically meaningful Medical FID supply a chest-X-ray DenseNet121.")
        print("  Tip: --medical-backbone xrv:densenet121-res224-all  (pip install torchxrayvision)")
    model = tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT)
    return _MedExtractor(model.eval()).to(device)


def _fid_score(
    gen: list[np.ndarray],
    real: list[np.ndarray],
    feature: int | nn.Module,
    device: str,
    uint8: bool = False,
) -> float | None:
    """
    Compute FID between two image lists using torchmetrics.

    feature : 2048 → standard FID  |  768 → sFID (pooled Mixed_6e)
              nn.Module → Medical FID (pass _MedExtractor)
    uint8   : True when feature is a custom module (normalize=False path).
    """
    if len(gen) < 2 or len(real) < 2:
        return None

    metric = FrechetInceptionDistance(feature=feature, normalize=not uint8).to(device)
    to_t   = _to_uint8 if uint8 else _to_float

    with torch.no_grad():
        for arrays, is_real in ((real, True), (gen, False)):
            for i in range(0, len(arrays), _BATCH):
                batch = torch.cat([to_t(a) for a in arrays[i : i + _BATCH]]).to(device)
                metric.update(batch, real=is_real)

    score = float(metric.compute())
    del metric
    torch.cuda.empty_cache()
    return score


def _medical_clip(
    gen_by_pathology: dict[str, list[np.ndarray]],
    device: str,
    model_name: str,
) -> tuple[float | None, dict[str, float | None]]:
    """
    Mean cosine similarity between BiomedCLIP image embeddings and the
    corresponding text-prompt embedding for each generated image.
    """
    try:
        import open_clip  # type: ignore[import-untyped]
    except ImportError:
        print("  open_clip not installed; skipping Medical CLIP.  pip install open_clip_torch")
        return None, {}

    print(f"  Loading {model_name} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    scores_by: dict[str, float | None] = {}
    all_scores: list[float] = []

    with torch.no_grad():
        for pathology, arrays in gen_by_pathology.items():
            if not arrays:
                scores_by[pathology] = None
                continue
            tokens  = tokenizer([f"Chest X-ray of a patient showing {pathology}"]).to(device)
            txt_emb = F.normalize(model.encode_text(tokens), dim=-1)

            path_scores = [
                float((F.normalize(model.encode_image(preprocess(Image.fromarray(a)).unsqueeze(0).to(device)), dim=-1) * txt_emb).sum())
                for a in arrays
            ]
            scores_by[pathology] = float(np.mean(path_scores))
            all_scores.extend(path_scores)

    return (float(np.mean(all_scores)) if all_scores else None), scores_by


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",           default="config.json")
    parser.add_argument("--gen-dir",          required=True,
                        help="Directory of generated PNGs (e.g. infer_diverse/).")
    parser.add_argument("--captions",         default=None,
                        help="JSONL captions file. Defaults to config captions_file.")
    parser.add_argument("--images-dir",       default=None,
                        help="Local directory of real images (patientXXX/studyX/viewX.png). "
                             "Skips ZIP extraction when provided.")
    parser.add_argument("--size",             type=int, default=256,
                        help="Resize images to this square size before computing metrics.")
    parser.add_argument("--n-real",           type=int, default=2400,
                        help="Max real images in total (split evenly across pathologies).")
    parser.add_argument("--medical-backbone", default="xrv:densenet121-res224-all",
                        help="DenseNet121 .pth trained on chest X-rays, or xrv:<weights_key>.")
    parser.add_argument("--med-clip-model",
                        default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--skip-med-clip",    action="store_true")
    parser.add_argument("--out",              default="eval_results.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    captions_file = args.captions or config.get("captions_file", "data/all_pathologies.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_dir = Path(args.gen_dir)

    # ── load images ───────────────────────────────────────────────────────────
    print(f"\nScanning {gen_dir} ...")
    gen_arrays: dict[str, list[np.ndarray]] = {
        p: [np.array(Image.open(f).convert("RGB").resize((args.size, args.size), Image.LANCZOS)) for f in fs]
        for p, fs in _parse_gen_dir(gen_dir).items()
    }
    total_gen = sum(len(v) for v in gen_arrays.values())
    print(f"Found {total_gen} generated images across {len(gen_arrays)} pathologies.")
    if total_gen < 50:
        print("  ! FID scores will be noisy — generate ≥2048 images for reliable results.")

    images_dir = Path(args.images_dir) if args.images_dir else None
    if images_dir:
        print(f"  Using local images dir: {images_dir}")

    n_per_pathology = max(1, args.n_real // max(len(gen_arrays), 1))
    print(f"\nLoading ≤{n_per_pathology} real images per pathology (≤{args.n_real} total) ...")
    real_arrays = _load_real(captions_file, n_per_pathology, args.size, images_dir)

    all_gen  = [a for v in gen_arrays.values()  for a in v]
    all_real = [a for v in real_arrays.values() for a in v]

    # ── build medical extractor once (shared across all FID calls) ────────────
    print("\nBuilding Medical FID extractor ...")
    med_ext = _build_med_extractor(args.medical_backbone, device)

    # ── [1] global FID ────────────────────────────────────────────────────────
    print("\n[1/4] Global FID (InceptionV3 2048-dim) ...")
    fid_global = _fid_score(all_gen, all_real, feature=2048, device=device)
    print(f"  FID : {fid_global:.2f}" if fid_global is not None else "  FID : N/A")

    # ── [2] global sFID ───────────────────────────────────────────────────────
    print("\n[2/4] Global sFID (InceptionV3 Mixed_6e 768-dim pooled) ...")
    sfid_global = _fid_score(all_gen, all_real, feature=768, device=device)
    print(f"  sFID: {sfid_global:.2f}" if sfid_global is not None else "  sFID: N/A")

    # ── [3] global Medical FID ────────────────────────────────────────────────
    print("\n[3/4] Global Medical FID (DenseNet121 1024-dim) ...")
    mfid_global = _fid_score(all_gen, all_real, feature=med_ext, device=device, uint8=True)
    print(f"  mFID: {mfid_global:.2f}" if mfid_global is not None else "  mFID: N/A")


    per: dict = {}
    for pathology in PATHOLOGIES:
        g = gen_arrays.get(pathology, [])
        r = real_arrays.get(pathology, [])
        entry: dict = {
            "n_gen": len(g), "n_real": len(r),
            "fid": None, "sfid": None, "med_fid": None,
            "med_clip": None,
        }
        if g and r:
            entry["fid"]     = _fid_score(g, r, feature=2048,    device=device)
            entry["sfid"]    = _fid_score(g, r, feature=768,     device=device)
            entry["med_fid"] = _fid_score(g, r, feature=med_ext, device=device, uint8=True)

        per[pathology] = entry

    # ── [5] Medical CLIP ──────────────────────────────────────────────────────
    clip_global: float | None = None
    if not args.skip_med_clip:
        print(f"\n[4/4] Medical CLIP ({args.med_clip_model}) ...")
        clip_global, clip_by = _medical_clip(gen_arrays, device, args.med_clip_model)
        for p, s in clip_by.items():
            if p in per:
                per[p]["med_clip"] = s
        print(f"  Mean CLIP: {clip_global:.4f}" if clip_global is not None else "  CLIP: N/A")
    else:
        print("\n[4/4] Medical CLIP skipped.")

    # ── aggregate ─────────────────────────────────────────────────────────────
    def _mean(key: str) -> float | None:
        vals = [v[key] for v in per.values() if v[key] is not None]
        return float(np.mean(vals)) if vals else None

    # ── table ─────────────────────────────────────────────────────────────────
    W = 76
    print("\n" + "─" * W)
    print(f"{'Pathology':<30} {'FID':>7} {'sFID':>7} {'mFID':>7} {'CLIP':>7} {'n':>4}")
    print("─" * W)

    def _fs(v: dict, k: str, fmt: str) -> str:
        return (fmt % v[k]) if v[k] is not None else "—"

    for pathology in PATHOLOGIES:
        v = per[pathology]
        print(
            f"{pathology:<30}"
            f" {_fs(v,'fid',     '%7.2f')}"
            f" {_fs(v,'sfid',    '%7.2f')}"
            f" {_fs(v,'med_fid', '%7.2f')}"
            f" {_fs(v,'med_clip','%7.4f')}"
            f" {v['n_gen']:>4}"
        )

    def _s(val: float | None, fmt: str) -> str:
        return (fmt % val) if val is not None else "N/A"

    print("─" * W)
    print(f"{'Global':<30}"
          f" {_s(fid_global,  '%7.2f')} {_s(sfid_global, '%7.2f')} {_s(mfid_global, '%7.2f')}")
    print(f"{'Mean per-class':<30}"
          f" {_s(_mean('fid'),     '%7.2f')}"
          f" {_s(_mean('sfid'),    '%7.2f')}"
          f" {_s(_mean('med_fid'), '%7.2f')}"
          f" {_s(_mean('med_clip'),'%7.4f')}")
    print("─" * W)

    # ── save ──────────────────────────────────────────────────────────────────
    results = {
        "gen_dir":          str(gen_dir),
        "captions_file":    captions_file,
        "n_real_per_class": args.n_real,
        "fid_global":       fid_global,
        "sfid_global":      sfid_global,
        "med_fid_global":   mfid_global,
        "mean_fid":         _mean("fid"),
        "mean_sfid":        _mean("sfid"),
        "mean_med_fid":     _mean("med_fid"),
        "med_clip_global":  clip_global,
        "per_pathology":    per,
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()