"""
mriup.py
Usage: python mriup.py <folder_containing_FrontLD>

Expects FrontLD/ with PNG slices named in sorted order.
Produces: TopLD/ RightLD/ FrontHD/ TopHD/ RightHD/ FrontMod2LD/ FrontMod2HD/ FrontCompleteHD/
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from reales4u2d import load_model, reales4u2d

TARGET_DEPTH = 256
SLICE_SIZE   = 256

# ── device detection ──────────────────────────────────────────────────────────


# ── helpers ───────────────────────────────────────────────────────────────────

def ensure(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)

def load_slices(folder: Path) -> list[Image.Image]:
    files = sorted(folder.glob("*.png"))
    return [Image.open(f).convert("L") for f in files]

def save_slice(img: Image.Image, folder: Path, name: str):
    img.save(folder / name)

def pad_to_depth(slices: list[Image.Image], depth: int) -> list[Image.Image]:
    n = len(slices)
    if n >= depth:
        return slices[:depth]
    pad_total  = depth - n
    pad_before = pad_total // 2
    pad_after  = pad_total - pad_before
    black = Image.new("L", (SLICE_SIZE, SLICE_SIZE), 0)
    return [black] * pad_before + slices + [black] * pad_after

def slices_to_volume(slices: list[Image.Image]) -> np.ndarray:
    return np.stack([np.array(s, dtype=np.float32) for s in slices], axis=0)

def volume_to_image(arr2d: np.ndarray) -> Image.Image:
    return Image.fromarray(arr2d.clip(0, 255).astype(np.uint8), mode="L")

def is_black_image(img: Image.Image) -> bool:
    return np.array(img).sum() == 0

def upscale_folder(src: Path, dst: Path):
    """Upscale per-image, skip already done."""
    ensure(dst)
    files = sorted(src.glob("*.png"))
    for f in files:
        out = dst / f.name
        if out.exists():
            continue
        img = Image.open(f).convert("L")
        if is_black_image(img):
            Image.new("L", (SLICE_SIZE * 2, SLICE_SIZE * 2), 0).save(out)
        else:
            reales4u2d(img).save(out)
    print(f"  {dst.name}: done ({len(files)} images)")

# ── axis mapping ──────────────────────────────────────────────────────────────
# vol[Z, Y, X]
# FrontLD[Z] → vol[Z, :, :]   X=horizontal, Y=vertical
# TopLD[Y]   → vol[:, Y, :]   X=horizontal, Z=vertical
# RightLD[X] → vol[:, :, X]   Y=horizontal, Z=vertical
#
# After 2x upscale, original slice Z maps to pixel-Z = Z*2
# Interleaved slice between Z=n and Z=n+1 → pixel-Z = n*2+1

# ── pipeline steps ────────────────────────────────────────────────────────────

def step_generate_top_right(base: Path, vol: np.ndarray):
    top_dir   = base / "TopLD"
    right_dir = base / "RightLD"
    Z, Y, X   = vol.shape

    if not top_dir.exists():
        ensure(top_dir)
        for y in range(Y):
            volume_to_image(vol[:, y, :]).save(top_dir / f"{y:04d}.png")
        print(f"TopLD generated: {Y} slices")
    else:
        print("TopLD exists, skipping")

    if not right_dir.exists():
        ensure(right_dir)
        for x in range(X):
            volume_to_image(vol[:, :, x]).save(right_dir / f"{x:04d}.png")
        print(f"RightLD generated: {X} slices")
    else:
        print("RightLD exists, skipping")


def step_upscale_all(base: Path):
    for name in ["FrontLD", "TopLD", "RightLD"]:
        src = base / name
        dst = base / name.replace("LD", "HD")
        print(f"Upscaling {name} → {dst.name} ...")
        upscale_folder(src, dst)


def step_build_frontmod2ld(base: Path):
    out_dir = base / "FrontMod2LD"
    ensure(out_dir)

    top_files   = sorted((base / "TopHD").glob("*.png"))
    right_files = sorted((base / "RightHD").glob("*.png"))

    print("Loading TopHD and RightHD into memory...")
    top_hd   = [np.array(Image.open(f).convert("L"), dtype=np.float32) for f in top_files]
    right_hd = [np.array(Image.open(f).convert("L"), dtype=np.float32) for f in right_files]

    num_Y = len(top_hd)
    num_X = len(right_hd)
    num_interleaved = TARGET_DEPTH - 1

    for n in range(num_interleaved):
        out_path = out_dir / f"{n:04d}.png"
        if out_path.exists():
            continue

        pz = n * 2 + 1

        # TopHD[Y][pz, x*2] → top_contrib[Y, X]
        top_contrib = np.zeros((num_Y, num_X), dtype=np.float32)
        for y in range(num_Y):
            top_contrib[y, :] = top_hd[y][pz, 0::2]

        # RightHD[X][pz, y*2] → right_contrib[Y, X]
        right_contrib = np.zeros((num_Y, num_X), dtype=np.float32)
        for x in range(num_X):
            right_contrib[:, x] = right_hd[x][pz, 0::2]

        volume_to_image((top_contrib + right_contrib) / 2.0).save(out_path)

    print(f"FrontMod2LD: {num_interleaved} slices")


def step_upscale_frontmod2(base: Path):
    print("Upscaling FrontMod2LD → FrontMod2HD ...")
    upscale_folder(base / "FrontMod2LD", base / "FrontMod2HD")


def step_combine(base: Path):
    out_dir = base / "FrontCompleteHD"
    ensure(out_dir)

    front_hd_files  = sorted((base / "FrontHD").glob("*.png"))
    frontmod2_files = sorted((base / "FrontMod2HD").glob("*.png"))

    seq_idx = 0
    for i, fhd in enumerate(front_hd_files):
        dst = out_dir / f"{seq_idx:04d}_FrontHD_{i}.png"
        if not dst.exists():
            Image.open(fhd).save(dst)
        seq_idx += 1

        if i < len(frontmod2_files):
            dst = out_dir / f"{seq_idx:04d}_FrontMod2HD_{i}.png"
            if not dst.exists():
                Image.open(frontmod2_files[i]).save(dst)
            seq_idx += 1

    print(f"FrontCompleteHD: {seq_idx} slices total")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python mriup.py <folder>")
        sys.exit(1)

    base = Path(sys.argv[1])
    front_ld = base / "FrontLD"
    if not front_ld.exists():
        print(f"FrontLD not found in: {base}")
        sys.exit(1)

    print("Loading upscale model...")
    load_model()

    print(f"Loading FrontLD from {front_ld} ...")
    raw_slices = load_slices(front_ld)
    print(f"  Found {len(raw_slices)} slices, padding to {TARGET_DEPTH}")
    if raw_slices:
        print(f"  Slice size: {raw_slices[0].size}")

    padded = pad_to_depth(raw_slices, TARGET_DEPTH)
    vol    = slices_to_volume(padded)
    print(f"  Volume shape: {vol.shape}")

    # Save padded FrontLD if needed
    if len(raw_slices) != TARGET_DEPTH:
        existing = sorted(front_ld.glob("*.png"))
        if len(existing) != TARGET_DEPTH:
            print("Saving padded FrontLD slices...")
            for i, s in enumerate(padded):
                s.save(front_ld / f"{i:04d}.png")

    step_generate_top_right(base, vol)
    step_upscale_all(base)
    step_build_frontmod2ld(base)
    step_upscale_frontmod2(base)
    step_combine(base)

    print("Done.")

if __name__ == "__main__":
    main()
