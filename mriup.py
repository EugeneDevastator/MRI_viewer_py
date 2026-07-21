"""
mri_upscale_pipeline.py
Usage: python mri_upscale_pipeline.py <FrontLD_folder>

Expects FrontLD/ with PNG slices named in sorted order.
Produces: TopLD/ RightLD/ FrontHD/ TopHD/ RightHD/ FrontMod2LD/ FrontMod2HD/ FrontCompleteHD/
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

# ── importer ──────────────────────────────────────────────────────────────────
from reales4u2d import load_model, reales4u2d

TARGET_DEPTH = 256
SLICE_SIZE   = 256  # assumed square

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
    pad_total = depth - n
    pad_before = pad_total // 2
    pad_after  = pad_total - pad_before
    black = Image.new("L", (SLICE_SIZE, SLICE_SIZE), 0)
    return [black] * pad_before + slices + [black] * pad_after

def slices_to_volume(slices: list[Image.Image]) -> np.ndarray:
    """Returns float32 array shape (Z, Y, X) values 0..255"""
    vol = np.stack([np.array(s, dtype=np.float32) for s in slices], axis=0)
    return vol  # (Z, Y, X)

def volume_to_image(arr2d: np.ndarray) -> Image.Image:
    return Image.fromarray(arr2d.clip(0, 255).astype(np.uint8), mode="L")

def is_black_image(img: Image.Image) -> bool:
    return np.array(img).sum() == 0

def upscale_folder(src: Path, dst: Path):
    ensure(dst)
    files = sorted(src.glob("*.png"))
    for f in files:
        out = dst / f.name
        if out.exists():
            continue
        img = Image.open(f).convert("L")
        if is_black_image(img):
            black = Image.new("L", (SLICE_SIZE * 2, SLICE_SIZE * 2), 0)
            black.save(out)
        else:
            result = reales4u2d(img)
            result.save(out)


# ── axis mapping ──────────────────────────────────────────────────────────────
# Volume axes: vol[Z, Y, X]
#
# FrontLD slice Z  → vol[Z, :, :]  image axes: X=horizontal, Y=vertical
# TopLD   slice Y  → vol[:, Y, :]  image axes: X=horizontal, Z=vertical
# RightLD slice X  → vol[:, :, X]  image axes: Y=horizontal, Z=vertical
#
# After 2x upscale:
#   TopHD  slice Y  has pixel-Z axis doubled  → original slice Z maps to pixel-Z = Z*2
#   RightHD slice X has pixel-Z axis doubled  → original slice Z maps to pixel-Z = Z*2
#
# Interleaved front slice between Z=n and Z=n+1:
#   pick pixel-Z = n*2+1 from TopHD[Y]   → gives row of (X,) for each Y
#   pick pixel-Z = n*2+1 from RightHD[X] → gives row of (Y,) for each X
#   average → 256x256 FrontMod2LD slice

# ── main steps ────────────────────────────────────────────────────────────────

def step_generate_top_right(base: Path, vol: np.ndarray):
    top_dir   = base / "TopLD"
    right_dir = base / "RightLD"

    if not top_dir.exists():
        ensure(top_dir)
        Z, Y, X = vol.shape
        for y in range(Y):
            img = volume_to_image(vol[:, y, :])   # shape (Z, X) → Z=vertical, X=horizontal
            save_slice(img, top_dir, f"{y:04d}.png")
        print(f"TopLD generated: {Y} slices")
    else:
        print("TopLD exists, skipping")

    if not right_dir.exists():
        ensure(right_dir)
        Z, Y, X = vol.shape
        for x in range(X):
            img = volume_to_image(vol[:, :, x])   # shape (Z, Y) → Z=vertical, Y=horizontal
            save_slice(img, right_dir, f"{x:04d}.png")
        print(f"RightLD generated: {X} slices")
    else:
        print("RightLD exists, skipping")


def step_upscale_all(base: Path):
    for name in ["FrontLD", "TopLD", "RightLD"]:
        src = base / name
        dst = base / name.replace("LD", "HD")
        if dst.exists():
            print(f"{dst.name} exists, skipping")
            continue
        print(f"Upscaling {name} → {dst.name} ...")
        upscale_folder(src, dst)


def step_build_frontmod2ld(base: Path):
    out_dir = base / "FrontMod2LD"
    if out_dir.exists():
        print("FrontMod2LD exists, skipping")
        return
    ensure(out_dir)

    top_hd_dir   = base / "TopHD"
    right_hd_dir = base / "RightHD"

    # Load TopHD slices: indexed by Y, each image shape (Z*2, X*2)
    top_files   = sorted(top_hd_dir.glob("*.png"))
    right_files = sorted(right_hd_dir.glob("*.png"))

    top_hd   = [np.array(Image.open(f).convert("L"), dtype=np.float32) for f in top_files]
    right_hd = [np.array(Image.open(f).convert("L"), dtype=np.float32) for f in right_files]

    # top_hd[Y]   shape: (Z*2, X*2)  axes: pixel-Z vertical, pixel-X horizontal
    # right_hd[X] shape: (Z*2, Y*2)  axes: pixel-Z vertical, pixel-Y horizontal

    num_Y = len(top_hd)    # = SLICE_SIZE
    num_X = len(right_hd)  # = SLICE_SIZE

    # Number of interleaved slices = TARGET_DEPTH - 1
    num_interleaved = TARGET_DEPTH - 1

    for n in range(num_interleaved):
        pz = n * 2 + 1  # odd pixel-Z index = between original slices n and n+1

        # From TopHD: for each Y, pick pixel-Z=pz row → shape (X*2,) per Y
        # We need original-resolution X (256), so take every other pixel: px = x*2
        # top_row[Y, X] = top_hd[Y][pz, x*2]
        top_contrib = np.zeros((num_Y, num_X), dtype=np.float32)
        for y in range(num_Y):
            row = top_hd[y][pz, :]          # shape (X*2,)
            top_contrib[y, :] = row[0::2]   # pick even pixel-X → original X positions

        # From RightHD: for each X, pick pixel-Z=pz row → shape (Y*2,) per X
        # right_row[X, Y] = right_hd[X][pz, y*2]
        right_contrib = np.zeros((num_Y, num_X), dtype=np.float32)
        for x in range(num_X):
            col = right_hd[x][pz, :]         # shape (Y*2,)
            right_contrib[:, x] = col[0::2]  # pick even pixel-Y → original Y positions

        averaged = ((top_contrib + right_contrib) / 2.0)
        img = volume_to_image(averaged)
        save_slice(img, out_dir, f"{n:04d}.png")

    print(f"FrontMod2LD generated: {num_interleaved} slices")


def step_upscale_frontmod2(base: Path):
    src = base / "FrontMod2LD"
    dst = base / "FrontMod2HD"
    if dst.exists():
        print("FrontMod2HD exists, skipping")
        return
    print("Upscaling FrontMod2LD → FrontMod2HD ...")
    upscale_folder(src, dst)


def step_combine(base: Path):
    out_dir = base / "FrontCompleteHD"
    if out_dir.exists():
        print("FrontCompleteHD exists, skipping")
        return
    ensure(out_dir)

    front_hd_files    = sorted((base / "FrontHD").glob("*.png"))
    frontmod2_files   = sorted((base / "FrontMod2HD").glob("*.png"))

    seq_idx = 0
    for i, fhd in enumerate(front_hd_files):
        # FrontHD slice
        dst_name = f"{seq_idx:04d}_FrontHD_{i}.png"
        Image.open(fhd).save(out_dir / dst_name)
        seq_idx += 1

        # Interleaved FrontMod2HD slice (one fewer than FrontHD)
        if i < len(frontmod2_files):
            dst_name = f"{seq_idx:04d}_FrontMod2HD_{i}.png"
            Image.open(frontmod2_files[i]).save(out_dir / dst_name)
            seq_idx += 1

    print(f"FrontCompleteHD: {seq_idx} slices total")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python mri_upscale_pipeline.py <FrontLD_folder>")
        sys.exit(1)

    front_ld = Path(sys.argv[1]+"/FrontLD/")
    if not front_ld.exists():
        print(f"Folder not found: {front_ld}")
        sys.exit(1)

    base = front_ld.parent

    print("Loading upscale model...")
    load_model()

    print(f"Loading FrontLD slices from {front_ld} ...")
    raw_slices = load_slices(front_ld)
    if raw_slices:
        print(f"  Slice size: {raw_slices[0].size}")
    print(f"  Found {len(raw_slices)} slices, padding to {TARGET_DEPTH}")
    padded = pad_to_depth(raw_slices, TARGET_DEPTH)
    vol    = slices_to_volume(padded)
    print(f"  Volume shape: {vol.shape}")  # (256, 256, 256)

    # Save padded FrontLD back if it differs (so TopLD/RightLD are consistent)
    padded_front = base / "FrontLD"
    if len(raw_slices) != TARGET_DEPTH:
        print("Saving padded FrontLD slices...")
        ensure(padded_front)
        existing = sorted(padded_front.glob("*.png"))
        if len(existing) != TARGET_DEPTH:
            for i, s in enumerate(padded):
                s.save(padded_front / f"{i:04d}.png")

    step_generate_top_right(base, vol)
    step_upscale_all(base)
    step_build_frontmod2ld(base)
    step_upscale_frontmod2(base)
    step_combine(base)

    print("Done.")

if __name__ == "__main__":
    main()
