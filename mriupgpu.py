"""
mriup.py
Usage: python mriup.py <folder_containing_FrontLD>

Expects FrontLD/ with PNG slices named in sorted order.
Produces: TopLD/ RightLD/ FrontHD/ TopHD/ RightHD/ FrontMod2HD/ FrontCompleteHD/
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

from reales4u2d import load_model, reales4u2d

TARGET_DEPTH = 256
SLICE_SIZE   = 256

# ── helpers ───────────────────────────────────────────────────────────────────

def ensure(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)

def load_slices(folder: Path) -> list:
    files = sorted(folder.glob("*.png"))
    return [Image.open(f).convert("L") for f in files]

def pad_to_depth(slices: list, depth: int) -> list:
    n = len(slices)
    if n >= depth:
        return slices[:depth]
    pad_total  = depth - n
    pad_before = pad_total // 2
    pad_after  = pad_total - pad_before
    black = Image.new("L", (SLICE_SIZE, SLICE_SIZE), 0)
    return [black] * pad_before + slices + [black] * pad_after

def slices_to_volume(slices: list) -> np.ndarray:
    return np.stack([np.array(s, dtype=np.float32) for s in slices], axis=0)

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
            Image.new("L", (SLICE_SIZE * 2, SLICE_SIZE * 2), 0).save(out)
        else:
            reales4u2d(img).save(out)
    print(f"  {dst.name}: done ({len(files)} images)")

# ── axis mapping ──────────────────────────────────────────────────────────────
# vol[Z, Y, X]
# FrontLD[Z] → vol[Z, :, :]   image axes: horizontal=X, vertical=Y
# TopLD[Y]   → vol[:, Y, :]   image axes: horizontal=X, vertical=Z
# RightLD[X] → vol[:, :, X]   image axes: horizontal=Y, vertical=Z
#
# HD images are 512x512. Original slice Z → pixel-Z = Z*2 in HD space.
# Interleaved slice between Z=n and Z=n+1 → pixel-Z = n*2+1

# ── pipeline steps ────────────────────────────────────────────────────────────

def step_generate_top_right(base: Path, vol: np.ndarray):
    top_dir   = base / "TopLD"
    right_dir = base / "RightLD"
    Z, Y, X   = vol.shape

    if not top_dir.exists():
        ensure(top_dir)
        for y in range(Y):
            # TopLD[y]: horizontal=X, vertical=Z → vol[:, y, :] shape (Z, X)
            volume_to_image(vol[:, y, :]).save(top_dir / f"{y:04d}.png")
        print(f"TopLD generated: {Y} slices")
    else:
        print("TopLD exists, skipping")

    if not right_dir.exists():
        ensure(right_dir)
        for x in range(X):
            # RightLD[x]: horizontal=Y, vertical=Z → vol[:, :, x] shape (Z, Y)
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


def step_build_frontmod2hd(base: Path):
    """
    Build interleaved front slices directly in HD space (512x512).

    For interleaved slice between Z=n and Z=n+1, pixel-Z in HD = n*2+1.
    Each 2x2 block at LD position (x, y) maps to HD pixels:

        (px, py)     = (x*2,   y*2)
        (px+1, py)   = (x*2+1, y*2)
        (px,   py+1) = (x*2,   y*2+1)
        (px+1, py+1) = (x*2+1, y*2+1)

    Known pixels:
        (px,   py)   = avg(TopHD[y*2][pz, x*2],   RightHD[x*2][pz, y*2])
        (px+1, py)   = TopHD[y*2][pz, x*2+1]
        (px,   py+1) = RightHD[x*2][pz, y*2+1]

    Unknown pixel (px+1, py+1): estimated from 6 neighbors:
        - FrontHD[n]   [py+1, px+1]  (Z-)
        - FrontHD[n+1] [py+1, px+1]  (Z+)
        - TopHD[y*2-1] [pz,   px+1]  (Y-)  clamped
        - TopHD[y*2+1] [pz,   px+1]  (Y+)  clamped
        - RightHD[x*2-1][pz,  py+1]  (X-)  clamped
        - RightHD[x*2+1][pz,  py+1]  (X+)  clamped

    TopHD has num_Y=256 files (LD Y rows), each 512x512.
    RightHD has num_X=256 files (LD X cols), each 512x512.
    FrontHD has TARGET_DEPTH=256 files, each 512x512.
    """
    out_dir = base / "FrontMod2HD"
    ensure(out_dir)

    front_hd_files = sorted((base / "FrontHD").glob("*.png"))
    top_files      = sorted((base / "TopHD").glob("*.png"))
    right_files    = sorted((base / "RightHD").glob("*.png"))

    num_Y = len(top_files)    # = SLICE_SIZE = 256  (one file per LD Y row)
    num_X = len(right_files)  # = SLICE_SIZE = 256  (one file per LD X col)
    HD    = SLICE_SIZE * 2    # 512

    print(f"Loading TopHD ({num_Y} files) and RightHD ({num_X} files) into memory...")
    # top_hd[y]  shape (512, 512): axis0=Z(HD), axis1=X(HD)
    top_hd   = np.stack([np.array(Image.open(f).convert("L"), dtype=np.float32)
                         for f in top_files],   axis=0)  # (num_Y, 512, 512)
    # right_hd[x] shape (512, 512): axis0=Z(HD), axis1=Y(HD)
    right_hd = np.stack([np.array(Image.open(f).convert("L"), dtype=np.float32)
                         for f in right_files], axis=0)  # (num_X, 512, 512)

    print(f"Loading FrontHD ({len(front_hd_files)} files) into memory...")
    front_hd = np.stack([np.array(Image.open(f).convert("L"), dtype=np.float32)
                         for f in front_hd_files], axis=0)  # (TARGET_DEPTH, 512, 512)

    num_interleaved = TARGET_DEPTH - 1

    # Precompute index arrays for vectorized pixel picking
    # LD coordinate grids
    ys = np.arange(SLICE_SIZE)  # 0..255
    xs = np.arange(SLICE_SIZE)  # 0..255
    # HD coordinate grids (meshgrid: Y varies along axis0, X along axis1)
    Y_ld, X_ld = np.meshgrid(ys, xs, indexing='ij')  # both shape (256, 256)

    py = Y_ld * 2   # HD Y coords of even rows
    px = X_ld * 2   # HD X coords of even cols

    # Neighbor indices for unknown pixel (px+1, py+1), clamped to [0, num_Y/X - 1]
    ty_lo = np.clip(Y_ld - 1, 0, num_Y - 1)   # TopHD Y- neighbor (LD index)
    ty_hi = np.clip(Y_ld + 1, 0, num_Y - 1)   # TopHD Y+ neighbor (LD index)
    rx_lo = np.clip(X_ld - 1, 0, num_X - 1)   # RightHD X- neighbor (LD index)
    rx_hi = np.clip(X_ld + 1, 0, num_X - 1)   # RightHD X+ neighbor (LD index)

    for n in range(num_interleaved):
        out_path = out_dir / f"{n:04d}.png"
        if out_path.exists():
            if n % 20 == 0:
                print(f"  FrontMod2HD: {n}/{num_interleaved} (skip)")
            continue

        pz = n * 2 + 1
        out = np.zeros((HD, HD), dtype=np.float32)

        # ── Known pixels ──────────────────────────────────────────────────────

        # (px, py): avg of TopHD[y*2][pz, x*2] and RightHD[x*2][pz, y*2]
        T_corner = top_hd[Y_ld * 2, pz, X_ld * 2]       # shape (256,256)
        R_corner = right_hd[X_ld * 2, pz, Y_ld * 2]     # shape (256,256)
        out[py, px] = (T_corner + R_corner) / 2.0

        # (px+1, py): TopHD[y*2][pz, x*2+1]
        out[py, px + 1] = top_hd[Y_ld * 2, pz, X_ld * 2 + 1]

        # (px, py+1): RightHD[x*2][pz, y*2+1]
        out[py + 1, px] = right_hd[X_ld * 2, pz, Y_ld * 2 + 1]

        # ── Unknown pixel (px+1, py+1): 6-neighbor average ───────────────────

        # Z- neighbor: FrontHD[n][py+1, px+1]
        n0 = front_hd[n,     py + 1, px + 1]
        # Z+ neighbor: FrontHD[n+1][py+1, px+1]
        n1 = front_hd[n + 1, py + 1, px + 1]
        # Y- neighbor: TopHD[ty_lo][pz, px+1]
        n2 = top_hd[ty_lo * 2, pz, px + 1]
        # Y+ neighbor: TopHD[ty_hi][pz, px+1]
        n3 = top_hd[ty_hi * 2, pz, px + 1]
        # X- neighbor: RightHD[rx_lo][pz, py+1]
        n4 = right_hd[rx_lo * 2, pz, py + 1]
        # X+ neighbor: RightHD[rx_hi][pz, py+1]
        n5 = right_hd[rx_hi * 2, pz, py + 1]

        out[py + 1, px + 1] = (n0 + n1 + n2 + n3 + n4 + n5) / 6.0

        volume_to_image(out).save(out_path)
        if n % 20 == 0:
            print(f"  FrontMod2HD: {n}/{num_interleaved}")

    print(f"FrontMod2HD: {num_interleaved} slices done")


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
    step_build_frontmod2hd(base)
    step_combine(base)

    print("Done.")

if __name__ == "__main__":
    main()
