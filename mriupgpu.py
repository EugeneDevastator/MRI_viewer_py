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
#
# top_hd   shape: (256, 512, 512)  → [ld_y,  hz, hx]
# right_hd shape: (256, 512, 512)  → [ld_x,  hz, hy]
# front_hd shape: (256, 512, 512)  → [ld_z,  hy, hx]

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


def step_build_frontmod2hd(base: Path):
    """
    Build interleaved front slices directly in HD space (512x512).

    Axis mapping:
        top_hd   [ld_y,  hz, hx]   ld_y in 0..255, hz/hx in 0..511
        right_hd [ld_x,  hz, hy]   ld_x in 0..255, hz/hy in 0..511
        front_hd [ld_z,  hy, hx]   ld_z in 0..255, hy/hx in 0..511

    For interleaved slice n (between FrontLD[n] and FrontLD[n+1]):
        pz = n*2+1  (HD Z coordinate)

    2x2 block for LD pixel (x, y):
        HD coords: px=x*2, py=y*2

        (px,   py)   = avg( top_hd[y, pz, px],   right_hd[x, pz, py]   )
        (px+1, py)   =      top_hd[y, pz, px+1]
        (px,   py+1) =      right_hd[x, pz, py+1]
        (px+1, py+1) = avg of 6 neighbors:
            front_hd[n,   py+1, px+1]   Z-
            front_hd[n+1, py+1, px+1]   Z+
            top_hd[y-1,   pz,   px+1]   Y-  (clamped)
            top_hd[y+1,   pz,   px+1]   Y+  (clamped)
            right_hd[x-1, pz,   py+1]   X-  (clamped)
            right_hd[x+1, pz,   py+1]   X+  (clamped)
    """
    out_dir = base / "FrontMod2HD"
    ensure(out_dir)

    front_hd_files = sorted((base / "FrontHD").glob("*.png"))
    top_files      = sorted((base / "TopHD").glob("*.png"))
    right_files    = sorted((base / "RightHD").glob("*.png"))

    num_Y = len(top_files)    # 256
    num_X = len(right_files)  # 256
    HD    = SLICE_SIZE * 2    # 512

    print(f"Loading TopHD ({num_Y} files) and RightHD ({num_X} files) into memory...")
    # top_hd[ld_y, hz, hx]
    top_hd   = np.stack([np.array(Image.open(f).convert("L"), dtype=np.float32)
                         for f in top_files],   axis=0)  # (256, 512, 512)
    # right_hd[ld_x, hz, hy]
    right_hd = np.stack([np.array(Image.open(f).convert("L"), dtype=np.float32)
                         for f in right_files], axis=0)  # (256, 512, 512)

    print(f"Loading FrontHD ({len(front_hd_files)} files) into memory...")
    # front_hd[ld_z, hy, hx]
    front_hd = np.stack([np.array(Image.open(f).convert("L"), dtype=np.float32)
                         for f in front_hd_files], axis=0)  # (256, 512, 512)

    num_interleaved = TARGET_DEPTH - 1

    # LD coordinate grids, shape (256, 256)
    ys = np.arange(SLICE_SIZE)
    xs = np.arange(SLICE_SIZE)
    Y_ld, X_ld = np.meshgrid(ys, xs, indexing='ij')  # Y_ld[y,x]=y, X_ld[y,x]=x

    py = Y_ld * 2   # HD Y coords, shape (256,256)
    px = X_ld * 2   # HD X coords, shape (256,256)

    # Clamped LD neighbor indices (stay in 0..255)
    ty_lo = np.clip(Y_ld - 1, 0, num_Y - 1)  # (256,256)
    ty_hi = np.clip(Y_ld + 1, 0, num_Y - 1)
    rx_lo = np.clip(X_ld - 1, 0, num_X - 1)
    rx_hi = np.clip(X_ld + 1, 0, num_X - 1)

    for n in range(num_interleaved):
        out_path = out_dir / f"{n:04d}.png"
        if out_path.exists():
            if n % 20 == 0:
                print(f"  FrontMod2HD: {n}/{num_interleaved} (skip)")
            continue

        pz = n * 2 + 1
        out = np.zeros((HD, HD), dtype=np.float32)

        # ── Known pixels ──────────────────────────────────────────────────────

        # (px, py) = avg(top_hd[y, pz, px], right_hd[x, pz, py])
        T_corner = top_hd[Y_ld,   pz, px]      # top_hd[ld_y, hz, hx]
        R_corner = right_hd[X_ld, pz, py]      # right_hd[ld_x, hz, hy]
        out[py, px] = (T_corner + R_corner) / 2.0

        # (px+1, py) = top_hd[y, pz, px+1]
        out[py, px + 1] = top_hd[Y_ld, pz, px + 1]

        # (px, py+1) = right_hd[x, pz, py+1]
        out[py + 1, px] = right_hd[X_ld, pz, py + 1]

        # ── Unknown pixel (px+1, py+1): 6-neighbor average ───────────────────

        n0 = front_hd[n,     py + 1, px + 1]          # Z-
        n1 = front_hd[n + 1, py + 1, px + 1]          # Z+
        n2 = top_hd[ty_lo,   pz,     px + 1]          # Y-
        n3 = top_hd[ty_hi,   pz,     px + 1]          # Y+
        n4 = right_hd[rx_lo, pz,     py + 1]          # X-
        n5 = right_hd[rx_hi, pz,     py + 1]          # X+

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
