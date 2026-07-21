"""
mriupgpu.py
Usage: python mriupgpu.py <folder_containing_FrontLD>
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

from reales4u2d import load_model, reales4u2d

TARGET_DEPTH = 512   # pad front slices to this many
SLICE_SIZE   = 512   # each slice is SLICE_SIZE x SLICE_SIZE
HD           = SLICE_SIZE * 2  # 1024

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
            w, h = img.size
            Image.new("L", (w * 2, h * 2), 0).save(out)
        else:
            reales4u2d(img).save(out)
    print(f"  {dst.name}: done ({len(files)} images)")

# ── axis mapping ──────────────────────────────────────────────────────────────
# vol[Z, Y, X]  shape = (TARGET_DEPTH, SLICE_SIZE, SLICE_SIZE)
#
# FrontLD[z] → vol[z, :, :]   saved as (SLICE_SIZE wide, SLICE_SIZE tall)
# TopLD[y]   → vol[:, y, :]   saved as (SLICE_SIZE wide, TARGET_DEPTH tall)
# RightLD[x] → vol[:, :, x]   saved as (SLICE_SIZE wide, TARGET_DEPTH tall)
#
# After upscale (2x each dimension):
# FrontHD[z]  → (HD wide,        HD tall)
# TopHD[y]    → (HD wide,        TARGET_DEPTH*2 tall)
# RightHD[x]  → (HD wide,        TARGET_DEPTH*2 tall)
#
# HD volume axes: [hz, hy, hx]
#   hz = z*2 for original front slices, z*2+1 for interleaved
#   hy = y*2 for original top rows
#   hx = x*2 for original right cols

# ── pipeline steps ────────────────────────────────────────────────────────────

def step_generate_top_right(base: Path, vol: np.ndarray):
    """
    vol shape: (Z, Y, X) = (TARGET_DEPTH, SLICE_SIZE, SLICE_SIZE)
    TopLD[y]:   image of shape (X wide, Z tall) = (SLICE_SIZE, TARGET_DEPTH)
    RightLD[x]: image of shape (Y wide, Z tall) = (SLICE_SIZE, TARGET_DEPTH)
    """
    top_dir   = base / "TopLD"
    right_dir = base / "RightLD"
    Z, Y, X   = vol.shape  # (TARGET_DEPTH, SLICE_SIZE, SLICE_SIZE)

    if not top_dir.exists():
        ensure(top_dir)
        for y in range(Y):
            # vol[:, y, :] shape = (Z, X) → image width=X, height=Z
            img = Image.fromarray(vol[:, y, :].clip(0,255).astype(np.uint8), mode="L")
            img.save(top_dir / f"{y:04d}.png")
        print(f"TopLD generated: {Y} slices, each {X}w x {Z}h")
    else:
        print("TopLD exists, skipping")

    if not right_dir.exists():
        ensure(right_dir)
        for x in range(X):
            # vol[:, :, x] shape = (Z, Y) → image width=Y, height=Z
            img = Image.fromarray(vol[:, :, x].clip(0,255).astype(np.uint8), mode="L")
            img.save(right_dir / f"{x:04d}.png")
        print(f"RightLD generated: {X} slices, each {Y}w x {Z}h")
    else:
        print("RightLD exists, skipping")


def step_upscale_all(base: Path):
    for name in ["FrontLD", "TopLD", "RightLD"]:
        src = base / name
        dst = base / name.replace("LD", "HD")
        print(f"Upscaling {name} → {dst.name} ...")
        upscale_folder(src, dst)


def step_build_frontmod2hd(base: Path):
    out_dir = base / "FrontMod2HD"
    ensure(out_dir)

    existing = sorted(out_dir.glob("*.png"))
    if len(existing) == TARGET_DEPTH:
        print("FrontMod2HD exists, skipping")
        return

    front_hd_files = sorted((base / "FrontHD").glob("*.png"))
    top_files      = sorted((base / "TopHD").glob("*.png"))
    right_files    = sorted((base / "RightHD").glob("*.png"))

    # HD volume dimensions
    HZ = TARGET_DEPTH * 2   # z axis in HD space
    HY = SLICE_SIZE   * 2   # y axis in HD space  (= HD)
    HX = SLICE_SIZE   * 2   # x axis in HD space  (= HD)

    print(f"Allocating HD volume {HZ}x{HY}x{HX} ...")
    vol   = np.zeros((HZ, HY, HX), dtype=np.float32)
    count = np.zeros((HZ, HY, HX), dtype=np.uint8)

    # FrontHD: vol[hz, hy, hx] at even Z positions
    # Each FrontHD image is HX wide x HY tall
    print(f"Loading FrontHD ({len(front_hd_files)} slices)...")
    for n, f in enumerate(front_hd_files):
        hz = n * 2
        if hz >= HZ: break
        arr = np.array(Image.open(f).convert("L"), dtype=np.float32)
        # arr shape: (HY, HX)
        vol[hz, :, :] += arr
        count[hz, :, :] += 1

    # TopHD: image[ld_y] has shape (HX wide, HZ tall) after upscale
    # axes: horizontal=hx, vertical=hz
    # placed at even Y positions: vol[:, hy, :] where hy = ld_y * 2
    print(f"Loading TopHD ({len(top_files)} slices)...")
    for ld_y, f in enumerate(top_files):
        hy = ld_y * 2
        if hy >= HY: break
        img = Image.open(f).convert("L")
        arr = np.array(img, dtype=np.float32)
        # arr shape: (HZ, HX)  height=HZ, width=HX
        if arr.shape != (HZ, HX):
            img = img.resize((HX, HZ), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32)
        mask = count[:, hy, :] == 0
        vol[:, hy, :][mask] += arr[mask]
        count[:, hy, :][mask] += 1

    # RightHD: image[ld_x] has shape (HY wide, HZ tall) after upscale
    # axes: horizontal=hy, vertical=hz
    # placed at even X positions: vol[:, :, hx] where hx = ld_x * 2
    print(f"Loading RightHD ({len(right_files)} slices)...")
    for ld_x, f in enumerate(right_files):
        hx = ld_x * 2
        if hx >= HX: break
        img = Image.open(f).convert("L")
        arr = np.array(img, dtype=np.float32)
        # arr shape: (HZ, HY)
        if arr.shape != (HZ, HY):
            img = img.resize((HY, HZ), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32)
        mask = count[:, :, hx] == 0
        vol[:, :, hx][mask] += arr[mask]
        count[:, :, hx][mask] += 1

    print("Averaging accumulated samples...")
    filled = count > 0
    vol[filled] /= count[filled].astype(np.float32)

    # Estimate all-odd voxels from 6 neighbors
    print("Estimating unknowns (all-odd coordinates)...")
    odd_z = np.arange(1, HZ - 1, 2)
    odd_y = np.arange(1, HY - 1, 2)
    odd_x = np.arange(1, HX - 1, 2)
    gz, gy, gx = np.meshgrid(odd_z, odd_y, odd_x, indexing='ij')
    gz = gz.ravel(); gy = gy.ravel(); gx = gx.ravel()

    neighbors = np.stack([
        vol[gz-1, gy,   gx  ],
        vol[gz+1, gy,   gx  ],
        vol[gz,   gy-1, gx  ],
        vol[gz,   gy+1, gx  ],
        vol[gz,   gy,   gx-1],
        vol[gz,   gy,   gx+1],
    ], axis=0)
    vol[gz, gy, gx] = neighbors.mean(axis=0)

    # Extract interleaved slices at odd Z, shrink then re-upscale
    print("Extracting interleaved slices (shrink → re-upscale)...")
    for n in range(TARGET_DEPTH):
        out_path = out_dir / f"{n:04d}.png"
        if out_path.exists():
            continue
        hz = n * 2 + 1
        slice_hd = volume_to_image(vol[hz, :, :])  # HX x HY

        if is_black_image(slice_hd):
            Image.new("L", (HD, HD), 0).save(out_path)
        else:
            slice_ld = slice_hd.resize((SLICE_SIZE, SLICE_SIZE), Image.LANCZOS)
            reales4u2d(slice_ld).save(out_path)

        if n % 20 == 0:
            print(f"  slice {n}/{TARGET_DEPTH}")

    print(f"FrontMod2HD: {TARGET_DEPTH} slices done")


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
        print("Usage: python mriupgpu.py <folder>")
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
    print(f"  Volume shape: {vol.shape}")  # (TARGET_DEPTH, SLICE_SIZE, SLICE_SIZE)

    # Save padded FrontLD if needed
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
