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
    Build interleaved front slices using a 512^3 volume approach:
    1. Load all HD slices into 512^3 array
    2. Fill FrontHD at even Z positions (priority)
    3. Fill TopHD and RightHD only into zero voxels
    4. Estimate unknowns at odd-Z, odd-X, odd-Y positions from 6 neighbors
    5. Extract interleaved front slices at odd Z positions
    """
    out_dir = base / "FrontMod2HD"
    ensure(out_dir)

    front_hd_files = sorted((base / "FrontHD").glob("*.png"))
    top_files      = sorted((base / "TopHD").glob("*.png"))
    right_files    = sorted((base / "RightHD").glob("*.png"))

    HD = SLICE_SIZE * 2  # 512

    print("Allocating 512^3 volume...")
    # vol[hz, hy, hx]  all float32
    vol = np.zeros((HD, HD, HD), dtype=np.float32)
    # mask: 1 = filled, 0 = empty
    filled = np.zeros((HD, HD, HD), dtype=np.uint8)

    # ── 1. FrontHD → even Z positions (priority) ─────────────────────────────
    # FrontHD[n] maps to hz = n*2
    print(f"Loading FrontHD ({len(front_hd_files)} slices)...")
    for n, f in enumerate(front_hd_files):
        hz = n * 2
        if hz >= HD:
            break
        arr = np.array(Image.open(f).convert("L"), dtype=np.float32)
        vol[hz, :, :] = arr
        filled[hz, :, :] = 1

    # ── 2. TopHD → vol[:, hy, hx] where hy = ld_y*2 ─────────────────────────
    # TopHD[ld_y] image axes: horizontal=hx, vertical=hz
    # maps to vol[hz, ld_y*2, hx]
    print(f"Loading TopHD ({len(top_files)} slices)...")
    for ld_y, f in enumerate(top_files):
        hy = ld_y * 2
        if hy >= HD:
            break
        arr = np.array(Image.open(f).convert("L"), dtype=np.float32)  # shape (512, 512) = (hz, hx)
        mask = filled[:, hy, :]  # shape (512, 512) = (hz, hx)
        update = mask == 0
        vol[:, hy, :][update] = arr[update]
        filled[:, hy, :][update] = 1

    # ── 3. RightHD → vol[:, hy, hx] where hx = ld_x*2 ──────────────────────
    # RightHD[ld_x] image axes: horizontal=hy, vertical=hz
    # maps to vol[hz, hy, ld_x*2]
    print(f"Loading RightHD ({len(right_files)} slices)...")
    for ld_x, f in enumerate(right_files):
        hx = ld_x * 2
        if hx >= HD:
            break
        arr = np.array(Image.open(f).convert("L"), dtype=np.float32)  # shape (512, 512) = (hz, hy)
        mask = filled[:, :, hx]  # shape (512, 512) = (hz, hy)
        update = mask == 0
        vol[:, :, hx][update] = arr[update]
        filled[:, :, hx][update] = 1

    # ── 4. Estimate unknowns: odd positions not yet filled ────────────────────
    # These are voxels where hz, hy, or hx is odd and not covered above.
    # We do a single pass: for each unfilled voxel, average available neighbors.
    print("Estimating unknown voxels from 6 neighbors...")

    # Find all unfilled positions
    unfilled_hz, unfilled_hy, unfilled_hx = np.where(filled == 0)
    print(f"  Unfilled voxels: {len(unfilled_hz)}")

    # For each unfilled voxel, collect 6 neighbors (clamp at borders)
    def get_neighbor(z, y, x):
        z = int(np.clip(z, 0, HD - 1))
        y = int(np.clip(y, 0, HD - 1))
        x = int(np.clip(x, 0, HD - 1))
        return vol[z, y, x], filled[z, y, x]

    # Vectorized 6-neighbor estimation
    for axis_pass in range(3):
        # Multiple passes help propagate values into corners
        # (odd,odd,odd) voxels need neighbors that may themselves be estimated
        unfilled_hz, unfilled_hy, unfilled_hx = np.where(filled == 0)
        if len(unfilled_hz) == 0:
            break
        print(f"  Pass {axis_pass+1}: {len(unfilled_hz)} unfilled voxels")

        for i in range(len(unfilled_hz)):
            z, y, x = int(unfilled_hz[i]), int(unfilled_hy[i]), int(unfilled_hx[i])
            neighbors = []
            for dz, dy, dx in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                nz = np.clip(z+dz, 0, HD-1)
                ny = np.clip(y+dy, 0, HD-1)
                nx = np.clip(x+dx, 0, HD-1)
                if filled[nz, ny, nx]:
                    neighbors.append(vol[nz, ny, nx])
            if neighbors:
                vol[z, y, x] = np.mean(neighbors)
                filled[z, y, x] = 1

    # ── 5. Extract interleaved front slices at odd Z positions ────────────────
    # Interleaved slice n sits between FrontLD[n] and FrontLD[n+1] → hz = n*2+1
    num_interleaved = TARGET_DEPTH - 1
    print(f"Extracting {num_interleaved} interleaved slices...")

    for n in range(num_interleaved):
        out_path = out_dir / f"{n:04d}.png"
        if out_path.exists():
            continue
        hz = n * 2 + 1
        slice_arr = vol[hz, :, :]  # shape (512, 512) = (hy, hx)
        volume_to_image(slice_arr).save(out_path)
        if n % 20 == 0:
            print(f"  {n}/{num_interleaved}")

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
