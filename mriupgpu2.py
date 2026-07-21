"""
mriupgpu.py
Usage: python mriupgpu.py <folder_containing_FrontLD>
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

from reales4u2d import load_model, reales4u2d

TARGET_DEPTH = 512   # pad front slices to this many (Z axis)
SLICE_SIZE   = 512   # each slice is SLICE_SIZE x SLICE_SIZE (resize if needed)
HD           = SLICE_SIZE * 2   # 1024
HZ           = TARGET_DEPTH * 2

# ── helpers ───────────────────────────────────────────────────────────────────

def ensure(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)

def load_slices(folder: Path) -> list:
    files = sorted(folder.glob("*.png"))
    imgs = []
    for f in files:
        img = Image.open(f).convert("L")
        if img.size != (SLICE_SIZE, SLICE_SIZE):
            img = img.resize((SLICE_SIZE, SLICE_SIZE), Image.LANCZOS)
        imgs.append(img)
    return imgs

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
    arrays = [np.array(s, dtype=np.float32) for s in slices]
    # Verify all same shape
    shapes = set(a.shape for a in arrays)
    if len(shapes) > 1:
        raise ValueError(f"Inconsistent slice shapes: {shapes}")
    return np.stack(arrays, axis=0)

def volume_to_image(arr2d: np.ndarray) -> Image.Image:
    return Image.fromarray(arr2d.clip(0, 255).astype(np.uint8), mode="L")

def is_black_image(img: Image.Image) -> bool:
    return np.array(img).sum() == 0

def upscale_folder(src: Path, dst: Path):
    ensure(dst)
    files = sorted(src.glob("*.png"))
    done  = 0
    for f in files:
        out = dst / f.name
        if out.exists():
            done += 1
            continue
        img = Image.open(f).convert("L")
        if is_black_image(img):
            w, h = img.size
            Image.new("L", (w * 2, h * 2), 0).save(out)
        else:
            reales4u2d(img).save(out)
        done += 1
        if done % 50 == 0:
            print(f"  {dst.name}: {done}/{len(files)}")
    print(f"  {dst.name}: done ({len(files)} images)")

# ── axis mapping ──────────────────────────────────────────────────────────────
# vol[Z, Y, X]  shape = (TARGET_DEPTH, SLICE_SIZE, SLICE_SIZE)
#
# FrontLD[z] → vol[z, :, :]
#   image: width=SLICE_SIZE (X), height=SLICE_SIZE (Y)
#   numpy arr shape: (SLICE_SIZE, SLICE_SIZE) = (Y, X)
#
# TopLD[y]   → vol[:, y, :]
#   numpy slice shape: (TARGET_DEPTH, SLICE_SIZE) = (Z, X)
#   image: width=SLICE_SIZE (X), height=TARGET_DEPTH (Z)
#   PIL image size: (SLICE_SIZE, TARGET_DEPTH)
#
# RightLD[x] → vol[:, :, x]
#   numpy slice shape: (TARGET_DEPTH, SLICE_SIZE) = (Z, Y)
#   image: width=SLICE_SIZE (Y), height=TARGET_DEPTH (Z)
#   PIL image size: (SLICE_SIZE, TARGET_DEPTH)
#
# After 2x upscale:
#   FrontHD[z]:  PIL size (HD, HD),            arr shape (HD, HD)
#   TopHD[y]:    PIL size (HD, TARGET_DEPTH*2), arr shape (TARGET_DEPTH*2, HD)
#   RightHD[x]:  PIL size (HD, TARGET_DEPTH*2), arr shape (TARGET_DEPTH*2, HD)
#
# HD volume: vol_hd[HZ, HY, HX]
#   HZ = TARGET_DEPTH*2, HY = HD, HX = HD
#   FrontHD[n] → vol_hd[n*2,  :,  :]   arr[row=hy, col=hx]
#   TopHD[y]   → vol_hd[:,  y*2,  :]   arr[row=hz, col=hx]
#   RightHD[x] → vol_hd[:,    :, x*2]  arr[row=hz, col=hy]

# ── pipeline steps ────────────────────────────────────────────────────────────

def step_generate_top_right(base: Path, vol: np.ndarray):
    """
    vol shape: (Z, Y, X) = (TARGET_DEPTH, SLICE_SIZE, SLICE_SIZE)
    TopLD[y]:   PIL size (SLICE_SIZE, TARGET_DEPTH)  i.e. width=X, height=Z
    RightLD[x]: PIL size (SLICE_SIZE, TARGET_DEPTH)  i.e. width=Y, height=Z
    """
    top_dir   = base / "TopLD"
    right_dir = base / "RightLD"
    Z, Y, X   = vol.shape

    print(f"Volume shape: Z={Z}, Y={Y}, X={X}")

    if not top_dir.exists():
        ensure(top_dir)
        for y in range(Y):
            # vol[:, y, :] → shape (Z, X) → PIL(width=X, height=Z)
            arr = vol[:, y, :].clip(0, 255).astype(np.uint8)  # shape (Z, X)
            img = Image.fromarray(arr, mode="L")               # PIL size (X, Z)
            assert img.size == (X, Z), f"TopLD size mismatch: {img.size} vs ({X},{Z})"
            img.save(top_dir / f"{y:04d}.png")
        print(f"TopLD generated: {Y} slices, PIL size ({X}w x {Z}h)")
    else:
        # Verify existing
        sample = sorted(top_dir.glob("*.png"))
        if sample:
            s = Image.open(sample[0])
            print(f"TopLD exists: {len(sample)} slices, PIL size {s.size} (expected ({X}w x {Z}h))")
        else:
            print("TopLD exists but empty!")

    if not right_dir.exists():
        ensure(right_dir)
        for x in range(X):
            # vol[:, :, x] → shape (Z, Y) → PIL(width=Y, height=Z)
            arr = vol[:, :, x].clip(0, 255).astype(np.uint8)  # shape (Z, Y)
            img = Image.fromarray(arr, mode="L")               # PIL size (Y, Z)
            assert img.size == (Y, Z), f"RightLD size mismatch: {img.size} vs ({Y},{Z})"
            img.save(right_dir / f"{x:04d}.png")
        print(f"RightLD generated: {X} slices, PIL size ({Y}w x {Z}h)")
    else:
        sample = sorted(right_dir.glob("*.png"))
        if sample:
            s = Image.open(sample[0])
            print(f"RightLD exists: {len(sample)} slices, PIL size {s.size} (expected ({Y}w x {Z}h))")
        else:
            print("RightLD exists but empty!")


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

    # Verify sizes from actual files
    def check_size(files, label):
        if not files: raise FileNotFoundError(f"No files in {label}")
        img = Image.open(files[0])
        print(f"  {label}[0] PIL size: {img.size}  (width, height)")
        return img.size  # (width, height)

    fhd_w, fhd_h = check_size(front_hd_files, "FrontHD")  # expect (HD, HD)
    thd_w, thd_h = check_size(top_files,       "TopHD")    # expect (HD*?, HZ*?)
    rhd_w, rhd_h = check_size(right_files,     "RightHD")  # expect (HD*?, HZ*?)

    # HD volume dimensions — derive from actual file sizes
    # FrontHD: arr shape (fhd_h, fhd_w) = (HY, HX)
    HX = fhd_w   # horizontal in front slice = X axis
    HY = fhd_h   # vertical in front slice   = Y axis
    # TopHD: arr shape (thd_h, thd_w) = (HZ, HX)  → HZ from height
    HZ_from_top = thd_h
    # RightHD: arr shape (rhd_h, rhd_w) = (HZ, HY)
    HZ_from_right = rhd_h
    HZ_actual = max(HZ_from_top, HZ_from_right, HZ)

    print(f"HD volume: HZ={HZ_actual}, HY={HY}, HX={HX}")
    print(f"  (from FrontHD: HX={HX}, HY={HY})")
    print(f"  (from TopHD height: {HZ_from_top}, RightHD height: {HZ_from_right}, constant HZ: {HZ})")

    vol   = np.zeros((HZ_actual, HY, HX), dtype=np.float32)
    count = np.zeros((HZ_actual, HY, HX), dtype=np.uint8)

    # ── Load FrontHD at even Z ────────────────────────────────────────────────
    # arr shape: (HY, HX)  → vol[hz, hy, hx]
    print(f"Loading FrontHD ({len(front_hd_files)} slices)...")
    for n, f in enumerate(front_hd_files):
        hz = n * 2
        if hz >= HZ_actual: break
        arr = np.array(Image.open(f).convert("L"), dtype=np.float32)
        if arr.shape != (HY, HX):
            arr = np.array(Image.open(f).convert("L").resize((HX, HY), Image.LANCZOS), dtype=np.float32)
        vol[hz, :, :] = arr
        count[hz, :, :] = 1

    # ── Load TopHD at even Y ──────────────────────────────────────────────────
    # TopHD[ld_y]: PIL size (thd_w, thd_h), arr shape (thd_h, thd_w) = (HZ, HX)
    # maps to vol[:, hy, :] where hy = ld_y * 2
    print(f"Loading TopHD ({len(top_files)} slices)...")
    for ld_y, f in enumerate(top_files):
        hy = ld_y * 2
        if hy >= HY: break
        img = Image.open(f).convert("L")
        arr = np.array(img, dtype=np.float32)  # shape (thd_h, thd_w)
        # Need shape (HZ_actual, HX)
        if arr.shape != (HZ_actual, HX):
            img = img.resize((HX, HZ_actual), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32)
        # Only fill zeros (FrontHD has priority)
        mask = count[:, hy, :] == 0   # shape (HZ_actual, HX)
        vol[:, hy, :][mask] = arr[mask]
        count[:, hy, :][mask] = 1

    # ── Load RightHD at even X ────────────────────────────────────────────────
    # RightHD[ld_x]: PIL size (rhd_w, rhd_h), arr shape (rhd_h, rhd_w) = (HZ, HY)
    # maps to vol[:, :, hx] where hx = ld_x * 2
    print(f"Loading RightHD ({len(right_files)} slices)...")
    for ld_x, f in enumerate(right_files):
        hx = ld_x * 2
        if hx >= HX: break
        img = Image.open(f).convert("L")
        arr = np.array(img, dtype=np.float32)  # shape (rhd_h, rhd_w)
        # Need shape (HZ_actual, HY)
        if arr.shape != (HZ_actual, HY):
            img = img.resize((HY, HZ_actual), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32)
        mask = count[:, :, hx] == 0   # shape (HZ_actual, HY)
        vol[:, :, hx][mask] = arr[mask]
        count[:, :, hx][mask] = 1

    # ── Estimate all-odd voxels from 6 neighbors ──────────────────────────────
    print("Estimating unknowns (all-odd coordinates)...")
    odd_z = np.arange(1, HZ_actual - 1, 2)
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

    # ── Extract interleaved slices at odd Z, shrink → re-upscale ─────────────
    print("Extracting interleaved slices (shrink → re-upscale)...")
    for n in range(TARGET_DEPTH):
        out_path = out_dir / f"{n:04d}.png"
        if out_path.exists():
            continue
        hz = n * 2 + 1
        if hz >= HZ_actual:
            Image.new("L", (HD, HD), 0).save(out_path)
            continue

        slice_hd = volume_to_image(vol[hz, :, :])  # PIL size (HX, HY)

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
        print(f"  Slice size after resize: {raw_slices[0].size}")

    padded = pad_to_depth(raw_slices, TARGET_DEPTH)
    vol    = slices_to_volume(padded)
    print(f"  Volume shape: {vol.shape}")  # (TARGET_DEPTH, SLICE_SIZE, SLICE_SIZE)

    # Save padded FrontLD if count changed
    existing = sorted(front_ld.glob("*.png"))
    if len(existing) != TARGET_DEPTH:
        print(f"Saving padded FrontLD slices ({len(existing)} → {TARGET_DEPTH})...")
        for i, s in enumerate(padded):
            s.save(front_ld / f"{i:04d}.png")

    step_generate_top_right(base, vol)
    step_upscale_all(base)
    step_build_frontmod2hd(base)
    step_combine(base)

    print("Done.")

if __name__ == "__main__":
    main()
