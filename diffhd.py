"""
diff_hd_check.py
Usage: python diff_hd_check.py <folder>

Checks TopHD and RightHD consistency against FrontHD.
Diffs should be small (upscaler variation) but not systematic.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def check_slice(name, actual_img, reconstructed):
    diff = actual_img - reconstructed
    abs_diff = np.abs(diff)
    print(f"\n{name} diff stats:")
    print(f"  max abs diff : {abs_diff.max():.4f}")
    print(f"  mean abs diff: {abs_diff.mean():.4f}")
    print(f"  nonzero pixels: {(abs_diff > 0).sum()} / {abs_diff.size}")
    print(f"  mean actual:       {actual_img.mean():.4f}")
    print(f"  mean reconstructed:{reconstructed.mean():.4f}")
    return diff, abs_diff

def main():
    base = Path(sys.argv[1])
    front_hd = base / "FrontHD"
    top_hd   = base / "TopHD"
    right_hd = base / "RightHD"

    front_files = sorted(front_hd.glob("*.png"))
    top_files   = sorted(top_hd.glob("*.png"))
    right_files = sorted(right_hd.glob("*.png"))

    print(f"FrontHD: {len(front_files)} slices")
    print(f"TopHD:   {len(top_files)} slices")
    print(f"RightHD: {len(right_files)} slices")

    HD = 512

    # --- TOP CHECK ---
    # TopHD[mid_y] should match FrontHD slices at row mid_y*2 (even HD rows)
    mid_y_ld = len(top_files) // 2
    mid_y_hd = mid_y_ld * 2
    print(f"\nChecking TopHD slice Y_ld={mid_y_ld} ({top_files[mid_y_ld].name})")
    print(f"  Comparing against FrontHD row {mid_y_hd}")
    top_img = np.array(Image.open(top_files[mid_y_ld]).convert("L"), dtype=np.float32)
    print(f"  TopHD image shape: {top_img.shape}  (expected {HD} x {HD})")

    # Reconstruct: for each front slice z_ld, pick row mid_y_hd from FrontHD[z_ld]
    # That gives us top_recon[z_hd=z_ld*2, :] — only even rows populated
    top_recon = np.zeros((HD, HD), dtype=np.float32)
    for z_ld, f in enumerate(front_files):
        z_hd = z_ld * 2
        if z_hd >= HD: break
        arr = np.array(Image.open(f).convert("L"), dtype=np.float32)
        # arr is 512x512 FrontHD, row mid_y_hd is the HD row for this Y
        top_recon[z_hd, :] = arr[mid_y_hd, :]

    # Only compare even rows (odd rows in TopHD are upscaler-interpolated, no FrontHD equivalent)
    even_rows = np.arange(0, HD, 2)
    top_even_actual = top_img[even_rows, :]
    top_even_recon  = top_recon[even_rows, :]
    top_diff_even, top_abs_even = check_slice("TopHD (even Z rows only)", top_even_actual, top_even_recon)

    # --- RIGHT CHECK ---
    mid_x_ld = len(right_files) // 2
    mid_x_hd = mid_x_ld * 2
    print(f"\nChecking RightHD slice X_ld={mid_x_ld} ({right_files[mid_x_ld].name})")
    print(f"  Comparing against FrontHD col {mid_x_hd}")
    right_img = np.array(Image.open(right_files[mid_x_ld]).convert("L"), dtype=np.float32)
    print(f"  RightHD image shape: {right_img.shape}  (expected {HD} x {HD})")

    right_recon = np.zeros((HD, HD), dtype=np.float32)
    for z_ld, f in enumerate(front_files):
        z_hd = z_ld * 2
        if z_hd >= HD: break
        arr = np.array(Image.open(f).convert("L"), dtype=np.float32)
        right_recon[z_hd, :] = arr[:, mid_x_hd]

    right_even_actual = right_img[even_rows, :]
    right_even_recon  = right_recon[even_rows, :]
    right_diff_even, right_abs_even = check_slice("RightHD (even Z rows only)", right_even_actual, right_even_recon)

    # --- PLOT ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0,0].imshow(top_img, cmap='gray', vmin=0, vmax=255)
    axes[0,0].set_title(f"TopHD[{mid_y_ld}] actual")
    axes[0,1].imshow(top_recon, cmap='gray', vmin=0, vmax=255)
    axes[0,1].set_title(f"Reconstructed from FrontHD row {mid_y_hd}")
    axes[0,2].imshow(top_abs_even, cmap='hot', vmin=0, vmax=max(top_abs_even.max(), 1))
    axes[0,2].set_title(f"Top abs diff even rows (max={top_abs_even.max():.1f})")
    axes[0,3].imshow(top_diff_even, cmap='bwr', vmin=-50, vmax=50)
    axes[0,3].set_title("Top signed diff (blue=neg, red=pos)")

    axes[1,0].imshow(right_img, cmap='gray', vmin=0, vmax=255)
    axes[1,0].set_title(f"RightHD[{mid_x_ld}] actual")
    axes[1,1].imshow(right_recon, cmap='gray', vmin=0, vmax=255)
    axes[1,1].set_title(f"Reconstructed from FrontHD col {mid_x_hd}")
    axes[1,2].imshow(right_abs_even, cmap='hot', vmin=0, vmax=max(right_abs_even.max(), 1))
    axes[1,2].set_title(f"Right abs diff even rows (max={right_abs_even.max():.1f})")
    axes[1,3].imshow(right_diff_even, cmap='bwr', vmin=-50, vmax=50)
    axes[1,3].set_title("Right signed diff (blue=neg, red=pos)")

    plt.tight_layout()
    out = base / "diff_hd_check.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    plt.show()

if __name__ == "__main__":
    main()
