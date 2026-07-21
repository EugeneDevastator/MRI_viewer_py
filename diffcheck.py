"""
diff_top_right_check.py
Usage: python diff_top_right_check.py <folder>

Checks TopLD and RightLD consistency against FrontLD.
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
    if abs_diff.max() == 0:
        print(f"  ✓ PERFECT MATCH")
    else:
        print(f"  ✗ MISMATCH")
    return abs_diff

def main():
    base = Path(sys.argv[1])
    front_ld = base / "FrontLD"
    top_ld   = base / "TopLD"
    right_ld = base / "RightLD"

    front_files = sorted(front_ld.glob("*.png"))
    top_files   = sorted(top_ld.glob("*.png"))
    right_files = sorted(right_ld.glob("*.png"))

    print(f"FrontLD: {len(front_files)} slices")
    print(f"TopLD:   {len(top_files)} slices")
    print(f"RightLD: {len(right_files)} slices")

    # --- TOP CHECK ---
    mid_y = len(top_files) // 2
    print(f"\nChecking TopLD slice Y={mid_y} ({top_files[mid_y].name})")
    top_img = np.array(Image.open(top_files[mid_y]).convert("L"), dtype=np.float32)
    print(f"TopLD image shape: {top_img.shape}  (expected Z x X)")
    top_recon = np.zeros_like(top_img)
    for z, f in enumerate(front_files):
        if z >= top_img.shape[0]: break
        arr = np.array(Image.open(f).convert("L"), dtype=np.float32)
        top_recon[z, :] = arr[mid_y, :]  # row mid_y from front slice Z
    top_diff = check_slice("TopLD", top_img, top_recon)

    # --- RIGHT CHECK ---
    mid_x = len(right_files) // 2
    print(f"\nChecking RightLD slice X={mid_x} ({right_files[mid_x].name})")
    right_img = np.array(Image.open(right_files[mid_x]).convert("L"), dtype=np.float32)
    print(f"RightLD image shape: {right_img.shape}  (expected Z x Y)")
    right_recon = np.zeros_like(right_img)
    for z, f in enumerate(front_files):
        if z >= right_img.shape[0]: break
        arr = np.array(Image.open(f).convert("L"), dtype=np.float32)
        right_recon[z, :] = arr[:, mid_x]  # column mid_x from front slice Z
    right_diff = check_slice("RightLD", right_img, right_recon)

    # --- PLOT ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0,0].imshow(top_img, cmap='gray', vmin=0, vmax=255)
    axes[0,0].set_title(f"TopLD[{mid_y}] actual")
    axes[0,1].imshow(top_recon, cmap='gray', vmin=0, vmax=255)
    axes[0,1].set_title(f"TopLD reconstructed from FrontLD row {mid_y}")
    axes[0,2].imshow(top_diff, cmap='hot', vmin=0, vmax=max(top_diff.max(), 1))
    axes[0,2].set_title(f"Top abs diff (max={top_diff.max():.1f})")

    axes[1,0].imshow(right_img, cmap='gray', vmin=0, vmax=255)
    axes[1,0].set_title(f"RightLD[{mid_x}] actual")
    axes[1,1].imshow(right_recon, cmap='gray', vmin=0, vmax=255)
    axes[1,1].set_title(f"RightLD reconstructed from FrontLD col {mid_x}")
    axes[1,2].imshow(right_diff, cmap='hot', vmin=0, vmax=max(right_diff.max(), 1))
    axes[1,2].set_title(f"Right abs diff (max={right_diff.max():.1f})")

    plt.tight_layout()
    out = base / "diff_check.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    plt.show()

if __name__ == "__main__":
    main()
