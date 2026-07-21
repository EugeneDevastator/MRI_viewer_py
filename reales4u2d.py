"""
reales4u2d.py
Upscale x4 with RealESRGAN_x4plus.pth, then downscale to x2 with Lanczos.
Usage: python reales4u2d.py image.png
As module: from reales4u2d import reales4u2d, load_model
"""

import sys
import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import spandrel

# Model sits next to this script
MODEL_PATH = Path(__file__).parent / "RealESRGAN_x4plus.pth"

_model = None  # cached model instance

def load_model():
    global _model
    if _model is None:
        _model = spandrel.ModelLoader().load_from_file(str(MODEL_PATH))
        _model.eval()
        _model.to("cpu")
    return _model


def reales4u2d(image: Image.Image) -> Image.Image:
    """
    Takes a PIL Image (any mode), returns PIL Image upscaled x4 then downscaled x2 (net x2).
    Preserves grayscale — returns grayscale if input was grayscale.
    """
    was_grayscale = image.mode == "L"

    # Work in RGB
    rgb = image.convert("RGB")
    w, h = rgb.size

    # To tensor [1, 3, H, W] float32 0..1
    arr = np.array(rgb).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    model = load_model()

    with torch.no_grad():
        out_tensor = model(tensor)

    # Back to PIL
    out_arr = out_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    out_arr = (out_arr * 255).astype(np.uint8)
    out_img = Image.fromarray(out_arr, mode="RGB")  # now 4x size

    # Downscale to 2x original with Lanczos
    target_w, target_h = w * 2, h * 2
    out_img = out_img.resize((target_w, target_h), Image.LANCZOS)

    # Restore grayscale if needed
    if was_grayscale:
        out_img = out_img.convert("L")

    return out_img


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reales4u2d.py <image>")
        sys.exit(1)

    src_path = Path(sys.argv[1])
    if not src_path.exists():
        print(f"File not found: {src_path}")
        sys.exit(1)

    print(f"Loading model...")
    load_model()

    print(f"Processing: {src_path.name}")
    img = Image.open(src_path)
    result = reales4u2d(img)

    out_path = src_path.parent / f"{src_path.stem}_realx4down2.png"
    result.save(out_path)
    print(f"Saved: {out_path}")
