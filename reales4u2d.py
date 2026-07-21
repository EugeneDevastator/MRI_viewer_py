"""
reales4u2d.py
Upscale x4 with RealESRGAN_x4plus.pth, then downscale to x2 with Lanczos.
Usage: python reales4u2d.py image.png
As module: from reales4u2d import reales4u2d, load_model
"""

import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import spandrel

MODEL_PATH = Path(__file__).parent / "RealESRGAN_x4plus.pth"

_model = None
_device = None


def get_device():
    print(f"  PyTorch version: {torch.version}")
    print(f"  CUDA built-in version: {torch.version.cuda}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"  CUDA device count: {count}")
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            vram_mb = props.total_memory // (1024 * 1024)
            print(f"    [{i}] {props.name} — {vram_mb} MB VRAM")
        best = max(range(count),
                   key=lambda i: torch.cuda.get_device_properties(i).total_memory)
        dev = torch.device(f"cuda:{best}")
        print(f"  Selected: cuda:{best} ({torch.cuda.get_device_name(best)})")
        return dev
    else:
        print("  CUDA not available — possible causes:")
        print("    - PyTorch installed without CUDA (cpu-only build)")
        print("    - CUDA toolkit version mismatch")
        print("    - Run: python -c \"import torch; print(torch.version)\"")
        print("    - Check if torch was installed via: pip install torch (cpu) vs pip install torch --index-url https://download.pytorch.org/whl/cu121")

    try:
        import torch_directml
        dml_count = torch_directml.device_count()
        print(f"  DirectML device count: {dml_count}")
        for i in range(dml_count):
            print(f"    [{i}] {torch_directml.device_name(i)}")
        dev = torch_directml.device(0)
        print(f"  Selected: DirectML device 0")
        return dev
    except ImportError:
        print("  DirectML: not installed (pip install torch-directml)")
    except Exception as e:
        print(f"  DirectML error: {e}")

    print("  Falling back to CPU")
    return torch.device("cpu")


def load_model():
    global _model, _device
    if _model is None:
        print("Detecting compute device...")
        _device = get_device()
        print(f"Loading model from: {MODEL_PATH}")
        _model = spandrel.ModelLoader().load_from_file(str(MODEL_PATH))
        _model.eval()
        _model.to(_device)
        print(f"Model loaded and ready on {_device}")
    return _model, _device


def reales4u2d(image: Image.Image) -> Image.Image:
    was_grayscale = image.mode == "L"
    rgb = image.convert("RGB")
    w, h = rgb.size

    arr = np.array(rgb).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    model, device = load_model()
    tensor = tensor.to(device)

    with torch.no_grad():
        out_tensor = model(tensor)

    out_arr = out_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    out_arr = (out_arr * 255).astype(np.uint8)
    out_img = Image.fromarray(out_arr, mode="RGB")

    target_w, target_h = w * 2, h * 2
    out_img = out_img.resize((target_w, target_h), Image.LANCZOS)

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

    load_model()
    img = Image.open(src_path)
    result = reales4u2d(img)
    out_path = src_path.parent / f"{src_path.stem}_realx4down2.png"
    result.save(out_path)
    print(f"Saved: {out_path}")