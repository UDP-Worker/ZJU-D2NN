# simulation/main.py
"""
Inference (simulation) for ONE MNIST 28x28 image using the full optical model.

What it does:
1) Load ONE 28x28 MNIST image (from a file OR from torchvision MNIST by index)
2) Encode it onto SLM1 (PHASE-ONLY): U_slm1 = exp(i * 2π * resized_image)
3) Load trained SLM2 phase from a checkpoint (.pt produced by ONN_train.py)
4) Forward propagate: SLM1 -> SLM2 -> Camera
5) Compute camera intensity image |U_cam|^2
6) Compute logits by 10 ROIs (loaded from checkpoint if present)
7) Pick the brightest ROI => predicted label
8) Save a PNG of the simulated camera intensity with the brightest ROI boxed and label annotated

Examples:
  # Use a local 28x28 image (png/jpg). It will be converted to grayscale and resized to 28x28.
  python main.py --ckpt ./checkpoints/onn_slm2.pt --image_path ./my_28x28.png

  # Use MNIST test-set index (downloads MNIST into ./data if needed)
  python main.py --ckpt ./checkpoints/onn_slm2.pt --mnist_index 0 --mnist_split test --data_dir ./data
"""

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure we can import system.py from the same folder
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import system  # noqa: E402


# -----------------------------
# Device selection (CUDA/MPS/CPU)
# -----------------------------
def select_device(no_cuda: bool) -> torch.device:
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------
# SLM1 encoding (phase-only)
# -----------------------------
def encode_mnist_to_slm1_phase_field(
    images: torch.Tensor,  # (B,1,28,28) in [0,1]
    out_hw=(system.H_SLM1, system.W_SLM1),
    phase_scale: float = 2.0 * math.pi,
) -> torch.Tensor:
    """
    Phase-only SLM1 encoding:
      phi1 = phase_scale * images_resized
      U_slm1 = exp(i*phi1)   (amplitude=1)
    Returns:
      U_slm1: (B,H_SLM1,W_SLM1) complex
    """
    assert images.ndim == 4 and images.shape[1:] == (1, 28, 28), "images must be (B,1,28,28)"
    img = images.clamp(0, 1).to(torch.float32)  # (B,1,28,28)
    img_big = F.interpolate(img, size=out_hw, mode="bilinear", align_corners=False)  # (B,1,H,W)
    img_big = img_big[:, 0, :, :]  # (B,H,W)
    phi1 = phase_scale * img_big
    U_slm1 = torch.exp(1j * phi1)
    return U_slm1.to(system.CDTYPE)


# -----------------------------
# ROI helpers
# -----------------------------
def make_default_rois(H: int, W: int, box_hw=(32, 32), x_range=(0.15, 0.85), y_center_frac=0.5) -> torch.Tensor:
    bh, bw = box_hw
    cy = int(round(H * y_center_frac))
    cy = max(bh // 2, min(H - 1 - bh // 2, cy))

    x0f, x1f = x_range
    xs = torch.linspace(W * x0f, W * x1f, steps=10).round().long()

    rois = []
    for x in xs.tolist():
        x = int(x)
        x = max(bw // 2, min(W - 1 - bw // 2, x))
        y0 = cy - bh // 2
        y1 = y0 + bh
        x0 = x - bw // 2
        x1 = x0 + bw
        rois.append([y0, y1, x0, x1])
    return torch.tensor(rois, dtype=torch.long)  # (10,4)


def logits_from_intensity(I_cam: torch.Tensor, rois: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """
    I_cam: (B,H,W) float
    rois: (10,4) long y0,y1,x0,x1
    returns: logits (B,10)
    """
    rois = rois.to(device=I_cam.device, dtype=torch.long)
    out = []
    for k in range(10):
        y0, y1, x0, x1 = rois[k].tolist()
        patch = I_cam[:, y0:y1, x0:x1]
        if mode == "sum":
            out.append(patch.sum(dim=(-2, -1)))
        else:
            out.append(patch.mean(dim=(-2, -1)))
    return torch.stack(out, dim=-1)


# -----------------------------
# Load ONE MNIST-like 28x28 image
# -----------------------------
def load_one_image_as_tensor(args) -> torch.Tensor:
    """
    Returns images: (1,1,28,28) float in [0,1]
    """
    if args.image_path:
        p = Path(args.image_path)
        if not p.exists():
            raise FileNotFoundError(f"--image_path not found: {p}")
        img = Image.open(p).convert("L").resize((28, 28), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0  # (28,28) in [0,1]
        t = torch.from_numpy(arr)[None, None, :, :]    # (1,1,28,28)
        return t

    # Else: load from torchvision MNIST by index
    from torchvision import datasets, transforms  # local import to avoid hard dependency if not used
    tfm = transforms.ToTensor()

    split = args.mnist_split.lower()
    if split not in ("train", "test"):
        raise ValueError("--mnist_split must be 'train' or 'test'")

    ds = datasets.MNIST(args.data_dir, train=(split == "train"), download=True, transform=tfm)
    idx = int(args.mnist_index)
    if idx < 0 or idx >= len(ds):
        raise ValueError(f"mnist_index out of range: {idx} (len={len(ds)})")

    img, _label = ds[idx]  # img: (1,28,28)
    return img.unsqueeze(0)  # (1,1,28,28)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint (.pt) from ONN_train.py")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Directory to save output images")
    parser.add_argument("--out_name", type=str, default="camera_pred.png", help="Output PNG filename")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU")

    # Input options (choose one)
    parser.add_argument("--image_path", type=str, default="", help="Path to a local image (will be converted to 28x28 grayscale)")
    parser.add_argument("--mnist_index", type=int, default=0, help="MNIST index (used if --image_path is empty)")
    parser.add_argument("--mnist_split", type=str, default="test", help="train or test (used if --image_path is empty)")
    parser.add_argument("--data_dir", type=str, default="./data", help="MNIST dataset directory (used if --image_path is empty)")

    # Readout
    parser.add_argument("--roi_mode", type=str, default="mean", choices=["mean", "sum"], help="ROI reduction mode")
    parser.add_argument("--fallback_roi_box", type=int, default=32, help="ROI box if ckpt has no rois")

    args = parser.parse_args()

    device = select_device(args.no_cuda)
    print(f"[INFO] device = {device}")

    # 1) Load checkpoint (SLM2 phase + ROIs)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "slm2_phase" not in ckpt:
        raise KeyError("Checkpoint missing key 'slm2_phase'. Please save slm2_phase in ONN_train.py.")
    slm2_phase = ckpt["slm2_phase"]  # (H_SLM2,W_SLM2) radians

    if not (isinstance(slm2_phase, torch.Tensor) and tuple(slm2_phase.shape) == (system.H_SLM2, system.W_SLM2)):
        raise ValueError(
            f"slm2_phase must have shape ({system.H_SLM2},{system.W_SLM2}), got {getattr(slm2_phase, 'shape', None)}"
        )
    slm2_phase = slm2_phase.to(device=device, dtype=torch.float32)

    if "rois" in ckpt and isinstance(ckpt["rois"], torch.Tensor) and ckpt["rois"].shape == (10, 4):
        rois = ckpt["rois"].to(device=device, dtype=torch.long)
        print("[INFO] Loaded ROIs from checkpoint.")
    else:
        rois = make_default_rois(system.H_CAM, system.W_CAM, box_hw=(args.fallback_roi_box, args.fallback_roi_box)).to(device)
        print("[WARN] No ROIs in checkpoint; using default ROIs.")

    # 2) Load one input image
    images = load_one_image_as_tensor(args).to(device=device, dtype=torch.float32)  # (1,1,28,28)

    # 3) Build optical system
    sys_model = system.OpticalSystem(dtype=system.CDTYPE).to(device)
    sys_model.eval()

    # 4) Forward simulation + prediction
    with torch.no_grad():
        U_slm1 = encode_mnist_to_slm1_phase_field(images).to(device)  # (1,H_SLM1,W_SLM1) complex

        U_slm2 = sys_model.forward_to_slm2(U_slm1)  # (1,H_SLM2,W_SLM2) complex
        U_slm2_mod = system.apply_phase(U_slm2, slm2_phase)  # multiply exp(i*phi2)

        U_cam = sys_model.forward_to_camera(U_slm2_mod)      # (1,H_CAM,W_CAM) complex
        I_cam = system.intensity(U_cam).to(torch.float32)    # (1,H_CAM,W_CAM)

        logits = logits_from_intensity(I_cam, rois, mode=args.roi_mode)  # (1,10)
        pred = int(logits.argmax(dim=1).item())

        # Find brightest ROI (same as pred)
        y0, y1, x0, x1 = rois[pred].tolist()

    print(f"[RESULT] predicted label = {pred}")

    # 5) Save camera image with bbox + label
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name

    # Convert intensity to numpy for plotting; use log for visualization (optional)
    I_np = I_cam[0].detach().cpu().numpy()
    I_vis = np.log1p(I_np)  # improves dynamic range for display

    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.imshow(I_vis, cmap="gray")
    rect = patches.Rectangle(
        (x0, y0), (x1 - x0), (y1 - y0),
        linewidth=2, edgecolor="lime", facecolor="none"
    )
    ax.add_patch(rect)
    ax.text(
        x0, max(0, y0 - 10),
        f"pred = {pred}",
        color="lime",
        fontsize=14,
        bbox=dict(facecolor="black", alpha=0.6, pad=4),
    )
    ax.set_title("Simulated Camera Intensity (log1p) with Brightest ROI")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    # (Optional) also save raw intensity as .npy
    raw_path = out_dir / (Path(args.out_name).stem + "_intensity.npy")
    np.save(raw_path, I_np)

    print(f"[INFO] saved camera PNG: {out_path}")
    print(f"[INFO] saved raw intensity: {raw_path}")


if __name__ == "__main__":
    main()