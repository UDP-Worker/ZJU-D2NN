# simulation/ONN_train.py
"""
Train SLM2 phase (the "weight" of the optical neural network) for MNIST classification.

- system.py defines device-level simulation (ASM free-space + thin lens + resampling).
- SLM1 is PHASE-ONLY and carries input images via phase encoding (NOT trained here).
- SLM2 is PHASE-ONLY and TRAINED (phi2 is learnable).

Readout:
- Use 10 ROIs on camera intensity as logits (energy/mean in each ROI).
"""

import math
import argparse
from pathlib import Path
import sys
from typing import Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

# Ensure we can import system.py from the same folder
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import system  # noqa: E402


# ============================================================
# 1) SLM1 encoding (phase-only)
# ============================================================

def encode_mnist_to_slm1_phase_field(
    images: torch.Tensor,
    out_hw=(system.H_SLM1, system.W_SLM1),
    phase_scale: float = 2.0 * math.pi,
) -> torch.Tensor:
    """
    Phase-only SLM1 encoding:
      images: (B,1,28,28), float in [0,1]
      phi1 = phase_scale * images_resized
      U_slm1 = exp(i*phi1)   (plane-wave amplitude = 1)

    return:
      U_slm1: (B, H_SLM1, W_SLM1) complex
    """
    assert images.ndim == 4 and images.shape[1] == 1, "images must be (B,1,H,W)"
    img = images.clamp(0, 1).to(torch.float32)  # (B,1,28,28)

    img_big = F.interpolate(img, size=out_hw, mode="bilinear", align_corners=False)  # (B,1,H,W)
    img_big = img_big[:, 0, :, :]  # (B,H,W)

    phi1 = phase_scale * img_big  # radians
    U_slm1 = torch.exp(1j * phi1)  # amplitude=1
    return U_slm1.to(system.CDTYPE)


# ============================================================
# 2) Readout (logits from camera intensity)
# ============================================================

def make_readout_rois(
    H: int,
    W: int,
    box_hw=(32, 32),
    x_range=(0.15, 0.85),
    y_center_frac=0.5,
) -> torch.Tensor:
    """
    Create 10 ROIs on camera plane.
    Returns tensor of shape (10, 4): [y0, y1, x0, x1] per class.

    Default: 10 boxes evenly spaced along x at center row.
    """
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


def make_roi_slices(rois: torch.Tensor) -> list[Tuple[slice, slice]]:
    """
    Convert ROI tensor to python slice pairs for fast indexing without GPU sync.
    """
    rois_cpu = rois.to(device="cpu", dtype=torch.long)
    slices: list[Tuple[slice, slice]] = []
    for y0, y1, x0, x1 in rois_cpu.tolist():
        slices.append((slice(int(y0), int(y1)), slice(int(x0), int(x1))))
    return slices


def readout_logits_from_intensity(
    I_cam: torch.Tensor,     # (B,H,W) float
    roi_slices: Sequence[Tuple[slice, slice]],
    mode: str = "mean",      # "mean" or "sum"
) -> torch.Tensor:
    """
    Convert camera intensity to logits (B,10) by ROI energy.
    """
    logits = []
    for y_slice, x_slice in roi_slices:
        patch = I_cam[:, y_slice, x_slice]  # (B,bh,bw)
        if mode == "sum":
            s = patch.sum(dim=(-2, -1))
        else:
            s = patch.mean(dim=(-2, -1))
        logits.append(s)

    return torch.stack(logits, dim=-1)  # (B,10)


# ============================================================
# 3) Model wrapper: only SLM2 is learnable
# ============================================================

class TrainONNSystem(nn.Module):
    """
    Wrap system.OpticalSystem and expose learnable SLM2 phase.

    The optics modules are frozen by design; only slm2_param is optimized.
    """
    def __init__(
        self,
        slm2_init: torch.Tensor | None = None,   # (H_SLM2,W_SLM2) radians
        slm2_learnable: bool = True,
        phi_range: float = 2.0 * math.pi,        # map to [0, phi_range)
        init_in_radians: bool = True,
    ):
        super().__init__()
        self.sys = system.OpticalSystem(dtype=system.CDTYPE)
        self.phi_range = float(phi_range)

        H2, W2 = system.H_SLM2, system.W_SLM2

        if slm2_init is None:
            phi0 = (2.0 * math.pi) * torch.rand((H2, W2), dtype=torch.float32)
        else:
            if not (isinstance(slm2_init, torch.Tensor) and tuple(slm2_init.shape) == (H2, W2)):
                raise ValueError(f"slm2_init must have shape ({H2},{W2})")
            slm2_init = slm2_init.to(torch.float32)
            if init_in_radians:
                phi0 = torch.remainder(slm2_init, 2.0 * math.pi)
            else:
                phi0 = (2.0 * math.pi) * torch.sigmoid(slm2_init)

        # Parametrize with unconstrained p, map to [0,2π) via sigmoid.
        t = (phi0 / (2.0 * math.pi)).clamp(1e-6, 1 - 1e-6)
        p0 = torch.log(t / (1.0 - t))

        if slm2_learnable:
            self.slm2_param = nn.Parameter(p0)
        else:
            self.register_buffer("slm2_param", p0, persistent=True)

        # Freeze system params (safety; system has no learnables by default)
        for p in self.sys.parameters():
            p.requires_grad_(False)

    def slm2_phase(self) -> torch.Tensor:
        """Return phase in [0,phi_range) radians, shape (H_SLM2,W_SLM2)."""
        return self.phi_range * torch.sigmoid(self.slm2_param)

    def forward(self, U_slm1: torch.Tensor) -> torch.Tensor:
        """
        U_slm1: (B,H_SLM1,W_SLM1) complex
        return: U_cam (B,H_CAM,W_CAM) complex
        """
        U_slm2 = self.sys.forward_to_slm2(U_slm1)          # (B,H2,W2)
        phi2 = self.slm2_phase()                           # (H2,W2)
        U_slm2_mod = system.apply_phase(U_slm2, phi2)      # (B,H2,W2)
        U_cam = self.sys.forward_to_camera(U_slm2_mod)     # (B,Hc,Wc)
        return U_cam


# ============================================================
# 4) Train / Eval
# ============================================================

@torch.no_grad()
def evaluate(model: TrainONNSystem, loader, roi_slices, device, show_progress: bool):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    non_blocking = (device.type == "cuda")
    pbar = iter_with_progress(loader, desc="Eval", enabled=show_progress)
    for images, labels in pbar:
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        U_slm1 = encode_mnist_to_slm1_phase_field(images)
        U_cam = model(U_slm1)
        I_cam = system.intensity(U_cam)

        logits = readout_logits_from_intensity(I_cam, roi_slices, mode="mean")
        loss = F.cross_entropy(logits, labels)

        pred = logits.argmax(dim=1)
        total_correct += int((pred == labels).sum().item())
        total_loss += float(loss.item()) * images.size(0)
        total_n += images.size(0)
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


def select_device(no_cuda: bool) -> torch.device:
    """
    Device selection with CUDA / MPS / CPU support.
    - If no_cuda is True: always CPU.
    - Else: prefer CUDA, then MPS, then CPU.
    """
    if no_cuda:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    # MPS (Apple Silicon)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def log_device_info(device: torch.device, no_cuda: bool) -> None:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        cap_major, cap_minor = torch.cuda.get_device_capability(device)
        print(f"[INFO] cuda device = {name} (cc {cap_major}.{cap_minor})")
        return
    if not no_cuda:
        print(
            "[WARN] CUDA not available. "
            f"torch.backends.cuda.is_built()={torch.backends.cuda.is_built()} "
            f"torch.version.cuda={torch.version.cuda} "
            f"device_count={torch.cuda.device_count()}"
        )


def iter_with_progress(loader, desc: str, enabled: bool):
    if not enabled or tqdm is None:
        return loader
    return tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_name", type=str, default="onn_slm2.pt")

    # training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=1)          # camera plane is huge
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_progress", action="store_true", help="disable tqdm progress bars")

    # readout ROI
    parser.add_argument("--roi_box", type=int, default=32, help="ROI box size (square), e.g. 16/32/64")

    # slm2 init
    parser.add_argument("--slm2_init", type=str, default="", help="optional .pt tensor (H_SLM2,W_SLM2) in radians")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = select_device(args.no_cuda)
    print(f"[INFO] device = {device}")
    log_device_info(device, args.no_cuda)

    # MNIST
    tfm = transforms.ToTensor()
    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=tfm)

    pin = (device.type == "cuda")  # pin_memory only helps CUDA
    non_blocking = pin
    persistent = args.num_workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )

    # slm2 init
    slm2_init = None
    if args.slm2_init:
        slm2_init = torch.load(args.slm2_init, map_location="cpu")
        print(f"[INFO] loaded slm2_init from: {args.slm2_init}")

    # model
    model = TrainONNSystem(slm2_init=slm2_init, slm2_learnable=True, init_in_radians=True).to(device)
    print("[INFO] system info:", model.sys.info())

    # ROIs on camera plane
    rois = make_readout_rois(
        H=system.H_CAM,
        W=system.W_CAM,
        box_hw=(args.roi_box, args.roi_box),
        x_range=(0.15, 0.85),
        y_center_frac=0.5,
    )
    roi_slices = make_roi_slices(rois)

    # optimizer (ONLY SLM2)
    optimizer = torch.optim.Adam([model.slm2_param], lr=args.lr)

    # train
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_n = 0

        pbar = iter_with_progress(
            train_loader,
            desc=f"Train {epoch:02d}/{args.epochs}",
            enabled=not args.no_progress,
        )
        for images, labels in pbar:
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            U_slm1 = encode_mnist_to_slm1_phase_field(images)
            U_cam = model(U_slm1)
            I_cam = system.intensity(U_cam)

            logits = readout_logits_from_intensity(I_cam, roi_slices, mode="mean")
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * images.size(0)
            running_n += images.size(0)
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(running_n, 1)
        test_loss, test_acc = evaluate(
            model,
            test_loader,
            roi_slices,
            device,
            show_progress=not args.no_progress,
        )

        print(f"[Epoch {epoch:02d}/{args.epochs}] "
              f"train_loss={train_loss:.4f}  test_loss={test_loss:.4f}  test_acc={test_acc*100:.2f}%")

    # save
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / args.save_name

    ckpt = {
        "slm2_param": model.slm2_param.detach().cpu(),
        "slm2_phase": model.slm2_phase().detach().cpu(),  # radians in [0,2π)
        "rois": rois.detach().cpu(),
        "system_info": model.sys.info(),
        "args": vars(args),
    }
    torch.save(ckpt, ckpt_path)
    print(f"[INFO] saved: {ckpt_path}")


if __name__ == "__main__":
    main()
