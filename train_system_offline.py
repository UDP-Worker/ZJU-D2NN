"""
Offline training for system identification in the SLM1 -> SLM2 -> camera pipeline.

Expected dataset format (.npz or .pt):
  - phase_slm1: (N, H, W) float32, phase in radians.
  - camera_intensity: (N, H, W) float32, grayscale intensity.

Optional keys accepted:
  - slm1_phase, phase
  - intensity, camera
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_system import build_first_half_model, build_second_half_model


def _pick_key(data, keys):
    for key in keys:
        if key in data:
            return key
    return None


def _squeeze_hw(x):
    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0, :, :]
    if x.ndim == 4 and x.shape[-1] == 1:
        return x[:, :, :, 0]
    if x.ndim == 2:
        return x[None, :, :]
    return x


def _to_grayscale(intensity):
    if intensity.ndim == 4 and intensity.shape[-1] == 3:
        # Assume BGR order from OpenCV.
        b = intensity[..., 0]
        g = intensity[..., 1]
        r = intensity[..., 2]
        return 0.114 * b + 0.587 * g + 0.299 * r
    return intensity


def load_dataset(path: str):
    data_path = Path(path)
    if data_path.suffix == ".npz":
        data = np.load(data_path)
        phase_key = _pick_key(data, ["phase_slm1", "slm1_phase", "phase"])
        intensity_key = _pick_key(data, ["camera_intensity", "intensity", "camera"])
        if phase_key is None or intensity_key is None:
            raise KeyError("Missing required keys in npz: phase_slm1 and camera_intensity.")
        phase = data[phase_key]
        intensity = data[intensity_key]
    elif data_path.suffix in (".pt", ".pth"):
        data = torch.load(data_path, map_location="cpu")
        if not isinstance(data, dict):
            raise ValueError("Torch file must contain a dict.")
        phase_key = _pick_key(data, ["phase_slm1", "slm1_phase", "phase"])
        intensity_key = _pick_key(data, ["camera_intensity", "intensity", "camera"])
        if phase_key is None or intensity_key is None:
            raise KeyError("Missing required keys in torch dict: phase_slm1 and camera_intensity.")
        phase = data[phase_key]
        intensity = data[intensity_key]
    else:
        raise ValueError(f"Unsupported data format: {data_path.suffix}")

    phase = _squeeze_hw(phase)
    intensity = _squeeze_hw(intensity)
    intensity = _to_grayscale(intensity)

    phase = torch.as_tensor(phase, dtype=torch.float32)
    intensity = torch.as_tensor(intensity, dtype=torch.float32)

    if phase.ndim != 3 or intensity.ndim != 3:
        raise ValueError("phase_slm1 and camera_intensity must be (N, H, W).")
    if phase.shape != intensity.shape:
        raise ValueError(f"Shape mismatch: phase {phase.shape} vs intensity {intensity.shape}")

    return phase, intensity


class PhaseIntensityDataset(Dataset):
    def __init__(self, phase, intensity):
        self.phase = phase
        self.intensity = intensity

    def __len__(self):
        return self.phase.shape[0]

    def __getitem__(self, idx):
        return self.phase[idx], self.intensity[idx]


class TrainSystem(nn.Module):
    def __init__(self, propagation1: nn.Module, propagation2: nn.Module, slm2_phase):
        super().__init__()
        self.propagation1 = propagation1
        self.propagation2 = propagation2
        if slm2_phase is None:
            self.slm2_phase_exp = None
        else:
            slm2_phase = slm2_phase.to(torch.float32)
            self.register_buffer("slm2_phase_exp", torch.exp(1j * slm2_phase), persistent=True)

    def forward(self, phase_slm1: torch.Tensor) -> torch.Tensor:
        input_field = torch.exp(1j * phase_slm1)
        field_slm2 = self.propagation1(input_field)
        if self.slm2_phase_exp is not None:
            field_slm2 = field_slm2 * self.slm2_phase_exp
        field_camera = self.propagation2(field_slm2)
        return field_camera


def normalize_intensity(x: torch.Tensor, mode: str, eps: float):
    if mode == "none":
        return x
    if mode == "sum":
        denom = x.sum(dim=(-2, -1), keepdim=True)
        return x / (denom + eps)
    if mode == "mean":
        denom = x.mean(dim=(-2, -1), keepdim=True)
        return x / (denom + eps)
    raise ValueError(f"Unknown normalize mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to .npz or .pt dataset.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_name", type=str, default="system_offline.pt")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "l1"])
    parser.add_argument("--normalize", type=str, default="sum", choices=["sum", "mean", "none"])
    parser.add_argument("--phase_scale", type=float, default=1.0, help="Scale for phase_slm1.")
    parser.add_argument("--intensity_scale", type=float, default=1.0, help="Scale for camera_intensity.")
    parser.add_argument("--slm2_phase", type=str, default="", help="Optional .npy or .pt for SLM2 phase (H,W).")

    parser.add_argument("--wavelength", type=float, required=True)
    parser.add_argument("--dx", type=float, required=True)
    parser.add_argument("--dy", type=float, required=True)
    parser.add_argument("--f_4f", type=float, required=True)
    parser.add_argument("--f_camera", type=float, required=True)
    parser.add_argument("--slm1_to_l1", type=float, default=None)
    parser.add_argument("--l1_to_slm2", type=float, default=None)
    parser.add_argument("--slm2_to_l2", type=float, default=None)
    parser.add_argument("--l2_to_cam_lens", type=float, required=True)
    parser.add_argument("--cam_lens_to_sensor", type=float, required=True)

    parser.add_argument("--learnable_slm1_to_l1", action="store_true")
    parser.add_argument("--learnable_l1_to_slm2", action="store_true")
    parser.add_argument("--learnable_slm2_to_l2", action="store_true")
    parser.add_argument("--learnable_l2_to_cam_lens", action="store_true")
    parser.add_argument("--learnable_cam_lens_to_sensor", action="store_true")
    parser.add_argument("--learnable_slm2_offset", action="store_true")
    parser.add_argument("--learnable_slm2_tilt", action="store_true")
    parser.add_argument("--learnable_camera_offset", action="store_true")
    parser.add_argument("--learnable_camera_tilt", action="store_true")

    parser.add_argument("--slm2_offset_xy", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--slm2_tilt_xy", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--camera_offset_xy", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--camera_tilt_xy", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--slm2_tilt_scale", type=float, default=1.0)
    parser.add_argument("--camera_tilt_scale", type=float, default=1.0)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    print(f"[INFO] device = {device}")

    phase, intensity = load_dataset(args.data)
    phase = phase * float(args.phase_scale)
    intensity = intensity * float(args.intensity_scale)

    H, W = phase.shape[-2], phase.shape[-1]

    slm2_phase = None
    if args.slm2_phase:
        slm2_path = Path(args.slm2_phase)
        if slm2_path.suffix in (".pt", ".pth"):
            slm2_phase = torch.load(slm2_path, map_location="cpu")
        else:
            slm2_phase = np.load(slm2_path)
        slm2_phase = torch.as_tensor(slm2_phase, dtype=torch.float32)
        if slm2_phase.shape != (H, W):
            raise ValueError(f"slm2_phase must be (H, W)=({H},{W}), got {slm2_phase.shape}")

    dataset = PhaseIntensityDataset(phase, intensity)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    propagation1 = build_first_half_model(
        wavelength=args.wavelength,
        dx=args.dx,
        dy=args.dy,
        f_4f=args.f_4f,
        slm1_to_l1=args.slm1_to_l1,
        l1_to_slm2=args.l1_to_slm2,
        learnable_slm1_to_l1=args.learnable_slm1_to_l1,
        learnable_l1_to_slm2=args.learnable_l1_to_slm2,
        slm2_offset_xy=args.slm2_offset_xy,
        slm2_tilt_xy=args.slm2_tilt_xy,
        learnable_slm2_offset=args.learnable_slm2_offset,
        learnable_slm2_tilt=args.learnable_slm2_tilt,
        slm2_tilt_scale=args.slm2_tilt_scale,
        dtype=torch.complex64,
    )
    propagation2 = build_second_half_model(
        wavelength=args.wavelength,
        dx=args.dx,
        dy=args.dy,
        f_4f=args.f_4f,
        f_camera=args.f_camera,
        slm2_to_l2=args.slm2_to_l2,
        l2_to_cam_lens=args.l2_to_cam_lens,
        cam_lens_to_sensor=args.cam_lens_to_sensor,
        learnable_slm2_to_l2=args.learnable_slm2_to_l2,
        learnable_l2_to_cam_lens=args.learnable_l2_to_cam_lens,
        learnable_cam_lens_to_sensor=args.learnable_cam_lens_to_sensor,
        camera_offset_xy=args.camera_offset_xy,
        camera_tilt_xy=args.camera_tilt_xy,
        learnable_camera_offset=args.learnable_camera_offset,
        learnable_camera_tilt=args.learnable_camera_tilt,
        camera_tilt_scale=args.camera_tilt_scale,
        dtype=torch.complex64,
    )

    model = TrainSystem(propagation1, propagation2, slm2_phase).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No learnable parameters. Enable learnable_* flags.")
    optimizer = torch.optim.Adam(params, lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for phase_batch, intensity_batch in loader:
            phase_batch = phase_batch.to(device)
            intensity_batch = intensity_batch.to(device)

            field_camera = model(phase_batch)
            pred_intensity = torch.abs(field_camera) ** 2

            pred_intensity = normalize_intensity(pred_intensity, args.normalize, eps=1e-12)
            target_intensity = normalize_intensity(intensity_batch, args.normalize, eps=1e-12)

            if args.loss == "mse":
                loss = F.mse_loss(pred_intensity, target_intensity)
            else:
                loss = F.l1_loss(pred_intensity, target_intensity)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * phase_batch.size(0)
            total_n += phase_batch.size(0)

        avg_loss = total_loss / max(total_n, 1)
        print(f"[Epoch {epoch:03d}/{args.epochs}] loss={avg_loss:.6f}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / args.save_name
    ckpt = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "H": H,
        "W": W,
    }
    torch.save(ckpt, ckpt_path)
    print(f"[INFO] Saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
