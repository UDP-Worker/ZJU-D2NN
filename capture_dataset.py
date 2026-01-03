"""
Capture offline dataset: SLM1 phase patterns paired with camera intensity images.

Output format (.npz):
  - phase_slm1: (N, H, W) float32, phase in radians.
  - camera_intensity: (N, Hc, Wc) float32, grayscale intensity.
"""

import argparse
import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import utils
from read_from_camera import read_from_camera


def _get_resolution(channel: int, height: Optional[int], width: Optional[int]):
    if height is not None and width is not None:
        return height, width
    outputs = utils.list_hdmi_outputs()
    for output in outputs:
        if int(output["index"]) == channel:
            w, h = output["resolution"]
            return int(h), int(w)
    raise ValueError(f"HDMI channel {channel} not found in list_hdmi_outputs().")


def _phase_to_uint8(phase: np.ndarray) -> np.ndarray:
    phase = np.mod(phase, 2 * math.pi)
    return np.round(phase / (2 * math.pi) * 255.0).astype(np.uint8)


def _center_crop(img: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if crop_h > h or crop_w > w:
        raise ValueError(f"Crop size ({crop_h},{crop_w}) exceeds image size ({h},{w}).")
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    return img[y0:y0 + crop_h, x0:x0 + crop_w]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=50, help="Number of samples to capture.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slm1_channel", type=int, default=1)
    parser.add_argument("--slm2_channel", type=int, default=2)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--slm2_phase", type=str, default="", help="Optional SLM2 phase file (.npy/.pt).")
    parser.add_argument("--settle_ms", type=int, default=200)
    parser.add_argument("--crop_h", type=int, default=None)
    parser.add_argument("--crop_w", type=int, default=None)
    parser.add_argument("--output", type=str, default="dataset.npz")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    slm1_h, slm1_w = _get_resolution(args.slm1_channel, args.height, args.width)

    if args.slm2_phase:
        slm2_h, slm2_w = _get_resolution(args.slm2_channel, None, None)
        slm2_path = Path(args.slm2_phase)
        if slm2_path.suffix in (".pt", ".pth"):
            import torch

            slm2_phase = torch.load(slm2_path, map_location="cpu")
            slm2_phase = np.asarray(slm2_phase, dtype=np.float32)
        else:
            slm2_phase = np.load(slm2_path).astype(np.float32)
        if slm2_phase.shape != (slm2_h, slm2_w):
            raise ValueError(f"SLM2 phase must be ({slm2_h},{slm2_w}), got {slm2_phase.shape}")
        slm2_frame = _phase_to_uint8(slm2_phase)
        utils.write_to_hdmi(slm2_frame, args.slm2_channel)
        time.sleep(args.settle_ms / 1000.0)

    phase_list = []
    intensity_list = []

    for idx in range(args.num):
        phase = rng.random((slm1_h, slm1_w)).astype(np.float32) * (2 * math.pi)
        frame = _phase_to_uint8(phase)
        utils.write_to_hdmi(frame, args.slm1_channel)
        time.sleep(args.settle_ms / 1000.0)

        cam_frame = read_from_camera()
        gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if args.crop_h is not None and args.crop_w is not None:
            gray = _center_crop(gray, args.crop_h, args.crop_w)

        phase_list.append(phase)
        intensity_list.append(gray)
        print(f"[INFO] Captured {idx + 1}/{args.num}")

    phase_arr = np.stack(phase_list, axis=0)
    intensity_arr = np.stack(intensity_list, axis=0)

    np.savez_compressed(args.output, phase_slm1=phase_arr, camera_intensity=intensity_arr)
    print(f"[INFO] Saved dataset to {args.output}")


if __name__ == "__main__":
    main()
