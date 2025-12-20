"""
Optimize the images loaded to SLM2 i.e. the weight of optical neural network
to recognize MNIST handwritten digits.

Loss is defined as the cross entropy loss, while the grey scale of certain area
on the camera is recognized as probability.
"""

import math
import utils
import torch
from torch import nn
import torch.nn.functional as F

class TrainONNSystem(nn.Module):
    def __init__(
            self,
            propagation1: nn.Module,  # SLM1 to SLM2
            propagation2: nn.Module,  # SLM2 to camera
            slm2_init: torch.Tensor,   # Initial phase pattern for SLM2
            slm2_learnable: bool = True,
    ):
        super().__init__()
        self.propagation1 = propagation1
        self.propagation2 = propagation2

        # SLM2 phase pattern
        if slm2_learnable:
            self.slm2_phase = nn.Parameter(slm2_init)
        else:
            self.register_buffer("slm2_phase", slm2_init, persistent=True)

    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """
        input_field: (..., H, W) complex field at SLM1 plane
        return: (..., H, W) complex field at camera plane
        """
        # Propagate from SLM1 to SLM2
        field_slm2 = self.propagation1(input_field)  # (..., H, W)

        # Apply SLM2 phase modulation
        slm2_phase_exp = torch.exp(1j * self.slm2_phase)  # (H, W)
        field_slm2_modulated = field_slm2 * slm2_phase_exp  # (..., H, W)

        # Propagate from SLM2 to camera
        field_camera = self.propagation2(field_slm2_modulated)  # (..., H, W)

        return field_camera
    
    
def calculate_loss(
    field_camera: torch.Tensor,      # (B, H, W) complex
    labels: torch.Tensor,            # (B,) long, 0..9
    readout_pixels: torch.Tensor,    # (10, 2) long, each [y, x]
    eps: float = 1e-12,
) -> torch.Tensor:


    intensity = torch.abs(field_camera) ** 2  # (B, H, W)

    readout_pixels = readout_pixels.to(device=intensity.device, dtype=torch.long)
    ys = readout_pixels[:, 0]
    xs = readout_pixels[:, 1]

    scores = intensity[:, ys, xs]  # (B, 10), non-negative

    probs = scores / (scores.sum(dim=1, keepdim=True) + eps)  # (B, 10)
    log_probs = torch.log(probs + eps)

    labels = labels.to(device=intensity.device, dtype=torch.long)
    loss = F.nll_loss(log_probs, labels)

    return loss