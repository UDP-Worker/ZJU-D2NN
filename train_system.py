import math

import torch
from torch import nn

"""
Train a neural network or use a normal function to describe the propagating process
from SLM1 to SLM2. Also train a neural network or use a normal function to describe
the propagating process from SLM2 to camera.

To train these two neural network, we can utilize programmable SLM2.
"""

class FreePropagation(nn.Module):
    """
    Free-space propagation using Angular Spectrum Method (ASM).

    input:  (..., H, W) complex field or real amplitude (will be cast to complex)
    output: (..., H, W) complex field after propagation distance z
    """
    def __init__(
            self,
            wavelength: float,   # meters
            dx: float,           # meters per pixel (x)
            dy: float,           # meters per pixel (y)
            distance: float,     # meters
            learnable_z: bool = False,
            dtype: torch.dtype = torch.complex64,
    ):
        super().__init__()
        self.wavelength = float(wavelength)
        self.dx = float(dx)
        self.dy = float(dy)
        self.cdtype = dtype

        # 用 log(z) 保证 z>0（更稳定）
        z0 = torch.tensor(float(distance), dtype=torch.float32)
        if learnable_z:
            self.log_z = nn.Parameter(torch.log(z0))
        else:
            self.register_buffer("log_z", torch.log(z0), persistent=True)

        # 简单缓存：不同分辨率/设备要重建频率网格
        self._grid_cache = {}

    def _get_z(self) -> torch.Tensor:
        return torch.exp(self.log_z)

    def _freq_grids(self, H: int, W: int, device, dtype_real=torch.float32):
        key = (H, W, device.type, device.index)
        if key in self._grid_cache:
            FX, FY = self._grid_cache[key]
            return FX, FY

        fx = torch.fft.fftfreq(W, d=self.dx, device=device, dtype=dtype_real)  # cycles/m
        fy = torch.fft.fftfreq(H, d=self.dy, device=device, dtype=dtype_real)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")  # (H,W)

        self._grid_cache[key] = (FX, FY)
        return FX, FY

    def forward(self, inputE: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(inputE):
            inputE = inputE.to(torch.float32).to(self.cdtype)

        device = inputE.device
        z = self._get_z().to(device=device, dtype=torch.float32)

        H, W = inputE.shape[-2], inputE.shape[-1]
        FX, FY = self._freq_grids(H, W, device=device, dtype_real=torch.float32)

        lam = self.wavelength
        k = 2.0 * math.pi / lam

        # kz = k * sqrt(1 - (λfx)^2 - (λfy)^2)
        # 这里用复数 sqrt，自动包含倏逝波（负值会变成虚数，产生指数衰减）
        arg = (1.0 - (lam * FX) ** 2 - (lam * FY) ** 2).to(torch.complex64)
        kz = k * torch.sqrt(arg)

        Htf = torch.exp(1j * kz * z)  # transfer function

        U1 = inputE
        U1_f = torch.fft.fft2(U1)
        U2 = torch.fft.ifft2(U1_f * Htf)

        return U2

