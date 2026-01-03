# simulation/system.py
# ------------------------------------------------------------
# Optical propagation system (simulation):
#   SLM1 (1024x768, 12.5um) -> free -> L1 (f=250mm) -> free -> SLM2 (1280x720, 6.3um)
#   SLM2 -> free -> L2 (f=250mm) -> free -> Sensor (camera, 3072x2048, 2.4um)
#
# - Uses Angular Spectrum Method (ASM) for free-space propagation.
# - Uses thin-lens quadratic phase for lenses.
# - NO OffsetProcess (no shift/tilt).
# - Different pixel pitches are handled via complex-field resampling (real/imag interpolated).
# - Camera has NO camera lens (just propagation to sensor).
# ------------------------------------------------------------

import math
import torch
from torch import nn
import torch.nn.functional as F

# ============================================================
# Parameters (edit here)
# ============================================================

# ---- Wavelength ----
WAVELENGTH = 532e-9  # meters (~532 nm)

# ---- Pixel pitches (square pixels) ----
PITCH_SLM1 = 12.5e-6  # meters / pixel
PITCH_SLM2 = 6.3e-6   # meters / pixel
PITCH_CAM  = 2.4e-6   # meters / pixel

# ---- Resolutions (W x H in common display specs; here we store H, W) ----
H_SLM1, W_SLM1 = 768, 1024
H_SLM2, W_SLM2 = 720, 1280
H_CAM,  W_CAM  = 2048, 3072

# ---- Sampling on each plane ----
DX1 = DY1 = PITCH_SLM1
DX2 = DY2 = PITCH_SLM2
DXC = DYC = PITCH_CAM

# ---- 4f lens focal length (given) ----
F_4F = 250e-3  # 250 mm = 0.25 m

# ---- Distances (meters) ----
# First half: ideal 4f half
SLM1_TO_L1 = F_4F
L1_TO_SLM2 = F_4F

# Second half: ideal 4f half to L2
SLM2_TO_L2 = F_4F

# L2 -> sensor distance (hard to know in real setup):
# A standard, reasonable simulation default is placing sensor at L2 back focal plane.
L2_TO_SENSOR = F_4F  # TODO: change if you want a different observation plane

# ---- Complex dtype ----
CDTYPE = torch.complex64


# ============================================================
# Small helpers
# ============================================================

def intensity(U: torch.Tensor) -> torch.Tensor:
    """Sensor intensity: I = |U|^2."""
    return U.real * U.real + U.imag * U.imag


# ============================================================
# Resampling between different pixel pitches / resolutions
# ============================================================

class ComplexResample(nn.Module):
    """
    Resample complex field from (H_in, W_in) to (H_out, W_out).
    Interpolate real and imaginary parts separately.

    Input:  (..., H_in, W_in) complex
    Output: (..., H_out, W_out) complex
    """
    def __init__(self, out_hw: tuple[int, int], mode: str = "bilinear", align_corners: bool = False):
        super().__init__()
        self.out_hw = out_hw
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(U):
            raise TypeError("ComplexResample expects a complex tensor (complex64/complex128).")

        *lead, H, W = U.shape
        B = 1
        for d in lead:
            B *= int(d)

        Ur = U.real.reshape(B, 1, H, W)
        Ui = U.imag.reshape(B, 1, H, W)
        X = torch.cat([Ur, Ui], dim=1)  # (B,2,H,W)

        if self.mode in ("bilinear", "bicubic", "trilinear"):
            Y = F.interpolate(X, size=self.out_hw, mode=self.mode, align_corners=self.align_corners)
        else:
            Y = F.interpolate(X, size=self.out_hw, mode=self.mode)

        H2, W2 = self.out_hw
        Yr = Y[:, 0:1].reshape(*lead, H2, W2)
        Yi = Y[:, 1:2].reshape(*lead, H2, W2)
        return torch.complex(Yr, Yi)


# ============================================================
# Core physics modules
# ============================================================

class FreePropagation(nn.Module):
    """
    Free-space propagation using Angular Spectrum Method (ASM).

    input:  (..., H, W) complex field before propagation
    output: (..., H, W) complex field after propagation distance z
    """
    def __init__(
        self,
        wavelength: float,
        dx: float,
        dy: float,
        distance: float,
        dtype: torch.dtype = torch.complex64,
    ):
        super().__init__()
        self.wavelength = float(wavelength)
        self.dx = float(dx)
        self.dy = float(dy)
        self.distance = float(distance)
        self.cdtype = dtype

        self._grid_cache = {}  # keyed by (H,W,device)

    def _freq_grids(self, H: int, W: int, device):
        key = (H, W, device.type, device.index)
        if key in self._grid_cache:
            return self._grid_cache[key]

        fx = torch.fft.fftfreq(W, d=self.dx, device=device, dtype=torch.float32)  # cycles/m
        fy = torch.fft.fftfreq(H, d=self.dy, device=device, dtype=torch.float32)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")  # (H,W)
        self._grid_cache[key] = (FX, FY)
        return FX, FY

    def forward(self, U1: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(U1):
            U1 = U1.to(torch.float32).to(self.cdtype)
        elif U1.dtype != self.cdtype:
            U1 = U1.to(self.cdtype)

        device = U1.device
        z = torch.tensor(self.distance, device=device, dtype=torch.float32)

        H, W = U1.shape[-2], U1.shape[-1]
        FX, FY = self._freq_grids(H, W, device)

        lam = self.wavelength
        k = 2.0 * math.pi / lam

        # kz = k * sqrt(1 - (λfx)^2 - (λfy)^2) with complex sqrt (evanescent included)
        arg = (1.0 - (lam * FX) ** 2 - (lam * FY) ** 2).to(torch.complex64)
        kz = k * torch.sqrt(arg)
        Htf = torch.exp(1j * kz * z)

        U1_f = torch.fft.fft2(U1)
        U2 = torch.fft.ifft2(U1_f * Htf)
        return U2


class LensPropagation(nn.Module):
    """
    Thin-lens phase modulation.

    input:  (..., H, W) complex field at lens plane
    output: (..., H, W) complex field after lens
    """
    def __init__(
        self,
        wavelength: float,
        dx: float,
        dy: float,
        focal_length: float,
        dtype: torch.dtype = torch.complex64,
    ):
        super().__init__()
        if float(focal_length) <= 0:
            raise ValueError("focal_length must be positive")

        self.wavelength = float(wavelength)
        self.dx = float(dx)
        self.dy = float(dy)
        self.f = float(focal_length)
        self.cdtype = dtype

        self._coord_cache = {}  # keyed by (H,W,device)

    def _coord_grids(self, H: int, W: int, device):
        key = (H, W, device.type, device.index)
        if key in self._coord_cache:
            return self._coord_cache[key]

        xs = (torch.arange(W, device=device, dtype=torch.float32) - (W // 2)) * self.dx
        ys = (torch.arange(H, device=device, dtype=torch.float32) - (H // 2)) * self.dy
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        self._coord_cache[key] = (X, Y)
        return X, Y

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(U):
            U = U.to(torch.float32).to(self.cdtype)
        elif U.dtype != self.cdtype:
            U = U.to(self.cdtype)

        device = U.device
        H, W = U.shape[-2], U.shape[-1]
        X, Y = self._coord_grids(H, W, device)

        lam = self.wavelength
        k = 2.0 * math.pi / lam
        phase = -(k / (2.0 * self.f)) * (X ** 2 + Y ** 2)
        H_lens = torch.exp(1j * phase)

        return U * H_lens


# ============================================================
# System construction
# ============================================================

def build_first_half_model(dtype: torch.dtype = CDTYPE) -> nn.Module:
    """
    SLM1 -> free(z=f) -> L1(f) -> resample to SLM2 grid -> free(z=f) -> SLM2
    """
    return nn.Sequential(
        FreePropagation(WAVELENGTH, DX1, DY1, SLM1_TO_L1, dtype=dtype),
        LensPropagation(WAVELENGTH, DX1, DY1, F_4F, dtype=dtype),
        ComplexResample(out_hw=(H_SLM2, W_SLM2), mode="bilinear", align_corners=False),
        FreePropagation(WAVELENGTH, DX2, DY2, L1_TO_SLM2, dtype=dtype),
    )


def build_second_half_model(dtype: torch.dtype = CDTYPE) -> nn.Module:
    """
    SLM2 -> free(z=f) -> L2(f) -> resample to camera grid -> free(z=L2_TO_SENSOR) -> sensor
    """
    return nn.Sequential(
        FreePropagation(WAVELENGTH, DX2, DY2, SLM2_TO_L2, dtype=dtype),
        LensPropagation(WAVELENGTH, DX2, DY2, F_4F, dtype=dtype),
        ComplexResample(out_hw=(H_CAM, W_CAM), mode="bilinear", align_corners=False),
        FreePropagation(WAVELENGTH, DXC, DYC, L2_TO_SENSOR, dtype=dtype),
    )


class OpticalSystem(nn.Module):
    """
    Full system: complex field at SLM1 plane -> sensor intensity.

    forward(U_slm1, return_fields=False):
        U_slm2 = first_half(U_slm1)
        U_cam_field = second_half(U_slm2)
        I_cam = |U_cam_field|^2
    """
    def __init__(self, dtype: torch.dtype = CDTYPE):
        super().__init__()
        self.first_half = build_first_half_model(dtype=dtype)
        self.second_half = build_second_half_model(dtype=dtype)

    def forward(self, U_slm1: torch.Tensor, return_fields: bool = False):
        U_slm2 = self.first_half(U_slm1)
        U_cam = self.second_half(U_slm2)
        I_cam = intensity(U_cam)
        if return_fields:
            return I_cam, U_cam, U_slm2
        return I_cam