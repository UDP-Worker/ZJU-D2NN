# simulation/system.py
# ------------------------------------------------------------
# Device-level optical propagation simulation (NO ONN logic here)
#
# Purpose:
#   - Define optics operators (free-space ASM, thin lens, complex resampling)
#   - Provide a clean "system" that maps:
#       U_slm1 (complex field on SLM1 plane) -> U_slm2 (complex on SLM2 plane)
#       U_slm2 (complex on SLM2 plane)      -> U_cam (complex on camera plane)
#   - SLM1/SLM2 are phase-only devices conceptually:
#       SLM1: carries input image via encoding into U_slm1 (done in ONN_train.py)
#       SLM2: learnable phase modulation phi_slm2 (applied in ONN_train.py):
#             U_slm2_mod = U_slm2 * exp(1j * phi_slm2)
#
# Your setup:
#   - Wavelength: ~532 nm
#   - Pixel pitches (square pixels):
#       SLM1: 12.5 um, res = 1024 x 768
#       SLM2:  6.3 um, res = 1280 x 720
#       CAM :  2.4 um, res = 3072 x 2048
#   - Optical layout (simulation):
#       SLM1 --(z=f)--> L1(f=250mm) --(resample)--> --(z=f)--> SLM2
#       SLM2 --(z=f)--> L2(f=250mm) --(resample)--> --(z=f)--> Sensor (no camera lens)
#
# Notes:
#   - No OffsetProcess (no shift/tilt).
#   - Different pitches handled by ComplexResample (interpolate real/imag separately).
#   - Camera measures intensity I = |U|^2 (helper provided).
# ------------------------------------------------------------

import math
import torch
from torch import nn
import torch.nn.functional as F

# ============================================================
# Parameters (all at top)
# ============================================================

# ---- Wavelength ----
WAVELENGTH = 532e-9  # meters (~532 nm)

# ---- Pixel pitches (square pixels) ----
PITCH_SLM1 = 12.5e-6  # meters / pixel
PITCH_SLM2 = 6.3e-6   # meters / pixel
PITCH_CAM  = 2.4e-6   # meters / pixel

# ---- Resolutions (store as H, W) ----
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
# First half
SLM1_TO_L1 = F_4F
L1_TO_SLM2 = F_4F

# Second half
SLM2_TO_L2 = F_4F

# L2 -> sensor distance
# In a standard 4f layout, putting the sensor at the back focal plane is common.
L2_TO_SENSOR = F_4F  # You may change if you want a different observation plane.

# ---- Complex dtype ----
CDTYPE = torch.complex64


# ============================================================
# Helpers
# ============================================================

def intensity(U: torch.Tensor) -> torch.Tensor:
    """Camera intensity: I = |U|^2."""
    return U.real * U.real + U.imag * U.imag


def apply_phase(U: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Ideal phase-only modulation: U_out = U * exp(i*phi)
    U:   (..., H, W) complex
    phi: (H, W) or (..., H, W) real, radians
    """
    if not torch.is_complex(U):
        raise TypeError("apply_phase expects a complex tensor U.")
    phi = phi.to(device=U.device, dtype=torch.float32)
    return U * torch.exp(1j * phi)


# ============================================================
# Resampling between different grids
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
        self._tf_cache = {}    # keyed by (H,W,device)

    def _freq_grids(self, H: int, W: int, device):
        key = (H, W, device.type, device.index)
        if key in self._grid_cache:
            return self._grid_cache[key]

        fx = torch.fft.fftfreq(W, d=self.dx, device=device, dtype=torch.float32)  # cycles/m
        fy = torch.fft.fftfreq(H, d=self.dy, device=device, dtype=torch.float32)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")  # (H,W)
        self._grid_cache[key] = (FX, FY)
        return FX, FY

    def _transfer_function(self, H: int, W: int, device):
        key = (H, W, device.type, device.index, self.cdtype)
        if key in self._tf_cache:
            return self._tf_cache[key]

        FX, FY = self._freq_grids(H, W, device)

        lam = self.wavelength
        k = 2.0 * math.pi / lam

        # kz = k * sqrt(1 - (λfx)^2 - (λfy)^2) with complex sqrt (evanescent included)
        arg = (1.0 - (lam * FX) ** 2 - (lam * FY) ** 2).to(self.cdtype)
        kz = k * torch.sqrt(arg)
        Htf = torch.exp(1j * kz * self.distance)

        self._tf_cache[key] = Htf
        return Htf

    def forward(self, U1: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(U1):
            U1 = U1.to(torch.float32).to(self.cdtype)
        elif U1.dtype != self.cdtype:
            U1 = U1.to(self.cdtype)

        device = U1.device
        H, W = U1.shape[-2], U1.shape[-1]
        Htf = self._transfer_function(H, W, device)

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
        self._lens_cache = {}   # keyed by (H,W,device)

    def _coord_grids(self, H: int, W: int, device):
        key = (H, W, device.type, device.index)
        if key in self._coord_cache:
            return self._coord_cache[key]

        xs = (torch.arange(W, device=device, dtype=torch.float32) - (W // 2)) * self.dx
        ys = (torch.arange(H, device=device, dtype=torch.float32) - (H // 2)) * self.dy
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        self._coord_cache[key] = (X, Y)
        return X, Y

    def _lens_phase(self, H: int, W: int, device):
        key = (H, W, device.type, device.index, self.cdtype)
        if key in self._lens_cache:
            return self._lens_cache[key]

        X, Y = self._coord_grids(H, W, device)
        lam = self.wavelength
        k = 2.0 * math.pi / lam
        phase = -(k / (2.0 * self.f)) * (X ** 2 + Y ** 2)
        H_lens = torch.exp(1j * phase)

        self._lens_cache[key] = H_lens
        return H_lens

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(U):
            U = U.to(torch.float32).to(self.cdtype)
        elif U.dtype != self.cdtype:
            U = U.to(self.cdtype)

        device = U.device
        H, W = U.shape[-2], U.shape[-1]
        H_lens = self._lens_phase(H, W, device)

        return U * H_lens


# ============================================================
# System builders
# ============================================================

def build_first_half_model(dtype: torch.dtype = CDTYPE) -> nn.Module:
    """
    SLM1 -> free(z=f) -> L1(f) -> resample to SLM2 grid -> free(z=f) -> SLM2

    Output is complex field on SLM2 grid (H_SLM2, W_SLM2).
    """
    return nn.Sequential(
        FreePropagation(WAVELENGTH, DX1, DY1, SLM1_TO_L1, dtype=dtype),
        LensPropagation(WAVELENGTH, DX1, DY1, F_4F, dtype=dtype),
        ComplexResample(out_hw=(H_SLM2, W_SLM2), mode="bilinear", align_corners=False),
        FreePropagation(WAVELENGTH, DX2, DY2, L1_TO_SLM2, dtype=dtype),
    )


def build_second_half_model(dtype: torch.dtype = CDTYPE) -> nn.Module:
    """
    SLM2 -> free(z=f) -> L2(f) -> resample to camera grid -> free(z=L2_TO_SENSOR) -> Sensor

    Output is complex field on camera grid (H_CAM, W_CAM).
    """
    return nn.Sequential(
        FreePropagation(WAVELENGTH, DX2, DY2, SLM2_TO_L2, dtype=dtype),
        LensPropagation(WAVELENGTH, DX2, DY2, F_4F, dtype=dtype),
        ComplexResample(out_hw=(H_CAM, W_CAM), mode="bilinear", align_corners=False),
        FreePropagation(WAVELENGTH, DXC, DYC, L2_TO_SENSOR, dtype=dtype),
    )


class OpticalSystem(nn.Module):
    """
    Device simulation wrapper.

    Typical usage in ONN_train.py:
        system = OpticalSystem().to(device)

        U_slm1 = encode_image_to_slm1_field(...)   # complex field on SLM1 grid
        U_slm2 = system.forward_to_slm2(U_slm1)
        U_slm2 = apply_phase(U_slm2, phi_slm2)     # learnable phase on SLM2 (in ONN_train.py)
        U_cam  = system.forward_to_camera(U_slm2)
        I_cam  = intensity(U_cam)
    """
    def __init__(self, dtype: torch.dtype = CDTYPE):
        super().__init__()
        self.first_half = build_first_half_model(dtype=dtype)
        self.second_half = build_second_half_model(dtype=dtype)

    @torch.no_grad()
    def info(self) -> dict:
        """Quick metadata for debugging."""
        return {
            "wavelength_m": WAVELENGTH,
            "f_4f_m": F_4F,
            "slm1_hw": (H_SLM1, W_SLM1),
            "slm2_hw": (H_SLM2, W_SLM2),
            "cam_hw": (H_CAM, W_CAM),
            "pitch_slm1_m": PITCH_SLM1,
            "pitch_slm2_m": PITCH_SLM2,
            "pitch_cam_m": PITCH_CAM,
            "distances_m": {
                "slm1_to_l1": SLM1_TO_L1,
                "l1_to_slm2": L1_TO_SLM2,
                "slm2_to_l2": SLM2_TO_L2,
                "l2_to_sensor": L2_TO_SENSOR,
            },
        }

    def forward_to_slm2(self, U_slm1: torch.Tensor) -> torch.Tensor:
        """
        U_slm1: (..., H_SLM1, W_SLM1) complex field on SLM1 plane
        return: (..., H_SLM2, W_SLM2) complex field right before SLM2 modulation
        """
        return self.first_half(U_slm1)

    def forward_to_camera(self, U_slm2: torch.Tensor) -> torch.Tensor:
        """
        U_slm2: (..., H_SLM2, W_SLM2) complex field AFTER SLM2 modulation (or before, if you want)
        return: (..., H_CAM, W_CAM) complex field on sensor plane
        """
        return self.second_half(U_slm2)

    def forward_camera_intensity(self, U_slm2: torch.Tensor) -> torch.Tensor:
        """Convenience: directly output camera intensity."""
        return intensity(self.forward_to_camera(U_slm2))
