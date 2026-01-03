# simulation/system.py
# ------------------------------------------------------------
# Optical propagation system for simulation:
#   SLM1 -> (free) -> L1 (f=250mm) -> (free) -> SLM2
#   SLM2 -> (free) -> L2 (f=250mm) -> (free) -> Sensor (camera)
#
# Notes:
# - Pure simulation: no calibration, no learnable offsets/tilts.
# - Camera has NO camera lens; sensor measures intensity |U|^2.
# - Unknown parameters are left as None with TODO marks.
# ------------------------------------------------------------

import math
import torch
from torch import nn

# ============================================================
# Parameters (edit here)
# ============================================================

# ---- Fundamental optical parameters ----
WAVELENGTH = None  # TODO: wavelength in meters, e.g. 532e-9 / 633e-9 / 1550e-9

# ---- Sampling pitch on the simulation grid (meters per pixel) ----
DX = None  # TODO: meters/pixel in x (SLM/camera plane sampling)
DY = None  # TODO: meters/pixel in y

# ---- 4f lens focal length (given) ----
F_4F = 250e-3  # 250 mm = 0.25 m

# ---- Distances (meters) ----
# First half: SLM1 -> L1 -> SLM2
# In an ideal 4f, these are typically f and f; we set defaults accordingly.
SLM1_TO_L1 = F_4F
L1_TO_SLM2 = F_4F

# Second half: SLM2 -> L2 -> Sensor (NO camera lens)
SLM2_TO_L2 = F_4F
L2_TO_SENSOR = None  # TODO: meters, distance from L2 to sensor plane

# ---- Numeric dtype ----
CDTYPE = torch.complex64  # complex64 is usually enough and faster


# ============================================================
# Utilities
# ============================================================

def _require_not_none(name: str, value):
    if value is None:
        raise ValueError(f"Parameter `{name}` is None. Please set it in system.py.")
    return value


def intensity(field: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """
    Sensor measures intensity: I = |U|^2.
    """
    I = field.real * field.real + field.imag * field.imag
    if eps > 0:
        I = I + eps
    return I


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
        learnable_z: bool = False,
        dtype: torch.dtype = torch.complex64,
    ):
        super().__init__()
        self.wavelength = float(wavelength)
        self.dx = float(dx)
        self.dy = float(dy)
        self.cdtype = dtype

        # log(z) ensures z > 0
        z0 = torch.tensor(float(distance), dtype=torch.float32)
        if learnable_z:
            self.log_z = nn.Parameter(torch.log(z0))
        else:
            self.register_buffer("log_z", torch.log(z0), persistent=True)

        # cache frequency grids per (H,W,device)
        self._grid_cache = {}

    def _get_z(self) -> torch.Tensor:
        return torch.exp(self.log_z)

    def _freq_grids(self, H: int, W: int, device, dtype_real=torch.float32):
        key = (H, W, device.type, device.index, dtype_real)
        if key in self._grid_cache:
            return self._grid_cache[key]

        fx = torch.fft.fftfreq(W, d=self.dx, device=device, dtype=dtype_real)  # cycles/m
        fy = torch.fft.fftfreq(H, d=self.dy, device=device, dtype=dtype_real)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")  # (H,W)

        self._grid_cache[key] = (FX, FY)
        return FX, FY

    def forward(self, inputE: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(inputE):
            inputE = inputE.to(torch.float32).to(self.cdtype)
        elif inputE.dtype != self.cdtype:
            inputE = inputE.to(self.cdtype)

        device = inputE.device
        z = self._get_z().to(device=device, dtype=torch.float32)

        H, W = inputE.shape[-2], inputE.shape[-1]
        FX, FY = self._freq_grids(H, W, device=device, dtype_real=torch.float32)

        lam = self.wavelength
        k = 2.0 * math.pi / lam

        # kz = k * sqrt(1 - (λfx)^2 - (λfy)^2)
        # Use complex sqrt to include evanescent waves (decay).
        arg = (1.0 - (lam * FX) ** 2 - (lam * FY) ** 2).to(torch.complex64)
        kz = k * torch.sqrt(arg)

        Htf = torch.exp(1j * kz * z)

        U1_f = torch.fft.fft2(inputE)
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
        learnable_f: bool = False,
        dtype: torch.dtype = torch.complex64,
    ):
        super().__init__()
        if float(focal_length) <= 0:
            raise ValueError("focal_length must be positive")

        self.wavelength = float(wavelength)
        self.dx = float(dx)
        self.dy = float(dy)
        self.cdtype = dtype

        # log(f) if needed; for simulation usually fixed
        f0 = torch.tensor(float(focal_length), dtype=torch.float32)
        if learnable_f:
            self.log_f = nn.Parameter(torch.log(f0))
        else:
            self.register_buffer("log_f", torch.log(f0), persistent=True)

        # cache coordinate grids per (H,W,device)
        self._coord_cache = {}

    def _get_f(self) -> torch.Tensor:
        return torch.exp(self.log_f)

    def _coord_grids(self, H: int, W: int, device, dtype_real=torch.float32):
        key = (H, W, device.type, device.index, dtype_real)
        if key in self._coord_cache:
            return self._coord_cache[key]

        xs = (torch.arange(W, device=device, dtype=dtype_real) - (W // 2)) * self.dx
        ys = (torch.arange(H, device=device, dtype=dtype_real) - (H // 2)) * self.dy
        Y, X = torch.meshgrid(ys, xs, indexing="ij")

        self._coord_cache[key] = (X, Y)
        return X, Y

    def forward(self, inputE: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(inputE):
            inputE = inputE.to(torch.float32).to(self.cdtype)
        elif inputE.dtype != self.cdtype:
            inputE = inputE.to(self.cdtype)

        device = inputE.device
        f = self._get_f().to(device=device, dtype=torch.float32)

        H, W = inputE.shape[-2], inputE.shape[-1]
        X, Y = self._coord_grids(H, W, device=device, dtype_real=torch.float32)

        lam = self.wavelength
        k = 2.0 * math.pi / lam
        phase = -(k / (2.0 * f)) * (X ** 2 + Y ** 2)
        H_lens = torch.exp(1j * phase)

        return inputE * H_lens


# ============================================================
# System builders
# ============================================================

def build_first_half_model(dtype: torch.dtype = CDTYPE) -> nn.Module:
    """
    SLM1 -> L1 -> SLM2
    """
    lam = _require_not_none("WAVELENGTH", WAVELENGTH)
    dx = _require_not_none("DX", DX)
    dy = _require_not_none("DY", DY)

    z1 = _require_not_none("SLM1_TO_L1", SLM1_TO_L1)
    z2 = _require_not_none("L1_TO_SLM2", L1_TO_SLM2)

    return nn.Sequential(
        FreePropagation(lam, dx, dy, z1, learnable_z=False, dtype=dtype),
        LensPropagation(lam, dx, dy, F_4F, learnable_f=False, dtype=dtype),
        FreePropagation(lam, dx, dy, z2, learnable_z=False, dtype=dtype),
    )


def build_second_half_model(dtype: torch.dtype = CDTYPE) -> nn.Module:
    """
    SLM2 -> L2 -> Sensor (no camera lens)
    """
    lam = _require_not_none("WAVELENGTH", WAVELENGTH)
    dx = _require_not_none("DX", DX)
    dy = _require_not_none("DY", DY)

    z3 = _require_not_none("SLM2_TO_L2", SLM2_TO_L2)
    z4 = _require_not_none("L2_TO_SENSOR", L2_TO_SENSOR)

    return nn.Sequential(
        FreePropagation(lam, dx, dy, z3, learnable_z=False, dtype=dtype),
        LensPropagation(lam, dx, dy, F_4F, learnable_f=False, dtype=dtype),
        FreePropagation(lam, dx, dy, z4, learnable_z=False, dtype=dtype),
    )


class OpticalSystem(nn.Module):
    """
    Full system: SLM1 plane complex field -> sensor intensity.

    forward(input_field):
        U_slm2 = first_half(input_field)
        U_sensor = second_half(U_slm2)
        I_sensor = |U_sensor|^2
    """
    def __init__(self, dtype: torch.dtype = CDTYPE):
        super().__init__()
        self.first_half = build_first_half_model(dtype=dtype)
        self.second_half = build_second_half_model(dtype=dtype)

    def forward(self, inputE: torch.Tensor, return_field: bool = False):
        U_slm2 = self.first_half(inputE)
        U_sensor = self.second_half(U_slm2)
        I = intensity(U_sensor)

        if return_field:
            return I, U_sensor, U_slm2
        return I