"""
Train a neural network or use a normal function to describe the propagating process
from SLM1 to SLM2. Also train a neural network or use a normal function to describe
the propagating process from SLM2 to camera.

To train these two neural network, we can utilize programmable SLM2.
"""

import math
import torch
from torch import nn

import pytorch_utils as ptu

class FreePropagation(nn.Module):
    """
    Free-space propagation using Angular Spectrum Method (ASM).

    input:  (..., H, W) complex field before propagation
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

class LensPropagation(nn.Module):
    """
    Thin-lens phase modulation.

    input:  (..., H, W) complex field at lens plane
    output: (..., H, W) complex field after lens
    """
    def __init__(
            self,
            wavelength: float,   # meters
            dx: float,           # meters per pixel (x)
            dy: float,           # meters per pixel (y)
            focal_length: float, # meters
            learnable_f: bool = False,
            dtype: torch.dtype = torch.complex64,
    ):
        super().__init__()
        self.wavelength = float(wavelength)
        self.dx = float(dx)
        self.dy = float(dy)
        self.cdtype = dtype

        f0 = torch.tensor(float(focal_length), dtype=torch.float32)
        if float(focal_length) <= 0:
            raise ValueError("focal_length must be positive")
        if learnable_f:
            self.log_f = nn.Parameter(torch.log(f0))
        else:
            self.register_buffer("log_f", torch.log(f0), persistent=True)

        self._coord_cache = {}

    def _get_f(self) -> torch.Tensor:
        return torch.exp(self.log_f)

    def _coord_grids(self, H: int, W: int, device, dtype_real=torch.float32):
        key = (H, W, device.type, device.index)
        if key in self._coord_cache:
            X, Y = self._coord_cache[key]
            return X, Y

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

class OffsetProcess(nn.Module):
    """
    Apply lateral offset (shift) and wavefront tilt.

    offset_x/offset_y: meters; tilt_x/tilt_y: radians
    """
    def __init__(
            self,
            wavelength: float,
            dx: float,
            dy: float,
            offset_x: float = 0.0,
            offset_y: float = 0.0,
            tilt_x: float = 0.0,
            tilt_y: float = 0.0,
            learnable_offset: bool = False,
            learnable_tilt: bool = False,
            tilt_scale: float = 1.0,
            dtype: torch.dtype = torch.complex64,
    ):
        super().__init__()
        self.wavelength = float(wavelength)
        self.dx = float(dx)
        self.dy = float(dy)
        self.tilt_scale = float(tilt_scale)
        self.cdtype = dtype

        ox0 = torch.tensor(float(offset_x), dtype=torch.float32)
        oy0 = torch.tensor(float(offset_y), dtype=torch.float32)
        if learnable_offset:
            self.offset_x = nn.Parameter(ox0)
            self.offset_y = nn.Parameter(oy0)
        else:
            self.register_buffer("offset_x", ox0, persistent=True)
            self.register_buffer("offset_y", oy0, persistent=True)

        tx0 = torch.tensor(float(tilt_x), dtype=torch.float32)
        ty0 = torch.tensor(float(tilt_y), dtype=torch.float32)
        if learnable_tilt:
            self.tilt_x = nn.Parameter(tx0)
            self.tilt_y = nn.Parameter(ty0)
        else:
            self.register_buffer("tilt_x", tx0, persistent=True)
            self.register_buffer("tilt_y", ty0, persistent=True)

        self._coord_cache = {}
        self._freq_cache = {}

    def _coord_grids(self, H: int, W: int, device, dtype_real=torch.float32):
        key = (H, W, device.type, device.index)
        if key in self._coord_cache:
            X, Y = self._coord_cache[key]
            return X, Y

        xs = (torch.arange(W, device=device, dtype=dtype_real) - (W // 2)) * self.dx
        ys = (torch.arange(H, device=device, dtype=dtype_real) - (H // 2)) * self.dy
        Y, X = torch.meshgrid(ys, xs, indexing="ij")

        self._coord_cache[key] = (X, Y)
        return X, Y

    def _freq_grids(self, H: int, W: int, device, dtype_real=torch.float32):
        key = (H, W, device.type, device.index)
        if key in self._freq_cache:
            FX, FY = self._freq_cache[key]
            return FX, FY

        fx = torch.fft.fftfreq(W, d=self.dx, device=device, dtype=dtype_real)
        fy = torch.fft.fftfreq(H, d=self.dy, device=device, dtype=dtype_real)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")

        self._freq_cache[key] = (FX, FY)
        return FX, FY

    def forward(self, inputE: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(inputE):
            inputE = inputE.to(torch.float32).to(self.cdtype)
        elif inputE.dtype != self.cdtype:
            inputE = inputE.to(self.cdtype)

        device = inputE.device
        H, W = inputE.shape[-2], inputE.shape[-1]

        field = inputE

        # Tilt: multiply a linear phase in spatial domain.
        tilt_x = self.tilt_x.to(device=device, dtype=torch.float32)
        tilt_y = self.tilt_y.to(device=device, dtype=torch.float32)
        if self.tilt_scale != 0.0:
            X, Y = self._coord_grids(H, W, device=device, dtype_real=torch.float32)
            k = 2.0 * math.pi / self.wavelength
            phase = self.tilt_scale * k * (torch.sin(tilt_x) * X + torch.sin(tilt_y) * Y)
            field = field * torch.exp(1j * phase)

        # Offset: shift the field by (offset_x, offset_y).
        offset_x = self.offset_x.to(device=device, dtype=torch.float32)
        offset_y = self.offset_y.to(device=device, dtype=torch.float32)
        FX, FY = self._freq_grids(H, W, device=device, dtype_real=torch.float32)
        shift_phase = torch.exp(-1j * 2.0 * math.pi * (FX * offset_x + FY * offset_y))
        field_f = torch.fft.fft2(field)
        field = torch.fft.ifft2(field_f * shift_phase)

        return field



def build_first_half_model(
        wavelength: float,
        dx: float,
        dy: float,
        f_4f: float,
        slm1_to_l1: float = None,
        l1_to_slm2: float = None,
        learnable_slm1_to_l1: bool = False,
        learnable_l1_to_slm2: bool = False,
        slm2_offset_xy=(0.0, 0.0),
        slm2_tilt_xy=(0.0, 0.0),
        learnable_slm2_offset: bool = True,
        learnable_slm2_tilt: bool = True,
        slm2_tilt_scale: float = 1.0,
        dtype: torch.dtype = torch.complex64,
) -> nn.Module:
    """
    Propagation from SLM1 to SLM2 (right before SLM2).
    """
    if slm1_to_l1 is None:
        slm1_to_l1 = f_4f
    if l1_to_slm2 is None:
        l1_to_slm2 = f_4f

    layers = []
    layers.append(FreePropagation(
        wavelength=wavelength, dx=dx, dy=dy,
        distance=slm1_to_l1, learnable_z=learnable_slm1_to_l1, dtype=dtype
    ))
    layers.append(LensPropagation(
        wavelength=wavelength, dx=dx, dy=dy,
        focal_length=f_4f, learnable_f=False, dtype=dtype
    ))
    layers.append(FreePropagation(
        wavelength=wavelength, dx=dx, dy=dy,
        distance=l1_to_slm2, learnable_z=learnable_l1_to_slm2, dtype=dtype
    ))
    layers.append(OffsetProcess(
        wavelength=wavelength, dx=dx, dy=dy,
        offset_x=float(slm2_offset_xy[0]), offset_y=float(slm2_offset_xy[1]),
        tilt_x=float(slm2_tilt_xy[0]), tilt_y=float(slm2_tilt_xy[1]),
        learnable_offset=learnable_slm2_offset,
        learnable_tilt=learnable_slm2_tilt,
        tilt_scale=slm2_tilt_scale,
        dtype=dtype,
    ))
    first_half_model = nn.Sequential(*layers)
    return first_half_model

def build_second_half_model(
        wavelength: float,
        dx: float,
        dy: float,
        f_4f: float,
        f_camera: float,
        slm2_to_l2: float = None,
        l2_to_cam_lens: float = None,
        cam_lens_to_sensor: float = None,
        learnable_slm2_to_l2: bool = False,
        learnable_l2_to_cam_lens: bool = True,
        learnable_cam_lens_to_sensor: bool = True,
        camera_offset_xy=(0.0, 0.0),
        camera_tilt_xy=(0.0, 0.0),
        learnable_camera_offset: bool = True,
        learnable_camera_tilt: bool = False,
        camera_tilt_scale: float = 1.0,
        dtype: torch.dtype = torch.complex64,
) -> nn.Module:
    """
    Propagation from SLM2 to camera (right after SLM2 -> camera plane).
    """
    if slm2_to_l2 is None:
        slm2_to_l2 = f_4f
    if l2_to_cam_lens is None:
        raise ValueError("l2_to_cam_lens must be provided (x' distance)")
    if cam_lens_to_sensor is None:
        raise ValueError("cam_lens_to_sensor must be provided (x'' distance)")

    layers = []
    layers.append(FreePropagation(
        wavelength=wavelength, dx=dx, dy=dy,
        distance=slm2_to_l2, learnable_z=learnable_slm2_to_l2, dtype=dtype
    ))
    layers.append(LensPropagation(
        wavelength=wavelength, dx=dx, dy=dy,
        focal_length=f_4f, learnable_f=False, dtype=dtype
    ))
    layers.append(FreePropagation(
        wavelength=wavelength, dx=dx, dy=dy,
        distance=l2_to_cam_lens, learnable_z=learnable_l2_to_cam_lens, dtype=dtype
    ))
    layers.append(LensPropagation(
        wavelength=wavelength, dx=dx, dy=dy,
        focal_length=f_camera, learnable_f=False, dtype=dtype
    ))
    layers.append(FreePropagation(
        wavelength=wavelength, dx=dx, dy=dy,
        distance=cam_lens_to_sensor, learnable_z=learnable_cam_lens_to_sensor, dtype=dtype
    ))
    layers.append(OffsetProcess(
        wavelength=wavelength, dx=dx, dy=dy,
        offset_x=float(camera_offset_xy[0]), offset_y=float(camera_offset_xy[1]),
        tilt_x=float(camera_tilt_xy[0]), tilt_y=float(camera_tilt_xy[1]),
        learnable_offset=learnable_camera_offset,
        learnable_tilt=learnable_camera_tilt,
        tilt_scale=camera_tilt_scale,
        dtype=dtype,
    ))
    second_half_model = nn.Sequential(*layers)
    return second_half_model
