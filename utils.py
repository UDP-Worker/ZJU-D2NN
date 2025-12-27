import sys
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from MvImport.MvCameraControl_class import *
import cv2
import numpy as np

# (height, width)
_SLM_PIXEL_SHAPE: Dict[int, Tuple[int, int]] = {
    1: (768, 1024),   # SLM1
    2: (720, 1280),   # SLM2
}
_WINDOW_TITLES = {1: "SLM1 / HDMI-1", 2: "SLM2 / HDMI-2"}
_REFRESH_HZ = 30

_display_threads: Dict[int, "_HDMIDisplayer"] = {}
_display_threads_lock = threading.Lock()
_IS_WINDOWS = sys.platform.startswith("win")


@dataclass(frozen=True)
class _MonitorInfo:
    index: int
    device_name: str
    display_name: str
    left: int
    top: int
    width: int
    height: int


def _enumerate_windows_monitors() -> List[_MonitorInfo]:
    if not _IS_WINDOWS:
        return []

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]

    class DISPLAY_DEVICEW(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("DeviceName", wintypes.WCHAR * 32),
            ("DeviceString", wintypes.WCHAR * 128),
            ("StateFlags", wintypes.DWORD),
            ("DeviceID", wintypes.WCHAR * 128),
            ("DeviceKey", wintypes.WCHAR * 128),
        ]

    MonitorEnumProc = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )

    monitors: List[_MonitorInfo] = []

    @MonitorEnumProc
    def _callback(hmonitor, hdc, lprect, lparam):
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(info)
        if not user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
            return True

        display_device = DISPLAY_DEVICEW()
        display_device.cb = ctypes.sizeof(DISPLAY_DEVICEW)

        display_name = info.szDevice
        device_name = info.szDevice
        if user32.EnumDisplayDevicesW(info.szDevice, 0, ctypes.byref(display_device), 0):
            if display_device.DeviceString:
                display_name = display_device.DeviceString
            if display_device.DeviceName:
                device_name = display_device.DeviceName

        rect = info.rcMonitor
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        monitors.append(
            _MonitorInfo(
                index=len(monitors) + 1,
                device_name=device_name,
                display_name=display_name,
                left=rect.left,
                top=rect.top,
                width=width,
                height=height,
            )
        )
        return True

    user32.EnumDisplayMonitors(None, None, _callback, 0)
    return monitors


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert arbitrary numeric array to uint8 in [0, 255]."""
    if frame.dtype == np.uint8:
        return frame
    frame = np.nan_to_num(frame, nan=0.0)
    min_val = float(frame.min())
    max_val = float(frame.max())
    if max_val == min_val:
        return np.zeros_like(frame, dtype=np.uint8)
    frame = (frame - min_val) / (max_val - min_val)
    frame = np.clip(frame, 0.0, 1.0)
    return (frame * 255.0).round().astype(np.uint8)


class _HDMIDisplayer(threading.Thread):
    """Lightweight background loop that keeps the most recent frame on screen."""

    def __init__(
        self,
        channel: int,
        expected_shape: Tuple[int, int],
        window_name: str,
        monitor: Optional[_MonitorInfo] = None,
    ):
        super().__init__(daemon=True)
        self.channel = channel
        self.expected_shape = expected_shape
        self.window_name = window_name
        self.monitor = monitor
        self._frame_lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stop = threading.Event()
        self._frame: Optional[np.ndarray] = None
        self._last_error: Optional[Exception] = None

    @property
    def error(self) -> Optional[Exception]:
        return self._last_error

    def update_frame(self, frame: np.ndarray) -> None:
        with self._frame_lock:
            self._frame = frame
        self._frame_ready.set()

    def stop(self) -> None:
        self._stop.set()
        self._frame_ready.set()

    def run(self) -> None:
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            if self.monitor is not None:
                cv2.moveWindow(self.window_name, self.monitor.left, self.monitor.top)
                cv2.resizeWindow(self.window_name, self.monitor.width, self.monitor.height)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except cv2.error as exc:  # pragma: no cover - GUI init errors are environment dependent
            self._last_error = exc
            return

        delay_ms = max(int(1000 / _REFRESH_HZ), 1)

        # Wait until we have something to show.
        self._frame_ready.wait()

        while not self._stop.is_set():
            self._frame_ready.clear()

            with self._frame_lock:
                frame = self._frame

            if frame is not None:
                try:
                    cv2.imshow(self.window_name, frame)
                except cv2.error as exc:  # pragma: no cover - GUI runtime errors
                    self._last_error = exc
                    break

            cv2.waitKey(delay_ms)

            # If a new frame arrives while we were showing the last one, loop quickly.
            if self._frame_ready.is_set():
                continue

        cv2.destroyWindow(self.window_name)


def decoding_char(ctypes_char_array):
    """安全地将相机返回的 ctypes 字符数组解码为 Python 字符串（支持中文）"""
    byte_str = memoryview(ctypes_char_array).tobytes()
    null_index = byte_str.find(b'\x00')
    if null_index != -1:
        byte_str = byte_str[:null_index]
    for encoding in ['gbk', 'utf-8', 'latin-1']:
        try:
            return byte_str.decode(encoding)
        except UnicodeDecodeError:
            continue
    return byte_str.decode('latin-1', errors='replace')


def list_hdmi_outputs() -> List[Dict[str, object]]:
    """
    List available outputs for HDMI display.

    On Windows this returns active monitors (HDMI/DP/VGA are not always distinguishable).
    """
    outputs: List[Dict[str, object]] = []
    if _IS_WINDOWS:
        monitors = _enumerate_windows_monitors()
        for monitor in monitors:
            outputs.append(
                {
                    "index": monitor.index,
                    "name": monitor.display_name or monitor.device_name,
                    "device": monitor.device_name,
                    "resolution": (monitor.width, monitor.height),
                    "position": (monitor.left, monitor.top),
                }
            )
        return outputs

    for channel, shape in _SLM_PIXEL_SHAPE.items():
        outputs.append(
            {
                "index": channel,
                "name": _WINDOW_TITLES.get(channel, f"HDMI-{channel}"),
                "resolution": (shape[1], shape[0]),
                "position": None,
            }
        )
    return outputs


def read_from_camera() -> np.ndarray:
    """
    从连接的 MV-CE060-10UC（或其他 USB 海康相机）抓取一帧图像，
    返回 BGR 格式的 np.ndarray（shape: H x W x 3, dtype: uint8）。

    已针对 BayerRG8 (PixelType = 0x01080009) 优化并测试可用。
    每次调用都会自动打开 → 取流 → 抓一帧 → 关闭，适合非实时高频采集场景。
    """
    cam = MvCamera()

    # 1. 枚举 USB 设备
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, deviceList)
    if ret != 0:
        raise RuntimeError(f"枚举相机失败: 0x{ret:x}")
    if deviceList.nDeviceNum == 0:
        raise RuntimeError("未检测到任何 USB 相机，请检查相机是否连接并安装驱动")

    # 使用第一台相机
    stDeviceInfo = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

    # 2. 创建句柄并打开设备
    ret = cam.MV_CC_CreateHandle(stDeviceInfo)
    if ret != 0:
        raise RuntimeError(f"创建相机句柄失败: 0x{ret:x}")
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        raise RuntimeError(f"打开相机失败: 0x{ret:x}")

    try:
        # 3. 设置连续采集模式
        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print(f"警告: 设置 TriggerMode 失败 (0x{ret:x})，可能已是 Off")

        # 4. 开始取流
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"开始取流失败: 0x{ret:x}")

        # 5. 获取一帧最大数据量
        stPayloadSize = MVCC_INTVALUE()
        memset(byref(stPayloadSize), 0, sizeof(stPayloadSize))
        ret = cam.MV_CC_GetIntValue("PayloadSize", stPayloadSize)
        if ret != 0:
            raise RuntimeError(f"获取 PayloadSize 失败: 0x{ret:x}")
        nPayloadSize = stPayloadSize.nCurValue

        # 6. 抓取一帧（超时 10 秒）
        data_buf = (c_ubyte * nPayloadSize)()
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

        ret = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 10000)
        if ret != 0:
            raise RuntimeError(f"抓取图像失败: 0x{ret:x}（常见原因：无光照、曝光过长、镜头盖未开）")

        # 7. 转为 numpy 并进行 Bayer 去马赛克
        img_buffer = np.frombuffer(data_buf, dtype=np.uint8, count=stFrameInfo.nFrameLen)

        pixel_type = stFrameInfo.enPixelType
        height = stFrameInfo.nHeight
        width = stFrameInfo.nWidth

        if pixel_type == 0x01080009:  # BayerRG8 —— 你的相机默认格式
            raw_bayer = img_buffer.reshape(height, width)
            frame = cv2.cvtColor(raw_bayer, cv2.COLOR_BAYER_BG2BGR)  # OpenCV 中 BayerRG 对应 BG
        elif pixel_type == PixelType_Gvsp_Mono8:
            gray = img_buffer.reshape(height, width)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif pixel_type in [PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_BGR8_Packed]:
            frame = img_buffer.reshape(height, width, 3)
            if pixel_type == PixelType_Gvsp_RGB8_Packed:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            raise RuntimeError(f"当前相机像素格式 0x{pixel_type:08x} 暂不支持，请在 MVS 客户端改为 BayerRG8 或 RGB8")

        return frame.copy()

    finally:
        # 8. 清理资源
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()

def write_to_hdmi(picture: np.ndarray, channel: int) -> None:
    """
    Push a numpy array to the HDMI output for the given channel.

    The latest frame is kept on screen by a background thread so this call
    returns immediately.
    """
    if _IS_WINDOWS:
        monitors = _enumerate_windows_monitors()
        if not monitors:
            raise RuntimeError("No active Windows displays found")
        if channel < 1 or channel > len(monitors):
            raise ValueError(f"Unsupported HDMI output {channel}. Choose 1-{len(monitors)}.")
        monitor = monitors[channel - 1]
        expected_shape = (monitor.height, monitor.width)
        window_name = f"HDMI-{channel} / {monitor.display_name}"
    else:
        if channel not in _SLM_PIXEL_SHAPE:
            raise NotImplementedError(f"Unsupported HDMI channel {channel}")
        expected_shape = _SLM_PIXEL_SHAPE[channel]
        window_name = _WINDOW_TITLES[channel]
        monitor = None

    frame = np.asarray(picture)

    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]

    if frame.ndim == 3 and frame.shape[-1] not in (1, 3):
        raise ValueError(f"Expected grayscale or 3-channel image for channel {channel}, got shape {frame.shape}")

    hw = frame.shape[:2]
    if hw == expected_shape:
        pass
    elif hw == (expected_shape[1], expected_shape[0]):
        # User supplied width x height; transpose to height x width.
        frame = frame.transpose(1, 0) if frame.ndim == 2 else frame.transpose(1, 0, 2)
        hw = frame.shape[:2]
    if hw != expected_shape:
        raise ValueError(f"Channel {channel} expects shape (H, W) {expected_shape}, got {hw}")

    frame = _to_uint8(frame)
    frame = np.ascontiguousarray(frame)

    # cv2.imshow can display grayscale or 3-channel BGR; ensure we always pass something displayable.
    if frame.ndim == 2:
        display_frame = frame
    elif frame.shape[-1] == 3:
        display_frame = frame
    else:
        display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    with _display_threads_lock:
        displayer = _display_threads.get(channel)
        if displayer is None or not displayer.is_alive():
            displayer = _HDMIDisplayer(
                channel=channel,
                expected_shape=expected_shape,
                window_name=window_name,
                monitor=monitor,
            )
            displayer.start()
            _display_threads[channel] = displayer

    if displayer.error:
        raise RuntimeError(f"HDMI displayer for channel {channel} failed to start") from displayer.error

    displayer.update_frame(display_frame)

def sample_images(input_pictures: np.ndarray) -> np.ndarray:
    """
    Write images from dataset coordinately to hdmi1 (SLM1, the picture loader),
    and remain (SLM2, the optical neural network unchanged), meanwhile, collect
    images from camera when different pictures are loaded.

    return: np.ndarray with shape (batch_size, height, width, grayscale)
    """

    return
