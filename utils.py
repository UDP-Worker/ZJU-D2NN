import threading
from typing import Dict, Optional, Tuple

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

    def __init__(self, channel: int, expected_shape: Tuple[int, int]):
        super().__init__(daemon=True)
        self.channel = channel
        self.expected_shape = expected_shape
        self.window_name = _WINDOW_TITLES[channel]
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


def read_from_camera() -> np.ndarray:
    return

def write_to_hdmi(picture: np.ndarray, channel: int) -> None:
    """
    Push a numpy array to the HDMI output for the given channel.

    The latest frame is kept on screen by a background thread so this call
    returns immediately.
    """
    if channel not in _SLM_PIXEL_SHAPE:
        raise NotImplementedError(f"Unsupported HDMI channel {channel}")

    expected_shape = _SLM_PIXEL_SHAPE[channel]
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
            displayer = _HDMIDisplayer(channel=channel, expected_shape=expected_shape)
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
