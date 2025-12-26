import itertools
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    _SLM_PIXEL_SHAPE,
    _display_threads,
    _display_threads_lock,
    _WINDOW_TITLES,
    write_to_hdmi,
)


def _gui_available() -> bool:
    if os.environ.get("HDMI_FORCE_GUI") == "1":
        return True
    if os.name == "nt":
        return True
    if os.environ.get("DISPLAY") is None and os.environ.get("WAYLAND_DISPLAY") is None:
        return False

    checks: list[tuple[str, list[str]]] = [
        ("xrandr", ["--query"]),
        ("xdpyinfo", []),
        ("xset", ["q"]),
    ]
    for cmd, args in checks:
        path = shutil.which(cmd)
        if not path:
            continue
        result = subprocess.run([path, *args], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return True
        combined = (result.stdout + result.stderr).lower()
        if "can't open display" in combined or "cannot open display" in combined:
            return False

    return False


def _query_xrandr_outputs() -> list[dict]:
    outputs: list[dict] = []
    xrandr = shutil.which("xrandr")
    if not xrandr:
        return outputs

    result = subprocess.run([xrandr, "--query"], capture_output=True, text=True, check=False)
    geometry_re = re.compile(r"(\\d+)x(\\d+)\\+(\\d+)\\+(\\d+)")
    for line in result.stdout.splitlines():
        if " connected" not in line or "HDMI" not in line:
            continue
        tokens = line.split()
        name = tokens[0]
        geometry = None
        match = geometry_re.search(line)
        if match:
            geometry = tuple(int(val) for val in match.groups())
        outputs.append(
            {
                "name": name,
                "connected": " connected" in line,
                "geometry": geometry,
                "raw": line.strip(),
            }
        )
    return outputs


def _list_hdmi_outputs() -> list[str]:
    outputs = [entry["name"] for entry in _query_xrandr_outputs()]
    if outputs:
        return outputs

    sysfs = Path("/sys/class/drm")
    if sysfs.exists():
        for entry in sysfs.iterdir():
            if "HDMI" not in entry.name:
                continue
            status_file = entry / "status"
            if not status_file.exists():
                continue
            status = status_file.read_text().strip()
            if status == "connected":
                outputs.append(entry.name)
    return outputs


def _select_hdmi_output() -> dict | None:
    outputs = _query_xrandr_outputs()
    if not outputs:
        names = _list_hdmi_outputs()
        if not names:
            return None
        return {"name": names[0], "connected": True, "geometry": None, "raw": names[0]}

    env_name = os.environ.get("HDMI_OUTPUT")
    if env_name:
        for entry in outputs:
            if entry["name"] == env_name:
                return entry
        print(f"HDMI_OUTPUT={env_name} not found, falling back to prompt.")

    env_index = os.environ.get("HDMI_OUTPUT_INDEX")
    if env_index:
        try:
            index = int(env_index)
            if 0 <= index < len(outputs):
                return outputs[index]
        except ValueError:
            print(f"Invalid HDMI_OUTPUT_INDEX={env_index}, falling back to prompt.")

    print("Available HDMI outputs:")
    for idx, entry in enumerate(outputs):
        geometry = entry["geometry"]
        geom_text = ""
        if geometry:
            width, height, x, y = geometry
            geom_text = f" {width}x{height}+{x}+{y}"
        print(f" [{idx}] {entry['name']}{geom_text}")

    selection = input("Select HDMI output index (press Enter for [0]): ").strip()
    if selection == "":
        return outputs[0]
    try:
        index = int(selection)
        return outputs[index]
    except (ValueError, IndexError):
        print("Invalid selection, using [0].")
        return outputs[0]


def _make_circle(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    frame = np.zeros((height, width), dtype=np.uint8)
    radius = min(height, width) // 4
    cv2.circle(frame, (width // 2, height // 2), radius, 255, -1)
    return frame


def _make_rectangle(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    frame = np.zeros((height, width), dtype=np.uint8)
    top_left = (width // 4, height // 4)
    bottom_right = (width * 3 // 4, height * 3 // 4)
    cv2.rectangle(frame, top_left, bottom_right, 255, -1)
    return frame


def _make_checkerboard(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    frame = np.zeros((height, width), dtype=np.uint8)
    block = max(min(height, width) // 8, 8)
    for y in range(0, height, block):
        for x in range(0, width, block):
            if (x // block + y // block) % 2 == 0:
                frame[y : y + block, x : x + block] = 255
    return frame


def _make_diagonal(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    frame = np.zeros((height, width), dtype=np.uint8)
    step = max(min(height, width) // 10, 10)
    for offset in range(-height, width, step):
        cv2.line(frame, (offset, 0), (offset + height, height), 255, 3)
    return frame


def _stop_hdmi_displayers() -> None:
    with _display_threads_lock:
        threads = list(_display_threads.values())
    for thread in threads:
        thread.stop()
    for thread in threads:
        thread.join(timeout=1.0)


def test_hdmi_output() -> None:
    if os.environ.get("RUN_HDMI_TEST") != "1":
        pytest.skip("Set RUN_HDMI_TEST=1 to enable HDMI output test.")
    if not _gui_available():
        pytest.skip("GUI display not available; set HDMI_FORCE_GUI=1 to bypass this check.")

    _stop_hdmi_displayers()

    selected_output = _select_hdmi_output()
    if selected_output:
        print(f"Selected HDMI output: {selected_output['name']}")
    else:
        print("No HDMI outputs detected.")

    channel = int(os.environ.get("HDMI_CHANNEL", "1"))
    if channel not in _SLM_PIXEL_SHAPE:
        raise ValueError(f"Unsupported HDMI channel {channel}. Available: {sorted(_SLM_PIXEL_SHAPE)}")
    shape = _SLM_PIXEL_SHAPE[channel]

    patterns = [
        ("circle", _make_circle),
        ("rectangle", _make_rectangle),
        ("checkerboard", _make_checkerboard),
        ("diagonal", _make_diagonal),
    ]

    preview_window = "HDMI Preview"
    cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
    if selected_output and selected_output.get("geometry"):
        _, _, x, y = selected_output["geometry"]
        cv2.moveWindow(preview_window, x, y)

    duration = float(os.environ.get("HDMI_TEST_SECONDS", "10"))
    run_forever = duration <= 0
    start_time = time.perf_counter()

    try:
        for name, builder in itertools.cycle(patterns):
            frame = builder(shape)
            write_to_hdmi(frame, channel, use_thread=False)
            preview_frame = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.imshow(preview_window, preview_frame)
            if selected_output and selected_output.get("geometry"):
                _, _, x, y = selected_output["geometry"]
                cv2.moveWindow(_WINDOW_TITLES[channel], x, y)
            print(f"Showing {name} pattern on HDMI channel {channel}. Press q or Esc to stop.")
            next_tick = time.perf_counter() + 5.0
            while True:
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):
                    return
                if not run_forever and time.perf_counter() - start_time >= duration:
                    return
                if time.perf_counter() >= next_tick:
                    break
    finally:
        cv2.destroyWindow(preview_window)
        _stop_hdmi_displayers()
