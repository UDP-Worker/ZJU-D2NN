import time

import cv2
import numpy as np

import utils


def _print_outputs(outputs):
    print("Available HDMI outputs:")
    for output in outputs:
        index = output["index"]
        name = output["name"]
        width, height = output["resolution"]
        position = output.get("position")
        if position:
            pos_text = f" @ {position[0]},{position[1]}"
        else:
            pos_text = ""
        print(f"  {index}: {name} ({width}x{height}){pos_text}")


def _prompt_selection(outputs):
    indices = {str(output["index"]) for output in outputs}
    while True:
        choice = input("Select output index: ").strip()
        if choice in indices:
            return int(choice)
        print("Invalid selection. Try again.")


def _make_circle(height, width):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    radius = min(height, width) // 4
    center = (width // 2, height // 2)
    cv2.circle(frame, center, radius, (0, 255, 0), thickness=-1)
    return frame


def _make_rectangle(height, width):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    margin = min(height, width) // 6
    cv2.rectangle(
        frame,
        (margin, margin),
        (width - margin, height - margin),
        (255, 0, 0),
        thickness=-1,
    )
    return frame


def _make_checkerboard(height, width, block=80):
    yy, xx = np.indices((height, width))
    board = ((xx // block + yy // block) % 2 * 255).astype(np.uint8)
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


def main():
    outputs = utils.list_hdmi_outputs()
    if not outputs:
        print("No HDMI outputs found.")
        return

    _print_outputs(outputs)
    selection = _prompt_selection(outputs)
    output_map = {output["index"]: output for output in outputs}
    chosen = output_map[selection]
    width, height = chosen["resolution"]

    print(f"Using output {selection} ({width}x{height}). Press Ctrl+C to stop.")

    frames = [
        _make_circle(height, width),
        _make_rectangle(height, width),
        _make_checkerboard(height, width),
    ]

    index = 0
    try:
        while True:
            utils.write_to_hdmi(frames[index % len(frames)], selection)
            index += 1
            time.sleep(0.8)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
