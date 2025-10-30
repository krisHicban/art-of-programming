#!/usr/bin/env python3
"""
Animate PNG frames using OpenCV, storing them in a single NumPy array to
highlight NumPy's speed for repeated rendering.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np


FRAME_NAMES = ("1.png", "2.png", "3.png")
DISPLAY_DELAY_SECONDS = 0.30  # Adjust for faster or slower playback


def load_frames(frame_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load frames into a stacked NumPy array.

    Returns:
        frames: Array shaped (frame_count, height, width, channels)
        load_times: Array of per-frame load durations in seconds
    """
    frames: list[np.ndarray] = []
    load_times: list[float] = []

    for name in FRAME_NAMES:
        frame_path = frame_dir / name

        start = time.perf_counter()
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        elapsed = time.perf_counter() - start

        if image is None:
            msg = f"Unable to load frame: {frame_path}"
            raise FileNotFoundError(msg)

        frames.append(image)
        load_times.append(elapsed)

    # Stack frames so we can index into them quickly during playback.
    return np.stack(frames, axis=0), np.asarray(load_times)


def cycle_frames(window_name: str, frames: np.ndarray, delay_seconds: float) -> None:
    """
    Display frames in a loop until Esc or q is pressed.
    """
    per_frame_elapsed: list[float] = []

    while True:
        for index, frame in enumerate(frames):
            start = time.perf_counter()
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(max(1, int(delay_seconds * 1000))) & 0xFF
            per_frame_elapsed.append(time.perf_counter() - start)

            # Exit if the user closed the window via the title-bar X button.
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                report_playback_stats(per_frame_elapsed)
                return

            if key in (27, ord("q")):  # Esc or q
                report_playback_stats(per_frame_elapsed)
                return

        # Report performance stats after each full cycle to track runtime speed.
        report_playback_stats(per_frame_elapsed)
        per_frame_elapsed.clear()


def report_playback_stats(elapsed_times: Iterable[float]) -> None:
    elapsed = list(elapsed_times)
    if not elapsed:
        return

    avg = sum(elapsed) / len(elapsed)
    print(
        f"[Playback] Frames: {len(elapsed)} | Mean: {avg * 1000:.2f} ms "
        f"| Max: {max(elapsed) * 1000:.2f} ms"
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    frame_dir = root / "frames"

    frames, load_times = load_frames(frame_dir)
    print(
        f"[Load] Loaded {len(frames)} frames "
        f"({frames.shape[1]}x{frames.shape[2]} pixels) into memory."
    )
    print(
        f"[Load] Mean: {load_times.mean() * 1000:.2f} ms "
        f"| Max: {load_times.max() * 1000:.2f} ms "
        f"| Array dtype: {frames.dtype}"
    )

    window_name = "NumPy Animation"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        cycle_frames(window_name, frames, DISPLAY_DELAY_SECONDS)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
