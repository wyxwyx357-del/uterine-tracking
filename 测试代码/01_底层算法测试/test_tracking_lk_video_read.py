from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from peristalsis_pipeline import tracking_lk


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], *, fps: float, reported: int):
        self.frames = list(frames)
        self.fps = fps
        self.reported = reported
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, key: int) -> float:
        return {
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_COUNT: float(self.reported),
            cv2.CAP_PROP_FRAME_WIDTH: 8.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 6.0,
        }.get(key, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True


def color_frames(count: int) -> list[np.ndarray]:
    return [np.zeros((6, 8, 3), dtype=np.uint8) for _ in range(count)]


def test_invalid_fps_fails_without_explicit_fallback() -> None:
    capture = FakeCapture(color_frames(2), fps=0.0, reported=2)
    with patch.object(tracking_lk.cv2, "VideoCapture", return_value=capture):
        with pytest.raises(ValueError, match="invalid FPS"):
            tracking_lk.read_video_gray(Path("case.mp4"), 1.0)
    assert capture.released


def test_explicit_fps_fallback_is_recorded() -> None:
    capture = FakeCapture(color_frames(2), fps=0.0, reported=2)
    with patch.object(tracking_lk.cv2, "VideoCapture", return_value=capture):
        frames, metadata = tracking_lk.read_video_gray(
            Path("case.mp4"), 0.5, fallback_fps=25.0
        )
    assert metadata["fps"] == 25.0
    assert metadata["read_frame_count"] == 2
    assert frames[0].shape == (3, 4)


def test_reported_frame_count_mismatch_fails() -> None:
    capture = FakeCapture(color_frames(2), fps=25.0, reported=3)
    with patch.object(tracking_lk.cv2, "VideoCapture", return_value=capture):
        with pytest.raises(ValueError, match="frame-count mismatch"):
            tracking_lk.read_video_gray(Path("case.mp4"), 1.0)
    assert capture.released


def test_invalid_resize_factor_fails_before_opening_video() -> None:
    with pytest.raises(ValueError, match="resize_factor"):
        tracking_lk.read_video_gray(Path("case.mp4"), 0.0)
