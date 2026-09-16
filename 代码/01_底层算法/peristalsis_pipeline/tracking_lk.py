"""Low-level Lucas-Kanade primitives for the maintained Step 1.4 pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


FB_ERROR_THRESHOLD_PX = 2.0
MAX_SINGLE_FRAME_DISPLACEMENT_PX = 8.0

LK_PARAMS = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    ),
}


def read_video_gray(
    video_path: Path,
    resize_factor: float,
    *,
    fallback_fps: float | None = None,
    require_reported_frame_count_match: bool = True,
) -> tuple[list[np.ndarray], dict[str, float | int]]:
    """Read resized grayscale frames and fail on ambiguous timing or truncation.

    ``fallback_fps`` is an explicit compatibility escape hatch for metadata-poor
    videos.  Formal callers should leave it as ``None`` so a missing FPS cannot
    silently change the RSR time scale.
    """
    if not np.isfinite(resize_factor) or resize_factor <= 0.0:
        raise ValueError("resize_factor must be finite and positive")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0.0:
        if fallback_fps is None:
            cap.release()
            raise ValueError(f"Video has invalid FPS metadata: {video_path}")
        if not np.isfinite(fallback_fps) or fallback_fps <= 0.0:
            cap.release()
            raise ValueError("fallback_fps must be finite and positive")
        fps = float(fallback_fps)

    original_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if resize_factor != 1.0:
                height, width = frame.shape[:2]
                resized_width = int(round(width * resize_factor))
                resized_height = int(round(height * resize_factor))
                if resized_width < 1 or resized_height < 1:
                    raise ValueError("resize_factor produces an empty frame")
                frame = cv2.resize(
                    frame,
                    (resized_width, resized_height),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        cap.release()

    if len(frames) < 2:
        raise ValueError(f"Video has too few readable frames: {video_path}")
    if (
        require_reported_frame_count_match
        and original_n > 0
        and len(frames) != original_n
    ):
        raise ValueError(
            f"Video frame-count mismatch for {video_path}: "
            f"reported={original_n}, read={len(frames)}"
        )

    return frames, {
        "fps": float(fps),
        "reported_frame_count": original_n,
        "read_frame_count": len(frames),
        "original_width": original_w,
        "original_height": original_h,
        "resized_width": int(frames[0].shape[1]),
        "resized_height": int(frames[0].shape[0]),
    }


def inside_points(points: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Return whether each floating-point coordinate lies inside the mask."""
    height, width = roi_mask.shape
    xy = np.round(points).astype(np.int32)
    inside_bounds = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    inside = np.zeros(len(points), dtype=bool)
    indices = np.flatnonzero(inside_bounds)
    if len(indices):
        valid_xy = xy[indices]
        inside[indices] = roi_mask[valid_xy[:, 1], valid_xy[:, 0]] > 0
    return inside
