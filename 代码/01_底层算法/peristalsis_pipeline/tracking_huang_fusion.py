"""Huang-2022-inspired global compensation and local deformation helpers.

This module deliberately keeps three quantities separate:

1. a rigid anatomical reference estimated from ten geometric-midline points;
2. one-frame local wall displacement measured around that reference;
3. exploratory deformation/propagation evidence derived from the local signal.

Local wall displacement is never fed into the next frame's reference position.
The current Step 1.4 forward/backward 2 px rule is not an admission gate here.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HuangFusionConfig:
    """Parameters for the paper-inspired pilot.

    The maintained videos are resized by 0.5.  A 21 px window on those frames
    is therefore the pixel-scale counterpart of the paper's 41 px window on
    the original frames.  Pyramid level and stopping criteria are explicit
    reproduction assumptions because Huang et al. did not report them.
    """

    midline_point_count: int = 10
    window_size_px: int = 21
    pyramid_max_level: int = 3
    iteration_count: int = 30
    iteration_epsilon: float = 0.01
    pcc_window_size_px: int = 21
    recording_pcc_threshold: float = 0.80
    physiology_low_cpm: float = 0.5
    physiology_high_cpm: float = 5.0
    minimum_signal_valid_fraction: float = 0.80
    minimum_formal_duration_s: float = 120.0

    def __post_init__(self) -> None:
        if self.midline_point_count < 2:
            raise ValueError("midline_point_count must be at least 2")
        for name, value in (
            ("window_size_px", self.window_size_px),
            ("pcc_window_size_px", self.pcc_window_size_px),
        ):
            if value < 3 or value % 2 == 0:
                raise ValueError(f"{name} must be an odd integer >= 3")
        if not 0.0 < self.recording_pcc_threshold < 1.0:
            raise ValueError("recording_pcc_threshold must lie in (0, 1)")
        if not 0.0 <= self.minimum_signal_valid_fraction <= 1.0:
            raise ValueError("minimum_signal_valid_fraction must lie in [0, 1]")

    def lk_parameters(self) -> dict:
        return {
            "winSize": (self.window_size_px, self.window_size_px),
            "maxLevel": self.pyramid_max_level,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.iteration_count,
                self.iteration_epsilon,
            ),
        }


def resample_polyline(points: np.ndarray, samples: int) -> np.ndarray:
    """Return equal-arc-length samples on an ordered polyline."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.r_[True, segment > 1e-6]]
    if len(points) < 2:
        raise ValueError("polyline has zero usable length")
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.r_[0.0, np.cumsum(segment)]
    target = np.linspace(0.0, float(distance[-1]), samples)
    result = np.column_stack(
        (
            np.interp(target, distance, points[:, 0]),
            np.interp(target, distance, points[:, 1]),
        )
    )
    return result.astype(np.float32)


def ordered_role_indices(nodes: list[dict], rail_name: str) -> np.ndarray:
    """Return point indices for one rail in cervix-to-fundus order."""
    pairs = [
        (int(node["section_order"]), point_idx)
        for point_idx, node in enumerate(nodes)
        if str(node.get("rail_name", "")) == rail_name
    ]
    pairs.sort()
    return np.asarray([point_idx for _, point_idx in pairs], dtype=np.int32)


def paired_wall_indices(
    nodes: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return anterior/posterior indices for sections containing both walls."""
    anterior = {
        int(node["section_order"]): point_idx
        for point_idx, node in enumerate(nodes)
        if str(node.get("rail_name", "")) == "anterior_wall"
    }
    posterior = {
        int(node["section_order"]): point_idx
        for point_idx, node in enumerate(nodes)
        if str(node.get("rail_name", "")) == "posterior_wall"
    }
    sections = np.asarray(sorted(set(anterior) & set(posterior)), dtype=np.int32)
    if len(sections) < 2:
        raise ValueError("at least two paired wall sections are required")
    return (
        np.asarray([anterior[int(s)] for s in sections], dtype=np.int32),
        np.asarray([posterior[int(s)] for s in sections], dtype=np.int32),
        sections,
    )


def apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 3):
        raise ValueError("matrix must have shape (2, 3)")
    return (points @ matrix[:, :2].T + matrix[:, 2]).astype(np.float32)


def compose_affine(after: np.ndarray, before: np.ndarray) -> np.ndarray:
    """Compose two 2-D affine matrices: result(points)=after(before(points))."""
    a = np.vstack((np.asarray(after, dtype=np.float64), [0.0, 0.0, 1.0]))
    b = np.vstack((np.asarray(before, dtype=np.float64), [0.0, 0.0, 1.0]))
    return (a @ b)[:2].astype(np.float32)


def _line_angle(points: np.ndarray) -> float:
    """Fit one line and orient it from the first to the last ordered point."""
    points = np.asarray(points, dtype=np.float64)
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    direction = vectors[0]
    ordered_direction = points[-1] - points[0]
    if np.dot(direction, ordered_direction) < 0.0:
        direction *= -1.0
    return float(np.arctan2(direction[1], direction[0]))


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def fit_mean_line_rigid(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Fit the paper-described mean translation plus line-angle rotation."""
    previous = np.asarray(previous, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if previous.shape != current.shape or previous.ndim != 2 or previous.shape[1] != 2:
        raise ValueError("previous/current must have matching shape (n, 2)")
    if len(previous) < 2:
        raise ValueError("at least two points are required for rotation")
    angle = _wrap_angle(_line_angle(current) - _line_angle(previous))
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    previous_center = np.mean(previous, axis=0)
    current_center = np.mean(current, axis=0)
    translation = current_center - rotation @ previous_center
    return np.column_stack((rotation, translation)).astype(np.float32)


def robust_mean_line_rigid(
    previous: np.ndarray, current: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Two-pass outlier-resistant sensitivity fit (project extension).

    This is kept separate from :func:`fit_mean_line_rigid`; it is not claimed
    as a Huang-2022 paper parameter and must not overwrite the primary branch.
    """
    previous = np.asarray(previous, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    first = fit_mean_line_rigid(previous, current)
    residual = np.linalg.norm(apply_affine(previous, first) - current, axis=1)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    limit = max(median + 3.0 * 1.4826 * mad, 0.25)
    inlier = residual <= limit
    if np.count_nonzero(inlier) < 2:
        inlier[:] = True
    return fit_mean_line_rigid(previous[inlier], current[inlier]), inlier


def refit_cumulative_global_transforms(
    midline_tracks: np.ndarray,
    transition_valid: np.ndarray,
    robust: bool,
) -> dict[str, np.ndarray]:
    """Refit cumulative transforms from saved midline tracks.

    ``robust=False`` reproduces the mean-line fit. ``robust=True`` is a
    separately stored project sensitivity branch.
    """
    tracks = np.asarray(midline_tracks, dtype=np.float32)
    valid = np.asarray(transition_valid, dtype=bool)
    if tracks.ndim != 3 or tracks.shape[2] != 2 or valid.shape != tracks.shape[:2]:
        raise ValueError("midline tracks/valid arrays have incompatible shapes")
    identity = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    transforms = np.repeat(identity[None], len(tracks), axis=0)
    inlier = np.zeros(valid.shape, dtype=bool)
    inlier[0] = valid[0]
    for frame_idx in range(1, len(tracks)):
        selected = valid[frame_idx]
        if np.count_nonzero(selected) < 2:
            transforms[frame_idx] = transforms[frame_idx - 1]
            continue
        if robust:
            incremental, local_inlier = robust_mean_line_rigid(
                tracks[frame_idx - 1, selected], tracks[frame_idx, selected]
            )
            inlier[frame_idx, np.flatnonzero(selected)[local_inlier]] = True
        else:
            incremental = fit_mean_line_rigid(
                tracks[frame_idx - 1, selected], tracks[frame_idx, selected]
            )
            inlier[frame_idx, selected] = True
        transforms[frame_idx] = compose_affine(
            incremental, transforms[frame_idx - 1]
        )
    return {"global_transform": transforms, "global_fit_inlier": inlier}


def interpolate_anchor_corrections(
    raw_transforms: np.ndarray,
    initial_reference_points: np.ndarray,
    anchor_indices: np.ndarray,
    anchor_reference_points: np.ndarray,
) -> dict[str, np.ndarray]:
    """Smoothly correct raw global pose using selected anatomical anchors.

    This is a project-fusion extension, not a reported Huang-2022 step.  The
    returned correction is kept separate so that it cannot be counted as
    local tissue deformation.
    """
    raw_transforms = np.asarray(raw_transforms, dtype=np.float32)
    anchor_indices = np.asarray(anchor_indices, dtype=np.int32)
    anchor_reference_points = np.asarray(anchor_reference_points, dtype=np.float32)
    if raw_transforms.ndim != 3 or raw_transforms.shape[1:] != (2, 3):
        raise ValueError("raw_transforms must have shape (frames, 2, 3)")
    if anchor_indices.ndim != 1 or len(anchor_indices) < 1:
        raise ValueError("at least one correction anchor is required")
    if anchor_reference_points.shape[0] != len(anchor_indices):
        raise ValueError("anchor point array does not match anchor indices")
    if np.any(np.diff(anchor_indices) <= 0):
        raise ValueError("anchor indices must be strictly increasing")
    if anchor_indices[0] < 0 or anchor_indices[-1] >= len(raw_transforms):
        raise ValueError("correction anchor outside transform range")

    anchor_corrections = []
    for frame_idx, expected in zip(anchor_indices, anchor_reference_points):
        raw_prediction = apply_affine(
            initial_reference_points, raw_transforms[int(frame_idx)]
        )
        anchor_corrections.append(
            fit_mean_line_rigid(raw_prediction, expected)
        )
    anchor_corrections = np.asarray(anchor_corrections, dtype=np.float32)
    anchor_angles = np.unwrap(
        np.arctan2(anchor_corrections[:, 1, 0], anchor_corrections[:, 0, 0])
    )
    anchor_translation = anchor_corrections[:, :, 2]
    frame_axis = np.arange(len(raw_transforms), dtype=np.float64)
    angle = np.interp(frame_axis, anchor_indices, anchor_angles)
    translation_x = np.interp(
        frame_axis, anchor_indices, anchor_translation[:, 0]
    )
    translation_y = np.interp(
        frame_axis, anchor_indices, anchor_translation[:, 1]
    )
    correction = np.zeros_like(raw_transforms)
    correction[:, 0, 0] = np.cos(angle)
    correction[:, 0, 1] = -np.sin(angle)
    correction[:, 1, 0] = np.sin(angle)
    correction[:, 1, 1] = np.cos(angle)
    correction[:, 0, 2] = translation_x
    correction[:, 1, 2] = translation_y
    corrected = np.stack(
        [
            compose_affine(correction[frame_idx], raw_transforms[frame_idx])
            for frame_idx in range(len(raw_transforms))
        ]
    ).astype(np.float32)
    return {
        "anchor_correction_transform": correction,
        "corrected_global_transform": corrected,
        "anchor_correction_at_constraints": anchor_corrections,
    }


def _inside_with_margin(
    points: np.ndarray, image_shape: tuple[int, int], radius: int
) -> np.ndarray:
    height, width = image_shape
    points = np.asarray(points, dtype=np.float32)
    return (
        np.all(np.isfinite(points), axis=1)
        & (points[:, 0] >= radius)
        & (points[:, 0] < width - radius)
        & (points[:, 1] >= radius)
        & (points[:, 1] < height - radius)
    )


def patch_pcc(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    previous_points: np.ndarray,
    current_points: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Pearson correlation of aligned local speckle windows."""
    previous_points = np.asarray(previous_points, dtype=np.float32)
    current_points = np.asarray(current_points, dtype=np.float32)
    if previous_points.shape != current_points.shape:
        raise ValueError("point arrays must have identical shapes")
    radius = window_size // 2
    valid = _inside_with_margin(previous_points, previous_image.shape, radius)
    valid &= _inside_with_margin(current_points, current_image.shape, radius)
    result = np.full(len(previous_points), np.nan, dtype=np.float32)
    for point_idx in np.flatnonzero(valid):
        first = cv2.getRectSubPix(
            previous_image,
            (window_size, window_size),
            tuple(float(v) for v in previous_points[point_idx]),
        ).astype(np.float32)
        second = cv2.getRectSubPix(
            current_image,
            (window_size, window_size),
            tuple(float(v) for v in current_points[point_idx]),
        ).astype(np.float32)
        first -= float(np.mean(first))
        second -= float(np.mean(second))
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator > 1e-6:
            result[point_idx] = float(np.sum(first * second) / denominator)
    return result


def track_global_midline(
    frames: list[np.ndarray],
    initial_midline: np.ndarray,
    config: HuangFusionConfig,
) -> dict[str, np.ndarray]:
    """Track ten midline points and estimate a cumulative rigid reference."""
    frame_count = len(frames)
    point_count = len(initial_midline)
    if frame_count < 2:
        raise ValueError("at least two frames are required")
    if point_count != config.midline_point_count:
        raise ValueError("initial_midline does not match configured point count")

    transforms = np.zeros((frame_count, 2, 3), dtype=np.float32)
    transforms[0] = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    tracks = np.full((frame_count, point_count, 2), np.nan, dtype=np.float32)
    references = np.full_like(tracks, np.nan)
    valid = np.zeros((frame_count, point_count), dtype=bool)
    pcc = np.full((frame_count, point_count), np.nan, dtype=np.float32)
    used_count = np.zeros(frame_count, dtype=np.int32)
    tracks[0] = np.asarray(initial_midline, dtype=np.float32)
    references[0] = tracks[0]
    valid[0] = True
    state = tracks[0].copy()
    margin = max(config.window_size_px, config.pcc_window_size_px) // 2

    for frame_idx in range(1, frame_count):
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            frames[frame_idx - 1],
            frames[frame_idx],
            state.reshape(-1, 1, 2),
            None,
            **config.lk_parameters(),
        )
        if next_points is None or status is None:
            next_points = np.full_like(state, np.nan)
            status_good = np.zeros(point_count, dtype=bool)
        else:
            next_points = next_points.reshape(-1, 2).astype(np.float32)
            status_good = status.reshape(-1).astype(bool)
        status_good &= _inside_with_margin(state, frames[frame_idx - 1].shape, margin)
        status_good &= _inside_with_margin(next_points, frames[frame_idx].shape, margin)
        used_count[frame_idx] = int(np.count_nonzero(status_good))

        if used_count[frame_idx] >= 2:
            incremental = fit_mean_line_rigid(
                state[status_good], next_points[status_good]
            )
        else:
            incremental = np.asarray(
                [[1, 0, 0], [0, 1, 0]], dtype=np.float32
            )
        transforms[frame_idx] = compose_affine(
            incremental, transforms[frame_idx - 1]
        )
        predicted = apply_affine(state, incremental)
        current_state = predicted
        current_state[status_good] = next_points[status_good]

        pcc_values = patch_pcc(
            frames[frame_idx - 1],
            frames[frame_idx],
            state,
            current_state,
            config.pcc_window_size_px,
        )
        pcc_values[~status_good] = np.nan
        tracks[frame_idx] = current_state
        references[frame_idx] = apply_affine(initial_midline, transforms[frame_idx])
        valid[frame_idx] = status_good
        pcc[frame_idx] = pcc_values
        state = current_state

    return {
        "midline_tracks": tracks,
        "midline_reference": references,
        "midline_lk_valid": valid,
        "midline_pcc": pcc,
        "midline_used_count": used_count,
        "global_transform": transforms,
    }


def track_local_wall_motion(
    frames: list[np.ndarray],
    initial_wall_points: np.ndarray,
    global_transforms: np.ndarray,
    config: HuangFusionConfig,
) -> dict[str, np.ndarray]:
    """Measure one-frame wall residuals without accumulating them."""
    frame_count = len(frames)
    point_count = len(initial_wall_points)
    reference = np.stack(
        [apply_affine(initial_wall_points, matrix) for matrix in global_transforms]
    ).astype(np.float32)
    measured = np.full_like(reference, np.nan)
    measured[0] = reference[0]
    residual = np.full_like(reference, np.nan)
    valid = np.zeros((frame_count, point_count), dtype=bool)
    pcc = np.full((frame_count, point_count), np.nan, dtype=np.float32)
    valid[0] = True
    margin = max(config.window_size_px, config.pcc_window_size_px) // 2

    for frame_idx in range(1, frame_count):
        previous_reference = reference[frame_idx - 1]
        current_reference = reference[frame_idx]
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            frames[frame_idx - 1],
            frames[frame_idx],
            previous_reference.reshape(-1, 1, 2),
            current_reference.reshape(-1, 1, 2).copy(),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            **config.lk_parameters(),
        )
        if tracked is None or status is None:
            tracked = np.full_like(current_reference, np.nan)
            status_good = np.zeros(point_count, dtype=bool)
        else:
            tracked = tracked.reshape(-1, 2).astype(np.float32)
            status_good = status.reshape(-1).astype(bool)
        status_good &= _inside_with_margin(
            previous_reference, frames[frame_idx - 1].shape, margin
        )
        status_good &= _inside_with_margin(
            tracked, frames[frame_idx].shape, margin
        )
        pcc_values = patch_pcc(
            frames[frame_idx - 1],
            frames[frame_idx],
            previous_reference,
            tracked,
            config.pcc_window_size_px,
        )
        pcc_values[~status_good] = np.nan
        measured[frame_idx] = tracked
        residual[frame_idx, status_good] = (
            tracked[status_good] - current_reference[status_good]
        )
        valid[frame_idx] = status_good
        pcc[frame_idx] = pcc_values

    return {
        "wall_reference": reference,
        "wall_measured": measured,
        "wall_local_residual": residual,
        "wall_lk_valid": valid,
        "wall_pcc": pcc,
    }


def deformation_from_wall_residuals(
    wall_reference: np.ndarray,
    wall_residual: np.ndarray,
    wall_valid: np.ndarray,
    paired_section_count: int,
    fps: float,
) -> dict[str, np.ndarray]:
    """Build anterior/posterior and cavity-width deformation activity maps."""
    anterior_reference = wall_reference[:, :paired_section_count]
    posterior_reference = wall_reference[:, paired_section_count:]
    anterior_residual = wall_residual[:, :paired_section_count]
    posterior_residual = wall_residual[:, paired_section_count:]
    anterior_valid = wall_valid[:, :paired_section_count]
    posterior_valid = wall_valid[:, paired_section_count:]

    across = posterior_reference - anterior_reference
    width = np.linalg.norm(across, axis=2)
    unit = np.divide(
        across,
        width[..., None],
        out=np.full_like(across, np.nan),
        where=width[..., None] > 1e-6,
    )
    anterior_inward_velocity = (
        np.sum(anterior_residual * unit, axis=2) * float(fps)
    )
    posterior_inward_velocity = (
        np.sum(posterior_residual * -unit, axis=2) * float(fps)
    )
    cavity_width_strain_rate = np.divide(
        np.sum((posterior_residual - anterior_residual) * unit, axis=2)
        * float(fps),
        width,
        out=np.full_like(width, np.nan, dtype=np.float32),
        where=width > 1e-6,
    )
    anterior_inward_velocity[~anterior_valid] = np.nan
    posterior_inward_velocity[~posterior_valid] = np.nan
    pair_valid = anterior_valid & posterior_valid
    cavity_width_strain_rate[~pair_valid] = np.nan
    return {
        "anterior_inward_velocity_px_s": anterior_inward_velocity.astype(np.float32),
        "posterior_inward_velocity_px_s": posterior_inward_velocity.astype(np.float32),
        "cavity_width_strain_rate_s": cavity_width_strain_rate.astype(np.float32),
        "anterior_deformation_valid": anterior_valid,
        "posterior_deformation_valid": posterior_valid,
        "cavity_deformation_valid": pair_valid,
    }


def fft_bandpass(
    values: np.ndarray,
    fps: float,
    low_cpm: float,
    high_cpm: float,
    minimum_valid_fraction: float,
    minimum_segment_s: float = 20.0,
) -> np.ndarray:
    """Gap-safe FFT band-pass retained for legacy exploratory maps.

    Finite runs are filtered independently.  Missing or short runs remain NaN,
    so this compatibility function cannot connect signal across an artifact gap.
    """
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive and finite")
    if not 0.0 < low_cpm < high_cpm:
        raise ValueError("cutoff frequencies must be positive and increasing")
    if not 0.0 <= minimum_valid_fraction <= 1.0:
        raise ValueError("minimum_valid_fraction must lie in [0, 1]")
    if minimum_segment_s <= 0.0:
        raise ValueError("minimum_segment_s must be positive")
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    elif values.ndim == 2:
        squeeze = False
    else:
        raise ValueError("values must be one- or two-dimensional")
    output = np.full_like(values, np.nan, dtype=np.float64)
    minimum_frames = max(4, int(np.ceil(minimum_segment_s * fps)))
    for column_idx in range(values.shape[1]):
        column = values[:, column_idx]
        valid = np.isfinite(column)
        if float(np.mean(valid)) < minimum_valid_fraction or np.count_nonzero(valid) < 2:
            continue
        padded = np.r_[False, valid, False].astype(np.int8)
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1)
        for start, end in zip(starts, ends):
            length = int(end - start)
            if length < minimum_frames:
                continue
            segment = column[start:end].copy()
            segment -= float(np.mean(segment))
            frequency = np.fft.rfftfreq(length, d=1.0 / float(fps))
            keep_frequency = (frequency >= low_cpm / 60.0) & (
                frequency <= high_cpm / 60.0
            )
            spectrum = np.fft.rfft(segment)
            spectrum[~keep_frequency] = 0.0
            output[start:end, column_idx] = np.fft.irfft(spectrum, n=length)
    result = output.astype(np.float32)
    return result[:, 0] if squeeze else result


def directional_energy_ratio(
    time_space: np.ndarray,
    fps: float,
    low_cpm: float,
    high_cpm: float,
) -> dict[str, float | str]:
    """Exploratory 2-D spectral direction ratio on cervix-to-fundus rows.

    ``time_space`` has shape (time, position), with position increasing from
    cervix to fundus.  Negative temporal/spatial-frequency products represent
    increasing-position (cervix-to-fundus) travel under this convention.
    """
    values = np.asarray(time_space, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 4:
        return {
            "cervix_to_fundus_energy": np.nan,
            "fundus_to_cervix_energy": np.nan,
            "energy_ratio": np.nan,
            "exploratory_label": "数据不足",
        }
    finite_fraction = np.mean(np.isfinite(values), axis=0)
    usable_columns = finite_fraction >= 0.5
    if np.count_nonzero(usable_columns) < 4:
        return {
            "cervix_to_fundus_energy": np.nan,
            "fundus_to_cervix_energy": np.nan,
            "energy_ratio": np.nan,
            "exploratory_label": "有效纵向位置不足",
        }
    values = values[:, usable_columns]
    for column_idx in range(values.shape[1]):
        column = values[:, column_idx]
        valid = np.isfinite(column)
        values[:, column_idx] = np.interp(
            np.arange(len(column)), np.flatnonzero(valid), column[valid]
        )
    values -= np.mean(values, axis=0, keepdims=True)
    window = np.hanning(values.shape[0])[:, None] * np.hanning(values.shape[1])[None, :]
    energy = np.abs(np.fft.fft2(values * window)) ** 2
    temporal_frequency = np.fft.fftfreq(values.shape[0], d=1.0 / float(fps))
    spatial_frequency = np.fft.fftfreq(
        values.shape[1], d=1.0 / max(values.shape[1] - 1, 1)
    )
    ft, fs = np.meshgrid(temporal_frequency, spatial_frequency, indexing="ij")
    physiological = (np.abs(ft) >= low_cpm / 60.0) & (
        np.abs(ft) <= high_cpm / 60.0
    ) & (np.abs(fs) > 1e-12)
    c2f = float(np.sum(energy[physiological & (ft * fs < 0.0)]))
    f2c = float(np.sum(energy[physiological & (ft * fs > 0.0)]))
    total = c2f + f2c
    ratio = (c2f - f2c) / total if total > 0.0 else np.nan
    if not np.isfinite(ratio):
        label = "频谱能量不足"
    elif ratio > 0.1:
        label = "仅探索：宫颈到宫底能量较高"
    elif ratio < -0.1:
        label = "仅探索：宫底到宫颈能量较高"
    else:
        label = "仅探索：无明显优势方向"
    return {
        "cervix_to_fundus_energy": c2f,
        "fundus_to_cervix_energy": f2c,
        "energy_ratio": float(ratio),
        "exploratory_label": label,
    }


def strict_propagation_gate(
    duration_s: float,
    recording_mean_pcc: float,
    signal_valid_fraction: float,
    config: HuangFusionConfig,
) -> tuple[bool, list[str]]:
    """Conservative gate; event-level propagation is intentionally absent."""
    reasons: list[str] = []
    if duration_s < config.minimum_formal_duration_s:
        reasons.append("视频短于最低频率一个完整周期120秒")
    if not np.isfinite(recording_mean_pcc) or recording_mean_pcc <= config.recording_pcc_threshold:
        reasons.append("整段平均PCC未超过0.8")
    if signal_valid_fraction < config.minimum_signal_valid_fraction:
        reasons.append("局部形变有效覆盖不足80%")
    reasons.append("尚未建立事件级稳健传播R2与持续时间验证")
    return False, reasons
