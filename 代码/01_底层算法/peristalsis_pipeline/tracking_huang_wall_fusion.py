"""Pairwise Huang-style motion measurement around frozen P3-v3 wall locations.

P3-v3 supplies an anatomical location for every frame and an LK search
initialization.  Its frame-to-frame displacement is never used as the
peristalsis signal.  For every adjacent frame pair, image optical flow is
measured afresh, rigid motion is estimated from ten midline points, and the
rigid component is subtracted from wall optical flow.  No local residual is
fed into the next frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .tracking_huang_fusion import (
    apply_affine,
    fit_mean_line_rigid,
    patch_pcc,
    resample_polyline,
)


@dataclass(frozen=True)
class HuangWallFusionConfig:
    """Explicit paper-inspired and engineering parameters for the pilot."""

    midline_point_count: int = 10
    window_size_px: int = 21
    pyramid_max_level: int = 3
    iteration_count: int = 30
    iteration_epsilon: float = 0.01
    pcc_window_size_px: int = 21
    recording_pcc_threshold: float = 0.80
    minimum_signal_valid_fraction: float = 0.80
    minimum_formal_duration_s: float = 120.0
    physiology_low_cpm: float = 0.5
    physiology_high_cpm: float = 5.0

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

    @property
    def support_radius_px(self) -> float:
        """Engineering support bound, not a Huang-paper threshold."""
        return self.window_size_px / 2.0

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


def _inside_with_margin(
    points: np.ndarray,
    image_shape: tuple[int, int],
    radius: int,
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


def _lk_with_initial(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    source_points: np.ndarray,
    initial_target_points: np.ndarray,
    config: HuangWallFusionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run LK only on finite in-bounds points and preserve original indexing."""
    source = np.asarray(source_points, dtype=np.float32)
    initial = np.asarray(initial_target_points, dtype=np.float32)
    if source.shape != initial.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source and initial targets must have matching shape (n, 2)")
    radius = max(config.window_size_px, config.pcc_window_size_px) // 2
    candidate = _inside_with_margin(source, previous_image.shape, radius)
    candidate &= _inside_with_margin(initial, current_image.shape, radius)
    measured = np.full_like(source, np.nan, dtype=np.float32)
    valid = np.zeros(len(source), dtype=bool)
    if not np.any(candidate):
        return measured, valid
    selected = np.flatnonzero(candidate)
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_image,
        current_image,
        source[selected].reshape(-1, 1, 2),
        initial[selected].reshape(-1, 1, 2).copy(),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        **config.lk_parameters(),
    )
    if tracked is None or status is None:
        return measured, valid
    tracked = tracked.reshape(-1, 2).astype(np.float32)
    status = status.reshape(-1).astype(bool)
    status &= _inside_with_margin(tracked, current_image.shape, radius)
    measured[selected] = tracked
    valid[selected] = status
    measured[~valid] = np.nan
    return measured, valid


def _forward_backward_error(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    source_points: np.ndarray,
    measured_points: np.ndarray,
    forward_valid: np.ndarray,
    config: HuangWallFusionConfig,
) -> np.ndarray:
    """Return an engineering diagnostic without applying a 2 px admission gate."""
    backward, backward_valid = _lk_with_initial(
        current_image,
        previous_image,
        measured_points,
        source_points,
        config,
    )
    valid = np.asarray(forward_valid, dtype=bool) & backward_valid
    error = np.full(len(source_points), np.nan, dtype=np.float32)
    error[valid] = np.linalg.norm(
        backward[valid] - np.asarray(source_points, dtype=np.float32)[valid],
        axis=1,
    )
    return error


def curve_tangents(points: np.ndarray) -> np.ndarray:
    """Unit tangents for an ordered wall curve."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("curve must have shape (n>=2, 2)")
    tangent = np.empty_like(points)
    tangent[0] = points[1] - points[0]
    tangent[-1] = points[-1] - points[-2]
    if len(points) > 2:
        tangent[1:-1] = points[2:] - points[:-2]
    length = np.linalg.norm(tangent, axis=1, keepdims=True)
    return np.divide(
        tangent,
        length,
        out=np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(points), 1)),
        where=length > 1e-6,
    ).astype(np.float32)


def point_to_polyline_distance(
    query_points: np.ndarray,
    curve_points: np.ndarray,
) -> np.ndarray:
    """Nearest Euclidean distance to a polyline, used as cross-curve distance."""
    query = np.asarray(query_points, dtype=np.float32)
    curve = np.asarray(curve_points, dtype=np.float32)
    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError("query_points must have shape (n, 2)")
    if curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 2:
        raise ValueError("curve_points must have shape (m>=2, 2)")
    output = np.full(len(query), np.nan, dtype=np.float32)
    finite_query = np.all(np.isfinite(query), axis=1)
    finite_segments = np.all(np.isfinite(curve[:-1]), axis=1)
    finite_segments &= np.all(np.isfinite(curve[1:]), axis=1)
    starts = curve[:-1][finite_segments]
    vectors = (curve[1:] - curve[:-1])[finite_segments]
    lengths_sq = np.sum(vectors * vectors, axis=1)
    usable = lengths_sq > 1e-8
    starts, vectors, lengths_sq = starts[usable], vectors[usable], lengths_sq[usable]
    if not len(starts):
        return output
    for point_idx in np.flatnonzero(finite_query):
        delta = query[point_idx] - starts
        fraction = np.clip(
            np.sum(delta * vectors, axis=1) / lengths_sq,
            0.0,
            1.0,
        )
        projection = starts + fraction[:, None] * vectors
        output[point_idx] = float(
            np.min(np.linalg.norm(query[point_idx] - projection, axis=1))
        )
    return output


def robust_mean_line_rigid(
    previous: np.ndarray,
    current: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-pass robust version of the paper-described rigid fit.

    The adaptive residual rule is a project-fusion safeguard.  It is not a
    2 px forward/backward gate and is not claimed as a Huang-paper parameter.
    """
    previous = np.asarray(previous, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    if previous.shape != current.shape or len(previous) < 2:
        raise ValueError("at least two matching points are required")
    first = fit_mean_line_rigid(previous, current)
    residual = np.linalg.norm(apply_affine(previous, first) - current, axis=1)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    adaptive_limit = median + 3.0 * 1.4826 * mad
    inlier = residual <= max(adaptive_limit, 0.25)
    if np.count_nonzero(inlier) < 2:
        inlier[:] = True
    return fit_mean_line_rigid(previous[inlier], current[inlier]), inlier


def _same_index_components(
    measured: np.ndarray,
    reference_curve: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tangent = curve_tangents(reference_curve)
    delta = np.asarray(measured, dtype=np.float32) - reference_curve
    tangential = np.abs(np.sum(delta * tangent, axis=1))
    normal = np.abs(delta[:, 0] * tangent[:, 1] - delta[:, 1] * tangent[:, 0])
    finite = np.all(np.isfinite(delta), axis=1)
    tangential[~finite] = np.nan
    normal[~finite] = np.nan
    return tangential.astype(np.float32), normal.astype(np.float32)


def estimate_pairwise_global_motion(
    frames: list[np.ndarray],
    p3_midline_tracks: np.ndarray,
    config: HuangWallFusionConfig,
) -> dict[str, np.ndarray]:
    """Estimate one rigid transform per adjacent pair; never accumulate it."""
    p3_midline_tracks = np.asarray(p3_midline_tracks, dtype=np.float32)
    if p3_midline_tracks.shape[0] != len(frames):
        raise ValueError("midline tracks and frames must have equal length")
    sampled = np.stack(
        [resample_polyline(points, config.midline_point_count) for points in p3_midline_tracks]
    ).astype(np.float32)
    frame_count, point_count = sampled.shape[:2]
    identity = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    transform = np.repeat(identity[None, ...], frame_count, axis=0)
    source = np.full_like(sampled, np.nan)
    measured = np.full_like(sampled, np.nan)
    predicted = np.full_like(sampled, np.nan)
    valid = np.zeros((frame_count, point_count), dtype=bool)
    pcc = np.full((frame_count, point_count), np.nan, dtype=np.float32)
    fb_error = np.full_like(pcc, np.nan)
    used_count = np.zeros(frame_count, dtype=np.int32)
    fit_inlier = np.zeros((frame_count, point_count), dtype=bool)
    source[0] = sampled[0]
    measured[0] = sampled[0]
    predicted[0] = sampled[0]

    for frame_idx in range(1, frame_count):
        source_points = sampled[frame_idx - 1]
        tracked, status = _lk_with_initial(
            frames[frame_idx - 1],
            frames[frame_idx],
            source_points,
            sampled[frame_idx],
            config,
        )
        if np.count_nonzero(status) >= 2:
            fitted, selected_inlier = robust_mean_line_rigid(
                source_points[status], tracked[status]
            )
            transform[frame_idx] = fitted
            fit_inlier[frame_idx, np.flatnonzero(status)[selected_inlier]] = True
        source[frame_idx] = source_points
        measured[frame_idx] = tracked
        predicted[frame_idx] = apply_affine(source_points, transform[frame_idx])
        valid[frame_idx] = status
        used_count[frame_idx] = int(np.count_nonzero(status))
        values = patch_pcc(
            frames[frame_idx - 1],
            frames[frame_idx],
            source_points,
            tracked,
            config.pcc_window_size_px,
        )
        values[~status] = np.nan
        pcc[frame_idx] = values
        fb_error[frame_idx] = _forward_backward_error(
            frames[frame_idx - 1],
            frames[frame_idx],
            source_points,
            tracked,
            status,
            config,
        )

    return {
        "p3_midline_points": sampled,
        "midline_pair_source": source,
        "midline_lk_measured": measured,
        "midline_global_prediction": predicted,
        "midline_lk_valid": valid,
        "midline_pcc": pcc,
        "midline_fb_error_px": fb_error,
        "midline_used_count": used_count,
        "midline_global_fit_inlier": fit_inlier,
        "pairwise_global_transform": transform,
    }


def measure_pairwise_wall_motion(
    frames: list[np.ndarray],
    p3_wall_tracks: np.ndarray,
    pairwise_global_transform: np.ndarray,
    anterior_count: int,
    config: HuangWallFusionConfig,
) -> dict[str, np.ndarray]:
    """Measure wall flow twice per pair and reset sources to P3 every frame."""
    reference = np.asarray(p3_wall_tracks, dtype=np.float32)
    transforms = np.asarray(pairwise_global_transform, dtype=np.float32)
    if reference.shape[0] != len(frames) or transforms.shape != (len(frames), 2, 3):
        raise ValueError("frame, wall-track, and transform lengths must agree")
    if anterior_count < 2 or reference.shape[1] != 2 * anterior_count:
        raise ValueError("paired anterior/posterior wall counts must match")
    frame_count, point_count = reference.shape[:2]
    source = np.full_like(reference, np.nan)
    global_prediction = np.full_like(reference, np.nan)
    measured = np.full_like(reference, np.nan)
    measured_global_initial = np.full_like(reference, np.nan)
    raw_displacement = np.full_like(reference, np.nan)
    local_residual = np.full_like(reference, np.nan)
    alternate_local_residual = np.full_like(reference, np.nan)
    lk_valid = np.zeros((frame_count, point_count), dtype=bool)
    alternate_valid = np.zeros_like(lk_valid)
    measurement_valid = np.zeros_like(lk_valid)
    pcc = np.full((frame_count, point_count), np.nan, dtype=np.float32)
    fb_error = np.full_like(pcc, np.nan)
    curve_distance = np.full_like(pcc, np.nan)
    same_index_tangential = np.full_like(pcc, np.nan)
    same_index_normal = np.full_like(pcc, np.nan)
    initialization_disagreement = np.full_like(pcc, np.nan)
    initialization_tangential = np.full_like(pcc, np.nan)
    initialization_normal = np.full_like(pcc, np.nan)
    topology_valid = np.zeros((frame_count, anterior_count), dtype=bool)
    source[0] = reference[0]
    global_prediction[0] = reference[0]
    measured[0] = reference[0]
    measured_global_initial[0] = reference[0]

    rail_slices = (slice(0, anterior_count), slice(anterior_count, point_count))
    for frame_idx in range(1, frame_count):
        previous_points = reference[frame_idx - 1]
        current_reference = reference[frame_idx]
        global_guess = apply_affine(previous_points, transforms[frame_idx])
        primary, primary_status = _lk_with_initial(
            frames[frame_idx - 1],
            frames[frame_idx],
            previous_points,
            current_reference,
            config,
        )
        alternate, alternate_status = _lk_with_initial(
            frames[frame_idx - 1],
            frames[frame_idx],
            previous_points,
            global_guess,
            config,
        )
        source[frame_idx] = previous_points
        global_prediction[frame_idx] = global_guess
        measured[frame_idx] = primary
        measured_global_initial[frame_idx] = alternate
        raw_displacement[frame_idx, primary_status] = (
            primary[primary_status] - previous_points[primary_status]
        )
        local_residual[frame_idx, primary_status] = (
            primary[primary_status] - global_guess[primary_status]
        )
        alternate_local_residual[frame_idx, alternate_status] = (
            alternate[alternate_status] - global_guess[alternate_status]
        )
        lk_valid[frame_idx] = primary_status
        alternate_valid[frame_idx] = alternate_status
        values = patch_pcc(
            frames[frame_idx - 1],
            frames[frame_idx],
            previous_points,
            primary,
            config.pcc_window_size_px,
        )
        values[~primary_status] = np.nan
        pcc[frame_idx] = values
        fb_error[frame_idx] = _forward_backward_error(
            frames[frame_idx - 1],
            frames[frame_idx],
            previous_points,
            primary,
            primary_status,
            config,
        )

        for rail_slice in rail_slices:
            target_curve = current_reference[rail_slice]
            curve_distance[frame_idx, rail_slice] = point_to_polyline_distance(
                primary[rail_slice], target_curve
            )
            tangential, normal = _same_index_components(
                primary[rail_slice], target_curve
            )
            same_index_tangential[frame_idx, rail_slice] = tangential
            same_index_normal[frame_idx, rail_slice] = normal
            difference = primary[rail_slice] - alternate[rail_slice]
            both = primary_status[rail_slice] & alternate_status[rail_slice]
            distance = np.linalg.norm(difference, axis=1)
            distance[~both] = np.nan
            initialization_disagreement[frame_idx, rail_slice] = distance
            tangent = curve_tangents(target_curve)
            along = np.abs(np.sum(difference * tangent, axis=1))
            across = np.abs(
                difference[:, 0] * tangent[:, 1]
                - difference[:, 1] * tangent[:, 0]
            )
            along[~both] = np.nan
            across[~both] = np.nan
            initialization_tangential[frame_idx, rail_slice] = along
            initialization_normal[frame_idx, rail_slice] = across

        measurement_valid[frame_idx] = primary_status & (
            curve_distance[frame_idx] <= config.support_radius_px
        )
        ant_measured = primary[:anterior_count]
        post_measured = primary[anterior_count:]
        ant_reference = current_reference[:anterior_count]
        post_reference = current_reference[anterior_count:]
        measured_width = post_measured - ant_measured
        reference_width = post_reference - ant_reference
        finite_pair = measurement_valid[frame_idx, :anterior_count]
        finite_pair &= measurement_valid[frame_idx, anterior_count:]
        topology_valid[frame_idx] = finite_pair & (
            np.sum(measured_width * reference_width, axis=1) > 0.0
        )

    return {
        "p3_wall_reference": reference,
        "wall_pair_source": source,
        "wall_global_prediction": global_prediction,
        "wall_lk_measured_p3_initial": measured,
        "wall_lk_measured_global_initial": measured_global_initial,
        "wall_raw_lk_displacement": raw_displacement,
        "wall_local_residual": local_residual,
        "wall_alternate_local_residual": alternate_local_residual,
        "wall_lk_valid": lk_valid,
        "wall_alternate_lk_valid": alternate_valid,
        "wall_measurement_valid": measurement_valid,
        "wall_pcc": pcc,
        "wall_fb_error_px": fb_error,
        "wall_to_p3_curve_distance_px": curve_distance,
        "wall_same_index_tangential_offset_px": same_index_tangential,
        "wall_same_index_normal_offset_px": same_index_normal,
        "initialization_disagreement_px": initialization_disagreement,
        "initialization_tangential_disagreement_px": initialization_tangential,
        "initialization_normal_disagreement_px": initialization_normal,
        "paired_wall_topology_valid": topology_valid,
    }


def deformation_from_local_wall_motion(
    p3_wall_reference: np.ndarray,
    wall_local_residual: np.ndarray,
    wall_measurement_valid: np.ndarray,
    paired_wall_topology_valid: np.ndarray,
    anterior_count: int,
    fps: float,
) -> dict[str, np.ndarray]:
    """Project local residuals into wall-normal ROI deformation signals."""
    reference = np.asarray(p3_wall_reference, dtype=np.float32)
    residual = np.asarray(wall_local_residual, dtype=np.float32)
    valid = np.asarray(wall_measurement_valid, dtype=bool)
    anterior_reference = reference[:, :anterior_count]
    posterior_reference = reference[:, anterior_count:]
    anterior_residual = residual[:, :anterior_count]
    posterior_residual = residual[:, anterior_count:]
    anterior_valid = valid[:, :anterior_count]
    posterior_valid = valid[:, anterior_count:]
    across = posterior_reference - anterior_reference
    width = np.linalg.norm(across, axis=2)
    unit = np.divide(
        across,
        width[..., None],
        out=np.full_like(across, np.nan),
        where=width[..., None] > 1e-6,
    )
    anterior_inward = np.sum(anterior_residual * unit, axis=2) * float(fps)
    posterior_inward = np.sum(posterior_residual * -unit, axis=2) * float(fps)
    pair_valid = anterior_valid & posterior_valid & paired_wall_topology_valid
    cavity_strain_rate = np.divide(
        -(anterior_inward + posterior_inward),
        width,
        out=np.full_like(width, np.nan, dtype=np.float32),
        where=width > 1e-6,
    )
    contraction_common = 0.5 * (anterior_inward + posterior_inward)
    opposing_wall = 0.5 * (anterior_inward - posterior_inward)
    anterior_inward[~anterior_valid] = np.nan
    posterior_inward[~posterior_valid] = np.nan
    cavity_strain_rate[~pair_valid] = np.nan
    contraction_common[~pair_valid] = np.nan
    opposing_wall[~pair_valid] = np.nan
    return {
        "anterior_inward_velocity_px_s": anterior_inward.astype(np.float32),
        "posterior_inward_velocity_px_s": posterior_inward.astype(np.float32),
        "cavity_width_strain_rate_s": cavity_strain_rate.astype(np.float32),
        "common_inward_contraction_velocity_px_s": contraction_common.astype(np.float32),
        "opposing_wall_velocity_px_s": opposing_wall.astype(np.float32),
        "cavity_width_px": width.astype(np.float32),
        "anterior_deformation_valid": anterior_valid,
        "posterior_deformation_valid": posterior_valid,
        "cavity_deformation_valid": pair_valid,
    }
