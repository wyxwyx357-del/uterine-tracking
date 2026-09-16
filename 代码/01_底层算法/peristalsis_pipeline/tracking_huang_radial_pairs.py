"""Pairwise LK tracking for same-wall Huang-inspired radial marker pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import radial_pair_geometry
from .tracking_huang_fusion import (
    HuangFusionConfig,
    apply_affine,
    patch_pcc,
    track_local_wall_motion,
)
from .tracking_huang_wall_fusion import (
    HuangWallFusionConfig,
    _forward_backward_error,
    _lk_with_initial,
    point_to_polyline_distance,
)


REANCHOR_MAX_DURATION = np.uint8(1)
REANCHOR_TRACKING_QUALITY = np.uint8(2)
REANCHOR_PAIR_TOPOLOGY = np.uint8(4)


@dataclass(frozen=True)
class P3GuidedTrackletConfig:
    """Engineering gates for the experimental material-point tracklet branch."""

    maximum_duration_s: float = 1.0
    minimum_pcc: float = 0.80
    maximum_fb_error_px: float = 2.0
    maximum_initialization_disagreement_px: float = 2.0
    maximum_curve_distance_px: float | None = None

    def __post_init__(self) -> None:
        if self.maximum_duration_s <= 0.0:
            raise ValueError("maximum_duration_s must be positive")
        if not -1.0 <= self.minimum_pcc <= 1.0:
            raise ValueError("minimum_pcc must lie in [-1, 1]")
        for name, value in (
            ("maximum_fb_error_px", self.maximum_fb_error_px),
            (
                "maximum_initialization_disagreement_px",
                self.maximum_initialization_disagreement_px,
            ),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if (
            self.maximum_curve_distance_px is not None
            and self.maximum_curve_distance_px < 0.0
        ):
            raise ValueError("maximum_curve_distance_px must be non-negative")


def radial_reference_from_wall_tracks(
    p3_wall_tracks: np.ndarray,
    anterior_count: int,
    offset_px: float,
) -> np.ndarray:
    """Return rails ordered as anterior inner/outer, posterior inner/outer."""
    tracks = np.asarray(p3_wall_tracks, dtype=np.float32)
    if tracks.ndim != 3 or tracks.shape[2] != 2:
        raise ValueError("p3_wall_tracks must have shape (frames, points, 2)")
    if anterior_count < 2 or tracks.shape[1] != 2 * anterior_count:
        raise ValueError("paired anterior/posterior wall counts must match")
    references = []
    for frame in tracks:
        result = radial_pair_geometry.build_radial_pair_candidates(
            frame[:anterior_count], frame[anterior_count:], offset_px
        )
        references.append(
            np.concatenate(
                (
                    result["anterior_inner"],
                    result["anterior_outer"],
                    result["posterior_inner"],
                    result["posterior_outer"],
                )
            )
        )
    return np.asarray(references, dtype=np.float32)


def adjacent_radial_pair_crossing_risk(
    inner: np.ndarray, outer: np.ndarray, epsilon: float = 1e-6
) -> np.ndarray:
    """Mark both members of adjacent radial segments that intersect or overlap."""
    first = np.asarray(inner, dtype=np.float32)
    second = np.asarray(outer, dtype=np.float32)
    if first.shape != second.shape or first.ndim < 2 or first.shape[-1] != 2:
        raise ValueError("inner and outer rails must have matching (..., S, 2) shape")
    if first.shape[-2] < 2:
        raise ValueError("at least two longitudinal sections are required")
    p1 = first[..., :-1, :]
    q1 = second[..., :-1, :]
    p2 = first[..., 1:, :]
    q2 = second[..., 1:, :]

    def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    o1 = cross(q1 - p1, p2 - p1)
    o2 = cross(q1 - p1, q2 - p1)
    o3 = cross(q2 - p2, p1 - p2)
    o4 = cross(q2 - p2, q1 - p2)
    x_overlap = (
        np.maximum(np.minimum(p1[..., 0], q1[..., 0]), np.minimum(p2[..., 0], q2[..., 0]))
        <= np.minimum(np.maximum(p1[..., 0], q1[..., 0]), np.maximum(p2[..., 0], q2[..., 0]))
        + epsilon
    )
    y_overlap = (
        np.maximum(np.minimum(p1[..., 1], q1[..., 1]), np.minimum(p2[..., 1], q2[..., 1]))
        <= np.minimum(np.maximum(p1[..., 1], q1[..., 1]), np.maximum(p2[..., 1], q2[..., 1]))
        + epsilon
    )
    finite = (
        np.all(np.isfinite(p1), axis=-1)
        & np.all(np.isfinite(q1), axis=-1)
        & np.all(np.isfinite(p2), axis=-1)
        & np.all(np.isfinite(q2), axis=-1)
    )
    intersects = (
        finite
        & (o1 * o2 <= epsilon)
        & (o3 * o4 <= epsilon)
        & x_overlap
        & y_overlap
    )
    risk = np.zeros(first.shape[:-1], dtype=bool)
    risk[..., :-1] |= intersects
    risk[..., 1:] |= intersects
    return risk


def polyline_unit_tangent(points: np.ndarray) -> np.ndarray:
    """Return a stable same-section tangent for one ordered P3 rail."""
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 2:
        raise ValueError("points must have shape (S, 2) with S >= 2")
    tangent = np.empty_like(values)
    tangent[0] = values[1] - values[0]
    tangent[-1] = values[-1] - values[-2]
    if len(values) > 2:
        tangent[1:-1] = values[2:] - values[:-2]
    length = np.linalg.norm(tangent, axis=1)
    return np.divide(
        tangent,
        length[:, None],
        out=np.full_like(tangent, np.nan),
        where=length[:, None] > 1e-6,
    )


def measure_pairwise_radial_points(
    frames: list[np.ndarray],
    p3_wall_tracks: np.ndarray,
    pairwise_global_transform: np.ndarray,
    anterior_count: int,
    offset_px: float,
    config: HuangWallFusionConfig,
) -> dict[str, np.ndarray]:
    """Track each adjacent frame pair and reset all sources to P3 geometry."""
    reference = radial_reference_from_wall_tracks(
        p3_wall_tracks, anterior_count, offset_px
    )
    transforms = np.asarray(pairwise_global_transform, dtype=np.float32)
    if len(reference) != len(frames) or transforms.shape != (len(frames), 2, 3):
        raise ValueError("frame, radial-reference, and transform lengths must agree")

    frame_count, point_count = reference.shape[:2]
    source = np.full_like(reference, np.nan)
    global_prediction = np.full_like(reference, np.nan)
    measured = np.full_like(reference, np.nan)
    measured_global_initial = np.full_like(reference, np.nan)
    local_residual = np.full_like(reference, np.nan)
    alternate_local_residual = np.full_like(reference, np.nan)
    lk_valid = np.zeros((frame_count, point_count), dtype=bool)
    alternate_lk_valid = np.zeros_like(lk_valid)
    measurement_valid = np.zeros_like(lk_valid)
    pcc = np.full((frame_count, point_count), np.nan, dtype=np.float32)
    fb_error = np.full_like(pcc, np.nan)
    curve_distance = np.full_like(pcc, np.nan)
    initialization_disagreement = np.full_like(pcc, np.nan)
    same_section_distance = np.full_like(pcc, np.nan)
    longitudinal_slip = np.full_like(pcc, np.nan)
    pair_valid = np.zeros((frame_count, 2, anterior_count), dtype=bool)
    reference_crossing = np.zeros_like(pair_valid)
    measured_crossing = np.zeros_like(pair_valid)

    source[0] = reference[0]
    global_prediction[0] = reference[0]
    measured[0] = reference[0]
    measured_global_initial[0] = reference[0]
    rail_slices = tuple(
        slice(rail * anterior_count, (rail + 1) * anterior_count)
        for rail in range(4)
    )
    for side, (inner_rail, outer_rail) in enumerate(((0, 1), (2, 3))):
        reference_crossing[:, side] = adjacent_radial_pair_crossing_risk(
            reference[:, rail_slices[inner_rail]],
            reference[:, rail_slices[outer_rail]],
        )

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
        local_residual[frame_idx, primary_status] = (
            primary[primary_status] - global_guess[primary_status]
        )
        alternate_local_residual[frame_idx, alternate_status] = (
            alternate[alternate_status] - global_guess[alternate_status]
        )
        lk_valid[frame_idx] = primary_status
        alternate_lk_valid[frame_idx] = alternate_status
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
        both_initials = primary_status & alternate_status
        initialization_disagreement[frame_idx, both_initials] = np.linalg.norm(
            primary[both_initials] - alternate[both_initials], axis=1
        )

        for rail_slice in rail_slices:
            curve_distance[frame_idx, rail_slice] = point_to_polyline_distance(
                primary[rail_slice], current_reference[rail_slice]
            )
            delta = primary[rail_slice] - current_reference[rail_slice]
            same_section_distance[frame_idx, rail_slice] = np.linalg.norm(
                delta, axis=1
            )
            tangent = polyline_unit_tangent(current_reference[rail_slice])
            longitudinal_slip[frame_idx, rail_slice] = np.abs(
                np.sum(delta * tangent, axis=1)
            )
        measurement_valid[frame_idx] = primary_status & (
            curve_distance[frame_idx] <= config.support_radius_px
        )

        for side, (inner_rail, outer_rail) in enumerate(((0, 1), (2, 3))):
            inner_slice = rail_slices[inner_rail]
            outer_slice = rail_slices[outer_rail]
            measured_vector = primary[outer_slice] - primary[inner_slice]
            reference_vector = (
                current_reference[outer_slice] - current_reference[inner_slice]
            )
            measured_crossing[frame_idx, side] = adjacent_radial_pair_crossing_risk(
                primary[inner_slice], primary[outer_slice]
            )
            pair_valid[frame_idx, side] = (
                measurement_valid[frame_idx, inner_slice]
                & measurement_valid[frame_idx, outer_slice]
                & (np.sum(measured_vector * reference_vector, axis=1) > 0.0)
            )

    return {
        "radial_reference": reference,
        "radial_pair_source": source,
        "radial_global_prediction": global_prediction,
        "radial_lk_measured_p3_initial": measured,
        "radial_lk_measured_global_initial": measured_global_initial,
        "radial_local_residual": local_residual,
        "radial_alternate_local_residual": alternate_local_residual,
        "radial_lk_valid": lk_valid,
        "radial_alternate_lk_valid": alternate_lk_valid,
        "radial_measurement_valid": measurement_valid,
        "radial_pcc": pcc,
        "radial_fb_error_px": fb_error,
        "radial_to_reference_curve_distance_px": curve_distance,
        "radial_initialization_disagreement_px": initialization_disagreement,
        "radial_same_section_reference_distance_px": same_section_distance,
        "radial_longitudinal_slip_px": longitudinal_slip,
        "radial_pair_reference_crossing_risk": reference_crossing,
        "radial_pair_measured_crossing_risk": measured_crossing,
        "radial_pair_any_crossing_risk": reference_crossing | measured_crossing,
        "radial_pair_geometric_valid": pair_valid.copy(),
        "radial_pair_valid": pair_valid,
        "radial_pair_valid_semantics": np.asarray(
            "lk_status_plus_curve_support_plus_pair_orientation_not_tissue_identity"
        ),
        "radial_identity_continuity_proven": np.asarray(False),
    }


def measure_p3_guided_tracklet_radial_points(
    frames: list[np.ndarray],
    p3_wall_tracks: np.ndarray,
    pairwise_global_transform: np.ndarray,
    anterior_count: int,
    offset_px: float,
    fps: float,
    config: HuangWallFusionConfig,
    tracklet_config: P3GuidedTrackletConfig | None = None,
) -> dict[str, np.ndarray]:
    """Track material points briefly while using P3 as a wall-location guardrail.

    A radial pair is re-anchored as one unit when its maximum duration is
    reached or either marker fails a quality/topology check.  The transition
    into a re-anchor is deliberately excluded from ``radial_pair_valid`` so
    that P3 coordinate changes cannot become RSR.
    """
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    gates = tracklet_config or P3GuidedTrackletConfig()
    reference = radial_reference_from_wall_tracks(
        p3_wall_tracks, anterior_count, offset_px
    )
    transforms = np.asarray(pairwise_global_transform, dtype=np.float32)
    if len(reference) != len(frames) or transforms.shape != (len(frames), 2, 3):
        raise ValueError("frame, radial-reference, and transform lengths must agree")

    frame_count, point_count = reference.shape[:2]
    maximum_tracklet_frames = max(1, int(round(gates.maximum_duration_s * fps)))
    curve_distance_limit = (
        config.support_radius_px
        if gates.maximum_curve_distance_px is None
        else gates.maximum_curve_distance_px
    )
    source = np.full_like(reference, np.nan)
    global_prediction = np.full_like(reference, np.nan)
    measured = np.full_like(reference, np.nan)
    measured_p3_initial = np.full_like(reference, np.nan)
    tracklet_points = np.full_like(reference, np.nan)
    local_residual = np.full_like(reference, np.nan)
    alternate_local_residual = np.full_like(reference, np.nan)
    lk_valid = np.zeros((frame_count, point_count), dtype=bool)
    alternate_lk_valid = np.zeros_like(lk_valid)
    measurement_valid = np.zeros_like(lk_valid)
    pcc = np.full((frame_count, point_count), np.nan, dtype=np.float32)
    fb_error = np.full_like(pcc, np.nan)
    curve_distance = np.full_like(pcc, np.nan)
    initialization_disagreement = np.full_like(pcc, np.nan)
    same_section_distance = np.full_like(pcc, np.nan)
    longitudinal_slip = np.full_like(pcc, np.nan)
    pair_valid = np.zeros((frame_count, 2, anterior_count), dtype=bool)
    reference_crossing = np.zeros_like(pair_valid)
    measured_crossing = np.zeros_like(pair_valid)
    pair_reanchor = np.zeros_like(pair_valid)
    reanchor_reason = np.zeros(pair_valid.shape, dtype=np.uint8)
    tracklet_age_frames = np.zeros(pair_valid.shape, dtype=np.int32)
    tracklet_id = np.zeros(pair_valid.shape, dtype=np.int32)

    source[0] = reference[0]
    global_prediction[0] = reference[0]
    measured[0] = reference[0]
    measured_p3_initial[0] = reference[0]
    tracklet_points[0] = reference[0]
    rail_slices = tuple(
        slice(rail * anterior_count, (rail + 1) * anterior_count)
        for rail in range(4)
    )
    for side, (inner_rail, outer_rail) in enumerate(((0, 1), (2, 3))):
        reference_crossing[:, side] = adjacent_radial_pair_crossing_risk(
            reference[:, rail_slices[inner_rail]],
            reference[:, rail_slices[outer_rail]],
        )

    for frame_idx in range(1, frame_count):
        previous_points = tracklet_points[frame_idx - 1]
        current_reference = reference[frame_idx]
        global_guess = apply_affine(previous_points, transforms[frame_idx])
        primary, primary_status = _lk_with_initial(
            frames[frame_idx - 1],
            frames[frame_idx],
            previous_points,
            global_guess,
            config,
        )
        alternate, alternate_status = _lk_with_initial(
            frames[frame_idx - 1],
            frames[frame_idx],
            previous_points,
            current_reference,
            config,
        )
        source[frame_idx] = previous_points
        global_prediction[frame_idx] = global_guess
        measured[frame_idx] = primary
        measured_p3_initial[frame_idx] = alternate
        lk_valid[frame_idx] = primary_status
        alternate_lk_valid[frame_idx] = alternate_status
        local_residual[frame_idx, primary_status] = (
            primary[primary_status] - global_guess[primary_status]
        )
        alternate_local_residual[frame_idx, alternate_status] = (
            alternate[alternate_status] - global_guess[alternate_status]
        )

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
        both_initials = primary_status & alternate_status
        initialization_disagreement[frame_idx, both_initials] = np.linalg.norm(
            primary[both_initials] - alternate[both_initials], axis=1
        )
        for rail_slice in rail_slices:
            curve_distance[frame_idx, rail_slice] = point_to_polyline_distance(
                primary[rail_slice], current_reference[rail_slice]
            )
            delta = primary[rail_slice] - current_reference[rail_slice]
            same_section_distance[frame_idx, rail_slice] = np.linalg.norm(
                delta, axis=1
            )
            tangent = polyline_unit_tangent(current_reference[rail_slice])
            longitudinal_slip[frame_idx, rail_slice] = np.abs(
                np.sum(delta * tangent, axis=1)
            )

        measurement_valid[frame_idx] = (
            both_initials
            & np.isfinite(pcc[frame_idx])
            & (pcc[frame_idx] >= gates.minimum_pcc)
            & np.isfinite(fb_error[frame_idx])
            & (fb_error[frame_idx] <= gates.maximum_fb_error_px)
            & np.isfinite(curve_distance[frame_idx])
            & (curve_distance[frame_idx] <= curve_distance_limit)
            & np.isfinite(initialization_disagreement[frame_idx])
            & (
                initialization_disagreement[frame_idx]
                <= gates.maximum_initialization_disagreement_px
            )
        )

        committed = primary.copy()
        for side, (inner_rail, outer_rail) in enumerate(((0, 1), (2, 3))):
            inner = slice(
                inner_rail * anterior_count, (inner_rail + 1) * anterior_count
            )
            outer = slice(
                outer_rail * anterior_count, (outer_rail + 1) * anterior_count
            )
            measured_vector = primary[outer] - primary[inner]
            reference_vector = current_reference[outer] - current_reference[inner]
            measured_crossing[frame_idx, side] = adjacent_radial_pair_crossing_risk(
                primary[inner], primary[outer]
            )
            quality_good = (
                measurement_valid[frame_idx, inner]
                & measurement_valid[frame_idx, outer]
            )
            topology_good = (
                np.sum(measured_vector * reference_vector, axis=1) > 0.0
            )
            duration_due = (
                tracklet_age_frames[frame_idx - 1, side]
                >= maximum_tracklet_frames
            )
            reasons = np.zeros(anterior_count, dtype=np.uint8)
            reasons[duration_due] |= REANCHOR_MAX_DURATION
            reasons[~quality_good] |= REANCHOR_TRACKING_QUALITY
            reasons[quality_good & ~topology_good] |= REANCHOR_PAIR_TOPOLOGY
            reset = reasons != 0

            pair_reanchor[frame_idx, side] = reset
            reanchor_reason[frame_idx, side] = reasons
            pair_valid[frame_idx, side] = quality_good & topology_good & ~reset
            tracklet_age_frames[frame_idx, side] = np.where(
                reset,
                0,
                tracklet_age_frames[frame_idx - 1, side] + 1,
            )
            tracklet_id[frame_idx, side] = (
                tracklet_id[frame_idx - 1, side] + reset.astype(np.int32)
            )
            reset_indices = np.flatnonzero(reset)
            if len(reset_indices):
                inner_indices = inner_rail * anterior_count + reset_indices
                outer_indices = outer_rail * anterior_count + reset_indices
                committed[inner_indices] = current_reference[inner_indices]
                committed[outer_indices] = current_reference[outer_indices]

        tracklet_points[frame_idx] = committed

    return {
        "radial_reference": reference,
        "radial_pair_source": source,
        "radial_global_prediction": global_prediction,
        "radial_lk_measured": measured,
        "radial_lk_measured_p3_initial": measured_p3_initial,
        "radial_tracklet_points": tracklet_points,
        "radial_local_residual": local_residual,
        "radial_alternate_local_residual": alternate_local_residual,
        "radial_lk_valid": lk_valid,
        "radial_alternate_lk_valid": alternate_lk_valid,
        "radial_measurement_valid": measurement_valid,
        "radial_pcc": pcc,
        "radial_fb_error_px": fb_error,
        "radial_to_reference_curve_distance_px": curve_distance,
        "radial_initialization_disagreement_px": initialization_disagreement,
        "radial_same_section_reference_distance_px": same_section_distance,
        "radial_longitudinal_slip_px": longitudinal_slip,
        "radial_pair_reference_crossing_risk": reference_crossing,
        "radial_pair_measured_crossing_risk": measured_crossing,
        "radial_pair_any_crossing_risk": reference_crossing | measured_crossing,
        "radial_pair_geometric_valid": pair_valid.copy(),
        "radial_pair_valid": pair_valid,
        "radial_pair_valid_semantics": np.asarray(
            "quality_gated_short_tracklet_not_long_term_tissue_identity_proof"
        ),
        "radial_identity_continuity_proven": np.asarray(False),
        "radial_pair_reanchor": pair_reanchor,
        "radial_pair_reanchor_reason": reanchor_reason,
        "radial_pair_tracklet_age_frames": tracklet_age_frames,
        "radial_pair_tracklet_id": tracklet_id,
        "tracklet_maximum_frames": np.asarray(maximum_tracklet_frames, dtype=np.int32),
    }


def measure_radial_pairs_from_initial_geometry(
    frames: list[np.ndarray],
    initial_anterior_wall: np.ndarray,
    initial_posterior_wall: np.ndarray,
    global_transforms: np.ndarray,
    offset_px: float,
    config: HuangFusionConfig,
) -> dict[str, np.ndarray]:
    """Measure radial pairs without any P3/per-frame wall coordinates.

    The two initial walls define four radial rails once.  Every frame's
    reference is obtained only by applying the cumulative rigid transform.
    Local LK residuals are measured between adjacent frames and are never fed
    back into the next reference.  This is the project's no-P3 reproduction
    candidate; physical marker placement still requires pixel calibration.
    """
    anterior = np.asarray(initial_anterior_wall, dtype=np.float32)
    posterior = np.asarray(initial_posterior_wall, dtype=np.float32)
    if anterior.shape != posterior.shape or anterior.ndim != 2 or anterior.shape[1] != 2:
        raise ValueError("initial walls must have matching shape (sections, 2)")
    candidates = radial_pair_geometry.build_radial_pair_candidates(
        anterior, posterior, offset_px
    )
    initial = np.concatenate(
        (
            candidates["anterior_inner"],
            candidates["anterior_outer"],
            candidates["posterior_inner"],
            candidates["posterior_outer"],
        )
    ).astype(np.float32)
    transforms = np.asarray(global_transforms, dtype=np.float32)
    if transforms.shape != (len(frames), 2, 3):
        raise ValueError("global_transforms must have shape (frames, 2, 3)")
    local = track_local_wall_motion(frames, initial, transforms, config)
    reference = np.asarray(local["wall_reference"], dtype=np.float32)
    measured = np.asarray(local["wall_measured"], dtype=np.float32)
    lk_valid = np.asarray(local["wall_lk_valid"], dtype=bool)
    pcc = np.asarray(local["wall_pcc"], dtype=np.float32)
    fb_error = np.full((len(frames), len(initial)), np.nan, dtype=np.float32)
    source = np.full_like(reference, np.nan)
    source[0] = reference[0]
    source[1:] = reference[:-1]
    for frame_idx in range(1, len(frames)):
        fb_error[frame_idx] = _forward_backward_error(
            frames[frame_idx - 1],
            frames[frame_idx],
            source[frame_idx],
            measured[frame_idx],
            lk_valid[frame_idx],
            config,
        )
    pair_valid = np.zeros((len(frames), 2, len(anterior)), dtype=bool)
    for side, (inner_rail, outer_rail) in enumerate(((0, 1), (2, 3))):
        inner = slice(inner_rail * len(anterior), (inner_rail + 1) * len(anterior))
        outer = slice(outer_rail * len(anterior), (outer_rail + 1) * len(anterior))
        measured_vector = measured[:, outer] - measured[:, inner]
        reference_vector = reference[:, outer] - reference[:, inner]
        pair_valid[:, side] = (
            lk_valid[:, inner]
            & lk_valid[:, outer]
            & (np.sum(measured_vector * reference_vector, axis=2) > 0.0)
        )
    pair_valid[0] = False
    return {
        "radial_reference": reference,
        "radial_pair_source": source,
        "radial_global_prediction": reference,
        "radial_lk_measured": measured,
        "radial_local_residual": np.asarray(local["wall_local_residual"], dtype=np.float32),
        "radial_lk_valid": lk_valid,
        "radial_measurement_valid": lk_valid,
        "radial_pcc": pcc,
        "radial_fb_error_px": fb_error,
        "radial_pair_valid": pair_valid,
        "initial_radial_points": initial,
    }


def radial_deformation_from_tracking(
    tracking: dict[str, np.ndarray],
    anterior_count: int,
    fps: float,
) -> dict[str, np.ndarray]:
    """Calculate anterior/posterior signed RSR and engineering diagnostics."""
    source = np.asarray(tracking["radial_pair_source"], dtype=np.float32)
    measured_key = (
        "radial_lk_measured"
        if "radial_lk_measured" in tracking
        else "radial_lk_measured_p3_initial"
    )
    measured = np.asarray(tracking[measured_key], dtype=np.float32)
    measured_alternate = np.asarray(
        tracking.get(
            "radial_lk_measured_global_initial",
            tracking.get("radial_lk_measured_p3_initial", measured),
        ),
        dtype=np.float32,
    )
    global_prediction = np.asarray(
        tracking["radial_global_prediction"], dtype=np.float32
    )
    pcc = np.asarray(tracking["radial_pcc"], dtype=np.float32)
    fb_error = np.asarray(
        tracking.get("radial_fb_error_px", np.full(pcc.shape, np.nan, dtype=np.float32)),
        dtype=np.float32,
    )
    valid = np.asarray(tracking["radial_pair_valid"], dtype=bool)
    frame_count = len(source)
    rsr = np.full((frame_count, 2, anterior_count), np.nan, dtype=np.float32)
    projected_rsr = np.full_like(rsr, np.nan)
    alternate_rsr = np.full_like(rsr, np.nan)
    pair_pcc = np.full_like(rsr, np.nan)
    pair_fb_error = np.full_like(rsr, np.nan)
    common_local_speed = np.full_like(rsr, np.nan)

    for side, (inner_rail, outer_rail) in enumerate(((0, 1), (2, 3))):
        inner = slice(inner_rail * anterior_count, (inner_rail + 1) * anterior_count)
        outer = slice(outer_rail * anterior_count, (outer_rail + 1) * anterior_count)
        rsr[:, side] = radial_pair_geometry.radial_strain_rate_from_positions(
            source[:, inner], source[:, outer], measured[:, inner], measured[:, outer], fps
        )
        alternate_rsr[:, side] = radial_pair_geometry.radial_strain_rate_from_positions(
            source[:, inner],
            source[:, outer],
            measured_alternate[:, inner],
            measured_alternate[:, outer],
            fps,
        )
        source_vector = source[:, outer] - source[:, inner]
        source_length = np.linalg.norm(source_vector, axis=2)
        unit = np.divide(
            source_vector,
            source_length[..., None],
            out=np.full_like(source_vector, np.nan),
            where=source_length[..., None] > 1e-6,
        )
        inner_local = measured[:, inner] - global_prediction[:, inner]
        outer_local = measured[:, outer] - global_prediction[:, outer]
        projected_rsr[:, side] = np.divide(
            np.sum((outer_local - inner_local) * unit, axis=2) * float(fps),
            source_length,
            out=np.full_like(source_length, np.nan),
            where=source_length > 1e-6,
        )
        common_local_speed[:, side] = (
            np.linalg.norm(0.5 * (inner_local + outer_local), axis=2) * float(fps)
        )
        pair_pcc[:, side] = np.minimum(pcc[:, inner], pcc[:, outer])
        pair_fb_error[:, side] = np.maximum(fb_error[:, inner], fb_error[:, outer])

    for values in (
        rsr,
        projected_rsr,
        alternate_rsr,
        pair_pcc,
        pair_fb_error,
        common_local_speed,
    ):
        values[~valid] = np.nan
    return {
        "radial_strain_rate_s": rsr,
        "projected_radial_strain_rate_s": projected_rsr,
        "alternate_initial_radial_strain_rate_s": alternate_rsr,
        "initialization_rsr_disagreement_s": np.abs(rsr - alternate_rsr),
        "radial_pair_pcc": pair_pcc,
        "radial_pair_fb_error_px": pair_fb_error,
        "radial_pair_common_local_speed_px_s": common_local_speed,
        "radial_pair_valid": valid,
    }
