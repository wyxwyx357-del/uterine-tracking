"""Image-supported correction of intermittent P3 wall-localization offsets.

The detector samples ultrasound intensity along each wall's outward normal.
It is deliberately independent of patient identifiers, timestamps, and manual
review rows.  Probe-shake frames are never corrected by this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .radial_pair_geometry import outward_unit_normals


@dataclass(frozen=True)
class P3ImageBoundaryConfig:
    """Project-development settings; these are not clinical thresholds."""

    profile_radius_px: int = 8
    tangent_radius_px: int = 3
    maximum_correction_px: float = 4.0
    maximum_frame_to_frame_correction_change_px: float = 0.5
    minimum_edge_drop_gray: float = 6.0
    minimum_edge_prominence: float = 1.2
    candidate_minimum_outward_offset_px: float = 3.0
    candidate_minimum_relative_offset_px: float = 3.0
    candidate_minimum_edge_drop_gray: float = 12.0
    candidate_minimum_edge_prominence: float = 0.8
    automatic_minimum_adjacent_sections: int = 3
    review_minimum_duration_s: float = 0.5
    isolated_neighbor_support_window_s: float = 0.25
    automatic_minimum_duration_s: float = 1.0
    automatic_endpoint_section_count: int = 3
    automatic_maximum_within_cluster_range_px: float = 2.0
    quality_gate_minimum_finite_pointframes: int = 8
    quality_gate_pcc_worsening_tolerance: float = 1e-4
    quality_gate_fb_worsening_tolerance_px: float = 1e-4
    quality_gate_valid_fraction_tolerance: float = 0.02

    def __post_init__(self) -> None:
        if self.profile_radius_px < 3:
            raise ValueError("profile_radius_px must be at least 3")
        if self.tangent_radius_px < 0:
            raise ValueError("tangent_radius_px must be non-negative")
        for name, value in (
            ("maximum_correction_px", self.maximum_correction_px),
            (
                "maximum_frame_to_frame_correction_change_px",
                self.maximum_frame_to_frame_correction_change_px,
            ),
            ("minimum_edge_drop_gray", self.minimum_edge_drop_gray),
            ("minimum_edge_prominence", self.minimum_edge_prominence),
            ("candidate_minimum_outward_offset_px", self.candidate_minimum_outward_offset_px),
            ("candidate_minimum_relative_offset_px", self.candidate_minimum_relative_offset_px),
            ("candidate_minimum_edge_drop_gray", self.candidate_minimum_edge_drop_gray),
            ("candidate_minimum_edge_prominence", self.candidate_minimum_edge_prominence),
            ("review_minimum_duration_s", self.review_minimum_duration_s),
            (
                "isolated_neighbor_support_window_s",
                self.isolated_neighbor_support_window_s,
            ),
            ("automatic_minimum_duration_s", self.automatic_minimum_duration_s),
            (
                "automatic_maximum_within_cluster_range_px",
                self.automatic_maximum_within_cluster_range_px,
            ),
            (
                "quality_gate_pcc_worsening_tolerance",
                self.quality_gate_pcc_worsening_tolerance,
            ),
            (
                "quality_gate_fb_worsening_tolerance_px",
                self.quality_gate_fb_worsening_tolerance_px,
            ),
            (
                "quality_gate_valid_fraction_tolerance",
                self.quality_gate_valid_fraction_tolerance,
            ),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.automatic_minimum_adjacent_sections < 2:
            raise ValueError("automatic_minimum_adjacent_sections must be at least 2")
        if self.automatic_endpoint_section_count < 1:
            raise ValueError("automatic_endpoint_section_count must be positive")
        if self.quality_gate_minimum_finite_pointframes < 1:
            raise ValueError("quality_gate_minimum_finite_pointframes must be positive")


def wall_outward_normals(wall_tracks: np.ndarray) -> np.ndarray:
    """Return normals with shape ``(frame, side, section, xy)``."""

    tracks = np.asarray(wall_tracks, dtype=np.float32)
    if tracks.ndim != 4 or tracks.shape[1] != 2 or tracks.shape[-1] != 2:
        raise ValueError("wall_tracks must have shape (frames, 2, sections, 2)")
    if tracks.shape[2] < 2 or not np.all(np.isfinite(tracks)):
        raise ValueError("wall tracks need at least two finite sections per side")
    result = np.empty_like(tracks)
    for frame_idx in range(len(tracks)):
        result[frame_idx, 0] = outward_unit_normals(
            tracks[frame_idx, 0], tracks[frame_idx, 1]
        )
        result[frame_idx, 1] = outward_unit_normals(
            tracks[frame_idx, 1], tracks[frame_idx, 0]
        )
    return result


def _sample_frame_profiles(
    frame: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    profile_offsets: np.ndarray,
    tangent_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tangents = np.stack((-normals[..., 1], normals[..., 0]), axis=-1)
    coordinates = (
        points[..., None, None, :]
        + profile_offsets[None, None, :, None, None] * normals[..., None, None, :]
        + tangent_offsets[None, None, None, :, None] * tangents[..., None, None, :]
    )
    map_x = coordinates[..., 0].reshape(-1).astype(np.float32)
    map_y = coordinates[..., 1].reshape(-1).astype(np.float32)
    sampled = cv2.remap(
        frame,
        map_x.reshape(1, -1),
        map_y.reshape(1, -1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    ).reshape(*points.shape[:2], len(profile_offsets), len(tangent_offsets))
    profiles = np.nanmedian(sampled, axis=-1).astype(np.float32)
    inside = (
        (coordinates[..., 0] >= 0)
        & (coordinates[..., 0] <= frame.shape[1] - 1)
        & (coordinates[..., 1] >= 0)
        & (coordinates[..., 1] <= frame.shape[0] - 1)
    )
    valid = np.all(inside, axis=(-1, -2))
    return profiles, valid


def extract_wall_boundary_evidence(
    frames: list[np.ndarray],
    wall_tracks: np.ndarray,
    config: P3ImageBoundaryConfig | None = None,
) -> dict[str, np.ndarray]:
    """Measure the strongest bright-to-dark edge along each outward normal."""

    cfg = config or P3ImageBoundaryConfig()
    tracks = np.asarray(wall_tracks, dtype=np.float32)
    if len(frames) != len(tracks):
        raise ValueError("frames and wall tracks must have equal length")
    if not frames:
        raise ValueError("frames cannot be empty")
    normals = wall_outward_normals(tracks)
    offsets = np.arange(
        -cfg.profile_radius_px, cfg.profile_radius_px + 1, dtype=np.float32
    )
    tangent_offsets = np.arange(
        -cfg.tangent_radius_px, cfg.tangent_radius_px + 1, dtype=np.float32
    )
    shape = tracks.shape[:3]
    edge_offset = np.full(shape, np.nan, dtype=np.float32)
    edge_drop = np.full(shape, np.nan, dtype=np.float32)
    edge_prominence = np.full(shape, np.nan, dtype=np.float32)
    profile_valid = np.zeros(shape, dtype=bool)

    for frame_idx, frame in enumerate(frames):
        gray = np.asarray(frame, dtype=np.float32)
        if gray.ndim != 2:
            raise ValueError("all frames must be grayscale")
        profiles, valid = _sample_frame_profiles(
            gray, tracks[frame_idx], normals[frame_idx], offsets, tangent_offsets
        )
        smooth = np.empty_like(profiles)
        smooth[..., 0] = profiles[..., 0]
        smooth[..., -1] = profiles[..., -1]
        smooth[..., 1:-1] = (
            0.25 * profiles[..., :-2]
            + 0.5 * profiles[..., 1:-1]
            + 0.25 * profiles[..., 2:]
        )
        # Positive values mean intensity falls while moving out of the cavity.
        drop = smooth[..., :-2] - smooth[..., 2:]
        best = np.argmax(drop, axis=-1)
        best_drop = np.take_along_axis(drop, best[..., None], axis=-1)[..., 0]
        candidate_offsets = offsets[1:-1]
        selected_offset = candidate_offsets[best]
        median_drop = np.median(drop, axis=-1)
        mad = np.median(np.abs(drop - median_drop[..., None]), axis=-1)
        prominence = (best_drop - median_drop) / np.maximum(1.4826 * mad, 1.0)
        valid &= np.isfinite(best_drop)
        edge_offset[frame_idx, valid] = selected_offset[valid]
        edge_drop[frame_idx, valid] = best_drop[valid]
        edge_prominence[frame_idx, valid] = prominence[valid]
        profile_valid[frame_idx] = valid

    supported = (
        profile_valid
        & (edge_drop >= cfg.minimum_edge_drop_gray)
        & (edge_prominence >= cfg.minimum_edge_prominence)
    )
    return {
        "outward_normal": normals,
        "boundary_edge_offset_px": edge_offset,
        "boundary_edge_drop_gray": edge_drop,
        "boundary_edge_prominence": edge_prominence,
        "boundary_profile_valid": profile_valid,
        "boundary_edge_supported": supported,
        "profile_offsets_px": offsets,
    }


def _retain_spatial_runs(
    candidate: np.ndarray,
    offsets: np.ndarray,
    minimum_length: int,
    maximum_range_px: float | None = None,
) -> np.ndarray:
    retained = np.zeros_like(candidate, dtype=bool)
    for frame_idx in range(len(candidate)):
        for side in range(candidate.shape[1]):
            padded = np.pad(candidate[frame_idx, side].astype(np.int8), (1, 1))
            starts = np.flatnonzero(np.diff(padded) == 1)
            stops = np.flatnonzero(np.diff(padded) == -1)
            for start, stop in zip(starts, stops):
                if stop - start < minimum_length:
                    continue
                if maximum_range_px is not None:
                    values = offsets[frame_idx, side, start:stop]
                    if np.ptp(values) > maximum_range_px:
                        continue
                retained[frame_idx, side, start:stop] = True
    return retained


def _retain_temporal_runs(candidate: np.ndarray, minimum_length: int) -> np.ndarray:
    retained = np.zeros_like(candidate, dtype=bool)
    for side in range(candidate.shape[1]):
        for section in range(candidate.shape[2]):
            padded = np.pad(candidate[:, side, section].astype(np.int8), (1, 1))
            starts = np.flatnonzero(np.diff(padded) == 1)
            stops = np.flatnonzero(np.diff(padded) == -1)
            for start, stop in zip(starts, stops):
                if stop - start >= minimum_length:
                    retained[start:stop, side, section] = True
    return retained


def _dilate_in_time(candidate: np.ndarray, radius: int) -> np.ndarray:
    dilated = np.zeros_like(candidate, dtype=bool)
    for shift in range(-radius, radius + 1):
        if shift < 0:
            dilated[-shift:] |= candidate[:shift]
        elif shift > 0:
            dilated[:-shift] |= candidate[shift:]
        else:
            dilated |= candidate
    return dilated


def _stabilized_cluster_correction(
    correction_mask: np.ndarray,
    offsets: np.ndarray,
    maximum_correction_px: float,
    maximum_change_px: float,
    eligible_frames: np.ndarray | None = None,
    forbidden_points: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth correction in time and across neighboring wall sections.

    The evidence-supported interval keeps the component target.  Smooth ramps
    are placed immediately before and after that interval so the first and last
    supported frames are not attenuated back toward zero.
    """

    candidate_correction = np.zeros(offsets.shape, dtype=np.float32)
    target_by_point = np.zeros(offsets.shape, dtype=np.float32)
    eligible = (
        np.ones(len(correction_mask), dtype=bool)
        if eligible_frames is None
        else np.asarray(eligible_frames, dtype=bool)
    )
    if eligible.shape != (len(correction_mask),):
        raise ValueError("eligible_frames must have one value per frame")
    forbidden = (
        np.zeros_like(correction_mask, dtype=bool)
        if forbidden_points is None
        else np.asarray(forbidden_points, dtype=bool)
    )
    if forbidden.shape != correction_mask.shape:
        raise ValueError("forbidden_points must match correction_mask")
    for side in range(correction_mask.shape[1]):
        component_count, labels = cv2.connectedComponents(
            correction_mask[:, side].astype(np.uint8), connectivity=8
        )
        for component_id in range(1, component_count):
            component = labels == component_id
            values = offsets[:, side][component]
            positive = values[np.isfinite(values) & (values > 0)]
            if not len(positive):
                continue
            target = float(
                np.clip(np.median(positive), 0.0, maximum_correction_px)
            )
            target_by_point[:, side][component] = target
            for section in range(component.shape[1]):
                padded = np.pad(component[:, section].astype(np.int8), (1, 1))
                starts = np.flatnonzero(np.diff(padded) == 1)
                stops = np.flatnonzero(np.diff(padded) == -1)
                for start, stop in zip(starts, stops):
                    ramp_frames = max(
                        2, int(np.ceil(1.5 * target / maximum_change_px))
                    )
                    candidate_correction[start:stop, side, section] = target
                    before_start = max(0, start - ramp_frames)
                    if before_start < start:
                        phase = np.linspace(
                            0.0, 1.0, start - before_start + 1, dtype=np.float32
                        )[:-1]
                        smoothstep = phase * phase * (3.0 - 2.0 * phase)
                        candidate_correction[
                            before_start:start, side, section
                        ] = np.maximum(
                            candidate_correction[before_start:start, side, section],
                            target * smoothstep,
                        )
                    after_stop = min(len(correction_mask), stop + ramp_frames)
                    if stop < after_stop:
                        phase = np.linspace(
                            1.0, 0.0, after_stop - stop + 1, dtype=np.float32
                        )[1:]
                        smoothstep = phase * phase * (3.0 - 2.0 * phase)
                        candidate_correction[stop:after_stop, side, section] = np.maximum(
                            candidate_correction[stop:after_stop, side, section],
                            target * smoothstep,
                        )

    candidate_correction[~eligible] = 0.0

    correction = np.zeros_like(candidate_correction)
    for side in range(correction_mask.shape[1]):
        padded = np.pad(
            candidate_correction[:, side], ((0, 0), (1, 1)), mode="edge"
        )
        correction[:, side] = (
            0.25 * padded[:, :-2]
            + 0.50 * padded[:, 1:-1]
            + 0.25 * padded[:, 2:]
        )
    mandatory_zero = forbidden | ~eligible[:, None, None]
    correction[mandatory_zero] = 0.0
    for frame_idx in range(1, len(correction)):
        correction[frame_idx] = np.minimum(
            correction[frame_idx], correction[frame_idx - 1] + maximum_change_px
        )
    for frame_idx in range(len(correction) - 2, -1, -1):
        correction[frame_idx] = np.minimum(
            correction[frame_idx], correction[frame_idx + 1] + maximum_change_px
        )
    applied = correction > 1e-6
    return correction, target_by_point, applied


def build_automatic_boundary_correction(
    wall_tracks: np.ndarray,
    boundary_evidence: dict[str, np.ndarray],
    artifact_grade: np.ndarray,
    fps: float,
    config: P3ImageBoundaryConfig | None = None,
    additional_excluded_frames: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Create a conservative, separate P3 correction track.

    Candidate locations are image-derived only.  Their final acceptance must
    be decided later by :func:`apply_tracking_quality_gate` after comparing
    original and candidate LK measurements.
    """

    cfg = config or P3ImageBoundaryConfig()
    tracks = np.asarray(wall_tracks, dtype=np.float32)
    offsets = np.asarray(boundary_evidence["boundary_edge_offset_px"], dtype=np.float32)
    drops = np.asarray(boundary_evidence["boundary_edge_drop_gray"], dtype=np.float32)
    prominence = np.asarray(
        boundary_evidence["boundary_edge_prominence"], dtype=np.float32
    )
    normals = np.asarray(boundary_evidence["outward_normal"], dtype=np.float32)
    grade = np.asarray(artifact_grade, dtype=np.int8)
    if offsets.shape != tracks.shape[:3] or normals.shape != tracks.shape:
        raise ValueError("boundary evidence shapes must match wall tracks")
    if grade.shape != (len(tracks),):
        raise ValueError("artifact_grade must have one value per frame")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    excluded = (
        np.zeros(len(tracks), dtype=bool)
        if additional_excluded_frames is None
        else np.asarray(additional_excluded_frames, dtype=bool)
    )
    if excluded.shape != (len(tracks),):
        raise ValueError("additional_excluded_frames must have one value per frame")

    def candidate_masks(
        stable_frames: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        stable_offsets = np.where(stable_frames[:, None, None], offsets, np.nan)
        baseline = np.nanmedian(stable_offsets, axis=0).astype(np.float32)
        relative = offsets - baseline[None]
        seed = (
            stable_frames[:, None, None]
            & np.isfinite(offsets)
            & (offsets >= cfg.candidate_minimum_outward_offset_px)
            & (relative >= cfg.candidate_minimum_relative_offset_px)
            & (drops >= cfg.candidate_minimum_edge_drop_gray)
            & (prominence >= cfg.candidate_minimum_edge_prominence)
        )
        review_minimum_frames = max(
            2, int(round(cfg.review_minimum_duration_s * fps))
        )
        review_spatial = _retain_spatial_runs(seed, offsets, minimum_length=2)
        review_cluster = _retain_temporal_runs(
            review_spatial, review_minimum_frames
        )
        persistent_single = _retain_temporal_runs(seed, review_minimum_frames)
        neighboring_seed = np.zeros_like(seed, dtype=bool)
        neighboring_seed[:, :, 1:] |= seed[:, :, :-1]
        neighboring_seed[:, :, :-1] |= seed[:, :, 1:]
        support_radius = max(
            1, int(round(cfg.isolated_neighbor_support_window_s * fps))
        )
        isolated_supported = persistent_single & _dilate_in_time(
            neighboring_seed, support_radius
        )
        review = review_cluster | isolated_supported
        coherent = _retain_spatial_runs(
            seed,
            offsets,
            minimum_length=cfg.automatic_minimum_adjacent_sections,
            maximum_range_px=cfg.automatic_maximum_within_cluster_range_px,
        )
        minimum_frames = max(
            2, int(round(cfg.automatic_minimum_duration_s * fps))
        )
        persistent = _retain_temporal_runs(coherent, minimum_frames)
        endpoint = np.zeros(tracks.shape[2], dtype=bool)
        endpoint[: cfg.automatic_endpoint_section_count] = True
        endpoint[-cfg.automatic_endpoint_section_count :] = True
        high_specificity = _retain_spatial_runs(
            persistent & endpoint[None, None, :],
            offsets,
            minimum_length=cfg.automatic_minimum_adjacent_sections,
            maximum_range_px=cfg.automatic_maximum_within_cluster_range_px,
        )
        return (
            baseline,
            relative,
            review,
            persistent,
            high_specificity,
            isolated_supported & ~review_cluster,
        )

    development_stable = grade < 3
    (
        development_baseline,
        development_relative,
        development_review_candidate,
        _,
        _,
        _,
    ) = candidate_masks(development_stable)
    stable = development_stable & ~excluded
    (
        temporal_baseline,
        relative_offset,
        review_candidate,
        persistent_consistent,
        high_specificity,
        isolated_supported,
    ) = candidate_masks(stable)
    automatic_candidate = review_candidate.copy()

    correction, correction_target, automatic = _stabilized_cluster_correction(
        automatic_candidate,
        offsets,
        cfg.maximum_correction_px,
        cfg.maximum_frame_to_frame_correction_change_px,
        stable,
    )
    corrected_tracks = tracks + correction[..., None] * normals
    correction_change = np.zeros_like(correction)
    correction_change[1:] = np.abs(correction[1:] - correction[:-1])
    transition = correction_change > 1e-6
    transition[1:] |= transition[:-1].copy()

    return {
        "p3_wall_tracks_original": tracks,
        "p3_wall_tracks_image_corrected": corrected_tracks.astype(np.float32),
        "boundary_temporal_baseline_offset_px": temporal_baseline,
        "boundary_relative_offset_px": relative_offset.astype(np.float32),
        "development_boundary_temporal_baseline_offset_px": development_baseline,
        "development_boundary_relative_offset_px": development_relative.astype(
            np.float32
        ),
        "development_review_candidate_without_automatic_grade3_exclusion": (
            development_review_candidate
        ),
        "image_boundary_review_candidate": review_candidate,
        "temporally_supported_isolated_candidate_mask": isolated_supported,
        "high_specificity_endpoint_correction_mask": high_specificity,
        "automatic_boundary_correction_mask": automatic,
        "automatic_boundary_correction_candidate_mask": automatic_candidate,
        "automatic_boundary_correction_excluded_frame": excluded,
        "automatic_boundary_correction_eligible_frame": stable,
        "development_candidate_excluded_by_automatic_grade3": (
            development_review_candidate & excluded[:, None, None]
        ),
        "automatic_boundary_correction_px": correction,
        "automatic_boundary_correction_target_px": correction_target,
        "automatic_boundary_correction_transition_risk": transition,
        "automatic_boundary_correction_applied": np.asarray(bool(np.any(automatic))),
        "automatic_correction_limited_to_stable_frames": np.asarray(True),
        "automatic_correction_limited_to_persistent_endpoint_clusters": np.asarray(False),
        "automatic_correction_limited_to_persistent_consistent_clusters": np.asarray(False),
        "automatic_correction_uses_shared_cluster_target": np.asarray(True),
        "automatic_correction_uses_smoothstep_temporal_ramp": np.asarray(True),
        "automatic_correction_uses_spatial_feather": np.asarray(True),
        "automatic_correction_maximum_change_px_per_frame": np.asarray(
            cfg.maximum_frame_to_frame_correction_change_px, dtype=np.float32
        ),
        "review_candidates_require_tracking_quality_gate": np.asarray(True),
        "automatic_grade3_frames_can_receive_boundary_correction": np.asarray(False),
        "automatic_correction_rule_status": np.asarray(
            "development_rule_with_automatic_grade3_safety_exclusion"
        ),
        "formal_automatic_boundary_correction_released": np.asarray(False),
        "thresholds_are_project_development_only": np.asarray(True),
        "external_generalization_validated": np.asarray(False),
    }


def apply_tracking_quality_gate(
    automatic_result: dict[str, np.ndarray],
    boundary_evidence: dict[str, np.ndarray],
    candidate_tracking: dict[str, np.ndarray],
    original_tracking: dict[str, np.ndarray],
    config: P3ImageBoundaryConfig | None = None,
) -> dict[str, np.ndarray]:
    """Keep only correction components supported by paired LK measurements.

    Internal wall components must improve at least one of PCC, forward-backward
    error, or valid-pair fraction without materially worsening the others.
    Endpoint components may be retained without a strict improvement because
    endpoint geometry is less constrained, but the same non-worsening limits
    still apply.  Candidate generation and acceptance remain separate so a
    strong grayscale edge alone can never force a correction.
    """

    cfg = config or P3ImageBoundaryConfig()
    result = dict(automatic_result)
    candidate = np.asarray(
        result["automatic_boundary_correction_candidate_mask"], dtype=bool
    )
    tracks = np.asarray(result["p3_wall_tracks_original"], dtype=np.float32)
    normals = np.asarray(boundary_evidence["outward_normal"], dtype=np.float32)
    offsets = np.asarray(
        boundary_evidence["boundary_edge_offset_px"], dtype=np.float32
    )

    quality_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("radial_pair_pcc", "radial_pair_fb_error_px", "radial_pair_valid"):
        candidate_values = np.asarray(candidate_tracking[name])
        original_values = np.asarray(original_tracking[name])[: len(tracks)]
        if candidate_values.shape != candidate.shape or original_values.shape != candidate.shape:
            raise ValueError(f"{name} shape must match correction candidates")
        quality_arrays[name] = (candidate_values, original_values)

    candidate_pcc, original_pcc = quality_arrays["radial_pair_pcc"]
    candidate_fb, original_fb = quality_arrays["radial_pair_fb_error_px"]
    candidate_valid, original_valid = quality_arrays["radial_pair_valid"]
    accepted_candidate = np.zeros_like(candidate, dtype=bool)
    rejection_reason = np.zeros(candidate.shape, dtype=np.int8)
    pcc_delta = np.full(candidate.shape, np.nan, dtype=np.float32)
    fb_delta = np.full(candidate.shape, np.nan, dtype=np.float32)
    valid_fraction_delta = np.full(candidate.shape, np.nan, dtype=np.float32)

    endpoint = np.zeros(candidate.shape[2], dtype=bool)
    endpoint[: cfg.automatic_endpoint_section_count] = True
    endpoint[-cfg.automatic_endpoint_section_count :] = True

    for side in range(candidate.shape[1]):
        component_count, labels = cv2.connectedComponents(
            candidate[:, side].astype(np.uint8), connectivity=8
        )
        for component_id in range(1, component_count):
            component = labels == component_id
            finite = (
                np.isfinite(candidate_pcc[:, side])
                & np.isfinite(original_pcc[:, side])
                & np.isfinite(candidate_fb[:, side])
                & np.isfinite(original_fb[:, side])
                & component
            )
            finite_count = int(np.count_nonzero(finite))
            if finite_count < cfg.quality_gate_minimum_finite_pointframes:
                rejection_reason[:, side][component] = 3
                continue

            component_pcc_delta = float(
                np.median(candidate_pcc[:, side][finite] - original_pcc[:, side][finite])
            )
            component_fb_delta = float(
                np.median(candidate_fb[:, side][finite] - original_fb[:, side][finite])
            )
            component_candidate_valid = float(
                np.mean(candidate_valid[:, side][component])
            )
            component_original_valid = float(
                np.mean(original_valid[:, side][component])
            )
            component_valid_delta = component_candidate_valid - component_original_valid
            pcc_delta[:, side][component] = component_pcc_delta
            fb_delta[:, side][component] = component_fb_delta
            valid_fraction_delta[:, side][component] = component_valid_delta

            within_tolerance = (
                component_pcc_delta >= -cfg.quality_gate_pcc_worsening_tolerance
                and component_fb_delta <= cfg.quality_gate_fb_worsening_tolerance_px
                and component_valid_delta >= -cfg.quality_gate_valid_fraction_tolerance
            )
            tracking_improved = (
                component_pcc_delta > 0
                or component_fb_delta < 0
                or component_valid_delta > 0
            )
            touches_endpoint = bool(np.any(component[:, endpoint]))
            accepted = within_tolerance and (tracking_improved or touches_endpoint)
            if accepted:
                accepted_candidate[:, side][component] = True
                rejection_reason[:, side][component] = 1 if tracking_improved else 2
            else:
                rejection_reason[:, side][component] = 4 if not within_tolerance else 5

    candidate_tracks = np.asarray(
        result["p3_wall_tracks_image_corrected"], dtype=np.float32
    ).copy()
    pre_gate_correction = np.asarray(
        result["automatic_boundary_correction_px"], dtype=np.float32
    ).copy()
    pre_gate_mask = np.asarray(
        result["automatic_boundary_correction_mask"], dtype=bool
    ).copy()
    correction, correction_target, automatic = _stabilized_cluster_correction(
        accepted_candidate,
        offsets,
        cfg.maximum_correction_px,
        cfg.maximum_frame_to_frame_correction_change_px,
        np.asarray(
            result["automatic_boundary_correction_eligible_frame"], dtype=bool
        ),
        candidate & ~accepted_candidate,
    )
    corrected_tracks = tracks + correction[..., None] * normals
    correction_change = np.zeros_like(correction)
    correction_change[1:] = np.abs(correction[1:] - correction[:-1])
    transition = correction_change > 1e-6
    transition[1:] |= transition[:-1].copy()

    result.update(
        p3_wall_tracks_candidate_image_corrected=candidate_tracks,
        pre_quality_gate_automatic_boundary_correction_px=pre_gate_correction,
        pre_quality_gate_automatic_boundary_correction_mask=pre_gate_mask,
        tracking_quality_accepted_boundary_correction_candidate_mask=accepted_candidate,
        tracking_quality_rejected_boundary_correction_candidate_mask=(
            candidate & ~accepted_candidate
        ),
        boundary_correction_quality_gate_reason=rejection_reason,
        boundary_correction_component_pcc_delta=pcc_delta,
        boundary_correction_component_fb_error_delta_px=fb_delta,
        boundary_correction_component_valid_fraction_delta=valid_fraction_delta,
        automatic_boundary_correction_px=correction,
        automatic_boundary_correction_target_px=correction_target,
        automatic_boundary_correction_mask=automatic,
        automatic_boundary_correction_transition_risk=transition,
        automatic_boundary_correction_applied=np.asarray(bool(np.any(automatic))),
        p3_wall_tracks_image_corrected=corrected_tracks.astype(np.float32),
        automatic_correction_uses_original_vs_candidate_lk_gate=np.asarray(True),
        automatic_correction_rule_status=np.asarray(
            "development_rule_with_spatiotemporal_candidates_and_lk_quality_gate"
        ),
    )
    return result


def apply_confirmed_boundary_overrides(
    automatic_result: dict[str, np.ndarray],
    boundary_evidence: dict[str, np.ndarray],
    manual_confirmed_mask: np.ndarray,
    artifact_grade: np.ndarray,
    maximum_correction_px: float = 4.0,
) -> dict[str, np.ndarray]:
    """Apply image-derived shifts only to manually confirmed stable points.

    Manual overrides are reported separately and never count as automatic
    detections.  A contiguous confirmed group receives its median positive
    image-boundary offset, which preserves local curve shape better than
    moving each point by a noisy independent edge estimate.
    """

    result = dict(automatic_result)
    tracks = np.asarray(result["p3_wall_tracks_original"], dtype=np.float32)
    normals = np.asarray(boundary_evidence["outward_normal"], dtype=np.float32)
    edge_offset = np.asarray(
        boundary_evidence["boundary_edge_offset_px"], dtype=np.float32
    )
    manual = np.asarray(manual_confirmed_mask, dtype=bool)
    grade = np.asarray(artifact_grade, dtype=np.int8)
    if manual.shape != tracks.shape[:3] or grade.shape != (len(tracks),):
        raise ValueError("manual mask or artifact grade shape does not match tracks")
    if not np.isfinite(maximum_correction_px) or maximum_correction_px <= 0:
        raise ValueError("maximum_correction_px must be positive and finite")

    automatic = np.asarray(result["automatic_boundary_correction_mask"], dtype=bool)
    manual_applied = manual & (grade < 3)[:, None, None] & ~automatic
    manual_offset = np.zeros(manual.shape, dtype=np.float32)
    for frame_idx in range(len(manual)):
        for side in range(manual.shape[1]):
            padded = np.pad(manual_applied[frame_idx, side].astype(np.int8), (1, 1))
            starts = np.flatnonzero(np.diff(padded) == 1)
            stops = np.flatnonzero(np.diff(padded) == -1)
            for start, stop in zip(starts, stops):
                values = edge_offset[frame_idx, side, start:stop]
                positive = values[np.isfinite(values) & (values > 0)]
                if not len(positive):
                    manual_applied[frame_idx, side, start:stop] = False
                    continue
                shift = float(np.clip(np.median(positive), 0.0, maximum_correction_px))
                manual_offset[frame_idx, side, start:stop] = shift

    final_offset = np.asarray(
        result["automatic_boundary_correction_px"], dtype=np.float32
    ).copy()
    final_offset[manual_applied] = manual_offset[manual_applied]
    final_tracks = tracks + final_offset[..., None] * normals
    transition = np.zeros_like(manual, dtype=bool)
    transition[1:] = np.abs(final_offset[1:] - final_offset[:-1]) > 1e-6
    transition[1:] |= transition[:-1].copy()
    result.update(
        manual_confirmed_boundary_correction_mask=manual_applied,
        manual_confirmed_boundary_correction_px=manual_offset,
        final_boundary_correction_px=final_offset,
        p3_wall_tracks_image_corrected=final_tracks.astype(np.float32),
        automatic_boundary_correction_transition_risk=transition,
        manual_confirmed_correction_applied=np.asarray(bool(np.any(manual_applied))),
        manual_confirmed_corrections_count_as_automatic=np.asarray(False),
    )
    return result
