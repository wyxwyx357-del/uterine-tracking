"""Multi-evidence jitter and artifact QC v2.1 for Step 1.4 radial RSR.

No single metric is treated as proof of probe motion.  Three independent
evidence groups are scored: global motion, tracking quality, and synchronized
pair motion. Relative within-case scores are not allowed to create a strong
automatic candidate unless absolute measurements and temporal context support
them. Thresholds in this module are project-development settings, not published
Huang thresholds. Raw RSR is never overwritten, and unreviewed automatic grade
3 candidates are retained as grade 2 in the analysis-action output.  The
current main application uses P3 only for per-frame anatomical localization;
P3 coordinate deltas are not used as RSR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArtifactQCConfig:
    """Initial development settings; these are not frozen clinical cutoffs."""

    robust_z_onset: float = 2.0
    robust_z_full: float = 5.0
    group_evidence_threshold: float = 0.50
    strong_group_threshold: float = 0.75
    severe_quality_threshold: float = 0.85
    simultaneous_fraction_onset: float = 0.25
    simultaneous_fraction_full: float = 0.75
    p3_distance_support_weight: float = 0.40
    p3_topology_support_weight: float = 0.25
    # Absolute support floors are development values on 0.5-resized images.
    # They were introduced because within-case MAD scoring over-reacted to tiny
    # deviations in the very stable CASE_002 recording. They are not paper or
    # clinical thresholds and must remain unfrozen until external validation.
    absolute_translation_step_px: float = 0.30
    absolute_rotation_step_deg: float = 0.10
    absolute_low_pcc: float = 0.97
    absolute_fb_error_p95_px: float = 0.05
    absolute_invalid_fraction: float = 0.10
    extreme_translation_step_px: float = 1.00
    extreme_rotation_step_deg: float = 0.30
    extreme_low_pcc: float = 0.85
    extreme_fb_error_p95_px: float = 1.00
    require_adjacent_support_for_grade3: bool = True
    allow_unreviewed_automatic_grade3_action: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.robust_z_onset < self.robust_z_full:
            raise ValueError("robust z limits must be increasing and positive")
        for value in (
            self.group_evidence_threshold,
            self.strong_group_threshold,
            self.severe_quality_threshold,
            self.simultaneous_fraction_onset,
            self.simultaneous_fraction_full,
            self.p3_distance_support_weight,
            self.p3_topology_support_weight,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("score/fraction limits must lie in [0, 1]")
        positive_values = (
            self.absolute_translation_step_px,
            self.absolute_rotation_step_deg,
            self.absolute_fb_error_p95_px,
            self.absolute_invalid_fraction,
            self.extreme_translation_step_px,
            self.extreme_rotation_step_deg,
            self.extreme_fb_error_p95_px,
        )
        if any(value <= 0 for value in positive_values):
            raise ValueError("absolute support limits must be positive")
        if not 0 < self.extreme_low_pcc < self.absolute_low_pcc < 1:
            raise ValueError("PCC support limits must satisfy 0 < extreme < absolute < 1")


def finite_frame_rms(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("values must contain a frame axis plus value axes")
    axes = tuple(range(1, values.ndim))
    finite = np.isfinite(values)
    count = np.sum(finite, axis=axes)
    squared = np.nansum(values * values, axis=axes)
    return np.sqrt(
        np.divide(
            squared,
            count,
            out=np.full(len(values), np.nan, dtype=np.float64),
            where=count > 0,
        )
    ).astype(np.float32)


def finite_frame_mean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    axes = tuple(range(1, values.ndim))
    finite = np.isfinite(values)
    count = np.sum(finite, axis=axes)
    return np.divide(
        np.nansum(values, axis=axes),
        count,
        out=np.full(len(values), np.nan, dtype=np.float64),
        where=count > 0,
    ).astype(np.float32)


def finite_frame_percentile(values: np.ndarray, percentile: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    flattened = values.reshape(len(values), -1)
    output = np.full(len(values), np.nan, dtype=np.float32)
    for frame_idx, row in enumerate(flattened):
        finite = row[np.isfinite(row)]
        if len(finite):
            output[frame_idx] = float(np.percentile(finite, percentile))
    return output


def robust_positive_z(values: np.ndarray) -> np.ndarray:
    """One-sided robust z score; non-finite values remain NaN."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    output = np.full(values.shape, np.nan, dtype=np.float32)
    if not np.any(finite):
        return output
    center = float(np.median(values[finite]))
    mad = float(np.median(np.abs(values[finite] - center)))
    scale = 1.4826 * mad
    if scale <= max(abs(center), 1.0) * 1e-9:
        # A nearly constant baseline can legitimately have MAD=0 (for
        # example, zero invalid fraction in almost every frame).  Preserve
        # exact-baseline frames as zero while treating positive departures as
        # maximally unusual instead of silently erasing the only excursion.
        output[finite] = np.where(values[finite] > center, np.inf, 0.0)
        return output
    output[finite] = np.maximum((values[finite] - center) / scale, 0.0)
    return output


def score_from_robust_z(z_score: np.ndarray, config: ArtifactQCConfig) -> np.ndarray:
    z_score = np.asarray(z_score, dtype=np.float32)
    return np.clip(
        (z_score - config.robust_z_onset)
        / (config.robust_z_full - config.robust_z_onset),
        0.0,
        1.0,
    ).astype(np.float32)


def anomaly_score(values: np.ndarray, config: ArtifactQCConfig) -> np.ndarray:
    return score_from_robust_z(robust_positive_z(values), config)


def pair_common_motion_components(
    local_residual: np.ndarray, pair_valid: np.ndarray, section_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return common-motion RMS and same-direction spatial coherence."""
    residual = np.asarray(local_residual, dtype=np.float32)
    if residual.shape[1:] != (4 * section_count, 2):
        raise ValueError("radial local residual shape does not match four rails")
    valid = np.asarray(pair_valid, dtype=bool)
    if valid.shape != (len(residual), 2, section_count):
        raise ValueError("pair validity shape must match two radial-pair walls")
    centers = []
    for side, (inner_rail, outer_rail) in enumerate(((0, 1), (2, 3))):
        inner = residual[
            :, inner_rail * section_count : (inner_rail + 1) * section_count
        ]
        outer = residual[
            :, outer_rail * section_count : (outer_rail + 1) * section_count
        ]
        center = 0.5 * (inner + outer)
        center[~valid[:, side]] = np.nan
        centers.append(center)
    vectors = np.concatenate(centers, axis=1)
    magnitude = np.linalg.norm(vectors, axis=2)
    common_rms = finite_frame_rms(magnitude)
    finite = np.all(np.isfinite(vectors), axis=2)
    count = np.sum(finite, axis=1)
    summed = np.nansum(vectors, axis=1)
    mean_vector = np.divide(
        summed,
        count[:, None],
        out=np.full_like(summed, np.nan),
        where=count[:, None] > 0,
    )
    mean_magnitude = np.divide(
        np.nansum(magnitude, axis=1),
        count,
        out=np.full(len(vectors), np.nan, dtype=np.float32),
        where=count > 0,
    )
    coherence = np.divide(
        np.linalg.norm(mean_vector, axis=1),
        mean_magnitude,
        out=np.zeros(len(vectors), dtype=np.float32),
        where=mean_magnitude > 1e-8,
    )
    return common_rms.astype(np.float32), np.clip(coherence, 0.0, 1.0)


def simultaneous_deformation_fraction(rsr: np.ndarray) -> np.ndarray:
    """Fraction of wall sections with a same-frame robust |RSR| excursion."""
    values = np.abs(np.asarray(rsr, dtype=np.float64))
    if values.ndim != 3:
        raise ValueError("rsr must have shape (frames, sides, sections)")
    flattened = values.reshape(len(values), -1)
    z = np.full_like(flattened, np.nan)
    for column_idx in range(flattened.shape[1]):
        z[:, column_idx] = robust_positive_z(flattened[:, column_idx])
    finite = np.isfinite(z)
    count = np.sum(finite, axis=1)
    return np.divide(
        np.sum((z >= 3.0) & finite, axis=1),
        count,
        out=np.zeros(len(values), dtype=np.float64),
        where=count > 0,
    ).astype(np.float32)


def grade_from_evidence(
    global_motion_score: np.ndarray,
    tracking_quality_score: np.ndarray,
    synchronized_motion_score: np.ndarray,
    manual_minimum_grade: np.ndarray,
    config: ArtifactQCConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 0-3 grade and count of independent automatic evidence groups."""
    motion = np.nan_to_num(np.asarray(global_motion_score), nan=0.0)
    quality = np.nan_to_num(np.asarray(tracking_quality_score), nan=0.0)
    sync = np.nan_to_num(np.asarray(synchronized_motion_score), nan=0.0)
    manual = np.asarray(manual_minimum_grade, dtype=np.int8)
    if not (motion.shape == quality.shape == sync.shape == manual.shape):
        raise ValueError("evidence arrays must have matching shapes")
    evidence = np.stack((motion, quality, sync), axis=1)
    count = np.sum(evidence >= config.group_evidence_threshold, axis=1).astype(
        np.int8
    )
    grade = np.zeros(len(motion), dtype=np.int8)
    grade[np.max(evidence, axis=1) >= config.group_evidence_threshold] = 1
    grade[count >= 2] = 2
    severe = (
        (motion >= config.strong_group_threshold)
        & (quality >= config.group_evidence_threshold)
    ) | (
        (quality >= config.severe_quality_threshold)
        & (
            (motion >= config.group_evidence_threshold)
            | (sync >= config.group_evidence_threshold)
        )
    )
    grade[severe] = 3
    grade = np.maximum(grade, manual)
    return grade.astype(np.int8), count


def apply_absolute_and_temporal_support(
    *,
    relative_grade: np.ndarray,
    global_translation_step_px: np.ndarray,
    global_rotation_step_deg: np.ndarray,
    mean_pcc: np.ndarray,
    fb_error_p95_px: np.ndarray,
    invalid_fraction: np.ndarray,
    synchronized_motion_score: np.ndarray,
    config: ArtifactQCConfig,
) -> dict[str, np.ndarray]:
    """Constrain relative grade-3 candidates using absolute and neighbor support.

    A very stable recording can have a tiny MAD, making clinically small image
    changes look like large relative outliers. Relative grades 0-2 are retained.
    Relative grade 3 requires absolute tracking-quality evidence together with
    either absolute global motion or synchronized-motion evidence. An isolated
    non-extreme candidate is downgraded to grade 2 for review.
    """
    relative = np.asarray(relative_grade, dtype=np.int8)
    translation = np.asarray(global_translation_step_px, dtype=np.float32)
    rotation = np.asarray(global_rotation_step_deg, dtype=np.float32)
    pcc = np.asarray(mean_pcc, dtype=np.float32)
    fb = np.asarray(fb_error_p95_px, dtype=np.float32)
    invalid = np.asarray(invalid_fraction, dtype=np.float32)
    sync = np.nan_to_num(np.asarray(synchronized_motion_score, dtype=np.float32))
    arrays = (translation, rotation, pcc, fb, invalid, sync)
    if any(array.shape != relative.shape for array in arrays):
        raise ValueError("absolute-support arrays must match relative grade shape")

    absolute_motion = (
        np.nan_to_num(translation, nan=0.0)
        >= config.absolute_translation_step_px
    ) | (
        np.nan_to_num(rotation, nan=0.0) >= config.absolute_rotation_step_deg
    )
    absolute_quality = (
        np.nan_to_num(pcc, nan=1.0) <= config.absolute_low_pcc
    ) | (
        np.nan_to_num(fb, nan=0.0) >= config.absolute_fb_error_p95_px
    ) | (
        np.nan_to_num(invalid, nan=0.0) >= config.absolute_invalid_fraction
    )
    extreme = (
        np.nan_to_num(translation, nan=0.0)
        >= config.extreme_translation_step_px
    ) | (
        np.nan_to_num(rotation, nan=0.0) >= config.extreme_rotation_step_deg
    ) | (
        np.nan_to_num(pcc, nan=1.0) <= config.extreme_low_pcc
    ) | (
        np.nan_to_num(fb, nan=0.0) >= config.extreme_fb_error_p95_px
    )
    combination_support = absolute_quality & (
        absolute_motion | (sync >= config.group_evidence_threshold)
    )
    candidate = relative.copy()
    candidate[(relative >= 3) & ~combination_support] = 2

    pre_temporal_grade3 = candidate >= 3
    neighbor = np.zeros(len(candidate), dtype=bool)
    if len(candidate) > 1:
        neighbor[:-1] |= pre_temporal_grade3[1:]
        neighbor[1:] |= pre_temporal_grade3[:-1]
    if config.require_adjacent_support_for_grade3:
        candidate[pre_temporal_grade3 & ~(neighbor | extreme)] = 2
    return {
        "automatic_artifact_grade": candidate.astype(np.int8),
        "absolute_motion_support": absolute_motion,
        "absolute_tracking_quality_support": absolute_quality,
        "absolute_combination_support": combination_support,
        "adjacent_grade3_support": neighbor,
        "extreme_absolute_artifact_support": extreme,
    }


def build_artifact_qc(
    *,
    global_translation_step_px: np.ndarray,
    global_rotation_step_deg: np.ndarray,
    pair_pcc: np.ndarray,
    pair_fb_error_px: np.ndarray,
    pair_valid: np.ndarray,
    p3_curve_distance_px: np.ndarray,
    p3_topology_risk: np.ndarray,
    radial_local_residual: np.ndarray,
    raw_rsr: np.ndarray,
    section_count: int,
    manual_minimum_grade: np.ndarray | None = None,
    manual_reviewed_automatic_grade3: np.ndarray | None = None,
    manual_review_decision_code: np.ndarray | None = None,
    config: ArtifactQCConfig | None = None,
) -> dict[str, np.ndarray]:
    config = config or ArtifactQCConfig()
    frame_count = len(raw_rsr)
    manual = (
        np.zeros(frame_count, dtype=np.int8)
        if manual_minimum_grade is None
        else np.asarray(manual_minimum_grade, dtype=np.int8)
    )
    if manual.shape != (frame_count,):
        raise ValueError("manual_minimum_grade must have one value per frame")
    decision = (
        np.zeros(frame_count, dtype=np.int8)
        if manual_review_decision_code is None
        else np.asarray(manual_review_decision_code, dtype=np.int8)
    )
    if decision.shape != (frame_count,):
        raise ValueError("manual_review_decision_code must have one value per frame")
    if np.any((decision < 0) | (decision > 3)):
        raise ValueError("manual_review_decision_code must contain only 0, 1, 2, or 3")
    reviewed = (
        manual >= 3
        if manual_reviewed_automatic_grade3 is None
        else np.asarray(manual_reviewed_automatic_grade3, dtype=bool) | (manual >= 3)
    )
    reviewed = reviewed | (decision > 0)
    if reviewed.shape != (frame_count,):
        raise ValueError(
            "manual_reviewed_automatic_grade3 must have one value per frame"
        )
    translation_score = anomaly_score(global_translation_step_px, config)
    rotation_score = anomaly_score(global_rotation_step_deg, config)
    global_score = np.maximum(
        np.nan_to_num(translation_score), np.nan_to_num(rotation_score)
    ).astype(np.float32)

    mean_pcc = finite_frame_mean(pair_pcc)
    pcc_drop_score = anomaly_score(-mean_pcc, config)
    fb_p95 = finite_frame_percentile(pair_fb_error_px, 95)
    fb_score = anomaly_score(fb_p95, config)
    invalid_fraction = 1.0 - np.mean(np.asarray(pair_valid, dtype=bool), axis=(1, 2))
    invalid_score = anomaly_score(invalid_fraction, config)
    p3_distance_p95 = finite_frame_percentile(p3_curve_distance_px, 95)
    p3_distance_score = anomaly_score(p3_distance_p95, config)
    topology_score = np.asarray(p3_topology_risk, dtype=np.float32)
    quality_score = np.maximum.reduce(
        (
            np.nan_to_num(pcc_drop_score),
            np.nan_to_num(fb_score),
            np.nan_to_num(invalid_score),
            config.p3_distance_support_weight
            * np.nan_to_num(p3_distance_score),
            config.p3_topology_support_weight * topology_score,
        )
    ).astype(np.float32)

    common_rms, coherence = pair_common_motion_components(
        radial_local_residual, pair_valid, section_count
    )
    common_score = anomaly_score(common_rms, config)
    coherent_common_score = (np.nan_to_num(common_score) * coherence).astype(
        np.float32
    )
    simultaneous_fraction = simultaneous_deformation_fraction(raw_rsr)
    simultaneity_score = np.clip(
        (simultaneous_fraction - config.simultaneous_fraction_onset)
        / (
            config.simultaneous_fraction_full
            - config.simultaneous_fraction_onset
        ),
        0.0,
        1.0,
    ).astype(np.float32)
    synchronized_score = np.maximum(
        coherent_common_score, simultaneity_score
    ).astype(np.float32)

    relative_grade, evidence_count = grade_from_evidence(
        global_score,
        quality_score,
        synchronized_score,
        np.zeros(frame_count, dtype=np.int8),
        config,
    )
    supported = apply_absolute_and_temporal_support(
        relative_grade=relative_grade,
        global_translation_step_px=global_translation_step_px,
        global_rotation_step_deg=global_rotation_step_deg,
        mean_pcc=mean_pcc,
        fb_error_p95_px=fb_p95,
        invalid_fraction=invalid_fraction,
        synchronized_motion_score=synchronized_score,
        config=config,
    )
    automatic_grade = supported["automatic_artifact_grade"]
    # Keep automatic detection separate from the action grade. Until thresholds
    # are frozen, an unreviewed automatic grade-3 candidate remains analysis
    # grade 2 and cannot blank RSR. A manual grade is a minimum action grade.
    automatic_action = (
        automatic_grade
        if config.allow_unreviewed_automatic_grade3_action
        else np.minimum(automatic_grade, 2)
    )
    grade = np.maximum(automatic_action, manual).astype(np.int8)
    # Posthoc decisions are actions, not three labels for the same grade-2 state.
    # Existing independent/manual minimum grades still take precedence on overlap.
    confirmed_grade3 = decision == 1
    keep_grade2 = decision == 2
    not_artifact = decision == 3
    grade[confirmed_grade3] = np.maximum(manual[confirmed_grade3], 3)
    grade[keep_grade2] = np.maximum(manual[keep_grade2], 2)
    grade[not_artifact] = manual[not_artifact]
    qc_rsr = np.asarray(raw_rsr, dtype=np.float32).copy()
    qc_rsr[grade >= 3] = np.nan
    return {
        "global_translation_step_px": np.asarray(
            global_translation_step_px, dtype=np.float32
        ),
        "global_rotation_step_deg": np.asarray(
            global_rotation_step_deg, dtype=np.float32
        ),
        "translation_anomaly_score": translation_score,
        "rotation_anomaly_score": rotation_score,
        "global_motion_anomaly_score": global_score,
        "pair_mean_pcc": mean_pcc,
        "pcc_drop_anomaly_score": pcc_drop_score,
        "pair_fb_error_p95_px": fb_p95,
        "fb_error_anomaly_score": fb_score,
        "pair_invalid_fraction": invalid_fraction.astype(np.float32),
        "invalid_fraction_anomaly_score": invalid_score,
        "p3_curve_distance_p95_px": p3_distance_p95,
        "p3_distance_anomaly_score": p3_distance_score,
        "p3_topology_risk": np.asarray(p3_topology_risk, dtype=bool),
        "tracking_quality_anomaly_score": quality_score,
        "pair_common_motion_rms_px_per_frame": common_rms,
        "pair_common_motion_coherence": coherence,
        "pair_common_motion_anomaly_score": common_score,
        "simultaneous_deformation_fraction": simultaneous_fraction,
        "simultaneous_deformation_score": simultaneity_score,
        "synchronized_motion_anomaly_score": synchronized_score,
        "automatic_evidence_group_count": evidence_count,
        "relative_only_artifact_candidate_grade": relative_grade,
        **supported,
        "automatic_artifact_grade": automatic_grade,
        "manual_minimum_grade": manual,
        "manual_review_decision_code": decision,
        "artifact_grade": grade,
        "automatic_grade3_reviewed": (automatic_grade >= 3) & reviewed,
        "automatic_grade3_pending_manual_review": (automatic_grade >= 3) & ~reviewed,
        "raw_radial_strain_rate_s": np.asarray(raw_rsr, dtype=np.float32),
        "qc_radial_strain_rate_grade3_blank_s": qc_rsr,
        "grade3_blank_frame": grade >= 3,
    }
