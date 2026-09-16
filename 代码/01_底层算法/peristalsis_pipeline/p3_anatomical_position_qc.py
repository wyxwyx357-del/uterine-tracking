"""Conservative point-level QC for P3 anatomical wall localization.

Automatic detections are review candidates only.  Suggested coordinates are
never applied to tracking or RSR by this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class P3AnatomicalPositionQCConfig:
    """Project-development gates; not Huang-paper or clinical thresholds."""

    local_motion_residual_px: float = 1.0
    spatial_curve_change_px: float = 1.0
    branch_normal_disagreement_px: float = 2.0

    def __post_init__(self) -> None:
        for name, value in (
            ("local_motion_residual_px", self.local_motion_residual_px),
            ("spatial_curve_change_px", self.spatial_curve_change_px),
            ("branch_normal_disagreement_px", self.branch_normal_disagreement_px),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


def _spatial_prediction(wall_tracks: np.ndarray) -> np.ndarray:
    prediction = np.full_like(wall_tracks, np.nan, dtype=np.float32)
    if wall_tracks.shape[2] > 2:
        prediction[:, :, 1:-1] = 0.5 * (
            wall_tracks[:, :, :-2] + wall_tracks[:, :, 2:]
        )
    return prediction


def _temporal_prediction(wall_tracks: np.ndarray) -> np.ndarray:
    prediction = np.full_like(wall_tracks, np.nan, dtype=np.float32)
    if wall_tracks.shape[0] > 2:
        prediction[1:-1] = 0.5 * (wall_tracks[:-2] + wall_tracks[2:])
    return prediction


def build_p3_anatomical_position_qc(
    wall_tracks: np.ndarray,
    branch_normal_disagreement_px: np.ndarray,
    topology_risk_frame: np.ndarray,
    recovered_display_only: np.ndarray,
    config: P3AnatomicalPositionQCConfig | None = None,
) -> dict[str, np.ndarray]:
    """Flag multi-evidence local P3 outliers and propose review-only positions.

    ``wall_tracks`` and point-level inputs are ordered ``(frame, side, section)``.
    A candidate requires an absolute local-motion residual, an absolute change
    in spatial curve smoothness, and either branch-normal disagreement or a
    topology-risk frame.  Recovery status is carried as evidence but cannot
    create a candidate by itself.
    """

    cfg = config or P3AnatomicalPositionQCConfig()
    tracks = np.asarray(wall_tracks, dtype=np.float32)
    branch = np.asarray(branch_normal_disagreement_px, dtype=np.float32)
    topology = np.asarray(topology_risk_frame, dtype=bool)
    recovered = np.asarray(recovered_display_only, dtype=bool)
    if tracks.ndim != 4 or tracks.shape[-1] != 2:
        raise ValueError("wall_tracks must have shape (frames, sides, sections, 2)")
    expected = tracks.shape[:3]
    if branch.shape != expected or recovered.shape != expected:
        raise ValueError("point-level P3 quality arrays must match wall tracks")
    if topology.shape != (tracks.shape[0],):
        raise ValueError("topology_risk_frame must have one value per frame")

    step = np.full_like(tracks, np.nan, dtype=np.float32)
    step[1:] = tracks[1:] - tracks[:-1]
    common_step = np.zeros((tracks.shape[0], 1, 1, 2), dtype=np.float32)
    common_step[1:] = np.nanmedian(step[1:], axis=(1, 2), keepdims=True)
    local_motion_residual = np.linalg.norm(step - common_step, axis=-1)

    spatial_prediction = _spatial_prediction(tracks)
    spatial_residual = np.linalg.norm(tracks - spatial_prediction, axis=-1)
    spatial_baseline = np.full((1, *tracks.shape[1:3]), np.nan, dtype=np.float32)
    if tracks.shape[2] > 2:
        spatial_baseline[:, :, 1:-1] = np.nanmedian(
            spatial_residual[:, :, 1:-1], axis=0, keepdims=True
        )
    spatial_curve_change = np.abs(spatial_residual - spatial_baseline)

    motion_support = local_motion_residual >= cfg.local_motion_residual_px
    spatial_support = spatial_curve_change >= cfg.spatial_curve_change_px
    branch_support = branch >= cfg.branch_normal_disagreement_px
    quality_support = branch_support | topology[:, None, None]
    automatic_candidate = motion_support & spatial_support & quality_support
    automatic_candidate &= np.all(np.isfinite(tracks), axis=-1)

    isolated = automatic_candidate.copy()
    isolated[:, :, 1:] &= ~automatic_candidate[:, :, :-1]
    isolated[:, :, :-1] &= ~automatic_candidate[:, :, 1:]
    temporal_prediction = _temporal_prediction(tracks)
    proposed = np.full_like(tracks, np.nan, dtype=np.float32)
    proposal_available = isolated & np.all(np.isfinite(spatial_prediction), axis=-1)
    proposal_available &= np.all(np.isfinite(temporal_prediction), axis=-1)
    proposed[proposal_available] = 0.5 * (
        spatial_prediction[proposal_available] + temporal_prediction[proposal_available]
    )

    return {
        "p3_local_motion_residual_px": local_motion_residual.astype(np.float32),
        "p3_spatial_curve_change_px": spatial_curve_change.astype(np.float32),
        "p3_branch_normal_disagreement_px": branch,
        "p3_recovered_display_only": recovered,
        "p3_topology_risk_frame": topology,
        "motion_support": motion_support,
        "spatial_support": spatial_support,
        "p3_quality_support": quality_support,
        "automatic_anatomical_position_candidate": automatic_candidate,
        "isolated_automatic_candidate": isolated,
        "suggested_position_available": proposal_available,
        "suggested_wall_position_review_only": proposed,
        "automatic_suggestion_applied_to_tracking": np.asarray(False),
        "automatic_suggestion_applied_to_rsr": np.asarray(False),
        "automatic_endpoint_section_detection_available": np.asarray(False),
    }
