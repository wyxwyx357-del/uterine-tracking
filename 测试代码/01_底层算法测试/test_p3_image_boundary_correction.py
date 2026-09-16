from __future__ import annotations

import cv2
import numpy as np

from peristalsis_pipeline.p3_image_boundary_correction import (
    P3ImageBoundaryConfig,
    apply_confirmed_boundary_overrides,
    apply_tracking_quality_gate,
    build_automatic_boundary_correction,
    extract_wall_boundary_evidence,
    wall_outward_normals,
)


def parallel_tracks(frames: int = 5, sections: int = 5) -> np.ndarray:
    tracks = np.zeros((frames, 2, sections, 2), dtype=np.float32)
    tracks[:, :, :, 0] = np.arange(sections, dtype=np.float32) * 12 + 20
    tracks[:, 0, :, 1] = 30
    tracks[:, 1, :, 1] = 70
    return tracks


def test_wall_normals_point_away_from_cavity() -> None:
    normals = wall_outward_normals(parallel_tracks())
    assert np.all(normals[:, 0, :, 1] < 0)
    assert np.all(normals[:, 1, :, 1] > 0)


def test_boundary_edge_offset_is_measured_along_outward_normal() -> None:
    tracks = parallel_tracks()
    frames = []
    for _ in range(len(tracks)):
        image = np.full((100, 100), 30, dtype=np.uint8)
        image[27:34] = 180
        image[67:75] = 180
        image = cv2.GaussianBlur(image, (3, 3), 0)
        frames.append(image)
    result = extract_wall_boundary_evidence(frames, tracks)
    np.testing.assert_allclose(result["boundary_edge_offset_px"][:, 0], 4, atol=1)
    np.testing.assert_allclose(result["boundary_edge_offset_px"][:, 1], 5, atol=1)
    assert np.all(result["boundary_edge_supported"])


def test_flat_image_has_no_supported_boundary() -> None:
    tracks = parallel_tracks()
    frames = [np.full((100, 100), 80, dtype=np.uint8) for _ in range(len(tracks))]
    result = extract_wall_boundary_evidence(frames, tracks)
    assert not np.any(result["boundary_edge_supported"])


def correction_evidence(tracks: np.ndarray) -> dict[str, np.ndarray]:
    shape = tracks.shape[:3]
    normals = wall_outward_normals(tracks)
    return {
        "boundary_edge_offset_px": np.zeros(shape, dtype=np.float32),
        "boundary_edge_drop_gray": np.full(shape, 20.0, dtype=np.float32),
        "boundary_edge_prominence": np.full(shape, 2.0, dtype=np.float32),
        "outward_normal": normals,
    }


def test_persistent_endpoint_cluster_is_moved_outward() -> None:
    tracks = parallel_tracks(frames=120, sections=6)
    evidence = correction_evidence(tracks)
    evidence["boundary_edge_offset_px"][30:70, 0, -3:] = 4.0
    result = build_automatic_boundary_correction(
        tracks, evidence, np.zeros(len(tracks), dtype=np.int8), fps=30.0
    )
    assert np.all(result["automatic_boundary_correction_mask"][30:70, 0, -3:])
    assert not np.any(result["automatic_boundary_correction_mask"][:, 1])
    np.testing.assert_allclose(
        result["p3_wall_tracks_image_corrected"][40, 0, -3:, 1],
        [27.0, 26.0, 26.0],
        atol=0.1,
    )
    np.testing.assert_allclose(
        result["p3_wall_tracks_image_corrected"][40, 0, -4, 1], 29.0, atol=0.1
    )


def test_persistent_internal_cluster_is_corrected_and_probe_shake_is_not() -> None:
    tracks = parallel_tracks(frames=120, sections=9)
    evidence = correction_evidence(tracks)
    evidence["boundary_edge_offset_px"][30:70, 0, 3:6] = 4.0
    grade = np.zeros(len(tracks), dtype=np.int8)
    result = build_automatic_boundary_correction(tracks, evidence, grade, fps=30.0)
    assert np.any(result["image_boundary_review_candidate"])
    assert np.any(result["automatic_boundary_correction_mask"])
    assert not np.any(result["high_specificity_endpoint_correction_mask"])

    evidence["boundary_edge_offset_px"][:] = 0
    evidence["boundary_edge_offset_px"][30:70, 0, -3:] = 4.0
    grade[30:70] = 3
    result = build_automatic_boundary_correction(tracks, evidence, grade, fps=30.0)
    assert not np.any(result["automatic_boundary_correction_mask"])


def test_unreviewed_automatic_grade3_frames_are_excluded_from_correction() -> None:
    tracks = parallel_tracks(frames=120, sections=6)
    evidence = correction_evidence(tracks)
    evidence["boundary_edge_offset_px"][30:70, 0, -3:] = 4.0
    excluded = np.zeros(len(tracks), dtype=bool)
    excluded[40:50] = True
    result = build_automatic_boundary_correction(
        tracks,
        evidence,
        np.zeros(len(tracks), dtype=np.int8),
        fps=30.0,
        additional_excluded_frames=excluded,
    )
    assert np.any(
        result[
            "development_review_candidate_without_automatic_grade3_exclusion"
        ][40:50]
    )
    assert not np.any(result["automatic_boundary_correction_mask"][40:50])
    assert np.all(result["automatic_boundary_correction_excluded_frame"][40:50])
    assert np.max(
        np.abs(np.diff(result["automatic_boundary_correction_px"], axis=0))
    ) <= P3ImageBoundaryConfig().maximum_frame_to_frame_correction_change_px + 1e-6
    assert not bool(result["formal_automatic_boundary_correction_released"])


def test_manual_override_moves_confirmed_group_but_not_probe_shake() -> None:
    tracks = parallel_tracks(frames=20, sections=6)
    evidence = correction_evidence(tracks)
    evidence["boundary_edge_offset_px"][5, 1, 2:5] = [3.0, 5.0, 4.0]
    evidence["boundary_edge_offset_px"][8, 1, 2:5] = [3.0, 5.0, 4.0]
    grade = np.zeros(len(tracks), dtype=np.int8)
    grade[8] = 3
    automatic = build_automatic_boundary_correction(tracks, evidence, grade, fps=30.0)
    manual = np.zeros(tracks.shape[:3], dtype=bool)
    manual[5, 1, 2:5] = True
    manual[8, 1, 2:5] = True
    result = apply_confirmed_boundary_overrides(automatic, evidence, manual, grade)
    np.testing.assert_allclose(
        result["manual_confirmed_boundary_correction_px"][5, 1, 2:5], 4.0
    )
    assert np.all(result["manual_confirmed_boundary_correction_mask"][5, 1, 2:5])
    assert not np.any(result["manual_confirmed_boundary_correction_mask"][8])


def test_persistent_single_point_needs_nearby_temporal_support() -> None:
    tracks = parallel_tracks(frames=60, sections=6)
    evidence = correction_evidence(tracks)
    evidence["boundary_edge_offset_px"][10:30, 0, 2] = 4.0
    evidence["boundary_edge_offset_px"][5:10, 0, 3] = 4.0
    result = build_automatic_boundary_correction(
        tracks,
        evidence,
        np.zeros(len(tracks), dtype=np.int8),
        fps=20.0,
    )
    assert result["temporally_supported_isolated_candidate_mask"][12, 0, 2]
    assert result["automatic_boundary_correction_candidate_mask"][12, 0, 2]
    assert not result["automatic_boundary_correction_candidate_mask"][25, 0, 2]


def tracking_quality(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    return {
        "radial_pair_pcc": np.full(shape, 0.99, dtype=np.float32),
        "radial_pair_fb_error_px": np.full(shape, 0.01, dtype=np.float32),
        "radial_pair_valid": np.ones(shape, dtype=bool),
    }


def test_tracking_quality_gate_rejects_worse_internal_correction() -> None:
    tracks = parallel_tracks(frames=60, sections=7)
    evidence = correction_evidence(tracks)
    evidence["boundary_edge_offset_px"][10:30, 0, 2:4] = 4.0
    automatic = build_automatic_boundary_correction(
        tracks, evidence, np.zeros(len(tracks), dtype=np.int8), fps=20.0
    )
    original = tracking_quality(tracks.shape[:3])
    candidate = tracking_quality(tracks.shape[:3])
    candidate["radial_pair_pcc"][10:30, 0, 2:4] -= 0.001
    candidate["radial_pair_fb_error_px"][10:30, 0, 2:4] += 0.001
    result = apply_tracking_quality_gate(
        automatic, evidence, candidate, original, P3ImageBoundaryConfig()
    )
    assert not np.any(result["automatic_boundary_correction_mask"])
    assert not np.any(
        result["automatic_boundary_correction_px"][
            result["tracking_quality_rejected_boundary_correction_candidate_mask"]
        ]
    )
    assert np.all(
        result["tracking_quality_rejected_boundary_correction_candidate_mask"][
            10:30, 0, 2:4
        ]
    )


def test_tracking_quality_gate_accepts_improved_internal_correction() -> None:
    tracks = parallel_tracks(frames=60, sections=7)
    evidence = correction_evidence(tracks)
    evidence["boundary_edge_offset_px"][10:30, 0, 2:4] = 4.0
    automatic = build_automatic_boundary_correction(
        tracks, evidence, np.zeros(len(tracks), dtype=np.int8), fps=20.0
    )
    original = tracking_quality(tracks.shape[:3])
    candidate = tracking_quality(tracks.shape[:3])
    candidate["radial_pair_pcc"][10:30, 0, 2:4] += 0.001
    result = apply_tracking_quality_gate(
        automatic, evidence, candidate, original, P3ImageBoundaryConfig()
    )
    assert np.any(result["automatic_boundary_correction_mask"])
    assert np.all(
        result["tracking_quality_accepted_boundary_correction_candidate_mask"][
            10:30, 0, 2:4
        ]
    )
