from __future__ import annotations

import numpy as np
import pytest

from peristalsis_pipeline.radial_pair_geometry import (
    build_radial_pair_candidates,
    outward_unit_normals,
    radial_strain_rate_from_positions,
)


def test_outward_normals_point_away_from_cavity() -> None:
    anterior = np.asarray([[0, 10], [10, 10], [20, 10]], dtype=np.float32)
    posterior = np.asarray([[0, 30], [10, 30], [20, 30]], dtype=np.float32)

    anterior_outward = outward_unit_normals(anterior, posterior)
    posterior_outward = outward_unit_normals(posterior, anterior)

    assert np.all(anterior_outward[:, 1] < 0)
    assert np.all(posterior_outward[:, 1] > 0)
    np.testing.assert_allclose(np.linalg.norm(anterior_outward, axis=1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(posterior_outward, axis=1), 1.0)


def test_candidate_pair_distance_matches_requested_offset() -> None:
    anterior = np.asarray([[0, 10], [10, 9], [20, 10]], dtype=np.float32)
    posterior = np.asarray([[0, 30], [10, 31], [20, 30]], dtype=np.float32)

    result = build_radial_pair_candidates(anterior, posterior, offset_px=21.0)

    for wall in ("anterior", "posterior"):
        distance = np.linalg.norm(
            result[f"{wall}_outer"] - result[f"{wall}_inner"], axis=1
        )
        np.testing.assert_allclose(distance, 21.0, atol=1e-5)


@pytest.mark.parametrize("offset", [0.0, -1.0, np.nan])
def test_candidate_offset_must_be_positive_and_finite(offset: float) -> None:
    anterior = np.asarray([[0, 10], [10, 10]], dtype=np.float32)
    posterior = np.asarray([[0, 30], [10, 30]], dtype=np.float32)
    with pytest.raises(ValueError):
        build_radial_pair_candidates(anterior, posterior, offset)


def test_common_translation_has_zero_radial_strain_rate() -> None:
    inner = np.asarray([[0, 0], [10, 0]], dtype=np.float32)
    outer = np.asarray([[0, 20], [10, 20]], dtype=np.float32)
    shift = np.asarray([3, -2], dtype=np.float32)
    result = radial_strain_rate_from_positions(
        inner, outer, inner + shift, outer + shift, fps=25.0
    )
    np.testing.assert_allclose(result, 0.0, atol=1e-6)


def test_pair_shortening_has_negative_radial_strain_rate() -> None:
    inner = np.asarray([[0, 0]], dtype=np.float32)
    outer = np.asarray([[0, 20]], dtype=np.float32)
    target_outer = np.asarray([[0, 19.8]], dtype=np.float32)
    result = radial_strain_rate_from_positions(
        inner, outer, inner, target_outer, fps=25.0
    )
    np.testing.assert_allclose(result, -0.25, atol=1e-5)


def test_rigid_rotation_has_zero_radial_strain_rate() -> None:
    inner = np.asarray([[10, 10]], dtype=np.float32)
    outer = np.asarray([[10, 30]], dtype=np.float32)
    rotation = np.asarray([[0, -1], [1, 0]], dtype=np.float32)
    result = radial_strain_rate_from_positions(
        inner,
        outer,
        inner @ rotation.T,
        outer @ rotation.T,
        fps=25.0,
    )
    np.testing.assert_allclose(result, 0.0, atol=1e-6)
