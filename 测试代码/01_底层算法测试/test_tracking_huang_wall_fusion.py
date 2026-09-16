from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "代码" / "01_底层算法"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from peristalsis_pipeline import tracking_huang_wall_fusion as fusion


class HuangWallFusionTest(unittest.TestCase):
    def setUp(self):
        self.config = fusion.HuangWallFusionConfig()

    def test_curve_distance_does_not_penalize_along_curve_slide(self):
        curve = np.asarray([[10, 20], [30, 20], [50, 20]], dtype=np.float32)
        query = np.asarray([[20, 20], [40, 23]], dtype=np.float32)
        distance = fusion.point_to_polyline_distance(query, curve)
        self.assertAlmostEqual(float(distance[0]), 0.0, places=5)
        self.assertAlmostEqual(float(distance[1]), 3.0, places=5)

    def test_pairwise_translation_is_removed_and_source_resets_to_p3(self):
        rng = np.random.default_rng(19)
        base = rng.integers(0, 255, size=(96, 96), dtype=np.uint8)
        frames = [
            cv2.warpAffine(
                base,
                np.asarray([[1, 0, shift], [0, 1, 0]], dtype=np.float32),
                (96, 96),
            )
            for shift in (0.0, 1.0, 2.0)
        ]
        initial = np.asarray(
            [[28, 34], [62, 34], [28, 58], [62, 58]], dtype=np.float32
        )
        p3 = np.stack([initial + [shift, 0] for shift in (0, 1, 2)])
        transforms = np.repeat(
            np.asarray([[[1, 0, 0], [0, 1, 0]]], dtype=np.float32),
            3,
            axis=0,
        )
        transforms[1:, 0, 2] = 1.0
        result = fusion.measure_pairwise_wall_motion(
            frames, p3, transforms, anterior_count=2, config=self.config
        )
        self.assertTrue(np.allclose(result["wall_pair_source"][2], p3[1]))
        valid = result["wall_measurement_valid"][1:]
        residual = result["wall_local_residual"][1:][valid]
        raw = result["wall_raw_lk_displacement"][1:][valid]
        self.assertTrue(np.all(valid))
        self.assertLess(float(np.max(np.linalg.norm(residual, axis=1))), 0.08)
        self.assertTrue(np.allclose(raw[:, 0], 1.0, atol=0.08))
        self.assertTrue(np.allclose(raw[:, 1], 0.0, atol=0.08))

    def test_p3_displacement_is_not_used_as_local_motion(self):
        rng = np.random.default_rng(23)
        image = rng.integers(0, 255, size=(96, 96), dtype=np.uint8)
        frames = [image, image.copy()]
        initial = np.asarray(
            [[28, 34], [62, 34], [28, 58], [62, 58]], dtype=np.float32
        )
        p3 = np.stack((initial, initial + [2.0, 0.0])).astype(np.float32)
        transforms = np.repeat(
            np.asarray([[[1, 0, 0], [0, 1, 0]]], dtype=np.float32),
            2,
            axis=0,
        )
        result = fusion.measure_pairwise_wall_motion(
            frames, p3, transforms, anterior_count=2, config=self.config
        )
        valid = result["wall_measurement_valid"][1]
        local = result["wall_local_residual"][1, valid]
        self.assertTrue(np.all(valid))
        self.assertLess(float(np.max(np.linalg.norm(local, axis=1))), 0.08)

    def test_pairwise_rotation_is_removed(self):
        rng = np.random.default_rng(29)
        base = rng.integers(0, 255, size=(120, 120), dtype=np.uint8)
        matrix = cv2.getRotationMatrix2D((60, 60), 2.0, 1.0).astype(np.float32)
        current = cv2.warpAffine(base, matrix, (120, 120))
        initial = np.asarray(
            [[35, 43], [82, 43], [35, 77], [82, 77]], dtype=np.float32
        )
        target = fusion.apply_affine(initial, matrix)
        p3 = np.stack((initial, target))
        transforms = np.stack(
            (
                np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
                matrix,
            )
        )
        result = fusion.measure_pairwise_wall_motion(
            [base, current], p3, transforms, anterior_count=2, config=self.config
        )
        valid = result["wall_measurement_valid"][1]
        residual = result["wall_local_residual"][1, valid]
        self.assertTrue(np.all(valid))
        self.assertLess(float(np.max(np.linalg.norm(residual, axis=1))), 0.15)

    def test_midline_pairwise_global_translation_is_not_accumulated(self):
        rng = np.random.default_rng(31)
        base = rng.integers(0, 255, size=(96, 96), dtype=np.uint8)
        frames = [
            cv2.warpAffine(
                base,
                np.asarray([[1, 0, shift], [0, 1, 0]], dtype=np.float32),
                (96, 96),
            )
            for shift in (0.0, 1.0, 2.0)
        ]
        initial = np.column_stack(
            (np.linspace(25, 70, 10), np.linspace(42, 50, 10))
        ).astype(np.float32)
        p3 = np.stack([initial + [shift, 0] for shift in (0, 1, 2)])
        result = fusion.estimate_pairwise_global_motion(frames, p3, self.config)
        self.assertTrue(
            np.allclose(result["pairwise_global_transform"][1:, 0, 2], 1.0, atol=0.08)
        )
        self.assertTrue(
            np.allclose(result["pairwise_global_transform"][1:, 1, 2], 0.0, atol=0.08)
        )

    def test_symmetric_inward_motion_is_cavity_narrowing(self):
        reference = np.asarray(
            [
                [[0, 0], [0, 10], [20, 0], [20, 10]],
                [[0, 0], [0, 10], [20, 0], [20, 10]],
            ],
            dtype=np.float32,
        )
        residual = np.full_like(reference, np.nan)
        residual[1] = np.asarray([[1, 0], [1, 0], [-1, 0], [-1, 0]])
        valid = np.asarray(
            [[False, False, False, False], [True, True, True, True]]
        )
        topology = np.asarray([[False, False], [True, True]])
        result = fusion.deformation_from_local_wall_motion(
            reference,
            residual,
            valid,
            topology,
            anterior_count=2,
            fps=1.0,
        )
        self.assertTrue(
            np.all(result["common_inward_contraction_velocity_px_s"][1] > 0)
        )
        self.assertTrue(np.allclose(result["opposing_wall_velocity_px_s"][1], 0))
        self.assertTrue(np.all(result["cavity_width_strain_rate_s"][1] < 0))

    def test_same_direction_wall_motion_is_not_symmetric_contraction(self):
        reference = np.asarray(
            [
                [[0, 0], [0, 10], [20, 0], [20, 10]],
                [[0, 0], [0, 10], [20, 0], [20, 10]],
            ],
            dtype=np.float32,
        )
        residual = np.full_like(reference, np.nan)
        residual[1] = np.asarray([[1, 0], [1, 0], [1, 0], [1, 0]])
        valid = np.asarray(
            [[False, False, False, False], [True, True, True, True]]
        )
        topology = np.asarray([[False, False], [True, True]])
        result = fusion.deformation_from_local_wall_motion(
            reference,
            residual,
            valid,
            topology,
            anterior_count=2,
            fps=1.0,
        )
        self.assertTrue(
            np.allclose(result["common_inward_contraction_velocity_px_s"][1], 0)
        )
        self.assertTrue(np.allclose(result["cavity_width_strain_rate_s"][1], 0))
        self.assertTrue(np.all(result["opposing_wall_velocity_px_s"][1] > 0))


if __name__ == "__main__":
    unittest.main()
