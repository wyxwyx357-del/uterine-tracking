from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "代码" / "01_底层算法"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from peristalsis_pipeline import tracking_artifact_qc_v2 as artifact


class ArtifactQCV2Test(unittest.TestCase):
    def test_pair_common_motion_excludes_invalid_pairs(self) -> None:
        sections = 2
        residual = np.zeros((1, 4 * sections, 2), dtype=np.float32)
        pair_vectors = (
            np.asarray([-100.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 0.0], dtype=np.float32),
        )
        for side, (inner_rail, outer_rail) in enumerate(((0, 1), (2, 3))):
            for section in range(sections):
                vector = pair_vectors[side * sections + section]
                residual[0, inner_rail * sections + section] = vector
                residual[0, outer_rail * sections + section] = vector

        all_valid = np.ones((1, 2, sections), dtype=bool)
        common_rms, coherence = artifact.pair_common_motion_components(
            residual, all_valid, sections
        )
        legacy_vectors = np.asarray(pair_vectors, dtype=np.float32)
        legacy_magnitudes = np.linalg.norm(legacy_vectors, axis=1)
        expected_rms = np.sqrt(np.mean(legacy_magnitudes**2))
        expected_coherence = np.linalg.norm(np.mean(legacy_vectors, axis=0)) / np.mean(
            legacy_magnitudes
        )
        self.assertAlmostEqual(float(common_rms[0]), float(expected_rms), places=6)
        self.assertAlmostEqual(
            float(coherence[0]), float(expected_coherence), places=6
        )

        pair_valid = all_valid.copy()
        pair_valid[0, 0, 0] = False
        common_rms, coherence = artifact.pair_common_motion_components(
            residual, pair_valid, sections
        )
        self.assertAlmostEqual(float(common_rms[0]), 1.0, places=6)
        self.assertAlmostEqual(float(coherence[0]), 1.0, places=6)

    def test_one_evidence_group_cannot_create_grade_three(self) -> None:
        motion = np.asarray([0.0, 1.0], dtype=np.float32)
        quality = np.zeros(2, dtype=np.float32)
        sync = np.zeros(2, dtype=np.float32)
        grade, count = artifact.grade_from_evidence(
            motion,
            quality,
            sync,
            np.zeros(2, dtype=np.int8),
            artifact.ArtifactQCConfig(),
        )
        self.assertEqual(int(count[1]), 1)
        self.assertEqual(int(grade[1]), 1)

    def test_motion_plus_quality_can_create_grade_three(self) -> None:
        grade, count = artifact.grade_from_evidence(
            np.asarray([0.8], dtype=np.float32),
            np.asarray([0.6], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.zeros(1, dtype=np.int8),
            artifact.ArtifactQCConfig(),
        )
        self.assertEqual(int(count[0]), 2)
        self.assertEqual(int(grade[0]), 3)

    def test_manual_grade_is_minimum_not_signal_rewrite(self) -> None:
        grade, _ = artifact.grade_from_evidence(
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.asarray([2, 3], dtype=np.int8),
            artifact.ArtifactQCConfig(),
        )
        np.testing.assert_array_equal(grade, [2, 3])

    def test_grade_three_blanks_only_qc_copy_and_preserves_raw(self) -> None:
        frames, sections = 40, 2
        raw = np.ones((frames, 2, sections), dtype=np.float32)
        manual = np.zeros(frames, dtype=np.int8); manual[20] = 3; manual[10] = 2
        residual = np.zeros((frames, 4 * sections, 2), dtype=np.float32)
        result = artifact.build_artifact_qc(
            global_translation_step_px=np.zeros(frames),
            global_rotation_step_deg=np.zeros(frames),
            pair_pcc=np.ones((frames, 2, sections), dtype=np.float32),
            pair_fb_error_px=np.zeros((frames, 2, sections), dtype=np.float32),
            pair_valid=np.ones((frames, 2, sections), dtype=bool),
            p3_curve_distance_px=np.zeros((frames, 2, sections), dtype=np.float32),
            p3_topology_risk=np.zeros(frames, dtype=bool),
            radial_local_residual=residual,
            raw_rsr=raw,
            section_count=sections,
            manual_minimum_grade=manual,
        )
        np.testing.assert_array_equal(result["raw_radial_strain_rate_s"], raw)
        np.testing.assert_array_equal(
            result["automatic_artifact_grade"], np.zeros(frames, dtype=np.int8)
        )
        self.assertEqual(int(result["artifact_grade"][10]), 2)
        self.assertEqual(int(result["artifact_grade"][20]), 3)
        self.assertTrue(np.all(np.isfinite(result["qc_radial_strain_rate_grade3_blank_s"][10])))
        self.assertTrue(np.all(np.isnan(result["qc_radial_strain_rate_grade3_blank_s"][20])))

    def test_high_pcc_alone_does_not_claim_no_jitter(self) -> None:
        frames, sections = 30, 2
        translation = np.zeros(frames, dtype=np.float32); translation[15] = 20.0
        fb = np.zeros((frames, 2, sections), dtype=np.float32); fb[15] = 10.0
        result = artifact.build_artifact_qc(
            global_translation_step_px=translation,
            global_rotation_step_deg=np.zeros(frames),
            pair_pcc=np.full((frames, 2, sections), 0.99, dtype=np.float32),
            pair_fb_error_px=fb,
            pair_valid=np.ones((frames, 2, sections), dtype=bool),
            p3_curve_distance_px=np.zeros((frames, 2, sections), dtype=np.float32),
            p3_topology_risk=np.zeros(frames, dtype=bool),
            radial_local_residual=np.zeros((frames, 4 * sections, 2), dtype=np.float32),
            raw_rsr=np.zeros((frames, 2, sections), dtype=np.float32),
            section_count=sections,
        )
        self.assertEqual(int(result["relative_only_artifact_candidate_grade"][15]), 3)
        self.assertEqual(int(result["automatic_artifact_grade"][15]), 3)
        self.assertEqual(int(result["artifact_grade"][15]), 2)
        self.assertTrue(result["automatic_grade3_pending_manual_review"][15])
        self.assertTrue(
            np.all(np.isfinite(result["qc_radial_strain_rate_grade3_blank_s"][15]))
        )

    def test_reviewed_negative_decision_clears_pending_without_blanking_rsr(self) -> None:
        frames, sections = 30, 2
        translation = np.zeros(frames, dtype=np.float32); translation[15] = 20.0
        fb = np.zeros((frames, 2, sections), dtype=np.float32); fb[15] = 10.0
        reviewed = np.zeros(frames, dtype=bool); reviewed[15] = True
        result = artifact.build_artifact_qc(
            global_translation_step_px=translation,
            global_rotation_step_deg=np.zeros(frames),
            pair_pcc=np.full((frames, 2, sections), 0.99, dtype=np.float32),
            pair_fb_error_px=fb,
            pair_valid=np.ones((frames, 2, sections), dtype=bool),
            p3_curve_distance_px=np.zeros((frames, 2, sections), dtype=np.float32),
            p3_topology_risk=np.zeros(frames, dtype=bool),
            radial_local_residual=np.zeros((frames, 4 * sections, 2), dtype=np.float32),
            raw_rsr=np.zeros((frames, 2, sections), dtype=np.float32),
            section_count=sections,
            manual_reviewed_automatic_grade3=reviewed,
        )
        self.assertTrue(result["automatic_grade3_reviewed"][15])
        self.assertFalse(result["automatic_grade3_pending_manual_review"][15])
        self.assertEqual(int(result["artifact_grade"][15]), 2)
        self.assertTrue(
            np.all(np.isfinite(result["qc_radial_strain_rate_grade3_blank_s"][15]))
        )

    def test_p3_topology_is_auxiliary_and_cannot_grade_a_frame_alone(self) -> None:
        frames, sections = 30, 2
        topology = np.zeros(frames, dtype=bool); topology[15] = True
        result = artifact.build_artifact_qc(
            global_translation_step_px=np.zeros(frames),
            global_rotation_step_deg=np.zeros(frames),
            pair_pcc=np.full((frames, 2, sections), 0.99, dtype=np.float32),
            pair_fb_error_px=np.zeros((frames, 2, sections), dtype=np.float32),
            pair_valid=np.ones((frames, 2, sections), dtype=bool),
            p3_curve_distance_px=np.zeros((frames, 2, sections), dtype=np.float32),
            p3_topology_risk=topology,
            radial_local_residual=np.zeros((frames, 4 * sections, 2), dtype=np.float32),
            raw_rsr=np.zeros((frames, 2, sections), dtype=np.float32),
            section_count=sections,
        )
        self.assertEqual(int(result["artifact_grade"][15]), 0)

    def test_stable_case_tiny_relative_outlier_cannot_remain_grade_three(self) -> None:
        frames, sections = 30, 2
        translation = np.zeros(frames, dtype=np.float32); translation[15] = 0.25
        rotation = np.zeros(frames, dtype=np.float32); rotation[15] = 0.08
        pcc = np.full((frames, 2, sections), 0.995, dtype=np.float32); pcc[15] = 0.99
        fb = np.full((frames, 2, sections), 0.004, dtype=np.float32); fb[15] = 0.02
        raw = np.zeros((frames, 2, sections), dtype=np.float32); raw[15] = 1.0
        result = artifact.build_artifact_qc(
            global_translation_step_px=translation,
            global_rotation_step_deg=rotation,
            pair_pcc=pcc,
            pair_fb_error_px=fb,
            pair_valid=np.ones((frames, 2, sections), dtype=bool),
            p3_curve_distance_px=np.zeros((frames, 2, sections), dtype=np.float32),
            p3_topology_risk=np.zeros(frames, dtype=bool),
            radial_local_residual=np.zeros((frames, 4 * sections, 2), dtype=np.float32),
            raw_rsr=raw,
            section_count=sections,
        )
        self.assertEqual(int(result["relative_only_artifact_candidate_grade"][15]), 3)
        self.assertFalse(result["absolute_motion_support"][15])
        self.assertFalse(result["absolute_tracking_quality_support"][15])
        self.assertEqual(int(result["automatic_artifact_grade"][15]), 2)
        self.assertEqual(int(result["artifact_grade"][15]), 2)

    def test_sustained_jitter_candidate_preserves_raw_until_manual_confirmation(self) -> None:
        frames, sections = 40, 2
        translation = np.zeros(frames, dtype=np.float32); translation[15:18] = 1.2
        pcc = np.full((frames, 2, sections), 0.995, dtype=np.float32); pcc[15:18] = 0.80
        fb = np.full((frames, 2, sections), 0.004, dtype=np.float32); fb[15:18] = 0.5
        raw = np.zeros((frames, 2, sections), dtype=np.float32); raw[15:18] = -0.4
        common = dict(
            global_translation_step_px=translation,
            global_rotation_step_deg=np.zeros(frames),
            pair_pcc=pcc,
            pair_fb_error_px=fb,
            pair_valid=np.ones((frames, 2, sections), dtype=bool),
            p3_curve_distance_px=np.zeros((frames, 2, sections), dtype=np.float32),
            p3_topology_risk=np.zeros(frames, dtype=bool),
            radial_local_residual=np.zeros((frames, 4 * sections, 2), dtype=np.float32),
            raw_rsr=raw,
            section_count=sections,
        )
        unreviewed = artifact.build_artifact_qc(**common)
        np.testing.assert_array_equal(
            unreviewed["automatic_artifact_grade"][15:18], [3, 3, 3]
        )
        np.testing.assert_array_equal(unreviewed["artifact_grade"][15:18], [2, 2, 2])
        np.testing.assert_array_equal(unreviewed["raw_radial_strain_rate_s"], raw)
        self.assertTrue(
            np.all(np.isfinite(unreviewed["qc_radial_strain_rate_grade3_blank_s"]))
        )

        manual = np.zeros(frames, dtype=np.int8); manual[15:18] = 3
        reviewed = artifact.build_artifact_qc(
            **common, manual_minimum_grade=manual
        )
        np.testing.assert_array_equal(reviewed["raw_radial_strain_rate_s"], raw)
        self.assertTrue(
            np.all(np.isnan(reviewed["qc_radial_strain_rate_grade3_blank_s"][15:18]))
        )

    def test_three_manual_decisions_have_distinct_action_grades(self) -> None:
        frames, sections = 40, 2
        translation = np.zeros(frames, dtype=np.float32); translation[15:18] = 1.2
        pcc = np.full((frames, 2, sections), 0.995, dtype=np.float32); pcc[15:18] = 0.80
        fb = np.full((frames, 2, sections), 0.004, dtype=np.float32); fb[15:18] = 0.5
        raw = np.ones((frames, 2, sections), dtype=np.float32)
        decisions = np.zeros(frames, dtype=np.int8)
        decisions[15:18] = [1, 2, 3]
        result = artifact.build_artifact_qc(
            global_translation_step_px=translation,
            global_rotation_step_deg=np.zeros(frames),
            pair_pcc=pcc,
            pair_fb_error_px=fb,
            pair_valid=np.ones((frames, 2, sections), dtype=bool),
            p3_curve_distance_px=np.zeros((frames, 2, sections), dtype=np.float32),
            p3_topology_risk=np.zeros(frames, dtype=bool),
            radial_local_residual=np.zeros((frames, 4 * sections, 2), dtype=np.float32),
            raw_rsr=raw,
            section_count=sections,
            manual_review_decision_code=decisions,
        )
        np.testing.assert_array_equal(
            result["automatic_artifact_grade"][15:18], [3, 3, 3]
        )
        np.testing.assert_array_equal(result["artifact_grade"][15:18], [3, 2, 0])
        np.testing.assert_array_equal(
            result["automatic_grade3_reviewed"][15:18], [True, True, True]
        )
        self.assertFalse(
            np.any(result["automatic_grade3_pending_manual_review"][15:18])
        )
        self.assertTrue(
            np.all(np.isnan(result["qc_radial_strain_rate_grade3_blank_s"][15]))
        )
        self.assertTrue(
            np.all(np.isfinite(result["qc_radial_strain_rate_grade3_blank_s"][16:18]))
        )

    def test_tracking_loss_alone_is_not_called_probe_jitter(self) -> None:
        frames, sections = 30, 2
        pcc = np.full((frames, 2, sections), 0.995, dtype=np.float32); pcc[15] = 0.50
        valid = np.ones((frames, 2, sections), dtype=bool); valid[15] = False
        result = artifact.build_artifact_qc(
            global_translation_step_px=np.zeros(frames),
            global_rotation_step_deg=np.zeros(frames),
            pair_pcc=pcc,
            pair_fb_error_px=np.zeros((frames, 2, sections), dtype=np.float32),
            pair_valid=valid,
            p3_curve_distance_px=np.zeros((frames, 2, sections), dtype=np.float32),
            p3_topology_risk=np.zeros(frames, dtype=bool),
            radial_local_residual=np.zeros((frames, 4 * sections, 2), dtype=np.float32),
            raw_rsr=np.zeros((frames, 2, sections), dtype=np.float32),
            section_count=sections,
        )
        self.assertEqual(int(result["automatic_evidence_group_count"][15]), 1)
        self.assertEqual(int(result["automatic_artifact_grade"][15]), 1)


if __name__ == "__main__":
    unittest.main()
