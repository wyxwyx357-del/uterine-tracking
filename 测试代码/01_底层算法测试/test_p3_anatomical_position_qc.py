from __future__ import annotations

from script_paths import script_path

import csv
import importlib.util
from pathlib import Path

import numpy as np

from peristalsis_pipeline.p3_anatomical_position_qc import (
    build_p3_anatomical_position_qc,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = script_path("run_step1_4_p3_anatomical_position_qc_v1_5case.py")
SPEC = importlib.util.spec_from_file_location("p3_anatomical_qc_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def base_tracks(frames: int = 7, sections: int = 5) -> np.ndarray:
    x = np.arange(sections, dtype=np.float32) * 10 + 20
    tracks = np.zeros((frames, 2, sections, 2), dtype=np.float32)
    tracks[:, 0, :, 0] = x
    tracks[:, 0, :, 1] = 40
    tracks[:, 1, :, 0] = x
    tracks[:, 1, :, 1] = 80
    for frame in range(frames):
        tracks[frame, :, :, 0] += frame
    return tracks


def test_common_translation_does_not_create_candidate() -> None:
    tracks = base_tracks()
    shape = tracks.shape[:3]
    result = build_p3_anatomical_position_qc(
        tracks,
        np.full(shape, 8.0, dtype=np.float32),
        np.zeros(shape[0], dtype=bool),
        np.ones(shape, dtype=bool),
    )
    assert not np.any(result["automatic_anatomical_position_candidate"])
    assert not bool(result["automatic_suggestion_applied_to_tracking"])
    assert not bool(result["automatic_suggestion_applied_to_rsr"])


def test_isolated_multi_evidence_jump_gets_review_only_suggestion() -> None:
    tracks = base_tracks()
    tracks[3, 0, 2, 1] += 4.0
    shape = tracks.shape[:3]
    branch = np.zeros(shape, dtype=np.float32)
    branch[3, 0, 2] = 4.0
    result = build_p3_anatomical_position_qc(
        tracks,
        branch,
        np.zeros(shape[0], dtype=bool),
        np.zeros(shape, dtype=bool),
    )
    assert result["automatic_anatomical_position_candidate"][3, 0, 2]
    assert result["suggested_position_available"][3, 0, 2]
    assert np.all(np.isfinite(result["suggested_wall_position_review_only"][3, 0, 2]))
    assert not np.any(result["automatic_anatomical_position_candidate"][3, 1])


def test_single_evidence_does_not_create_candidate() -> None:
    tracks = base_tracks()
    tracks[3, 0, 2, 1] += 4.0
    shape = tracks.shape[:3]
    result = build_p3_anatomical_position_qc(
        tracks,
        np.zeros(shape, dtype=np.float32),
        np.zeros(shape[0], dtype=bool),
        np.zeros(shape, dtype=bool),
    )
    assert not np.any(result["automatic_anatomical_position_candidate"])


def test_only_confirmed_manual_rows_create_pair_exclusion(tmp_path: Path) -> None:
    path = tmp_path / "manual.csv"
    fieldnames = [
        "case_id", "start_s", "end_s", "start_frame", "end_frame", "side",
        "section_start_1based", "section_end_1based", "apply_to_qc", "description",
        "review_status", "independent_for_algorithm_validation",
    ]
    rows = [
        {
            "case_id": "CASE", "start_s": "", "end_s": "", "start_frame": "2",
            "end_frame": "4", "side": "后壁", "section_start_1based": "2",
            "section_end_1based": "3", "apply_to_qc": "true", "description": "bad",
            "review_status": "confirmed", "independent_for_algorithm_validation": "true",
        },
        {
            "case_id": "CASE", "start_s": "", "end_s": "", "start_frame": "0",
            "end_frame": "9", "side": "前壁", "section_start_1based": "1",
            "section_end_1based": "5", "apply_to_qc": "true", "description": "pending",
            "review_status": "pending", "independent_for_algorithm_validation": "true",
        },
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    mask, applied = runner.load_manual_pair_mask(path, "CASE", 10, 5, 10.0)
    assert len(applied) == 1
    assert int(np.count_nonzero(mask)) == 6
    assert np.all(mask[2:5, 1, 1:3])
    assert not np.any(mask[:, 0])
