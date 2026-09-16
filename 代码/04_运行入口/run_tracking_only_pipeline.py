#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone tracking-only pipeline migrated from end@a9cf11.

This entry intentionally stops after corrected P3/radial tracking and tracking
QC. Core algorithms, thresholds and formulas are imported from unchanged
modules copied from the source repository. The only new code here is pipeline
orchestration and file I/O for an independent repository.

Current standalone semantics match the 310-case new-patient batch with empty
manual-review files: automatic artifact/P3 candidates are recorded but are not
promoted to manual exclusions by this entry.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "代码" / "01_底层算法"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from peristalsis_pipeline import (  # noqa: E402
    tracking_artifact_qc_v2,
    tracking_huang_radial_pairs,
    tracking_huang_wall_fusion,
    tracking_lk,
)
from peristalsis_pipeline.p3_anatomical_position_qc import (  # noqa: E402
    P3AnatomicalPositionQCConfig,
    build_p3_anatomical_position_qc,
)
from peristalsis_pipeline.p3_image_boundary_correction import (  # noqa: E402
    P3ImageBoundaryConfig,
    apply_confirmed_boundary_overrides,
    apply_tracking_quality_gate,
    build_automatic_boundary_correction,
    extract_wall_boundary_evidence,
)


SOURCE_REPOSITORY = "wyxwyx357-del/end"
SOURCE_COMMIT = "a9cf11b95293e395e0a90a1e3340393d782a2d5d"
RUN_TAG = "new_patient_tracking"
RADIAL_OFFSET_PX = 21.0
STAGES = (
    "01_wall_tracking_reference",
    "02_radial_tracking_raw",
    "03_base_artifact_qc",
    "04_p3_boundary_correction",
    "05_radial_tracking_corrected",
    "06_p3_position_qc",
    "07_artifact_qc",
)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return convert(value.item()) if value.ndim == 0 else value.tolist()
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=convert),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def manifest_rows(label_dir: Path, case_id: str) -> list[dict[str, str]]:
    path = label_dir / f"{case_id}__five_anchor_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required_roles = {
        "video_start",
        "front_half_center",
        "middle",
        "back_half_center",
        "video_end",
    }
    if len(rows) != 5 or {row.get("anchor_name", "") for row in rows} != required_roles:
        raise ValueError(f"{case_id}: five-anchor manifest is incomplete")
    if {row.get("case_id", "") for row in rows} != {case_id}:
        raise ValueError(f"{case_id}: case_id mismatch in five-anchor manifest")
    videos = {row.get("source_video", "").strip() for row in rows}
    if len(videos) != 1 or "" in videos:
        raise ValueError(f"{case_id}: anchors do not reference exactly one source video")
    return rows


def source_video_from_manifest(label_dir: Path, case_id: str) -> Path:
    rows = manifest_rows(label_dir, case_id)
    video = Path(rows[0]["source_video"])
    if not video.is_file():
        raise FileNotFoundError(video)
    return video


def ordered_ids(nodes: list[dict[str, Any]], rail_name: str) -> np.ndarray:
    rows = sorted(
        (int(node["section_order"]), int(node["point_id"]))
        for node in nodes
        if str(node.get("rail_name", "")) == rail_name
    )
    if len(rows) < 2:
        raise ValueError(f"rail {rail_name!r} has too few points")
    return np.asarray([point_id for _, point_id in rows], dtype=np.int32)


def p3_risk_fields(source: dict[str, np.ndarray], frame_count: int) -> dict[str, np.ndarray]:
    measurement = np.asarray(source["tracking_measurement_valid"][:frame_count], dtype=bool)
    feature = np.asarray(source["feature_candidate_valid"][:frame_count], dtype=bool)
    geometry = np.asarray(source["geometry_candidate_valid"][:frame_count], dtype=bool)
    recovered = np.asarray(source["recovered_display_only"][:frame_count], dtype=bool)
    branch = np.asarray(source["branch_disagreement_px"][:frame_count], dtype=np.float32)
    edges = np.asarray(source["topology_bad_edge_count"][:frame_count], dtype=np.int32)
    cells = np.asarray(source["topology_bad_cell_count"][:frame_count], dtype=np.int32)
    return {
        "tracking_measurement_valid": measurement,
        "feature_candidate_valid": feature,
        "geometry_candidate_valid": geometry,
        "recovered_display_only": recovered,
        "branch_disagreement_px": branch,
        "topology_bad_edge_count": edges,
        "topology_bad_cell_count": cells,
        "p3_topology_risk_frame": (edges > 0) | (cells > 0),
        "p3_tracking_measurement_valid_fraction": np.mean(measurement, axis=1).astype(np.float32),
        "p3_recovered_display_only_fraction": np.mean(recovered, axis=1).astype(np.float32),
    }


def pair_maximum(values: np.ndarray, sections: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    output = np.full((len(values), 2, sections), np.nan, dtype=np.float32)
    for side, (inner_rail, outer_rail) in enumerate(((0, 1), (2, 3))):
        inner = values[:, inner_rail * sections : (inner_rail + 1) * sections]
        outer = values[:, outer_rail * sections : (outer_rail + 1) * sections]
        output[:, side] = np.maximum(inner, outer)
    return output


def global_steps(tracking: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    midline = np.asarray(tracking["p3_midline_points"], dtype=np.float32)
    center = np.mean(midline, axis=1)
    translation = np.r_[np.nan, np.linalg.norm(np.diff(center, axis=0), axis=1)].astype(np.float32)
    transform = np.asarray(tracking["pairwise_global_transform"], dtype=np.float32)
    angle = np.abs(np.arctan2(transform[:, 1, 0], transform[:, 0, 0]))
    rotation = (angle * 180.0 / np.pi).astype(np.float32)
    if len(rotation):
        rotation[0] = np.nan
    return translation, rotation


def measure_radial(
    frames: list[np.ndarray],
    wall_tracks: np.ndarray,
    pairwise_global_transform: np.ndarray,
    sections: int,
    fps: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    config = tracking_huang_wall_fusion.HuangWallFusionConfig()
    wall_flat = np.concatenate((wall_tracks[:, 0], wall_tracks[:, 1]), axis=1)
    tracking = tracking_huang_radial_pairs.measure_pairwise_radial_points(
        frames,
        wall_flat,
        pairwise_global_transform,
        anterior_count=sections,
        offset_px=RADIAL_OFFSET_PX,
        config=config,
    )
    deformation = tracking_huang_radial_pairs.radial_deformation_from_tracking(
        tracking,
        anterior_count=sections,
        fps=fps,
    )
    return tracking, deformation


def build_radial_archive(
    case_id: str,
    frames: list[np.ndarray],
    fps: float,
    resize_factor: float,
    p3_source: dict[str, np.ndarray],
    nodes: list[dict[str, Any]],
    wall_tracks: np.ndarray,
    global_result: dict[str, np.ndarray],
    extra: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    anterior = ordered_ids(nodes, "anterior_wall")
    posterior = ordered_ids(nodes, "posterior_wall")
    if len(anterior) != len(posterior):
        raise ValueError(f"{case_id}: anterior/posterior section count mismatch")
    tracking, deformation = measure_radial(
        frames,
        wall_tracks,
        global_result["pairwise_global_transform"],
        len(anterior),
        fps,
    )
    risk = p3_risk_fields(p3_source, len(frames))
    archive: dict[str, np.ndarray] = {
        **global_result,
        **tracking,
        **deformation,
        **risk,
        "radial_offset_px": np.asarray(RADIAL_OFFSET_PX, dtype=np.float32),
        "fps": np.asarray(fps, dtype=np.float32),
        "resize_factor": np.asarray(resize_factor, dtype=np.float32),
        "longitudinal_section_count": np.asarray(len(anterior), dtype=np.int32),
        "case_id": np.asarray(case_id),
        "p3_role": np.asarray("per_frame_anatomical_localization_not_direct_deformation"),
        "deformation_measurement_role": np.asarray("adjacent_frame_lk_radial_pair_length_change"),
        "p3_coordinate_delta_used_as_rsr": np.asarray(False),
        "formal_propagation_released": np.asarray(False),
    }
    if extra:
        archive.update(extra)
    return archive


def artifact_qc_from_tracking(
    tracking: dict[str, np.ndarray],
    position_qc: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    sections = int(tracking["longitudinal_section_count"])
    frame_count = len(tracking["radial_strain_rate_s"])
    translation, rotation = global_steps(tracking)
    p3_distance = pair_maximum(tracking["radial_to_reference_curve_distance_px"], sections)
    zeros_grade = np.zeros(frame_count, dtype=np.int8)
    zeros_reviewed = np.zeros(frame_count, dtype=bool)
    qc = tracking_artifact_qc_v2.build_artifact_qc(
        global_translation_step_px=translation,
        global_rotation_step_deg=rotation,
        pair_pcc=np.asarray(tracking["radial_pair_pcc"], dtype=np.float32),
        pair_fb_error_px=np.asarray(tracking["radial_pair_fb_error_px"], dtype=np.float32),
        pair_valid=np.asarray(tracking["radial_pair_valid"], dtype=bool),
        p3_curve_distance_px=p3_distance,
        p3_topology_risk=np.asarray(tracking["p3_topology_risk_frame"], dtype=bool),
        radial_local_residual=np.asarray(tracking["radial_local_residual"], dtype=np.float32),
        raw_rsr=np.asarray(tracking["radial_strain_rate_s"], dtype=np.float32),
        section_count=sections,
        manual_minimum_grade=zeros_grade,
        manual_reviewed_automatic_grade3=zeros_reviewed,
        manual_review_decision_code=zeros_grade,
        config=tracking_artifact_qc_v2.ArtifactQCConfig(),
    )
    expected_shape = np.asarray(tracking["radial_strain_rate_s"]).shape
    if position_qc is None:
        automatic_position = np.zeros(expected_shape, dtype=bool)
        manual_position = np.zeros(expected_shape, dtype=bool)
    else:
        automatic_position = np.asarray(
            position_qc["automatic_anatomical_position_candidate"], dtype=bool
        )
        manual_position = np.zeros(expected_shape, dtype=bool)
    qc["automatic_anatomical_position_candidate"] = automatic_position
    qc["manual_anatomical_position_invalid_pair"] = manual_position
    transition = np.asarray(
        tracking.get(
            "automatic_boundary_correction_transition_risk",
            np.zeros(expected_shape, dtype=bool),
        ),
        dtype=bool,
    )
    qc["image_boundary_correction_transition_risk"] = transition
    sensitivity = np.asarray(qc["qc_radial_strain_rate_grade3_blank_s"], dtype=np.float32).copy()
    sensitivity[transition] = np.nan
    qc["qc_radial_strain_rate_with_boundary_transition_blank_s"] = sensitivity
    qc["image_boundary_transition_can_exclude_qc_rsr"] = np.asarray(False)
    qc["image_boundary_transition_is_sensitivity_branch"] = np.asarray(True)
    qc["automatic_position_candidates_can_exclude_rsr"] = np.asarray(False)
    qc["manual_confirmed_position_invalid_can_exclude_qc_rsr"] = np.asarray(True)
    qc["artifact_qc_version"] = np.asarray("2.1")
    qc["threshold_source"] = np.asarray(
        "project_development_relative_plus_absolute_support_not_literature_cutoffs"
    )
    qc["unreviewed_automatic_grade3_can_blank"] = np.asarray(False)
    qc["formal_propagation_released"] = np.asarray(False)
    return qc


def stage_dirs(case_root: Path) -> dict[str, Path]:
    return {name: case_root / name for name in STAGES}


def run_wall_tracking(
    case_id: str,
    label_dir: Path,
    output: Path,
    resize_factor: float,
    sections: int,
    skip_videos: bool,
) -> tuple[Path, Path]:
    script = PROJECT_ROOT / "代码" / "03_实验与历史代码" / "run_final_sparse_anchor_wall_tracking.py"
    command = [
        sys.executable,
        "-B",
        str(script),
        "--cases",
        case_id,
        "--external-label-dir",
        str(label_dir),
        "--output-dir",
        str(output),
        "--include-endpoint-anchors",
        "--run-tag",
        RUN_TAG,
        "--allow-overwrite",
        "--resize-factor",
        str(resize_factor),
        "--sections",
        str(sections),
    ]
    if skip_videos:
        command.append("--skip-videos")
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    tracking = output / f"{case_id}_{RUN_TAG}_sparse_anchor_tracks.npz"
    nodes = output / f"{case_id}_{RUN_TAG}_nodes.json"
    if not tracking.is_file() or not nodes.is_file():
        raise FileNotFoundError(f"{case_id}: wall tracking outputs missing")
    return tracking, nodes


def process_case(
    case_id: str,
    label_dir: Path,
    output_root: Path,
    resize_factor: float,
    sections_requested: int,
    skip_videos: bool,
) -> dict[str, Any]:
    video = source_video_from_manifest(label_dir, case_id)
    frames, info = tracking_lk.read_video_gray(video, resize_factor)
    fps = float(info["fps"])
    case_root = output_root / case_id
    dirs = stage_dirs(case_root)
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    wall_path, node_path = run_wall_tracking(
        case_id,
        label_dir,
        dirs[STAGES[0]],
        resize_factor,
        sections_requested,
        skip_videos,
    )
    p3 = load_npz(wall_path)
    nodes = json.loads(node_path.read_text(encoding="utf-8"))
    p3_tracks = np.asarray(p3["tracks"][: len(frames)], dtype=np.float32)
    midline = ordered_ids(nodes, "midline")
    anterior = ordered_ids(nodes, "anterior_wall")
    posterior = ordered_ids(nodes, "posterior_wall")
    if len(anterior) != len(posterior):
        raise ValueError(f"{case_id}: paired wall section count mismatch")
    wall = np.stack((p3_tracks[:, anterior], p3_tracks[:, posterior]), axis=1)

    config = tracking_huang_wall_fusion.HuangWallFusionConfig()
    global_result = tracking_huang_wall_fusion.estimate_pairwise_global_motion(
        frames, p3_tracks[:, midline], config
    )
    raw = build_radial_archive(
        case_id,
        frames,
        fps,
        resize_factor,
        p3,
        nodes,
        wall,
        global_result,
    )
    raw_path = dirs[STAGES[1]] / f"{case_id}_径向点对原始数据.npz"
    np.savez_compressed(raw_path, **raw)

    base_qc = artifact_qc_from_tracking(raw)
    base_qc_path = dirs[STAGES[2]] / f"{case_id}_artifact_qc_v2_1.npz"
    np.savez_compressed(base_qc_path, **base_qc, case_id=np.asarray(case_id), fps=np.asarray(fps, dtype=np.float32))

    boundary_config = P3ImageBoundaryConfig()
    evidence = extract_wall_boundary_evidence(frames, wall, boundary_config)
    grade = np.asarray(base_qc["artifact_grade"], dtype=np.int8)
    automatic_grade = np.asarray(base_qc["automatic_artifact_grade"], dtype=np.int8)
    correction_grade = np.maximum(grade, automatic_grade)
    correction = build_automatic_boundary_correction(
        wall,
        evidence,
        grade,
        fps,
        boundary_config,
        additional_excluded_frames=automatic_grade >= 3,
    )
    candidate_tracking, candidate_deformation = measure_radial(
        frames,
        correction["p3_wall_tracks_image_corrected"],
        raw["pairwise_global_transform"],
        len(anterior),
        fps,
    )
    correction = apply_tracking_quality_gate(
        correction,
        evidence,
        {**candidate_tracking, **candidate_deformation},
        raw,
        boundary_config,
    )
    correction = apply_confirmed_boundary_overrides(
        correction,
        evidence,
        np.zeros(wall.shape[:3], dtype=bool),
        correction_grade,
        boundary_config.maximum_correction_px,
    )
    corrected_wall = np.asarray(correction["p3_wall_tracks_image_corrected"], dtype=np.float32)
    boundary_path = dirs[STAGES[3]] / f"{case_id}_p3_image_boundary_evidence_v2.npz"
    np.savez_compressed(
        boundary_path,
        **evidence,
        **correction,
        pairwise_global_transform=raw["pairwise_global_transform"],
        p3_midline_points=raw["p3_midline_points"],
        artifact_grade=grade,
        automatic_artifact_grade_before_boundary_correction=automatic_grade,
        boundary_correction_exclusion_grade=correction_grade,
        fps=np.asarray(fps, dtype=np.float32),
        resize_factor=np.asarray(resize_factor, dtype=np.float32),
        longitudinal_section_count=np.asarray(len(anterior), dtype=np.int32),
        radial_offset_px=np.asarray(RADIAL_OFFSET_PX, dtype=np.float32),
        case_id=np.asarray(case_id),
        formal_propagation_released=np.asarray(False),
    )

    corrected = build_radial_archive(
        case_id,
        frames,
        fps,
        resize_factor,
        p3,
        nodes,
        corrected_wall,
        {
            "pairwise_global_transform": raw["pairwise_global_transform"],
            "p3_midline_points": raw["p3_midline_points"],
        },
        extra={
            "p3_wall_tracks_image_corrected": corrected_wall,
            "automatic_boundary_correction_mask": np.asarray(
                correction["automatic_boundary_correction_mask"], dtype=bool
            ),
            "automatic_boundary_correction_transition_risk": np.asarray(
                correction["automatic_boundary_correction_transition_risk"], dtype=bool
            ),
        },
    )
    corrected_path = dirs[STAGES[4]] / f"{case_id}_p3_image_boundary_evidence_v2.npz"
    np.savez_compressed(corrected_path, **corrected)

    branch = np.stack(
        (
            np.asarray(p3["branch_normal_disagreement_px"][:, anterior], dtype=np.float32),
            np.asarray(p3["branch_normal_disagreement_px"][:, posterior], dtype=np.float32),
        ),
        axis=1,
    )
    recovered = np.stack(
        (
            np.asarray(p3["recovered_display_only"][:, anterior], dtype=bool),
            np.asarray(p3["recovered_display_only"][:, posterior], dtype=bool),
        ),
        axis=1,
    )
    topology = (
        np.asarray(p3["topology_bad_edge_count"], dtype=np.int32) > 0
    ) | (
        np.asarray(p3["topology_bad_cell_count"], dtype=np.int32) > 0
    )
    position = build_p3_anatomical_position_qc(
        corrected_wall,
        branch,
        topology,
        recovered,
        P3AnatomicalPositionQCConfig(),
    )
    position["manual_anatomical_position_invalid_pair"] = np.zeros(
        position["automatic_anatomical_position_candidate"].shape, dtype=bool
    )
    position_path = dirs[STAGES[5]] / f"{case_id}_p3_anatomical_position_qc_v1.npz"
    np.savez_compressed(
        position_path,
        **position,
        case_id=np.asarray(case_id),
        fps=np.asarray(fps, dtype=np.float32),
        automatic_candidates_can_exclude_rsr=np.asarray(False),
        formal_propagation_released=np.asarray(False),
    )

    final_qc = artifact_qc_from_tracking(corrected, position)
    final_qc_path = dirs[STAGES[6]] / f"{case_id}_artifact_qc_v2_1.npz"
    np.savez_compressed(
        final_qc_path,
        **final_qc,
        case_id=np.asarray(case_id),
        fps=np.asarray(fps, dtype=np.float32),
        manual_review_status=np.asarray("PENDING_MANUAL_REVIEW"),
    )

    radial_valid = np.asarray(corrected["radial_pair_valid"], dtype=bool)
    pcc = np.asarray(corrected["radial_pair_pcc"], dtype=np.float32)
    fb = np.asarray(corrected["radial_pair_fb_error_px"], dtype=np.float32)
    usable_fb = fb[1:][np.isfinite(fb[1:])]
    summary = {
        "case_id": case_id,
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_video": str(video),
        "frame_count": len(frames),
        "fps": fps,
        "resize_factor": resize_factor,
        "requested_sections": sections_requested,
        "final_paired_sections": len(anterior),
        "radial_offset_px": RADIAL_OFFSET_PX,
        "corrected_radial_pair_valid_percent": round(
            100.0 * float(np.mean(radial_valid[1:])), 6
        ),
        "corrected_pair_pcc_mean": (
            round(float(np.nanmean(pcc[1:])), 6)
            if np.any(np.isfinite(pcc[1:]))
            else None
        ),
        "corrected_pair_fb_p95_px": (
            round(float(np.percentile(usable_fb, 95)), 6) if len(usable_fb) else None
        ),
        "boundary_correction_point_count": int(
            np.count_nonzero(correction["automatic_boundary_correction_mask"])
        ),
        "p3_position_automatic_candidate_count": int(
            np.count_nonzero(position["automatic_anatomical_position_candidate"])
        ),
        "final_automatic_grade3_pending_frame_count": int(
            np.count_nonzero(final_qc["automatic_grade3_pending_manual_review"])
        ),
        "manual_review_status": "PENDING_MANUAL_REVIEW",
        "tracking_scope_complete": True,
        "feature_extraction_run": False,
        "direction_or_propagation_run": False,
        "prediction_model_run": False,
    }
    save_json(case_root / "tracking_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--resize-factor", type=float, default=0.5)
    parser.add_argument("--sections", type=int, default=10)
    parser.add_argument("--skip-videos", action="store_true")
    args = parser.parse_args()

    label_dir = args.external_label_dir.resolve()
    output_root = args.output_dir.resolve()
    if not label_dir.is_dir():
        raise FileNotFoundError(label_dir)
    if args.resize_factor <= 0 or not math.isfinite(args.resize_factor):
        raise ValueError("--resize-factor must be positive and finite")
    if args.sections < 3:
        raise ValueError("--sections must be >= 3")
    if len(set(args.cases)) != len(args.cases):
        raise ValueError("--cases contains duplicate case IDs")
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    failures = []
    for case_id in args.cases:
        try:
            print(f"[{case_id}] tracking start", flush=True)
            summaries.append(
                process_case(
                    case_id,
                    label_dir,
                    output_root,
                    args.resize_factor,
                    args.sections,
                    args.skip_videos,
                )
            )
            print(f"[{case_id}] tracking complete", flush=True)
        except Exception as exc:
            failures.append(
                {
                    "case_id": case_id,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                }
            )
            print(f"[{case_id}] FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    write_csv(output_root / "tracking_batch_summary.csv", summaries)
    write_csv(output_root / "tracking_batch_failures.csv", failures)
    save_json(
        output_root / "tracking_batch_provenance.json",
        {
            "source_repository": SOURCE_REPOSITORY,
            "source_commit": SOURCE_COMMIT,
            "cases_requested": args.cases,
            "cases_complete": [row["case_id"] for row in summaries],
            "cases_failed": [row["case_id"] for row in failures],
            "scope": "wall tracking through corrected radial tracking and tracking QC only",
            "manual_review_semantics": "automatic candidates pending; no manual confirmation rows applied",
        },
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
