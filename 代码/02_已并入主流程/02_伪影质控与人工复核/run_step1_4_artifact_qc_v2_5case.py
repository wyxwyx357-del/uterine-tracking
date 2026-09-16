#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Five-case artifact QC v2.1 adapted to P3-localized radial LK measurements."""

from __future__ import annotations

from peristalsis_pipeline.project_layout import output_dir

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "代码" / "01_底层算法"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from peristalsis_pipeline import tracking_artifact_qc_v2, tracking_lk


CASES = (
    "CASE_003",
    "CASE_002",
    "CASE_001",
    "CASE_005",
    "CASE_004",
)
DEFAULT_TRACKING_DIR = (
    output_dir(PROJECT_ROOT, "step1_4_p3_assisted_radial_tracking_v1_5case")
)
DEFAULT_P3_DIR = PROJECT_ROOT / "输入" / "04_冻结基线" / "step1_4_five_anchor"
DEFAULT_P3_POSITION_QC_DIR = (
    output_dir(PROJECT_ROOT, "step1_4_p3_anatomical_position_qc_v1_5case")
)
DEFAULT_MANUAL_INTERVALS = PROJECT_ROOT / "输入" / "02_人工标签" / "03_质控人工记录" / "人工抖动与伪影时段_v2_1.csv"
DEFAULT_POSTHOC_CONFIRMATIONS = (
    PROJECT_ROOT / "输入" / "02_人工标签" / "03_质控人工记录" / "p3辅助主轨_自动3级候选人工确认.csv"
)
DEFAULT_FROZEN_CONFIG = (
    PROJECT_ROOT / "输入" / "04_冻结基线"
    / "artifact_qc_v2_1_frozen"
    / "artifact_qc_v2_1_project_freeze.json"
)
DEFAULT_OUTPUT = (
    output_dir(PROJECT_ROOT, "step1_4_p3_assisted_artifact_qc_v2_1_5case")
)
GRADE_NAMES = {
    0: "0级_无明显抖动证据",
    1: "1级_轻微或单组异常",
    2: "2级_疑似伪影或待复核候选_保留RSR",
    3: "3级_人工确认明显抖动或离面风险_质控副本留空",
}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def apply_manual_position_exclusion(
    qc_rsr: np.ndarray, manual_invalid_pair: np.ndarray
) -> np.ndarray:
    """Blank only the QC copy at manually confirmed point-level locations."""
    values = np.asarray(qc_rsr, dtype=np.float32)
    invalid = np.asarray(manual_invalid_pair, dtype=bool)
    if values.shape != invalid.shape:
        raise ValueError("manual P3 position mask must match radial RSR shape")
    result = values.copy()
    result[invalid] = np.nan
    return result


def load_nodes(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def wall_ids(nodes: list[dict[str, Any]], rail_name: str) -> np.ndarray:
    ordered = sorted(
        (
            int(node["section_order"]),
            point_idx,
        )
        for point_idx, node in enumerate(nodes)
        if str(node.get("rail_name")) == rail_name
    )
    if len(ordered) < 2:
        raise ValueError(f"P3 rail {rail_name!r} has too few points")
    return np.asarray([point_idx for _, point_idx in ordered], dtype=np.int32)


def p3_reference_distance(
    tracking: dict[str, np.ndarray], sections: int
) -> np.ndarray:
    """Return LK-measured radial-point distance from its P3-generated rail."""
    distance = np.asarray(
        tracking["radial_to_reference_curve_distance_px"], dtype=np.float32
    )
    return pair_maximum(distance, sections)


def pair_minimum(values: np.ndarray, sections: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    output = np.full((len(values), 2, sections), np.nan, dtype=np.float32)
    for side, (inner_rail, outer_rail) in enumerate(((0, 1), (2, 3))):
        inner = values[:, inner_rail * sections : (inner_rail + 1) * sections]
        outer = values[:, outer_rail * sections : (outer_rail + 1) * sections]
        output[:, side] = np.minimum(inner, outer)
    return output


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
    translation = np.r_[
        np.nan, np.linalg.norm(np.diff(center, axis=0), axis=1)
    ].astype(np.float32)
    transform = np.asarray(tracking["pairwise_global_transform"], dtype=np.float32)
    angle = np.abs(np.arctan2(transform[:, 1, 0], transform[:, 0, 0]))
    rotation = angle * 180.0 / np.pi
    rotation[0] = np.nan
    return translation, rotation.astype(np.float32)


def manual_interval_frame_bounds(
    row: dict[str, str], frame_count: int, fps: float
) -> tuple[int, int]:
    if row.get("start_frame", "").strip() and row.get("end_frame", "").strip():
        start = int(row["start_frame"])
        end = int(row["end_frame"])
    else:
        start = int(math.floor(float(row["start_s"]) * fps))
        end = int(math.ceil(float(row["end_s"]) * fps))
    start = max(0, start)
    end = min(frame_count - 1, end)
    if end < start:
        raise ValueError(f"manual interval has end before start: {row}")
    return start, end


def load_manual_grade(
    path: Path, case_id: str, frame_count: int, fps: float
) -> tuple[np.ndarray, list[dict[str, str]]]:
    grade, _, relevant = load_manual_review(path, case_id, frame_count, fps)
    return grade, relevant


def manual_review_frame_bounds(
    row: dict[str, str], frame_count: int, fps: float
) -> tuple[int, int]:
    review_row = dict(row)
    has_start = bool(row.get("review_start_frame", "").strip())
    has_end = bool(row.get("review_end_frame", "").strip())
    if has_start != has_end:
        raise ValueError("review_start_frame and review_end_frame must be provided together")
    if has_start:
        review_row["start_frame"] = row.get("review_start_frame", "")
        review_row["end_frame"] = row.get("review_end_frame", "")
    return manual_interval_frame_bounds(review_row, frame_count, fps)


def load_manual_review(
    path: Path, case_id: str, frame_count: int, fps: float
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    grade = np.zeros(frame_count, dtype=np.int8)
    reviewed = np.zeros(frame_count, dtype=bool)
    relevant: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] != case_id:
                continue
            relevant.append(row)
            apply_to_qc = row["apply_to_qc"].strip().lower() == "true"
            minimum_grade_text = row.get("minimum_grade", "").strip()
            minimum_grade = int(minimum_grade_text) if minimum_grade_text else 0
            confirmed = row.get("review_status", "").strip().lower() == "confirmed"
            explicitly_resolved = (
                row.get("resolves_automatic_grade3_review", "").strip().lower()
                == "true"
            )
            if confirmed and (
                explicitly_resolved or (apply_to_qc and minimum_grade >= 3)
            ):
                review_start, review_end = manual_review_frame_bounds(
                    row, frame_count, fps
                )
                reviewed[review_start : review_end + 1] = True
                if row.get("review_decision", "").strip() == "keep_grade2":
                    grade[review_start : review_end + 1] = np.maximum(
                        grade[review_start : review_end + 1], 2
                    )
            if apply_to_qc:
                start, end = manual_interval_frame_bounds(row, frame_count, fps)
                grade[start : end + 1] = np.maximum(
                    grade[start : end + 1], minimum_grade
                )
    return grade, reviewed, relevant


def manual_review_trace_arrays(
    rows: list[dict[str, str]], frame_count: int, fps: float
) -> dict[str, np.ndarray]:
    """Preserve review provenance while limiting grade actions to action bounds."""
    decision_codes = {
        "confirmed_grade3": 1,
        "keep_grade2": 2,
        "not_artifact": 3,
    }
    code = np.zeros(frame_count, dtype=np.int8)
    reviewed_code = np.zeros(frame_count, dtype=np.int8)
    recorded = np.zeros(frame_count, dtype=bool)
    reviewer = np.full(frame_count, "", dtype="<U128")
    review_date = np.full(frame_count, "", dtype="<U32")
    for row in rows:
        if row.get("review_status", "").strip().lower() != "confirmed":
            continue
        decision = row.get("review_decision", "").strip()
        if decision not in decision_codes:
            continue
        review_start, review_end = manual_review_frame_bounds(row, frame_count, fps)
        new_code = decision_codes[decision]
        occupied = reviewed_code[review_start : review_end + 1] != 0
        if np.any(
            occupied
            & (reviewed_code[review_start : review_end + 1] != new_code)
        ):
            raise ValueError("conflicting manual review decisions overlap in time")
        reviewed_code[review_start : review_end + 1] = new_code
        recorded[review_start : review_end + 1] = True
        reviewer[review_start : review_end + 1] = row.get("reviewer", "").strip()
        review_date[review_start : review_end + 1] = row.get(
            "review_date", ""
        ).strip()
        if decision == "confirmed_grade3":
            action_start, action_end = manual_interval_frame_bounds(
                row, frame_count, fps
            )
        else:
            action_start, action_end = review_start, review_end
        code[action_start : action_end + 1] = new_code
    return {
        "manual_review_decision_code": code,
        "manual_review_recorded": recorded,
        "manual_review_reviewer": reviewer,
        "manual_review_date": review_date,
    }


def load_posthoc_confirmation(
    path: Path, case_id: str
) -> dict[str, str] | None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        matches = [
            row for row in csv.DictReader(handle) if row["case_id"] == case_id
        ]
    if len(matches) > 1:
        raise ValueError(f"duplicate posthoc confirmation for {case_id}")
    if not matches:
        return None
    row = matches[0]
    if row["artifact_qc_version"].strip() != "2.1":
        raise ValueError(f"posthoc confirmation version mismatch for {case_id}")
    if row["independent_for_algorithm_validation"].strip().lower() != "false":
        raise ValueError("posthoc confirmation cannot be marked independent")
    return row


def load_frozen_qc_config(
    path: Path,
) -> tuple[tracking_artifact_qc_v2.ArtifactQCConfig, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_qc_version") != "2.1":
        raise ValueError("frozen artifact QC version must be 2.1")
    if payload.get("freeze_scope") != "current_project_engineering_rule_for_future_locked_evaluation":
        raise ValueError("unexpected artifact QC freeze scope")
    if payload.get("clinical_or_literature_threshold") is not False:
        raise ValueError("project freeze must not claim a clinical/literature threshold")
    return tracking_artifact_qc_v2.ArtifactQCConfig(**payload["config"]), payload


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else math.nan


def contiguous_segments(mask: np.ndarray, fps: float) -> list[tuple[float, float]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    return [
        (float(group[0] / fps), float(group[-1] / fps))
        for group in np.split(indices, splits)
    ]


def reason_for_frame(qc: dict[str, np.ndarray], frame_idx: int) -> str:
    reasons = []
    if qc["manual_minimum_grade"][frame_idx] > 0:
        reasons.append("人工已知异常")
    if qc["global_motion_anomaly_score"][frame_idx] >= 0.5:
        reasons.append("整体运动突变")
    if qc["tracking_quality_anomaly_score"][frame_idx] >= 0.5:
        reasons.append("追踪质量异常")
    if qc["synchronized_motion_anomaly_score"][frame_idx] >= 0.5:
        reasons.append("多点同步运动或同步形变")
    if qc["automatic_grade3_pending_manual_review"][frame_idx]:
        reasons.append("自动3级候选待人工确认_当前按2级保留")
    elif (
        qc["relative_only_artifact_candidate_grade"][frame_idx] >= 3
        and qc["automatic_artifact_grade"][frame_idx] < 3
    ):
        reasons.append("相对异常但绝对量或邻帧支持不足_已降级")
    return "、".join(reasons) if reasons else "无明显组合证据"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def chinese_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def add_chinese_text(image: np.ndarray, lines: list[str], color: tuple[int, int, int]) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((4, 4, image.shape[1] - 4, 78), fill=(0, 0, 0, 165))
    font = chinese_font(15)
    for line_idx, line in enumerate(lines):
        draw.text((10, 8 + line_idx * 22), line, font=font, fill=(*color, 255))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def write_review_video(
    path: Path,
    frames: list[np.ndarray],
    tracking: dict[str, np.ndarray],
    p3: dict[str, np.ndarray],
    p3_nodes: list[dict[str, Any]],
    qc: dict[str, np.ndarray],
    sections: int,
    fps: float,
) -> None:
    height, width = frames[0].shape
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create {path}")
    measured = np.asarray(
        tracking["radial_lk_measured_p3_initial"], dtype=np.float32
    )
    reference = np.asarray(tracking["radial_reference"], dtype=np.float32)
    wall_tracks = review_wall_tracks(tracking, p3, p3_nodes)
    bgr = {0: (0, 180, 0), 1: (0, 220, 255), 2: (0, 140, 255), 3: (0, 0, 255)}
    rgb = {0: (150, 255, 150), 1: (255, 255, 0), 2: (255, 170, 0), 3: (255, 80, 80)}
    try:
        for frame_idx, gray in enumerate(frames):
            canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for side in range(2):
                cv2.polylines(
                    canvas,
                    [np.round(wall_tracks[frame_idx, side]).astype(np.int32)],
                    False,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            points = measured[frame_idx]
            if frame_idx == 0:
                points = reference[frame_idx]
            for side, (inner_rail, outer_rail) in enumerate(((0, 1), (2, 3))):
                inner = points[
                    inner_rail * sections : (inner_rail + 1) * sections
                ]
                outer = points[
                    outer_rail * sections : (outer_rail + 1) * sections
                ]
                for first, second in zip(inner, outer):
                    if np.all(np.isfinite((first, second))):
                        cv2.line(
                            canvas,
                            tuple(np.round(first).astype(int)),
                            tuple(np.round(second).astype(int)),
                            (0, 210, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.circle(
                            canvas,
                            tuple(np.round(first).astype(int)),
                            2,
                            (0, 235, 255),
                            -1,
                            cv2.LINE_AA,
                        )
                        cv2.circle(
                            canvas,
                            tuple(np.round(second).astype(int)),
                            2,
                            (0, 210, 0),
                            -1,
                            cv2.LINE_AA,
                        )
            grade = int(qc["artifact_grade"][frame_idx])
            cv2.rectangle(canvas, (1, 1), (width - 2, height - 2), bgr[grade], 3)
            canvas = add_chinese_text(
                canvas,
                [
                    f"帧 {frame_idx}  时间 {frame_idx/fps:.3f}秒  处理:{GRADE_NAMES[grade]}  自动候选:{int(qc['automatic_artifact_grade'][frame_idx])}级",
                    f"整体 {qc['global_motion_anomaly_score'][frame_idx]:.2f}  追踪 {qc['tracking_quality_anomaly_score'][frame_idx]:.2f}  同步 {qc['synchronized_motion_anomaly_score'][frame_idx]:.2f}",
                    f"原因：{reason_for_frame(qc, frame_idx)}；青线=计算所用P3，黄点=LK内点，绿线/点=LK径向点对",
                ],
                rgb[grade],
            )
            writer.write(canvas)
    finally:
        writer.release()


def review_wall_tracks(
    tracking: dict[str, np.ndarray],
    p3: dict[str, np.ndarray],
    p3_nodes: list[dict[str, Any]],
) -> np.ndarray:
    """Return the P3 wall tracks that generated the displayed LK references."""
    if "p3_wall_tracks_image_corrected" in tracking:
        tracks = np.asarray(
            tracking["p3_wall_tracks_image_corrected"], dtype=np.float32
        )
        if tracks.ndim != 4 or tracks.shape[1] != 2 or tracks.shape[-1] != 2:
            raise ValueError("corrected P3 wall tracks must have shape (F, 2, S, 2)")
        return tracks
    p3_tracks = np.asarray(p3["tracks"], dtype=np.float32)
    ant_ids = wall_ids(p3_nodes, "anterior_wall")
    post_ids = wall_ids(p3_nodes, "posterior_wall")
    return np.stack((p3_tracks[:, ant_ids], p3_tracks[:, post_ids]), axis=1)


def analyze_case(
    case_id: str,
    tracking_dir: Path,
    tracking_filename_template: str,
    p3_dir: Path,
    p3_position_qc_dir: Path,
    manual_path: Path,
    posthoc_path: Path,
    qc_config: tracking_artifact_qc_v2.ArtifactQCConfig,
    freeze_metadata: dict[str, Any],
    output_dir: Path,
    skip_video: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tracking_path = tracking_dir / tracking_filename_template.format(case_id=case_id)
    p3_path = p3_dir / f"{case_id}_step1_4_5a_sparse_anchor_tracks.npz"
    node_path = p3_dir / f"{case_id}_step1_4_5a_nodes.json"
    tracking = load_npz(tracking_path)
    p3 = load_npz(p3_path)
    nodes = load_nodes(node_path)
    fps = float(tracking["fps"])
    sections = int(tracking["longitudinal_section_count"])
    frame_count = len(tracking["radial_strain_rate_s"])
    if "radial_pair_fb_error_px" not in tracking:
        raise KeyError(
            "P3-assisted input lacks radial_pair_fb_error_px; rerun the P3-assisted tracker"
        )
    translation, rotation = global_steps(tracking)
    pair_pcc = np.asarray(tracking["radial_pair_pcc"], dtype=np.float32)
    pair_fb = np.asarray(tracking["radial_pair_fb_error_px"], dtype=np.float32)
    p3_distance = p3_reference_distance(tracking, sections)
    topology = np.asarray(tracking["p3_topology_risk_frame"], dtype=bool)
    manual_grade, manual_reviewed, manual_rows = load_manual_review(
        manual_path, case_id, frame_count, fps
    )
    manual_trace = manual_review_trace_arrays(manual_rows, frame_count, fps)
    posthoc_row = load_posthoc_confirmation(posthoc_path, case_id)
    qc = tracking_artifact_qc_v2.build_artifact_qc(
        global_translation_step_px=translation,
        global_rotation_step_deg=rotation,
        pair_pcc=pair_pcc,
        pair_fb_error_px=pair_fb,
        pair_valid=tracking["radial_pair_valid"],
        p3_curve_distance_px=p3_distance,
        p3_topology_risk=topology,
        radial_local_residual=tracking["radial_local_residual"],
        raw_rsr=tracking["radial_strain_rate_s"],
        section_count=sections,
        manual_minimum_grade=manual_grade,
        manual_reviewed_automatic_grade3=manual_reviewed,
        manual_review_decision_code=manual_trace["manual_review_decision_code"],
        config=qc_config,
    )
    posthoc_confirmed = np.zeros(frame_count, dtype=bool)
    if (
        posthoc_row is not None
        and posthoc_row["confirm_all_current_automatic_grade3"].strip().lower()
        == "true"
    ):
        posthoc_confirmed = qc["automatic_artifact_grade"] >= 3
        reviewed_manual_grade = manual_grade.copy()
        reviewed_manual_grade[posthoc_confirmed] = np.maximum(
            reviewed_manual_grade[posthoc_confirmed], 3
        )
        reviewed_automatic_grade3 = manual_reviewed | posthoc_confirmed
        qc = tracking_artifact_qc_v2.build_artifact_qc(
            global_translation_step_px=translation,
            global_rotation_step_deg=rotation,
            pair_pcc=pair_pcc,
            pair_fb_error_px=pair_fb,
            pair_valid=tracking["radial_pair_valid"],
            p3_curve_distance_px=p3_distance,
            p3_topology_risk=topology,
            radial_local_residual=tracking["radial_local_residual"],
            raw_rsr=tracking["radial_strain_rate_s"],
            section_count=sections,
            manual_minimum_grade=reviewed_manual_grade,
            manual_reviewed_automatic_grade3=reviewed_automatic_grade3,
            manual_review_decision_code=manual_trace["manual_review_decision_code"],
            config=qc_config,
        )
    qc["posthoc_confirmed_automatic_grade3"] = posthoc_confirmed
    qc.update(manual_trace)
    position_qc_path = p3_position_qc_dir / f"{case_id}_p3_anatomical_position_qc_v1.npz"
    position_qc_available = position_qc_path.is_file()
    expected_pair_shape = np.asarray(tracking["radial_strain_rate_s"]).shape
    manual_position_invalid = np.zeros(expected_pair_shape, dtype=bool)
    automatic_position_candidate = np.zeros(expected_pair_shape, dtype=bool)
    if position_qc_available:
        position_qc = load_npz(position_qc_path)
        manual_position_invalid = np.asarray(
            position_qc["manual_anatomical_position_invalid_pair"], dtype=bool
        )
        automatic_position_candidate = np.asarray(
            position_qc["automatic_anatomical_position_candidate"], dtype=bool
        )
        if manual_position_invalid.shape != expected_pair_shape:
            raise ValueError(f"P3 point-QC pair shape mismatch: {case_id}")
        if automatic_position_candidate.shape != expected_pair_shape:
            raise ValueError(f"P3 automatic candidate shape mismatch: {case_id}")
    qc["p3_position_qc_available"] = np.asarray(position_qc_available)
    qc["automatic_anatomical_position_candidate"] = automatic_position_candidate
    qc["manual_anatomical_position_invalid_pair"] = manual_position_invalid
    qc["qc_radial_strain_rate_grade3_blank_s"] = apply_manual_position_exclusion(
        qc["qc_radial_strain_rate_grade3_blank_s"], manual_position_invalid
    )
    boundary_transition = np.asarray(
        tracking.get(
            "automatic_boundary_correction_transition_risk",
            np.zeros(expected_pair_shape, dtype=bool),
        ),
        dtype=bool,
    )
    if boundary_transition.shape != expected_pair_shape:
        raise ValueError(f"P3 image-boundary transition shape mismatch: {case_id}")
    qc["image_boundary_correction_transition_risk"] = boundary_transition
    qc["qc_radial_strain_rate_with_boundary_transition_blank_s"] = apply_manual_position_exclusion(
        qc["qc_radial_strain_rate_grade3_blank_s"], boundary_transition
    )
    qc["image_boundary_transition_can_exclude_qc_rsr"] = np.asarray(False)
    qc["image_boundary_transition_is_sensitivity_branch"] = np.asarray(True)
    qc["automatic_position_candidates_can_exclude_rsr"] = np.asarray(False)
    qc["manual_confirmed_position_invalid_can_exclude_qc_rsr"] = np.asarray(True)
    pair_geometry_crossing = np.asarray(
        tracking.get(
            "radial_pair_any_crossing_risk",
            np.zeros(expected_pair_shape, dtype=bool),
        ),
        dtype=bool,
    )
    if pair_geometry_crossing.shape != expected_pair_shape:
        raise ValueError(f"radial pair crossing-risk shape mismatch: {case_id}")
    qc["radial_pair_geometry_crossing_risk"] = pair_geometry_crossing
    qc["geometry_crossing_can_exclude_qc_rsr"] = np.asarray(False)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{case_id}_artifact_qc_v2_1.npz",
        **qc,
        case_id=np.asarray(case_id),
        fps=np.asarray(fps, dtype=np.float32),
        thresholds_are_frozen=np.asarray(True),
        project_engineering_rules_frozen=np.asarray(False),
        clinical_or_literature_thresholds_frozen=np.asarray(False),
        independent_external_validation_complete=np.asarray(False),
        freeze_scope=np.asarray(str(freeze_metadata["freeze_scope"])),
        freeze_date=np.asarray(str(freeze_metadata["freeze_date"])),
        artifact_qc_version=np.asarray("2.1"),
        threshold_source=np.asarray("project_development_relative_plus_absolute_support_not_literature_cutoffs"),
        unreviewed_automatic_grade3_can_blank=np.asarray(False),
        posthoc_confirmation_applied=np.asarray(bool(np.any(posthoc_confirmed))),
        posthoc_confirmation_independent_for_validation=np.asarray(False),
        absolute_translation_step_px=np.asarray(
            qc_config.absolute_translation_step_px, dtype=np.float32
        ),
        absolute_rotation_step_deg=np.asarray(
            qc_config.absolute_rotation_step_deg, dtype=np.float32
        ),
        absolute_low_pcc=np.asarray(qc_config.absolute_low_pcc, dtype=np.float32),
        absolute_fb_error_p95_px=np.asarray(
            qc_config.absolute_fb_error_p95_px, dtype=np.float32
        ),
        absolute_invalid_fraction=np.asarray(
            qc_config.absolute_invalid_fraction, dtype=np.float32
        ),
        tracking_source=np.asarray("p3_assisted_anatomical_reference"),
        p3_role=np.asarray("per_frame_anatomical_localization_not_direct_deformation"),
        deformation_measurement_role=np.asarray("adjacent_frame_lk_radial_pair_length_change"),
        p3_coordinate_delta_used_as_rsr=np.asarray(False),
        p3_assisted_application_review_pending=np.asarray(
            bool(np.any(qc["automatic_grade3_pending_manual_review"]))
        ),
        surrounding_tissue_score_available=np.asarray(False),
        formal_propagation_released=np.asarray(False),
    )
    frame_rows = []
    for frame_idx in range(frame_count):
        grade = int(qc["artifact_grade"][frame_idx])
        frame_rows.append(
            {
                "病例": case_id,
                "帧": frame_idx,
                "时间_s": round(frame_idx / fps, 6),
                "自动伪影等级": int(qc["automatic_artifact_grade"][frame_idx]),
                "仅相对分数候选等级": int(qc["relative_only_artifact_candidate_grade"][frame_idx]),
                "伪影等级": grade,
                "等级说明": GRADE_NAMES[grade],
                "自动异常证据组数量": int(qc["automatic_evidence_group_count"][frame_idx]),
                "人工最低等级": int(qc["manual_minimum_grade"][frame_idx]),
                "人工复核动作代码_0无动作_1确认3级_2保留2级_3非伪影": int(
                    qc["manual_review_decision_code"][frame_idx]
                ),
                "人工复核是否已记录": bool(
                    qc["manual_review_recorded"][frame_idx]
                ),
                "人工复核人": str(qc["manual_review_reviewer"][frame_idx]),
                "人工复核日期": str(qc["manual_review_date"][frame_idx]),
                "自动3级是否已人工复核": bool(
                    qc["automatic_grade3_reviewed"][frame_idx]
                ),
                "绝对整体运动支持": bool(qc["absolute_motion_support"][frame_idx]),
                "绝对追踪质量支持": bool(qc["absolute_tracking_quality_support"][frame_idx]),
                "相邻3级候选支持": bool(qc["adjacent_grade3_support"][frame_idx]),
                "极端绝对异常支持": bool(qc["extreme_absolute_artifact_support"][frame_idx]),
                "自动3级是否待人工确认": bool(qc["automatic_grade3_pending_manual_review"][frame_idx]),
                "整体运动异常分数": round(float(qc["global_motion_anomaly_score"][frame_idx]), 6),
                "追踪质量异常分数": round(float(qc["tracking_quality_anomaly_score"][frame_idx]), 6),
                "同步共同运动异常分数": round(float(qc["synchronized_motion_anomaly_score"][frame_idx]), 6),
                "整体平移步长_px": float(qc["global_translation_step_px"][frame_idx]),
                "整体旋转步长_deg": float(qc["global_rotation_step_deg"][frame_idx]),
                "点对平均PCC": float(qc["pair_mean_pcc"][frame_idx]),
                "点对正反误差P95_px": float(qc["pair_fb_error_p95_px"][frame_idx]),
                "径向测量点与P3参考曲线距离P95_px_仅质控": float(qc["p3_curve_distance_p95_px"][frame_idx]),
                "P3拓扑风险_仅质控": bool(qc["p3_topology_risk"][frame_idx]),
                "P3局部位置自动候选点对数_不自动删除": int(
                    np.count_nonzero(automatic_position_candidate[frame_idx])
                ),
                "P3局部位置人工确认排除点对数": int(
                    np.count_nonzero(manual_position_invalid[frame_idx])
                ),
                "径向线交叉风险点对数_不自动删除": int(
                    np.count_nonzero(pair_geometry_crossing[frame_idx])
                ),
                "P3自动纠偏过渡风险点对数_仅敏感性": int(
                    np.count_nonzero(boundary_transition[frame_idx])
                ),
                "点对共同运动一致性": float(qc["pair_common_motion_coherence"][frame_idx]),
                "多点同时形变比例": float(qc["simultaneous_deformation_fraction"][frame_idx]),
                "质控后RSR是否有新增留空": bool(
                    np.any(
                        np.isfinite(qc["raw_radial_strain_rate_s"][frame_idx])
                        & ~np.isfinite(
                            qc["qc_radial_strain_rate_grade3_blank_s"][frame_idx]
                        )
                    )
                ),
                "3级是否整帧留空": bool(qc["grade3_blank_frame"][frame_idx]),
                "质控后RSR新增留空点对数": int(
                    np.count_nonzero(
                        np.isfinite(qc["raw_radial_strain_rate_s"][frame_idx])
                        & ~np.isfinite(
                            qc["qc_radial_strain_rate_grade3_blank_s"][frame_idx]
                        )
                    )
                ),
                "判定依据": reason_for_frame(qc, frame_idx),
            }
        )
    write_csv(output_dir / f"{case_id}_artifact_qc_frames.csv", frame_rows)
    interval_rows: list[dict[str, Any]] = []
    for interval in manual_rows:
        start_s = float(interval["start_s"])
        end_s = float(interval["end_s"])
        start, end = manual_interval_frame_bounds(interval, frame_count, fps)
        indices = np.arange(start, end + 1, dtype=np.int32)
        automatic = qc["automatic_artifact_grade"][indices]
        final = qc["artifact_grade"][indices]
        interval_rows.append(
            {
                "病例": case_id,
                "人工开始_s": start_s,
                "人工结束_s": end_s,
                "人工描述": interval["manual_description"],
                "时间确定性": interval["time_certainty"],
                "人工复核状态": interval.get("review_status", ""),
                "是否独立于当前自动规则": interval.get(
                    "independent_for_algorithm_validation", ""
                ),
                "是否叠加人工最低等级": interval["apply_to_qc"],
                "人工最低等级": interval["minimum_grade"],
                "区间帧数": len(indices),
                "自动0级帧数": int(np.count_nonzero(automatic == 0)),
                "自动1级帧数": int(np.count_nonzero(automatic == 1)),
                "自动2级帧数": int(np.count_nonzero(automatic == 2)),
                "自动3级帧数": int(np.count_nonzero(automatic == 3)),
                "自动2级及以上百分比": round(
                    100 * float(np.mean(automatic >= 2)), 4
                ),
                "自动3级百分比": round(100 * float(np.mean(automatic >= 3)), 4),
                "叠加人工后最高等级": int(np.max(final)),
                "说明": "自动列不包含人工覆盖，可用于检查算法是否真正命中",
            }
        )
    interval_path = output_dir / f"{case_id}_manual_interval_comparison.csv"
    if interval_rows:
        write_csv(interval_path, interval_rows)
    elif interval_path.exists():
        interval_path.unlink()
    video = PROJECT_ROOT / "输入" / "01_原始视频" / f"{case_id}.mp4"
    if not skip_video:
        frames, _ = tracking_lk.read_video_gray(
            video, float(tracking["resize_factor"])
        )
        write_review_video(
            output_dir / f"{case_id}_artifact_qc_review_cn.mp4",
            frames,
            tracking,
            p3,
            nodes,
            qc,
            sections,
            fps,
        )
    grade = qc["artifact_grade"]
    automatic_grade = qc["automatic_artifact_grade"]
    manual_grade3 = qc["manual_minimum_grade"] >= 3
    manual_grade2 = qc["manual_minimum_grade"] == 2
    independent_manual_grade3 = np.zeros(frame_count, dtype=bool)
    posthoc_confirmed_grade3 = np.asarray(
        qc["posthoc_confirmed_automatic_grade3"], dtype=bool
    ).copy()
    provisional_review_rows = 0
    for interval in manual_rows:
        review_status = interval.get("review_status", "").strip().lower()
        if "provisional" in review_status or "pending" in review_status:
            provisional_review_rows += 1
        if interval["apply_to_qc"].strip().lower() != "true":
            continue
        minimum_grade = interval.get("minimum_grade", "").strip()
        if not minimum_grade or int(minimum_grade) < 3:
            continue
        start, end = manual_interval_frame_bounds(interval, frame_count, fps)
        independent = (
            interval.get("independent_for_algorithm_validation", "")
            .strip()
            .lower()
            == "true"
        )
        target = independent_manual_grade3 if independent else posthoc_confirmed_grade3
        target[start : end + 1] = True
    grade3_segments = contiguous_segments(grade >= 3, fps)
    grade2plus_segments = contiguous_segments(grade >= 2, fps)
    raw_rsr = qc["raw_radial_strain_rate_s"]
    qc_rsr = qc["qc_radial_strain_rate_grade3_blank_s"]
    formal_new_blank = np.isfinite(raw_rsr) & ~np.isfinite(qc_rsr)
    formal_new_blank_frame = np.any(formal_new_blank, axis=(1, 2))
    jointly_finite = np.isfinite(raw_rsr) & np.isfinite(qc_rsr)
    shared_max_difference = (
        float(np.max(np.abs(raw_rsr[jointly_finite] - qc_rsr[jointly_finite])))
        if np.any(jointly_finite)
        else math.nan
    )
    summary = {
        "病例": case_id,
        "总帧数": frame_count,
        "0级百分比": round(100 * float(np.mean(grade == 0)), 4),
        "1级百分比": round(100 * float(np.mean(grade == 1)), 4),
        "2级百分比": round(100 * float(np.mean(grade == 2)), 4),
        "3级百分比": round(100 * float(np.mean(grade == 3)), 4),
        "3级留空帧数": int(np.count_nonzero(grade >= 3)),
        "自动3级候选帧数": int(np.count_nonzero(automatic_grade >= 3)),
        "自动3级待人工确认帧数": int(
            np.count_nonzero(qc["automatic_grade3_pending_manual_review"])
        ),
        "自动3级待人工确认且RSR保留帧数": int(
            np.count_nonzero(
                qc["automatic_grade3_pending_manual_review"]
                & ~formal_new_blank_frame
            )
        ),
        "自动3级待人工确认且原有限值完整保留帧数": int(
            np.count_nonzero(
                qc["automatic_grade3_pending_manual_review"]
                & ~formal_new_blank_frame
            )
        ),
        "3级时段_s": "；".join(f"{start:.3f}-{end:.3f}" for start, end in grade3_segments),
        "2级及以上时段_s": "；".join(f"{start:.3f}-{end:.3f}" for start, end in grade2plus_segments),
        "原始RSR有限百分比": round(100 * float(np.mean(np.isfinite(raw_rsr))), 4),
        "质控后RSR有限百分比": round(100 * float(np.mean(np.isfinite(qc_rsr))), 4),
        "原始与质控共同有效值最大差异_1_s": shared_max_difference,
        "质控RSR是否插值": False,
        "人工时段数量": len(manual_rows),
        "全部人工确认3级帧数": int(np.count_nonzero(manual_grade3)),
        "独立于当前算法的人工3级帧数": int(
            np.count_nonzero(independent_manual_grade3)
        ),
        "独立人工3级帧自动判为3级百分比": (
            round(
                100
                * float(
                    np.mean(automatic_grade[independent_manual_grade3] >= 3)
                ),
                4,
            )
            if np.any(independent_manual_grade3)
            else ""
        ),
        "后验候选人工确认3级帧数_不用于算法召回率": int(
            np.count_nonzero(posthoc_confirmed_grade3)
        ),
        "后验确认新增3级帧数_排除独立人工重叠": int(
            np.count_nonzero(
                posthoc_confirmed_grade3 & ~independent_manual_grade3
            )
        ),
        "后验确认自动3级时段_s": "；".join(
            f"{start:.3f}-{end:.3f}"
            for start, end in contiguous_segments(posthoc_confirmed_grade3, fps)
        ),
        "暂定支持但尚未应用的复核时段数量": provisional_review_rows,
        "人工2级帧数": int(np.count_nonzero(manual_grade2)),
        "218后验复核confirmed_grade3帧数": int(
            np.count_nonzero(qc["manual_review_decision_code"] == 1)
        ),
        "218后验复核keep_grade2帧数": int(
            np.count_nonzero(qc["manual_review_decision_code"] == 2)
        ),
        "218后验复核not_artifact帧数": int(
            np.count_nonzero(qc["manual_review_decision_code"] == 3)
        ),
        "人工2级帧自动判为2级及以上百分比": (
            round(100 * float(np.mean(automatic_grade[manual_grade2] >= 2)), 4)
            if np.any(manual_grade2)
            else ""
        ),
        "P3用途": "逐帧贴壁定位参考；P3坐标差不直接进入RSR",
        "周围组织独立分数是否可用": False,
        "P3逐点解剖位置质控是否接入": bool(position_qc_available),
        "P3局部位置自动候选点对数_不自动删除": int(
            np.count_nonzero(automatic_position_candidate)
        ),
        "P3局部位置人工确认排除点对帧数": int(
            np.count_nonzero(manual_position_invalid)
        ),
        "径向线交叉风险点对帧数_不自动删除": int(
            np.count_nonzero(pair_geometry_crossing)
        ),
        "P3自动纠偏过渡风险点对帧数_仅敏感性": int(
            np.count_nonzero(boundary_transition)
        ),
        "当前项目工程规则v2.1是否冻结": False,
        "是否临床或文献阈值": False,
        "是否完成独立外部验证": False,
        "未复核自动3级是否能清空RSR": bool(
            np.any(qc["automatic_grade3_pending_manual_review"] & formal_new_blank_frame)
        ),
        "正式传播是否释放": False,
    }
    write_csv(output_dir / f"{case_id}_artifact_qc_summary.csv", [summary])
    return summary, interval_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Five-case P3-assisted radial LK artifact QC"
    )
    parser.add_argument("--cases", nargs="*", choices=CASES, default=list(CASES))
    parser.add_argument("--tracking-dir", type=Path, default=DEFAULT_TRACKING_DIR)
    parser.add_argument(
        "--tracking-filename-template",
        default="{case_id}_径向点对原始数据.npz",
    )
    parser.add_argument("--p3-dir", type=Path, default=DEFAULT_P3_DIR)
    parser.add_argument(
        "--p3-position-qc-dir", type=Path, default=DEFAULT_P3_POSITION_QC_DIR
    )
    parser.add_argument("--manual-intervals", type=Path, default=DEFAULT_MANUAL_INTERVALS)
    parser.add_argument(
        "--posthoc-confirmations",
        type=Path,
        default=DEFAULT_POSTHOC_CONFIRMATIONS,
    )
    parser.add_argument(
        "--frozen-config", type=Path, default=DEFAULT_FROZEN_CONFIG
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    qc_config, freeze_metadata = load_frozen_qc_config(
        args.frozen_config.resolve()
    )
    results = [
        analyze_case(
            case,
            args.tracking_dir.resolve(),
            args.tracking_filename_template,
            args.p3_dir.resolve(),
            args.p3_position_qc_dir.resolve(),
            args.manual_intervals.resolve(),
            args.posthoc_confirmations.resolve(),
            qc_config,
            freeze_metadata,
            output,
            args.skip_video,
        )
        for case in args.cases
    ]
    summaries = [summary for summary, _ in results]
    interval_rows = [row for _, rows in results for row in rows]
    write_csv(output / "artifact_qc_v2_1_five_case_summary.csv", summaries)
    write_csv(output / "人工时段与自动检测对照.csv", interval_rows)
    with args.posthoc_confirmations.resolve().open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        posthoc_source_rows = [
            row
            for row in csv.DictReader(handle)
            if row["case_id"] in args.cases
        ]
    posthoc_source_by_case = {row["case_id"]: row for row in posthoc_source_rows}
    summary_by_case = {row["病例"]: row for row in summaries}
    current_review_rows = []
    for case_id in args.cases:
        source_row = posthoc_source_by_case[case_id]
        summary = summary_by_case[case_id]
        automatic_grade3 = int(summary["自动3级候选帧数"])
        pending_grade3 = int(summary["自动3级待人工确认帧数"])
        current_review_rows.append(
            {
                "case_id": case_id,
                "artifact_qc_version": source_row["artifact_qc_version"],
                "病例级全部确认开关": source_row[
                    "confirm_all_current_automatic_grade3"
                ],
                "当前自动3级帧数": automatic_grade3,
                "当前自动3级已复核帧数": automatic_grade3 - pending_grade3,
                "当前自动3级待复核帧数": pending_grade3,
                "当前复核是否完成": pending_grade3 == 0,
                "详细人工记录是否为权威来源": True,
                "是否独立于当前算法验证": False,
                "病例级开关原备注": source_row.get("note", ""),
            }
        )
    write_csv(output / "后验自动3级候选人工确认状态.csv", current_review_rows)
    (output / "README_中文结果说明.md").write_text(
        "# P3辅助贴壁定位主轨：抖动与伪影质控v2.1适配候选\n\n"
        "输入来自P3逐帧贴壁定位后的相邻帧LK径向测量。P3只用于放置每帧径向测量参考，"
        "P3坐标自身的变化不直接进入RSR公式。\n\n"
        "本结果联合整体运动、追踪质量、点对同步共同运动三组证据，不用单一指标判定抖动。\n\n"
        "v2.1在逐病例相对异常分数之外增加绝对量门槛和相邻帧支持，避免稳定病例的小波动被放大。"
        "0级无明显组合证据；1级只有轻微或单组异常；2级为疑似伪影或未复核自动3级候选，"
        "仍保留ROI局部形变但附风险；只有人工确认的3级明显抖动或离面风险才整帧留空。"
        "人工确认的P3局部位置错误只排除对应点对。原始RSR始终完整保留。\n\n"
        "逐帧表同时保存“自动伪影等级”和叠加人工最低等级后的“伪影等级”，"
        "因此可以检查程序是否真的识别了人工已知时段，不能用人工覆盖结果冒充自动命中。\n\n"
        "人工时段与自动结果另见《人工时段与自动检测对照.csv》。其中“是否独立于当前自动规则=False”"
        "表示用户看过v2.1候选后作出的后验确认，只能决定是否清空质控RSR，不能用于计算自动检出率。\n\n"
        "P3用于逐帧解剖定位及传递拓扑风险；真正的局部径向变化由相邻帧LK测量。"
        "恢复显示坐标可在人工确认贴壁后用于定位，但不能据此宣称始终追踪同一组织点。"
        "当前没有独立周围组织ROI，因此同步分数只是径向点对共同运动的代理，不能写成真正周围组织分数。\n\n"
        "P3逐点解剖位置质控已经接入：自动候选只提示复核，不自动删除；只有人工CSV中confirmed且"
        "apply_to_qc=true的具体壁段和帧才从质控RSR排除，原始RSR不覆盖。\n\n"
        "P3自动图像边界纠偏的过渡风险只保存为独立敏感性分支"
        "qc_radial_strain_rate_with_boundary_transition_blank_s，不参与正式QC置空。\n\n"
        "本次沿用既有v2.1数值门槛，但输入追踪主线已经从no-P3切换到P3辅助定位，"
        "因此当前属于适配候选。是否仍有未复核自动3级应以五病例汇总和逐帧表为准；"
        "未复核候选不能自动清空RSR。它不是Huang论文或临床阈值，"
        "也未完成独立外部验证。正式传播继续关闭。\n",
        encoding="utf-8",
    )
    print(f"complete: {output}")


if __name__ == "__main__":
    main()
