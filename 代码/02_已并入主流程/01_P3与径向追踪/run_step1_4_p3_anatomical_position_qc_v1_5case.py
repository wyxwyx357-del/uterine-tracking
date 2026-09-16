#!/usr/bin/env python3
"""Five-case point-level review of intermittent P3 anatomical misplacement."""

from __future__ import annotations

from peristalsis_pipeline.project_layout import output_dir

import argparse
import csv
import json
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

from peristalsis_pipeline import tracking_lk
from peristalsis_pipeline.p3_anatomical_position_qc import (
    P3AnatomicalPositionQCConfig,
    build_p3_anatomical_position_qc,
)


CASES = (
    "CASE_001",
    "CASE_002",
    "CASE_003",
    "CASE_004",
    "CASE_005",
)
DEFAULT_TRACKING_DIR = output_dir(PROJECT_ROOT, "step1_4_p3_assisted_radial_tracking_v1_5case")
DEFAULT_P3_DIR = PROJECT_ROOT / "输入" / "04_冻结基线" / "step1_4_five_anchor"
DEFAULT_MANUAL_CSV = PROJECT_ROOT / "输入" / "02_人工标签" / "03_质控人工记录" / "P3局部解剖位置人工复核.csv"
DEFAULT_OUTPUT = output_dir(PROJECT_ROOT, "step1_4_p3_anatomical_position_qc_v1_5case")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def wall_ids(nodes: list[dict[str, Any]], role: str) -> np.ndarray:
    rows = [
        (int(node["section_order"]), int(node["point_id"]))
        for node in nodes
        if node["rail_name"] == role
    ]
    return np.asarray([point_id for _, point_id in sorted(rows)], dtype=np.int32)


def frame_bounds(row: dict[str, str], frame_count: int, fps: float) -> tuple[int, int]:
    if row.get("start_frame", "").strip() and row.get("end_frame", "").strip():
        start = int(row["start_frame"])
        end = int(row["end_frame"])
    else:
        start = int(np.floor(float(row["start_s"]) * fps))
        end = int(np.ceil(float(row["end_s"]) * fps))
    return max(0, min(start, frame_count - 1)), max(0, min(end, frame_count - 1))


def load_manual_pair_mask(
    path: Path,
    case_id: str,
    frame_count: int,
    section_count: int,
    fps: float,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    mask = np.zeros((frame_count, 2, section_count), dtype=bool)
    if not path.is_file():
        return mask, []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("case_id") == case_id]
    applied = []
    side_names = {"anterior": 0, "front": 0, "前壁": 0, "posterior": 1, "back": 1, "后壁": 1}
    for row in rows:
        if row.get("apply_to_qc", "").strip().lower() != "true":
            continue
        if row.get("review_status", "").strip().lower() != "confirmed":
            continue
        side_text = row.get("side", "").strip().lower()
        if side_text not in side_names:
            raise ValueError(f"Unsupported side in {path}: {row.get('side')}")
        side = side_names[side_text]
        section_start = int(row["section_start_1based"]) - 1
        section_end = int(row["section_end_1based"]) - 1
        if not (0 <= section_start <= section_end < section_count):
            raise ValueError(f"Section range outside 1..{section_count}: {row}")
        start, end = frame_bounds(row, frame_count, fps)
        if end < start:
            raise ValueError(f"End precedes start: {row}")
        mask[start : end + 1, side, section_start : section_end + 1] = True
        applied.append(row)
    return mask, applied


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def add_text(image: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((3, 3, image.shape[1] - 3, 70), fill=(0, 0, 0, 160))
    text_font = font(15)
    for row, line in enumerate(lines):
        draw.text((8, 5 + row * 20), line, font=text_font, fill="white")
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def write_review_video(
    path: Path,
    frames: list[np.ndarray],
    tracking: dict[str, np.ndarray],
    result: dict[str, np.ndarray],
    manual_mask: np.ndarray,
    fps: float,
    sections: int,
) -> None:
    height, width = frames[0].shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"Cannot create review video: {path}")
    reference = tracking["radial_reference"]
    measured = tracking["radial_lk_measured_p3_initial"]
    automatic = result["automatic_anatomical_position_candidate"]
    suggested = result["suggested_wall_position_review_only"]
    try:
        for frame_idx, gray in enumerate(frames):
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            points = reference[frame_idx] if frame_idx == 0 else measured[frame_idx]
            for side, (inner_rail, outer_rail) in enumerate(((0, 1), (2, 3))):
                inner_slice = slice(inner_rail * sections, (inner_rail + 1) * sections)
                outer_slice = slice(outer_rail * sections, (outer_rail + 1) * sections)
                p3_inner = reference[frame_idx, inner_slice]
                cv2.polylines(image, [np.round(p3_inner).astype(np.int32)], False, (255, 255, 0), 1, cv2.LINE_AA)
                for section in range(sections):
                    inner = points[inner_slice][section]
                    outer = points[outer_slice][section]
                    if not np.all(np.isfinite((inner, outer))):
                        continue
                    auto = bool(automatic[frame_idx, side, section])
                    manual = bool(manual_mask[frame_idx, side, section])
                    color = (255, 0, 255) if manual else ((0, 0, 255) if auto else (0, 140, 255))
                    inner_xy = tuple(np.round(inner).astype(int))
                    outer_xy = tuple(np.round(outer).astype(int))
                    cv2.line(image, inner_xy, outer_xy, color, 1, cv2.LINE_AA)
                    cv2.circle(image, inner_xy, 2, (0, 235, 255), -1, cv2.LINE_AA)
                    cv2.circle(image, outer_xy, 2, (0, 220, 0), -1, cv2.LINE_AA)
                    cv2.putText(image, str(section + 1), (inner_xy[0] + 2, inner_xy[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
                    proposal = suggested[frame_idx, side, section]
                    if np.all(np.isfinite(proposal)):
                        proposal_xy = tuple(np.round(proposal).astype(int))
                        cv2.drawMarker(image, proposal_xy, (255, 255, 255), cv2.MARKER_CROSS, 7, 1)
            image = add_text(
                image,
                [
                    f"P3局部位置质控｜帧 {frame_idx}/{len(frames)-1}｜时间 {frame_idx/fps:.3f}秒",
                    f"自动候选点对 {int(np.count_nonzero(automatic[frame_idx]))}｜人工排除点对 {int(np.count_nonzero(manual_mask[frame_idx]))}",
                    "数字=壁段编号　红=自动候选　紫=人工确认排除　白十字=仅供复核的纠偏建议",
                ],
            )
            writer.write(image)
    finally:
        writer.release()


def analyze_case(
    case_id: str,
    tracking_dir: Path,
    tracking_filename_template: str,
    p3_dir: Path,
    manual_csv: Path,
    output_dir: Path,
    config: P3AnatomicalPositionQCConfig,
    skip_video: bool,
) -> dict[str, Any]:
    tracking = load_npz(
        tracking_dir / tracking_filename_template.format(case_id=case_id)
    )
    p3 = load_npz(p3_dir / f"{case_id}_step1_4_5a_sparse_anchor_tracks.npz")
    nodes = json.loads((p3_dir / f"{case_id}_step1_4_5a_nodes.json").read_text(encoding="utf-8"))
    anterior = wall_ids(nodes, "anterior_wall")
    posterior = wall_ids(nodes, "posterior_wall")
    if len(anterior) != len(posterior):
        raise ValueError(f"Wall section count mismatch: {case_id}")
    ids = np.stack((anterior, posterior))
    if "p3_wall_tracks_image_corrected" in tracking:
        tracks = np.asarray(
            tracking["p3_wall_tracks_image_corrected"], dtype=np.float32
        )
        if tracks.shape[1:3] != ids.shape:
            raise ValueError(
                f"Corrected wall shape {tracks.shape[1:3]} does not match P3 ids {ids.shape}"
            )
    else:
        tracks = np.stack(
            (p3["tracks"][:, anterior], p3["tracks"][:, posterior]), axis=1
        )
    branch = np.stack((p3["branch_normal_disagreement_px"][:, anterior], p3["branch_normal_disagreement_px"][:, posterior]), axis=1)
    recovered = np.stack((p3["recovered_display_only"][:, anterior], p3["recovered_display_only"][:, posterior]), axis=1)
    topology = (p3["topology_bad_edge_count"] > 0) | (p3["topology_bad_cell_count"] > 0)
    result = build_p3_anatomical_position_qc(tracks, branch, topology, recovered, config)
    fps = float(tracking["fps"])
    manual_mask, manual_rows = load_manual_pair_mask(manual_csv, case_id, len(tracks), len(anterior), fps)
    result["manual_anatomical_position_invalid_pair"] = manual_mask
    result["qc_pair_exclusion_applied"] = manual_mask.copy()
    result["wall_point_ids"] = ids
    np.savez_compressed(
        output_dir / f"{case_id}_p3_anatomical_position_qc_v1.npz",
        **result,
        case_id=np.asarray(case_id),
        fps=np.asarray(fps, dtype=np.float32),
        automatic_candidates_can_exclude_rsr=np.asarray(False),
        manual_confirmed_points_can_exclude_qc_rsr=np.asarray(True),
        thresholds_are_project_development_only=np.asarray(True),
        formal_propagation_released=np.asarray(False),
    )
    candidate_rows = []
    automatic = result["automatic_anatomical_position_candidate"]
    for frame_idx, side, section in np.argwhere(automatic):
        candidate_rows.append(
            {
                "病例": case_id,
                "帧": int(frame_idx),
                "时间_s": round(float(frame_idx) / fps, 6),
                "侧别": "前壁" if side == 0 else "后壁",
                "壁段编号_从1开始": int(section + 1),
                "局部运动残差_px": float(result["p3_local_motion_residual_px"][frame_idx, side, section]),
                "空间曲线偏离变化_px": float(result["p3_spatial_curve_change_px"][frame_idx, side, section]),
                "正反法向差异_px": float(result["p3_branch_normal_disagreement_px"][frame_idx, side, section]),
                "P3拓扑风险帧": bool(topology[frame_idx]),
                "是否有纠偏建议": bool(result["suggested_position_available"][frame_idx, side, section]),
                "是否自动用于追踪或RSR": False,
                "人工复核结论": "待复核",
            }
        )
    write_csv(output_dir / f"{case_id}_p3局部位置自动候选.csv", candidate_rows)
    if not skip_video:
        video = PROJECT_ROOT / "输入" / "01_原始视频" / f"{case_id}.mp4"
        frames, _ = tracking_lk.read_video_gray(video, float(tracking["resize_factor"]))
        write_review_video(output_dir / f"{case_id}_p3局部位置复核.mp4", frames, tracking, result, manual_mask, fps, len(anterior))
    return {
        "病例": case_id,
        "每侧壁段数量": len(anterior),
        "自动局部位置候选点对数": int(np.count_nonzero(automatic)),
        "自动候选涉及帧数": int(np.count_nonzero(np.any(automatic, axis=(1, 2)))),
        "仅供复核纠偏建议数": int(np.count_nonzero(result["suggested_position_available"])),
        "人工确认排除点对帧数": int(np.count_nonzero(manual_mask)),
        "人工确认记录数": len(manual_rows),
        "自动候选是否自动修改P3": False,
        "自动候选是否自动排除RSR": False,
        "人工确认异常是否从质控RSR排除": True,
        "首尾壁段是否支持自动候选": False,
        "正式传播是否释放": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", choices=CASES, default=list(CASES))
    parser.add_argument("--tracking-dir", type=Path, default=DEFAULT_TRACKING_DIR)
    parser.add_argument(
        "--tracking-filename-template",
        default="{case_id}_径向点对原始数据.npz",
    )
    parser.add_argument("--p3-dir", type=Path, default=DEFAULT_P3_DIR)
    parser.add_argument("--manual-csv", type=Path, default=DEFAULT_MANUAL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = P3AnatomicalPositionQCConfig()
    summaries = [
        analyze_case(
            case,
            args.tracking_dir.resolve(),
            args.tracking_filename_template,
            args.p3_dir.resolve(),
            args.manual_csv.resolve(),
            output,
            config,
            args.skip_video,
        )
        for case in args.cases
    ]
    write_csv(output / "p3_anatomical_position_qc_five_case_summary.csv", summaries)
    (output / "README_中文结果说明.md").write_text(
        "# P3局部解剖位置质控v1\n\n"
        "自动候选要求局部运动、同侧曲线连续性及P3正反/拓扑质量多项证据同时异常。"
        "它只用于定位人工复核，不会自动修改P3，也不会自动删除RSR。\n\n"
        "复核视频中的数字是从1开始的壁段编号；红色为自动候选，紫色为人工CSV确认排除，"
        "白色十字为仅供比较的纠偏建议。人工确认异常后填写"
        "输入/02_人工标签/03_质控人工记录/P3局部解剖位置人工复核.csv，"
        "再次运行本步骤和伪影质控，只有确认记录会从质控RSR排除。原始P3和原始RSR始终保留。\n"
        "首尾壁段因为各自缺少一侧相邻壁点，不进入自动候选，必须人工查看。\n",
        encoding="utf-8",
    )
    print(f"complete: {output}")


if __name__ == "__main__":
    main()
