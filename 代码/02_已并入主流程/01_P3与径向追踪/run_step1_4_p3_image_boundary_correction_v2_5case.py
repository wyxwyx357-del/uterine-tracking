#!/usr/bin/env python3
"""Detect image-supported P3 wall offsets and build a separate correction branch."""

from __future__ import annotations

from peristalsis_pipeline.project_layout import output_dir

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "代码" / "01_底层算法"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from peristalsis_pipeline import (
    tracking_huang_fusion,
    tracking_huang_radial_pairs,
    tracking_huang_wall_fusion,
    tracking_lk,
)
from peristalsis_pipeline.p3_image_boundary_correction import (
    P3ImageBoundaryConfig,
    apply_confirmed_boundary_overrides,
    apply_tracking_quality_gate,
    build_automatic_boundary_correction,
    extract_wall_boundary_evidence,
)


CASES = (
    "CASE_001",
    "CASE_002",
    "CASE_003",
    "CASE_004",
    "CASE_005",
)
BASELINE_ROOT = PROJECT_ROOT / "输入" / "04_冻结基线" / "step1_4_five_anchor"
ARTIFACT_ROOT = (
    output_dir(PROJECT_ROOT, "step1_4_p3_assisted_artifact_qc_v2_1_5case")
)
DEFAULT_OUTPUT = (
    output_dir(PROJECT_ROOT, "step1_4_p3_image_boundary_correction_v2_5case")
)
TRACKING_ROOT = (
    output_dir(PROJECT_ROOT, "step1_4_p3_assisted_radial_tracking_v1_5case")
)
MANUAL_REVIEW_CSV = PROJECT_ROOT / "输入" / "02_人工标签" / "03_质控人工记录" / "P3局部解剖位置人工复核.csv"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def load_wall_tracks(case_id: str) -> tuple[np.ndarray, float, float, list[np.ndarray]]:
    baseline = load_npz(BASELINE_ROOT / f"{case_id}_step1_4_5a_sparse_anchor_tracks.npz")
    nodes = json.loads(
        (BASELINE_ROOT / f"{case_id}_step1_4_5a_nodes.json").read_text(encoding="utf-8")
    )
    anterior, posterior, _ = tracking_huang_fusion.paired_wall_indices(nodes)
    tracks = np.asarray(baseline["tracks"], dtype=np.float32)
    wall_tracks = np.stack((tracks[:, anterior], tracks[:, posterior]), axis=1)
    meta_path = (
        PROJECT_ROOT / "输入" / "03_上游解剖节点"
        / "step1_3_middle_wall_band_tracking_points_smoke"
        / case_id
        / f"{case_id}_tracking_points.json"
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    resize_factor = float(meta["parameters"]["resize_factor"])
    video_path = (
        PROJECT_ROOT / "输入" / "01_原始视频"
        / f"{case_id}.mp4"
    )
    frames, video_info = tracking_lk.read_video_gray(video_path, resize_factor)
    if len(frames) != len(wall_tracks):
        raise ValueError(
            f"video/P3 frame count mismatch for {case_id}: "
            f"video={len(frames)}, P3={len(wall_tracks)}"
        )
    return wall_tracks, float(video_info["fps"]), resize_factor, frames


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_manual_confirmed_mask(
    case_id: str, frame_count: int, section_count: int, fps: float
) -> np.ndarray:
    mask = np.zeros((frame_count, 2, section_count), dtype=bool)
    if not MANUAL_REVIEW_CSV.is_file():
        return mask
    with MANUAL_REVIEW_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("case_id") == case_id]
    side_map = {"前壁": 0, "anterior": 0, "后壁": 1, "posterior": 1}
    for row in rows:
        if row.get("apply_to_qc", "").strip().lower() != "true":
            continue
        if row.get("review_status", "").strip().lower() != "confirmed":
            continue
        side_text = row.get("side", "").strip().lower()
        if side_text not in side_map:
            raise ValueError(f"人工复核表侧别无法识别：{row.get('side')}")
        side = side_map[side_text]
        start_section = int(row["section_start_1based"]) - 1
        stop_section = int(row["section_end_1based"])
        if not (0 <= start_section < stop_section <= section_count):
            raise ValueError(f"人工复核表壁段超出范围：{row}")
        if row.get("start_frame", "").strip() and row.get("end_frame", "").strip():
            start = int(row["start_frame"])
            stop = int(row["end_frame"])
        else:
            start = int(np.floor(float(row["start_s"]) * fps))
            stop = int(np.ceil(float(row["end_s"]) * fps))
        start = max(0, min(start, frame_count - 1))
        stop = max(0, min(stop, frame_count - 1))
        mask[start : stop + 1, side, start_section:stop_section] = True
    return mask


def finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if len(finite) else float("nan")


def write_review_video(
    path: Path,
    frames: list[np.ndarray],
    correction: dict[str, np.ndarray],
    tracking: dict[str, np.ndarray],
    fps: float,
) -> None:
    height, width = frames[0].shape
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise OSError(f"无法创建复核视频：{path}")
    original = correction["p3_wall_tracks_original"]
    corrected = correction["p3_wall_tracks_image_corrected"]
    automatic = correction["automatic_boundary_correction_mask"]
    manual_applied = correction["manual_confirmed_boundary_correction_mask"]
    candidate = correction["image_boundary_review_candidate"]
    reference = tracking["radial_reference"]
    measured = tracking["radial_lk_measured_p3_initial"]
    pair_valid = tracking["radial_pair_valid"]
    sections = original.shape[2]
    try:
        for frame_idx, gray in enumerate(frames):
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for side, (inner_rail, outer_rail) in enumerate(((0, 1), (2, 3))):
                inner_slice = slice(inner_rail * sections, (inner_rail + 1) * sections)
                outer_slice = slice(outer_rail * sections, (outer_rail + 1) * sections)
                for section in range(sections):
                    old = original[frame_idx, side, section]
                    new = corrected[frame_idx, side, section]
                    if not np.all(np.isfinite((old, new))):
                        continue
                    old_xy = tuple(np.round(old).astype(int))
                    new_xy = tuple(np.round(new).astype(int))
                    cv2.circle(image, old_xy, 2, (255, 255, 0), -1, cv2.LINE_AA)
                    if automatic[frame_idx, side, section]:
                        cv2.arrowedLine(
                            image, old_xy, new_xy, (255, 0, 255), 2, cv2.LINE_AA, tipLength=0.3
                        )
                    elif manual_applied[frame_idx, side, section]:
                        cv2.arrowedLine(
                            image, old_xy, new_xy, (180, 0, 180), 2, cv2.LINE_AA, tipLength=0.3
                        )
                    elif candidate[frame_idx, side, section]:
                        cv2.circle(image, old_xy, 5, (0, 80, 255), 1, cv2.LINE_AA)
                    inner = measured[frame_idx, inner_slice][section]
                    outer = measured[frame_idx, outer_slice][section]
                    if frame_idx == 0 or not pair_valid[frame_idx, side, section]:
                        inner = reference[frame_idx, inner_slice][section]
                        outer = reference[frame_idx, outer_slice][section]
                    inner_xy = tuple(np.round(inner).astype(int))
                    outer_xy = tuple(np.round(outer).astype(int))
                    cv2.line(image, inner_xy, outer_xy, (0, 150, 255), 1, cv2.LINE_AA)
                    cv2.circle(image, inner_xy, 2, (0, 235, 255), -1, cv2.LINE_AA)
                    cv2.circle(image, outer_xy, 2, (0, 220, 0), -1, cv2.LINE_AA)
                    cv2.putText(
                        image,
                        str(section + 1),
                        (inner_xy[0] + 2, inner_xy[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.28,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
            moved = int(np.count_nonzero(automatic[frame_idx]))
            manual_moved = int(np.count_nonzero(manual_applied[frame_idx]))
            review = int(np.count_nonzero(candidate[frame_idx] & ~automatic[frame_idx]))
            cv2.rectangle(image, (3, 3), (width - 3, 57), (0, 0, 0), -1)
            cv2.putText(
                image,
                f"P3 image-boundary correction | frame {frame_idx} | {frame_idx/fps:.3f}s",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"auto={moved} manual={manual_moved} review={review} | cyan=original arrows=move",
                (8, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(image)
    finally:
        writer.release()


def analyze(
    case_id: str,
    output: Path,
    config: P3ImageBoundaryConfig,
    skip_video: bool,
) -> dict[str, object]:
    print(f"[{case_id}] 读取视频并提取P3法向灰度边界", flush=True)
    tracks, fps, resize_factor, frames = load_wall_tracks(case_id)
    evidence = extract_wall_boundary_evidence(frames, tracks, config)
    artifact_path = ARTIFACT_ROOT / f"{case_id}_artifact_qc_v2_1.npz"
    artifact = load_npz(artifact_path)
    grade = np.asarray(artifact["artifact_grade"][: len(tracks)], dtype=np.int8)
    automatic_grade = np.asarray(
        artifact["automatic_artifact_grade"][: len(tracks)], dtype=np.int8
    )
    automatic_grade3_exclusion = automatic_grade >= 3
    correction_exclusion_grade = np.maximum(grade, automatic_grade)
    stable = correction_exclusion_grade < 3
    correction = build_automatic_boundary_correction(
        tracks,
        evidence,
        grade,
        fps,
        config,
        additional_excluded_frames=automatic_grade3_exclusion,
    )
    tracking_matches = sorted(TRACKING_ROOT.glob(f"{case_id}_*原始数据.npz"))
    if len(tracking_matches) != 1:
        raise RuntimeError(
            f"expected exactly one base tracking archive for {case_id}, "
            f"found {len(tracking_matches)}: {tracking_matches}"
        )
    original_tracking_path = tracking_matches[0]
    original_tracking = load_npz(original_tracking_path)
    global_transform = np.asarray(
        original_tracking["pairwise_global_transform"][: len(tracks)], dtype=np.float32
    )

    def measure_wall_tracks(
        wall_tracks: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        wall_flat = np.concatenate((wall_tracks[:, 0], wall_tracks[:, 1]), axis=1)
        measured_tracking = tracking_huang_radial_pairs.measure_pairwise_radial_points(
            frames,
            wall_flat,
            global_transform,
            anterior_count=tracks.shape[2],
            offset_px=float(original_tracking["radial_offset_px"]),
            config=tracking_huang_wall_fusion.HuangWallFusionConfig(),
        )
        measured_deformation = (
            tracking_huang_radial_pairs.radial_deformation_from_tracking(
                measured_tracking, anterior_count=tracks.shape[2], fps=fps
            )
        )
        return measured_tracking, measured_deformation

    candidate_tracking, candidate_deformation = measure_wall_tracks(
        correction["p3_wall_tracks_image_corrected"]
    )
    correction = apply_tracking_quality_gate(
        correction,
        evidence,
        {**candidate_tracking, **candidate_deformation},
        original_tracking,
        config,
    )
    manual_mask = load_manual_confirmed_mask(
        case_id, len(tracks), tracks.shape[2], fps
    )
    correction = apply_confirmed_boundary_overrides(
        correction,
        evidence,
        manual_mask,
        correction_exclusion_grade,
        maximum_correction_px=config.maximum_correction_px,
    )
    corrected_tracks = correction["p3_wall_tracks_image_corrected"]
    tracking, deformation = measure_wall_tracks(corrected_tracks)
    corrected_rsr = deformation["radial_strain_rate_s"]
    qc_corrected_rsr = corrected_rsr.copy()
    qc_corrected_rsr[grade >= 3] = np.nan
    transition = correction["automatic_boundary_correction_transition_risk"]
    qc_corrected_rsr[transition] = np.nan
    original_rsr = np.asarray(
        original_tracking["radial_strain_rate_s"][: len(tracks)], dtype=np.float32
    )
    rsr_difference = corrected_rsr - original_rsr
    rsr_affected = (
        correction["automatic_boundary_correction_mask"]
        | correction["manual_confirmed_boundary_correction_mask"]
    )
    rsr_affected = rsr_affected.copy()
    rsr_affected[1:] |= rsr_affected[:-1]
    archive = {**evidence, **correction, **tracking, **deformation}
    for key in (
        "pairwise_global_transform",
        "p3_midline_points",
        "p3_topology_risk_frame",
        "p3_topology_bad_edge_count",
        "p3_topology_bad_cell_count",
    ):
        if key in original_tracking:
            archive[key] = np.asarray(original_tracking[key])[: len(tracks)]
    archive.update(
        artifact_grade=grade,
        automatic_artifact_grade_before_boundary_correction=automatic_grade,
        boundary_correction_exclusion_grade=correction_exclusion_grade,
        qc_image_corrected_radial_strain_rate_s=qc_corrected_rsr,
        original_main_radial_strain_rate_s=original_rsr,
        image_correction_rsr_difference_s=rsr_difference,
        fps=np.asarray(fps, dtype=np.float32),
        resize_factor=np.asarray(resize_factor, dtype=np.float32),
        longitudinal_section_count=np.asarray(tracks.shape[2], dtype=np.int32),
        radial_offset_px=np.asarray(float(original_tracking["radial_offset_px"]), dtype=np.float32),
        p3_coordinate_delta_used_as_rsr=np.asarray(False),
        deformation_measurement_role=np.asarray(
            "adjacent_frame_lk_after_image_boundary_relocalization"
        ),
        original_main_outputs_overwritten=np.asarray(False),
        formal_propagation_released=np.asarray(False),
    )
    np.savez_compressed(
        output / f"{case_id}_p3_image_boundary_evidence_v2.npz", **archive
    )
    if not skip_video:
        write_review_video(
            output / f"{case_id}_P3自动贴壁纠偏复核视频.mp4",
            frames,
            correction,
            {**tracking, **deformation},
            fps,
        )
    offset = evidence["boundary_edge_offset_px"]
    supported = evidence["boundary_edge_supported"] & stable[:, None, None]
    auto = correction["automatic_boundary_correction_mask"]
    manual_applied = correction["manual_confirmed_boundary_correction_mask"]
    review = correction["image_boundary_review_candidate"]
    quality_accepted = correction[
        "tracking_quality_accepted_boundary_correction_candidate_mask"
    ]
    quality_rejected = correction[
        "tracking_quality_rejected_boundary_correction_candidate_mask"
    ]
    affected = auto | manual_applied
    affected_frames = np.flatnonzero(np.any(affected, axis=(1, 2)))
    affected_rsr_p95 = finite_percentile(np.abs(rsr_difference[rsr_affected]), 95)
    affected_rsr_max = finite_percentile(np.abs(rsr_difference[rsr_affected]), 100)
    corrected_pcc = finite_percentile(deformation["radial_pair_pcc"][1:], 50)
    original_pcc = finite_percentile(original_tracking["radial_pair_pcc"][1:], 50)
    return {
        "病例": case_id,
        "帧数": len(frames),
        "每侧壁段数": tracks.shape[2],
        "稳定帧百分比": round(100 * float(np.mean(stable)), 4),
        "图像边界证据支持点帧百分比": round(100 * float(np.mean(supported)), 4),
        "支持证据的法向偏移中位数_px": round(float(np.nanmedian(offset[supported])), 4)
        if np.any(supported)
        else "",
        "图像边界待复核点帧数": int(np.count_nonzero(review & ~auto)),
        "LK质量门控接受候选点帧数": int(np.count_nonzero(quality_accepted)),
        "LK质量门控拒绝候选点帧数": int(np.count_nonzero(quality_rejected)),
        "因自动3级安全排除的原开发候选点帧数": int(
            np.count_nonzero(
                correction["development_candidate_excluded_by_automatic_grade3"]
            )
        ),
        "自动移动P3点帧数": int(np.count_nonzero(auto)),
        "人工确认补充移动P3点帧数": int(np.count_nonzero(manual_applied)),
        "自动移动涉及帧数": int(len(affected_frames)),
        "自动移动时段_s": (
            f"{affected_frames[0]/fps:.3f}–{affected_frames[-1]/fps:.3f}"
            if len(affected_frames)
            else ""
        ),
        "纠偏前后RSR差异P95_1_s": round(
            finite_percentile(np.abs(rsr_difference), 95), 6
        ),
        "纠偏涉及点对RSR差异P95_1_s": round(
            affected_rsr_p95, 6
        ) if np.isfinite(affected_rsr_p95) else "",
        "纠偏涉及点对RSR最大差异_1_s": round(
            affected_rsr_max, 6
        ) if np.isfinite(affected_rsr_max) else "",
        "纠偏转换风险点帧数": int(np.count_nonzero(transition)),
        "纠偏后有效点对百分比": round(
            100 * float(np.mean(deformation["radial_pair_valid"][1:])), 4
        ),
        "纠偏后点对PCC中位数": round(corrected_pcc, 6),
        "原始点对PCC中位数": round(original_pcc, 6),
        "纠偏后正反误差P95_px": round(
            finite_percentile(deformation["radial_pair_fb_error_px"][1:], 95), 6
        ),
        "本轮是否自动移动P3": bool(np.any(auto)),
        "本轮是否应用人工确认补充纠偏": bool(np.any(manual_applied)),
        "原始主结果是否覆盖": False,
        "是否已完成外部泛化验证": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", choices=CASES, default=list(CASES))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"拒绝覆盖非空目录：{output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        analyze(case, output, P3ImageBoundaryConfig(), args.skip_video)
        for case in args.cases
    ]
    write_csv(output / "P3图像边界证据五病例汇总.csv", rows)
    (output / "README_中文结果说明.md").write_text(
        "# P3图像边界保守纠偏当前工程主轨 v2（开发期规则）\n\n"
        "原始P3为默认结果。程序沿每个P3点的外向法线读取原始超声灰度，寻找由亮到暗的"
        "候选壁边界；灰度候选本身不再直接触发最终移动。所有规则均不包含患者名、固定时间"
        "或固定壁段编号。\n\n"
        "候选规则为：伪影等级低于3级、相对平时外移至少3 px、图像边界证据达到门槛并"
        "连续至少0.5秒。同帧相邻壁段可共同形成候选；持续单点只有在邻近壁段于前后0.25秒"
        "内提供支持时才形成候选。\n\n"
        "每组候选先分别用原始P3和候选P3执行相邻帧LK。只有候选分支的PCC、正反误差和"
        "有效点对比例没有实质变差，并至少有一项改善时才自动移动；解剖末端允许在各项均"
        "未实质变差时保守接受。被拒绝的候选保持原始P3并在复核视频中显示为空圈。\n\n"
        "同一连续候选组使用统一的中位移动目标，最大移动4 px。时间上采用缓启动和缓停止的"
        "平滑曲线，空间上向相邻壁段逐渐减弱，避免壁线折角；移动量发生变化的帧继续作为"
        "转换风险从纠偏QC RSR中留空。宫颈端和宫底端使用末端保持，不会因边界外补零而"
        "减小最末端纠正量。\n\n"
        "复核视频中：青色是原始P3点；洋红箭头表示自动移动；紫色箭头表示人工确认后的补充移动；"
        "黄色/绿色是纠偏后送入LK的径向点对；"
        "橙色空圈是只报警、不移动的候选。探头抖动3级帧禁止自动纠偏。\n\n"
        "纠偏后的RSR仍由相邻帧LK点对长度变化计算，不使用P3坐标差。"
        "纠偏启停转换只作为下游独立敏感性标志，不自动清空正式QC RSR；"
        "只有人工确认3级帧可整帧留空，原始RSR完整保留。\n\n"
        "当前规则只在现有五病例中进行内部工程复核，尚无未参与开发的新病例，因此不能"
        "声称已证明外部泛化；新病例仍需抽查自动移动和被拒绝候选。"
        "正式传播方向、速度和频率仍未开启。\n",
        encoding="utf-8",
    )
    print(f"完成：{output}", flush=True)


if __name__ == "__main__":
    main()
