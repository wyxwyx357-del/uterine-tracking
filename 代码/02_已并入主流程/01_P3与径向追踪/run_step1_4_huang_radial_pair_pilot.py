#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3辅助贴壁定位的相邻帧径向LK测量：单病例ROI局部形变证据。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
from peristalsis_pipeline.project_layout import output_dir


DEFAULT_CASE = "CASE_002"
DEFAULT_OUTPUT = output_dir(PROJECT_ROOT, "step1_4_huang_radial_pair_pilot_1case")
BASELINE_ROOT = PROJECT_ROOT / "输入" / "04_冻结基线" / "step1_4_five_anchor"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def finite_values(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).ravel()
    return result[np.isfinite(result)]


def safe_mean(values: np.ndarray) -> float:
    finite = finite_values(values)
    return float(np.mean(finite)) if len(finite) else math.nan


def safe_rms(values: np.ndarray) -> float:
    finite = finite_values(values)
    return float(np.sqrt(np.mean(finite * finite))) if len(finite) else math.nan


def safe_percentile(values: np.ndarray, percentile: float) -> float:
    finite = finite_values(values)
    return float(np.percentile(finite, percentile)) if len(finite) else math.nan


def input_paths(case_id: str) -> tuple[Path, Path, Path, Path]:
    video = (
        PROJECT_ROOT / "输入" / "01_原始视频"
        / f"{case_id}.mp4"
    )
    meta = (
        PROJECT_ROOT / "输入" / "03_上游解剖节点"
        / "step1_3_middle_wall_band_tracking_points_smoke"
        / case_id
        / f"{case_id}_tracking_points.json"
    )
    baseline = BASELINE_ROOT / f"{case_id}_step1_4_5a_sparse_anchor_tracks.npz"
    nodes = BASELINE_ROOT / f"{case_id}_step1_4_5a_nodes.json"
    for path in (video, meta, baseline, nodes):
        if not path.is_file():
            raise FileNotFoundError(path)
    return video, meta, baseline, nodes


def case_output_paths(output_dir: Path, case_id: str) -> tuple[Path, ...]:
    return (
        output_dir / f"{case_id}_径向点对原始数据.npz",
        output_dir / f"{case_id}_前后壁径向应变率时空图.png",
        output_dir / f"{case_id}_径向点对追踪复核视频.mp4",
        output_dir / f"{case_id}_径向点对质量汇总.csv",
    )


def frame_mean_pcc(pair_pcc: np.ndarray) -> np.ndarray:
    values = np.asarray(pair_pcc, dtype=np.float64)
    finite = np.isfinite(values)
    counts = np.sum(finite, axis=(1, 2))
    return np.divide(
        np.nansum(values, axis=(1, 2)),
        counts,
        out=np.full(len(values), np.nan, dtype=np.float64),
        where=counts > 0,
    )


def p3_risk_fields(
    baseline: dict[str, np.ndarray], frame_count: int
) -> dict[str, np.ndarray]:
    point_keys = (
        "tracking_measurement_valid",
        "recovered_display_only",
    )
    frame_keys = (
        "topology_bad_edge_count",
        "topology_bad_cell_count",
    )
    available = all(key in baseline for key in (*point_keys, *frame_keys))
    if not available:
        return {
            "p3_quality_flags_available": np.asarray(False),
            "p3_tracking_measurement_valid_fraction": np.full(
                frame_count, np.nan, dtype=np.float32
            ),
            "p3_recovered_display_only_fraction": np.full(
                frame_count, np.nan, dtype=np.float32
            ),
            "p3_topology_bad_edge_count": np.full(frame_count, -1, dtype=np.int32),
            "p3_topology_bad_cell_count": np.full(frame_count, -1, dtype=np.int32),
            "p3_topology_risk_frame": np.zeros(frame_count, dtype=bool),
        }

    measurement = np.asarray(baseline["tracking_measurement_valid"][:frame_count])
    recovered = np.asarray(baseline["recovered_display_only"][:frame_count])
    bad_edges = np.asarray(
        baseline["topology_bad_edge_count"][:frame_count], dtype=np.int32
    )
    bad_cells = np.asarray(
        baseline["topology_bad_cell_count"][:frame_count], dtype=np.int32
    )
    if measurement.ndim != 2 or recovered.shape != measurement.shape:
        raise ValueError("P3点级质量标记形状与轨迹不一致")
    if bad_edges.shape != (frame_count,) or bad_cells.shape != (frame_count,):
        raise ValueError("P3逐帧拓扑标记形状与视频不一致")
    return {
        "p3_quality_flags_available": np.asarray(True),
        "p3_tracking_measurement_valid_fraction": np.mean(
            measurement, axis=1, dtype=np.float64
        ).astype(np.float32),
        "p3_recovered_display_only_fraction": np.mean(
            recovered, axis=1, dtype=np.float64
        ).astype(np.float32),
        "p3_topology_bad_edge_count": bad_edges,
        "p3_topology_bad_cell_count": bad_cells,
        "p3_topology_risk_frame": (bad_edges > 0) | (bad_cells > 0),
    }


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def add_text(image: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((3, 3, image.shape[1] - 3, 68), fill=(0, 0, 0, 155))
    text_font = font(15)
    for row, line in enumerate(lines):
        draw.text((8, 5 + row * 20), line, font=text_font, fill="white")
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def write_review_video(
    output_path: Path,
    frames: list[np.ndarray],
    tracking: dict[str, np.ndarray],
    anterior_count: int,
    fps: float,
) -> None:
    height, width = frames[0].shape
    temporary = output_path.parent / "_radial_pair_review_temporary.mp4"
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise OSError(f"无法创建复核视频: {temporary}")
    reference = tracking["radial_reference"]
    measured = tracking["radial_lk_measured_p3_initial"]
    pair_valid = tracking["radial_pair_valid"]
    pair_pcc = tracking["radial_pair_pcc"]
    try:
        for frame_idx, gray in enumerate(frames):
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for side, (inner_rail, outer_rail) in enumerate(((0, 1), (2, 3))):
                inner_slice = slice(
                    inner_rail * anterior_count, (inner_rail + 1) * anterior_count
                )
                outer_slice = slice(
                    outer_rail * anterior_count, (outer_rail + 1) * anterior_count
                )
                inner_points = measured[frame_idx, inner_slice]
                outer_points = measured[frame_idx, outer_slice]
                for section in range(anterior_count):
                    valid = bool(pair_valid[frame_idx, side, section])
                    if frame_idx == 0 or not valid:
                        inner_point = reference[frame_idx, inner_slice][section]
                        outer_point = reference[frame_idx, outer_slice][section]
                    else:
                        inner_point = inner_points[section]
                        outer_point = outer_points[section]
                    if not np.all(np.isfinite((inner_point, outer_point))):
                        continue
                    inner_xy = tuple(np.round(inner_point).astype(int))
                    outer_xy = tuple(np.round(outer_point).astype(int))
                    color = (0, 135, 255) if valid or frame_idx == 0 else (0, 0, 255)
                    cv2.line(image, inner_xy, outer_xy, color, 1, cv2.LINE_AA)
                    cv2.circle(image, inner_xy, 2, (0, 235, 255), -1, cv2.LINE_AA)
                    cv2.circle(
                        image,
                        outer_xy,
                        2,
                        (0, 220, 0) if valid or frame_idx == 0 else (0, 0, 255),
                        -1,
                        cv2.LINE_AA,
                    )
            coverage = float(np.mean(pair_valid[frame_idx])) if frame_idx else 1.0
            mean_pcc = safe_mean(pair_pcc[frame_idx])
            image = add_text(
                image,
                [
                    f"径向点对追踪复核｜帧 {frame_idx}/{len(frames)-1}",
                    f"有效点对 {coverage*100:.1f}%｜点对PCC {mean_pcc:.3f}",
                    "黄色=内层点　绿色=外层点　橙线=有效点对　红色=本帧无效",
                ],
            )
            writer.write(image)
    finally:
        writer.release()
    temporary.replace(output_path)


def map_limit(values: np.ndarray) -> float:
    limit = safe_percentile(np.abs(values), 98)
    return max(limit, 1e-6) if np.isfinite(limit) else 1.0


def write_time_space_figure(
    output_path: Path,
    raw: np.ndarray,
    filtered: np.ndarray,
    fps: float,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    duration = (len(raw) - 1) / float(fps)
    frequencies_cpm = np.fft.rfftfreq(len(raw), d=1.0 / float(fps)) * 60.0
    kept = frequencies_cpm[(frequencies_cpm >= 0.5) & (frequencies_cpm <= 5.0)]
    kept_text = "、".join(f"{value:.2f}" for value in kept) or "无"
    figure, axes = plt.subplots(4, 1, figsize=(12, 11), constrained_layout=True)
    panels = (
        (raw[:, 0], "前壁原始径向应变率", map_limit(raw)),
        (raw[:, 1], "后壁原始径向应变率", map_limit(raw)),
        (
            filtered[:, 0],
            "前壁探索性滤波（低PCC复核帧不进入滤波输入）",
            map_limit(filtered),
        ),
        (
            filtered[:, 1],
            "后壁探索性滤波（低PCC复核帧不进入滤波输入）",
            map_limit(filtered),
        ),
    )
    for axis, (values, title, limit) in zip(axes, panels):
        image = axis.imshow(
            values.T,
            aspect="auto",
            origin="lower",
            extent=(0, duration, 0, 1),
            cmap="RdYlBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.set_ylabel("宫颈 → 宫底")
        figure.colorbar(image, ax=axis, label="1/s；负值=径向压缩")
    axes[-1].set_xlabel("时间（秒）")
    figure.suptitle(
        "Huang径向点对融合测试｜仅为ROI局部形变证据｜"
        f"短视频滤波可用频点：{kept_text}次/分钟｜不作正式传播解释"
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_summary(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def analyze_case(
    case_id: str,
    output_dir: Path,
    offset_px: float,
    max_frames: int | None,
    skip_video: bool,
) -> dict[str, Any]:
    video_path, meta_path, baseline_path, nodes_path = input_paths(case_id)
    resize_factor = float(read_json(meta_path)["parameters"]["resize_factor"])
    frames, video_info = tracking_lk.read_video_gray(video_path, resize_factor)
    if max_frames is not None:
        frames = frames[: max(2, min(max_frames, len(frames)))]
    fps = float(video_info["fps"])
    baseline = load_npz(baseline_path)
    tracks = np.asarray(baseline["tracks"][: len(frames)], dtype=np.float32)
    nodes = read_json(nodes_path)
    midline_ids = tracking_huang_fusion.ordered_role_indices(nodes, "midline")
    anterior_ids, posterior_ids, _ = tracking_huang_fusion.paired_wall_indices(nodes)
    wall_tracks = tracks[:, np.concatenate((anterior_ids, posterior_ids))]
    config = tracking_huang_wall_fusion.HuangWallFusionConfig()

    print(f"[{case_id}] 中线10点估计相邻帧整体运动", flush=True)
    global_result = tracking_huang_wall_fusion.estimate_pairwise_global_motion(
        frames, tracks[:, midline_ids], config
    )
    print(f"[{case_id}] 追踪前后壁21像素径向点对", flush=True)
    tracking = tracking_huang_radial_pairs.measure_pairwise_radial_points(
        frames,
        wall_tracks,
        global_result["pairwise_global_transform"],
        anterior_count=len(anterior_ids),
        offset_px=offset_px,
        config=config,
    )
    deformation = tracking_huang_radial_pairs.radial_deformation_from_tracking(
        tracking, anterior_count=len(anterior_ids), fps=fps
    )
    raw_rsr = deformation["radial_strain_rate_s"]
    pair_pcc = deformation["radial_pair_pcc"]
    mean_pcc_by_frame = frame_mean_pcc(pair_pcc)
    low_pcc_review_frame = mean_pcc_by_frame < config.recording_pcc_threshold
    low_pcc_review_frame[0] = False

    legacy_filtered_rsr = np.stack(
        [
            tracking_huang_fusion.fft_bandpass(
                raw_rsr[:, side],
                fps,
                config.physiology_low_cpm,
                config.physiology_high_cpm,
                config.minimum_signal_valid_fraction,
            )
            for side in range(2)
        ],
        axis=1,
    )
    qc_filter_input = np.asarray(raw_rsr, dtype=np.float32).copy()
    qc_filter_input[low_pcc_review_frame] = np.nan
    filtered_rsr = np.stack(
        [
            tracking_huang_fusion.fft_bandpass(
                qc_filter_input[:, side],
                fps,
                config.physiology_low_cpm,
                config.physiology_high_cpm,
                config.minimum_signal_valid_fraction,
            )
            for side in range(2)
        ],
        axis=1,
    )
    filter_sensitivity = np.abs(filtered_rsr - legacy_filtered_rsr)
    p3_risk = p3_risk_fields(baseline, len(frames))

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{case_id}_径向点对原始数据.npz"
    archive = {**global_result, **tracking, **deformation, **p3_risk}
    archive.update(
        filtered_radial_strain_rate_s=filtered_rsr,
        filtered_radial_strain_rate_legacy_all_frames_s=legacy_filtered_rsr,
        low_pcc_review_frame=low_pcc_review_frame,
        filter_sensitivity_absolute_difference_s=filter_sensitivity,
        radial_offset_px=np.asarray(offset_px, dtype=np.float32),
        fps=np.asarray(fps, dtype=np.float32),
        resize_factor=np.asarray(resize_factor, dtype=np.float32),
        longitudinal_section_count=np.asarray(len(anterior_ids), dtype=np.int32),
        tracking_source=np.asarray("p3_assisted_anatomical_reference"),
        p3_role=np.asarray("per_frame_anatomical_localization_not_direct_deformation"),
        deformation_measurement_role=np.asarray("adjacent_frame_lk_radial_pair_length_change"),
        p3_coordinate_delta_used_as_rsr=np.asarray(False),
        no_p3_used_as_primary=np.asarray(False),
        formal_propagation_released=np.asarray(False),
    )
    np.savez_compressed(npz_path, **archive)
    figure_path = output_dir / f"{case_id}_前后壁径向应变率时空图.png"
    write_time_space_figure(figure_path, raw_rsr, filtered_rsr, fps)
    if not skip_video:
        write_review_video(
            output_dir / f"{case_id}_径向点对追踪复核视频.mp4",
            frames,
            {**tracking, **deformation},
            len(anterior_ids),
            fps,
        )

    valid = deformation["radial_pair_valid"]
    frame_valid_fraction = np.mean(valid[1:], axis=(1, 2))
    frame_mean_pcc_values = mean_pcc_by_frame[1:]
    low_pcc_frames = np.flatnonzero(low_pcc_review_frame)
    common_speed = deformation["radial_pair_common_local_speed_px_s"]
    relative_speed = raw_rsr * float(offset_px)
    common_rms = safe_rms(common_speed[1:])
    relative_rms = safe_rms(relative_speed[1:])
    frequencies_cpm = np.fft.rfftfreq(len(frames), d=1.0 / fps) * 60.0
    kept_frequencies = frequencies_cpm[
        (frequencies_cpm >= config.physiology_low_cpm)
        & (frequencies_cpm <= config.physiology_high_cpm)
    ]
    summary: dict[str, Any] = {
        "病例": case_id,
        "证据层": "P3辅助贴壁定位_相邻帧LK_ROI局部形变证据",
        "主追踪来源": "P3逐帧解剖定位＋相邻帧LK径向测量",
        "P3坐标变化是否直接作为RSR": False,
        "no-P3是否作为主结果": False,
        "视频时长_s": round((len(frames) - 1) / fps, 4),
        "径向间距_px": offset_px,
        "每侧径向点对数量": len(anterior_ids),
        "前壁有效覆盖百分比": round(100 * float(np.mean(valid[1:, 0])), 4),
        "后壁有效覆盖百分比": round(100 * float(np.mean(valid[1:, 1])), 4),
        "径向点对平均PCC": round(safe_mean(pair_pcc[1:]), 6),
        "录像级PCC是否超过0.8": bool(safe_mean(pair_pcc[1:]) > 0.8),
        "单帧最低有效覆盖百分比": round(
            100 * float(np.min(frame_valid_fraction)), 4
        ),
        "单帧平均PCC最低值_仅定位复核": round(
            safe_mean([np.nanmin(frame_mean_pcc_values)]), 6
        ),
        "单帧平均PCC低于0.8帧数_不自动删除": int(len(low_pcc_frames)),
        "低PCC复核时刻_s": "、".join(
            f"{frame_idx / fps:.3f}" for frame_idx in low_pcc_frames
        ),
        "点对正反误差P95_px": round(
            safe_percentile(deformation["radial_pair_fb_error_px"][1:], 95), 6
        ),
        "低PCC帧是否从探索性滤波输入排除": bool(len(low_pcc_frames)),
        "低PCC滤波敏感性差异P95_1_s": round(
            safe_percentile(filter_sensitivity[1:], 95), 6
        ),
        "低PCC滤波敏感性最大差异_1_s": round(
            safe_percentile(filter_sensitivity[1:], 100), 6
        ),
        "P3质量标记是否可用": bool(p3_risk["p3_quality_flags_available"]),
        "P3正式测量有效百分比_仅传递风险": round(
            100 * safe_mean(p3_risk["p3_tracking_measurement_valid_fraction"]), 4
        ),
        "P3恢复显示坐标百分比_仅传递风险": round(
            100 * safe_mean(p3_risk["p3_recovered_display_only_fraction"]), 4
        ),
        "P3拓扑风险帧数_不自动删除": int(
            np.count_nonzero(p3_risk["p3_topology_risk_frame"])
        ),
        "双初值RSR差异P95_1_s": round(
            safe_percentile(deformation["initialization_rsr_disagreement_s"][1:], 95),
            6,
        ),
        "距离法与投影法RSR差异P95_1_s": round(
            safe_percentile(
                np.abs(raw_rsr - deformation["projected_radial_strain_rate_s"])[1:],
                95,
            ),
            6,
        ),
        "相对形变速度RMS_px_s": round(relative_rms, 6),
        "共同局部速度RMS_px_s": round(common_rms, 6),
        "相对形变与共同运动RMS比": round(
            relative_rms / common_rms if common_rms > 1e-9 else math.nan, 6
        ),
        "滤波实际可用离散频点_cpm": "、".join(
            f"{value:.4f}" for value in kept_frequencies
        ),
        "是否足以分辨0.5次每分钟": bool((len(frames) - 1) / fps >= 120.0),
        "正式传播是否释放": False,
        "说明": "P3只定位每帧壁线；RSR来自相邻帧LK点对长度变化。未分析正式方向、速度、频率或CF/FC；R系列未接入",
    }
    write_summary(output_dir / f"{case_id}_径向点对质量汇总.csv", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default=DEFAULT_CASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offset-px", type=float, default=21.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    existing = [
        path for path in case_output_paths(output_dir, args.case_id) if path.exists()
    ]
    if existing and not args.allow_overwrite:
        raise FileExistsError(
            "拒绝覆盖已有病例结果；如已确认需要覆盖，请增加 --allow-overwrite："
            f"{existing[0]}"
        )
    summary = analyze_case(
        args.case_id,
        output_dir,
        args.offset_px,
        args.max_frames,
        args.skip_video,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
