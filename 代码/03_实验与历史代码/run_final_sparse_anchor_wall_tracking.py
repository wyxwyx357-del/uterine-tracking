#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final sparse-anchor, true-wall, segmented bidirectional tracking.

This is the single maintained Step 1.4 entry point. It combines independent
anatomical line anchors, DUSTrack-compatible RSTC fusion, true-wall nodes,
terminal-fundus deduplication, and display-only smoothing of the anterior
fundus line. By default it preserves the original three-anchor behavior; the
optional endpoint-anchor mode adds manual labels at the first and last frames
so the complete video is bounded by manual anchors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "代码" / "01_底层算法"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from peristalsis_pipeline import tracking_lk, tracking_mesh


CASES = (
    "CASE_003",
    "CASE_002",
    "CASE_001",
    "CASE_005",
    "CASE_004",
)
ANCHOR_ORDER = ("front_half_center", "middle", "back_half_center")
ENDPOINT_ANCHOR_ORDER = (
    "video_start",
    "front_half_center",
    "middle",
    "back_half_center",
    "video_end",
)
MAX_BRANCH_DISAGREEMENT_PX = 2.0
MIN_FRAME_VALID_FRACTION = 0.80
DISPLAY_SCALE = 2
RSTC_EPSILON = 0.01
TRACK_POINT_RADIUS = 3
TRACK_POINT_HALO_RADIUS = 5
TRACK_LINE_THICKNESS = 1
TRUE_WALL_U = {0: 0.0, 1: 0.5, 2: 1.0}
RAIL_NAMES = {0: "anterior_wall", 1: "midline", 2: "posterior_wall"}
RAIL_COLORS = {
    0: (255, 255, 0),
    1: (0, 255, 255),
    2: (255, 0, 255),
}
NARROW_REFERENCE_U_GAP = 0.35
MIN_FUNDUS_SPACING_RATIO = 0.60
SMOOTH_POINT_COUNT = 6
SAMPLES_PER_SEGMENT = 12
MAX_CURVE_DEVIATION_PX = 0.75


def module_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def shape_map(data: dict[str, Any], source: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for shape in data.get("shapes", []):
        label = str(shape.get("label", ""))
        if label in result:
            raise ValueError(f"Duplicate LabelMe label {label!r}: {source}")
        result[label] = shape
    required = {
        "ant_wall": "linestrip",
        "post_wall": "linestrip",
        "cervix_point": "point",
        "fundus_point": "point",
    }
    for label, expected_type in required.items():
        if label not in result:
            raise ValueError(f"Missing LabelMe label {label!r}: {source}")
        actual_type = str(result[label].get("shape_type", ""))
        if actual_type != expected_type:
            raise ValueError(
                f"Label {label!r} must be {expected_type}, got {actual_type}: {source}"
            )
        minimum = 2 if expected_type == "linestrip" else 1
        if len(result[label].get("points", [])) < minimum:
            raise ValueError(f"Too few points for {label!r}: {source}")
    return result


def orient_curve(
    points: np.ndarray, cervix: np.ndarray, fundus: np.ndarray
) -> np.ndarray:
    """Orient a polyline from cervix to fundus using both endpoints."""
    points = np.asarray(points, dtype=np.float32)
    forward_cost = float(
        np.linalg.norm(points[0] - cervix) + np.linalg.norm(points[-1] - fundus)
    )
    reverse_cost = float(
        np.linalg.norm(points[-1] - cervix) + np.linalg.norm(points[0] - fundus)
    )
    return points if forward_cost <= reverse_cost else points[::-1].copy()


def resample_curve(points: np.ndarray, samples: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if samples < 2:
        raise ValueError("samples must be >= 2")
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.r_[True, segment > 1e-6]
    points = points[keep]
    if len(points) < 2:
        raise ValueError("Polyline has zero usable length")
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment)]
    targets = np.linspace(0.0, float(cumulative[-1]), samples)
    result = np.empty((samples, 2), dtype=np.float32)
    result[:, 0] = np.interp(targets, cumulative, points[:, 0])
    result[:, 1] = np.interp(targets, cumulative, points[:, 1])
    return result


def orient_sections(
    ant: np.ndarray,
    post: np.ndarray,
    cervix: np.ndarray,
    fundus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    center = 0.5 * (ant + post)
    forward_cost = float(
        np.linalg.norm(center[0] - cervix) + np.linalg.norm(center[-1] - fundus)
    )
    reverse_cost = float(
        np.linalg.norm(center[-1] - cervix) + np.linalg.norm(center[0] - fundus)
    )
    if reverse_cost < forward_cost:
        return ant[::-1].copy(), post[::-1].copy()
    return ant, post


def load_manifest(
    root: Path,
    include_endpoint_anchors: bool = False,
    cases: tuple[str, ...] | list[str] = CASES,
) -> dict[tuple[str, str], dict[str, str]]:
    sparse_path = (
        root / "输入" / "02_人工标签" / "02_P3五锚点标签"
        / "sparse_anchor_frame_manifest.csv"
    )
    with sparse_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {(row["case_id"], row["anchor_name"]): row for row in rows}
    expected = {
        (case_id, anchor)
        for case_id in cases
        for anchor in ("front_half_center", "back_half_center")
    }
    if set(result) != expected:
        raise ValueError(
            f"Sparse-anchor manifest mismatch: missing={sorted(expected - set(result))}, "
            f"extra={sorted(set(result) - expected)}"
        )
    if not include_endpoint_anchors:
        return result

    endpoint_path = (
        root / "输入" / "02_人工标签" / "02_P3五锚点标签"
        / "endpoint_anchor_frame_manifest.csv"
    )
    with endpoint_path.open("r", encoding="utf-8-sig", newline="") as handle:
        endpoint_rows = list(csv.DictReader(handle))
    endpoint_result = {
        (row["case_id"], row["anchor_name"]): row for row in endpoint_rows
    }
    endpoint_expected = {
        (case_id, anchor)
        for case_id in cases
        for anchor in ("video_start", "video_end")
    }
    if set(endpoint_result) != endpoint_expected:
        raise ValueError(
            "Endpoint-anchor manifest mismatch: "
            f"missing={sorted(endpoint_expected - set(endpoint_result))}, "
            f"extra={sorted(set(endpoint_result) - endpoint_expected)}"
        )
    result.update(endpoint_result)
    return result


def load_external_manifests(
    label_dir: Path,
    cases: tuple[str, ...] | list[str],
) -> dict[tuple[str, str], dict[str, str]]:
    """Load one five-anchor extraction manifest per real case."""

    required = set(ENDPOINT_ANCHOR_ORDER)
    result: dict[tuple[str, str], dict[str, str]] = {}
    for case_id in cases:
        path = label_dir / f"{case_id}__five_anchor_manifest.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        anchors = {str(row.get("anchor_name", "")) for row in rows}
        if anchors != required or len(rows) != len(required):
            raise ValueError(
                f"{path}: expected exactly five anchors {sorted(required)}, "
                f"found {sorted(anchors)}"
            )
        source_videos: set[Path] = set()
        for row in rows:
            if str(row.get("case_id", "")) != case_id:
                raise ValueError(f"{path}: case_id mismatch")
            image = Path(str(row.get("output_png", "")))
            if not image.is_absolute():
                image = label_dir / image
            json_path = image.with_suffix(".json")
            if not image.is_file() or not json_path.is_file():
                raise FileNotFoundError(
                    f"{path}: missing PNG/JSON pair for {image.name}"
                )
            video = Path(str(row.get("source_video", "")))
            if not video.is_file():
                raise FileNotFoundError(f"{path}: source video not found: {video}")
            source_videos.add(video.resolve())
            normalized = dict(row)
            normalized["output_png"] = str(image.resolve())
            normalized["source_video"] = str(video.resolve())
            result[(case_id, str(row["anchor_name"]))] = normalized
        if len(source_videos) != 1:
            raise ValueError(f"{path}: anchors refer to different source videos")
    return result


def input_paths(root: Path, case_id: str) -> tuple[Path, Path]:
    video = (
        root / "输入" / "01_原始视频"
        / f"{case_id}.mp4"
    )
    meta = (
        root / "输入" / "03_上游解剖节点"
        / "step1_3_middle_wall_band_tracking_points_smoke"
        / case_id
        / f"{case_id}_tracking_points.json"
    )
    for path in (video, meta):
        if not path.is_file():
            raise FileNotFoundError(path)
    return video, meta


def load_manual_anchor(
    json_path: Path,
    frame_idx: int,
    anchor_name: str,
    resize_factor: float,
    n_sections: int,
) -> dict[str, Any]:
    data = load_json(json_path)
    shapes = shape_map(data, json_path)
    cervix_original = np.asarray(shapes["cervix_point"]["points"][0], dtype=np.float32)
    fundus_original = np.asarray(shapes["fundus_point"]["points"][0], dtype=np.float32)
    ant_original = orient_curve(
        np.asarray(shapes["ant_wall"]["points"], dtype=np.float32),
        cervix_original,
        fundus_original,
    )
    post_original = orient_curve(
        np.asarray(shapes["post_wall"]["points"], dtype=np.float32),
        cervix_original,
        fundus_original,
    )
    ant = resample_curve(ant_original * resize_factor, n_sections)
    post = resample_curve(post_original * resize_factor, n_sections)
    return {
        "name": anchor_name,
        "frame_idx": int(frame_idx),
        "json_path": json_path,
        "image_path": json_path.parent / str(data["imagePath"]),
        "cervix": cervix_original * resize_factor,
        "fundus": fundus_original * resize_factor,
        "ant": ant,
        "post": post,
        "ant_raw": ant_original * resize_factor,
        "post_raw": post_original * resize_factor,
        "image_width": int(data.get("imageWidth", 0)),
        "image_height": int(data.get("imageHeight", 0)),
    }


def load_middle_anchor(
    root: Path,
    case_id: str,
    meta: dict[str, Any],
    frame_idx: int,
    resize_factor: float,
) -> dict[str, Any]:
    json_path = (
        root / "输入" / "02_人工标签" / "01_中帧解剖标签"
        / f"{case_id}_middle_flow2.json"
    )
    data = load_json(json_path)
    shapes = shape_map(data, json_path)
    cervix = np.asarray(
        [meta["cervix_point_resized"]["x"], meta["cervix_point_resized"]["y"]],
        dtype=np.float32,
    )
    fundus = np.asarray(
        [meta["fundus_point_resized"]["x"], meta["fundus_point_resized"]["y"]],
        dtype=np.float32,
    )
    sections = sorted(
        meta["sections_resized"], key=lambda item: int(item["long_axis_index"])
    )
    ant = np.asarray(
        [[item["ant"]["x"], item["ant"]["y"]] for item in sections],
        dtype=np.float32,
    )
    post = np.asarray(
        [[item["post"]["x"], item["post"]["y"]] for item in sections],
        dtype=np.float32,
    )
    ant, post = orient_sections(ant, post, cervix, fundus)

    cervix_original = np.asarray(shapes["cervix_point"]["points"][0], dtype=np.float32)
    fundus_original = np.asarray(shapes["fundus_point"]["points"][0], dtype=np.float32)
    ant_raw = orient_curve(
        np.asarray(shapes["ant_wall"]["points"], dtype=np.float32),
        cervix_original,
        fundus_original,
    )
    post_raw = orient_curve(
        np.asarray(shapes["post_wall"]["points"], dtype=np.float32),
        cervix_original,
        fundus_original,
    )
    return {
        "name": "middle",
        "frame_idx": int(frame_idx),
        "json_path": json_path,
        "image_path": json_path.parent / str(data["imagePath"]),
        "cervix": cervix,
        "fundus": fundus,
        "ant": ant,
        "post": post,
        "ant_raw": ant_raw * resize_factor,
        "post_raw": post_raw * resize_factor,
        "image_width": int(data.get("imageWidth", 0)),
        "image_height": int(data.get("imageHeight", 0)),
    }


def build_shared_nodes(
    anchors: list[dict[str, Any]], mesh_module
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], np.ndarray]:
    n_sections = len(anchors[0]["ant"])
    if any(
        len(anchor["ant"]) != n_sections or len(anchor["post"]) != n_sections
        for anchor in anchors
    ):
        raise ValueError("Anchor section count mismatch")
    full_width = np.stack(
        [
            np.linalg.norm(anchor["post"] - anchor["ant"], axis=1)
            for anchor in anchors
        ]
    )
    narrow = (
        np.min(full_width * NARROW_REFERENCE_U_GAP, axis=0)
        < mesh_module.MIN_INDEPENDENT_SPACING_PX
    )
    nodes: list[dict[str, Any]] = []
    for section_idx in range(n_sections):
        width_indices = (1,) if narrow[section_idx] else (0, 1, 2)
        for width_idx in width_indices:
            nodes.append(
                {
                    "point_id": len(nodes),
                    "section_order": section_idx,
                    "long_axis_index": section_idx,
                    "width_index": width_idx,
                    "u_ant_to_post_norm": TRUE_WALL_U[width_idx],
                    "rail_name": RAIL_NAMES[width_idx],
                    "independent_tracking": True,
                    "true_wall_boundary_node": width_idx in (0, 2),
                }
            )

    points_by_anchor: dict[str, np.ndarray] = {}
    for anchor in anchors:
        points = []
        for node in nodes:
            section_idx = int(node["section_order"])
            u_value = float(node["u_ant_to_post_norm"])
            point = (
                (1.0 - u_value) * anchor["ant"][section_idx]
                + u_value * anchor["post"][section_idx]
            )
            points.append(point)
        points_by_anchor[str(anchor["name"])] = np.asarray(points, dtype=np.float32)

    middle_points = points_by_anchor["middle"]
    for node, point in zip(nodes, middle_points):
        node["x_resized_middle"] = float(point[0])
        node["y_resized_middle"] = float(point[1])
    return nodes, points_by_anchor, narrow


def midline_spacing_ratios(
    poses: np.ndarray, nodes: list[dict[str, Any]]
) -> np.ndarray:
    section = np.asarray([int(node["section_order"]) for node in nodes])
    rail = np.asarray([int(node["width_index"]) for node in nodes])
    point_ids = np.flatnonzero(rail == 1)
    point_ids = point_ids[np.argsort(section[point_ids])]
    if len(point_ids) < 2:
        raise ValueError("At least two midline nodes are required")
    spacing = np.linalg.norm(
        np.diff(np.asarray(poses)[:, point_ids, :], axis=1), axis=2
    )
    median = np.median(spacing, axis=1)
    return np.divide(
        spacing[:, -1],
        median,
        out=np.full_like(median, np.nan),
        where=median > 1e-6,
    )


def fundus_endpoint_keep_sections(
    poses: np.ndarray,
    nodes: list[dict[str, Any]],
    narrow_sections: np.ndarray,
    min_ratio: float = MIN_FUNDUS_SPACING_RATIO,
) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
    """Drop a redundant terminal, midline-only fundus section."""
    if not 0.0 < min_ratio < 1.0:
        raise ValueError("min_ratio must be between zero and one")
    keep_sections = list(range(len(narrow_sections)))
    removed: list[int] = []
    before = midline_spacing_ratios(poses, nodes)
    current_nodes = nodes
    current_poses = np.asarray(poses)

    while len(keep_sections) >= 3:
        terminal = keep_sections[-1]
        terminal_nodes = [
            node
            for node in current_nodes
            if int(node["section_order"]) == terminal
        ]
        if not (
            bool(narrow_sections[terminal])
            and len(terminal_nodes) == 1
            and int(terminal_nodes[0]["width_index"]) == 1
        ):
            break
        ratios = midline_spacing_ratios(current_poses, current_nodes)
        if np.all(ratios >= min_ratio):
            break
        removed.append(terminal)
        keep_sections.pop()
        keep_set = set(keep_sections)
        keep_mask = np.asarray(
            [
                int(node["section_order"]) in keep_set
                for node in current_nodes
            ],
            dtype=bool,
        )
        current_poses = current_poses[:, keep_mask]
        current_nodes = [
            node for node, keep in zip(current_nodes, keep_mask) if keep
        ]

    after = midline_spacing_ratios(current_poses, current_nodes)
    return keep_sections, removed, before, after


def subset_tracking_points(
    result: dict[str, np.ndarray],
    nodes: list[dict[str, Any]],
    points_by_anchor: dict[str, np.ndarray],
    keep_sections: list[int],
) -> tuple[
    dict[str, np.ndarray],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    np.ndarray,
]:
    keep_set = set(keep_sections)
    point_ids = np.asarray(
        [
            point_idx
            for point_idx, node in enumerate(nodes)
            if int(node["section_order"]) in keep_set
        ],
        dtype=np.int32,
    )
    n_points = len(nodes)
    filtered_result: dict[str, np.ndarray] = {}
    for key, value in result.items():
        array = np.asarray(value)
        if array.ndim >= 2 and array.shape[1] == n_points:
            filtered_result[key] = array[:, point_ids, ...]
        else:
            filtered_result[key] = value
    filtered_nodes: list[dict[str, Any]] = []
    for new_id, old_id in enumerate(point_ids):
        node = dict(nodes[int(old_id)])
        node["point_id"] = new_id
        filtered_nodes.append(node)
    filtered_anchors = {
        name: np.asarray(points)[point_ids]
        for name, points in points_by_anchor.items()
    }
    return filtered_result, filtered_nodes, filtered_anchors, point_ids


def smooth_interpolating_curve(
    points: np.ndarray,
    samples_per_segment: int = SAMPLES_PER_SEGMENT,
    max_deviation_px: float = MAX_CURVE_DEVIATION_PX,
) -> tuple[np.ndarray, float]:
    """Chord-length cubic Hermite curve through every supplied point."""
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 2:
        return points.copy(), 0.0
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least two")
    if max_deviation_px < 0:
        raise ValueError("max_deviation_px must be non-negative")

    segment_length = np.maximum(
        np.linalg.norm(np.diff(points, axis=0), axis=1), 1e-6
    )
    parameter = np.r_[0.0, np.cumsum(segment_length)].astype(np.float32)
    tangent = np.empty_like(points)
    tangent[0] = (points[1] - points[0]) / segment_length[0]
    tangent[-1] = (points[-1] - points[-2]) / segment_length[-1]
    if len(points) > 2:
        tangent[1:-1] = (points[2:] - points[:-2]) / (
            parameter[2:] - parameter[:-2]
        )[:, None]

    samples: list[np.ndarray] = []
    max_observed = 0.0
    for segment_idx in range(len(points) - 1):
        p0 = points[segment_idx]
        p1 = points[segment_idx + 1]
        dt = float(parameter[segment_idx + 1] - parameter[segment_idx])
        for u in np.linspace(
            0.0, 1.0, samples_per_segment, endpoint=False, dtype=np.float32
        ):
            u2 = u * u
            u3 = u2 * u
            smooth = (
                (2.0 * u3 - 3.0 * u2 + 1.0) * p0
                + (u3 - 2.0 * u2 + u) * dt * tangent[segment_idx]
                + (-2.0 * u3 + 3.0 * u2) * p1
                + (u3 - u2) * dt * tangent[segment_idx + 1]
            )
            linear = (1.0 - u) * p0 + u * p1
            delta = smooth - linear
            deviation = float(np.linalg.norm(delta))
            if deviation > max_deviation_px and deviation > 1e-9:
                smooth = linear + delta * (max_deviation_px / deviation)
                deviation = max_deviation_px
            max_observed = max(max_observed, deviation)
            samples.append(smooth.astype(np.float32))
    samples.append(points[-1].copy())
    return np.asarray(samples, dtype=np.float32), max_observed


def crossing_flip_count(ant: np.ndarray, post: np.ndarray) -> int:
    center = 0.5 * (ant + post)
    tangent = np.gradient(center, axis=0)
    width = post - ant
    signed = tangent[:, 0] * width[:, 1] - tangent[:, 1] * width[:, 0]
    usable = np.linalg.norm(width, axis=1) > 2.0
    signed = signed[usable & (np.abs(signed) > 1e-4)]
    if not len(signed):
        return 0
    reference_sign = 1.0 if float(np.median(signed)) >= 0 else -1.0
    return int(np.count_nonzero(np.sign(signed) != reference_sign))


def anchor_validation_row(
    case_id: str, anchor: dict[str, Any], narrow: np.ndarray
) -> dict[str, Any]:
    ant = np.asarray(anchor["ant"])
    post = np.asarray(anchor["post"])
    width = np.linalg.norm(post - ant, axis=1)
    cervix = np.asarray(anchor["cervix"])
    fundus = np.asarray(anchor["fundus"])
    endpoint_distances = {
        "ant_start_to_cervix_px_original": 2.0 * float(np.linalg.norm(ant[0] - cervix)),
        "post_start_to_cervix_px_original": 2.0 * float(np.linalg.norm(post[0] - cervix)),
        "ant_end_to_fundus_px_original": 2.0 * float(np.linalg.norm(ant[-1] - fundus)),
        "post_end_to_fundus_px_original": 2.0 * float(np.linalg.norm(post[-1] - fundus)),
    }
    return {
        "case_id": case_id,
        "anchor_name": anchor["name"],
        "frame_idx": int(anchor["frame_idx"]),
        "json_path": str(anchor["json_path"]),
        "n_sections": len(ant),
        "n_shared_narrow_sections": int(np.count_nonzero(narrow)),
        "ant_arc_length_px_original": round(
            2.0 * float(np.sum(np.linalg.norm(np.diff(ant, axis=0), axis=1))), 4
        ),
        "post_arc_length_px_original": round(
            2.0 * float(np.sum(np.linalg.norm(np.diff(post, axis=0), axis=1))), 4
        ),
        "paired_width_min_px_original": round(2.0 * float(np.min(width)), 4),
        "paired_width_median_px_original": round(2.0 * float(np.median(width)), 4),
        "paired_width_max_px_original": round(2.0 * float(np.max(width)), 4),
        "crossing_orientation_flip_count": crossing_flip_count(ant, post),
        **{key: round(value, 4) for key, value in endpoint_distances.items()},
        "status": "pass",
    }


def draw_anchor_overlay(
    gray: np.ndarray,
    anchor: dict[str, Any],
    narrow: np.ndarray,
    case_id: str,
) -> np.ndarray:
    vis = cv2.resize(
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
    )
    colors = {"ant": (0, 255, 0), "post": (255, 0, 255)}
    for key in ("ant", "post"):
        points = np.round(np.asarray(anchor[key]) * DISPLAY_SCALE).astype(np.int32)
        cv2.polylines(vis, [points], False, colors[key], 2, cv2.LINE_AA)
        for section_idx, point in enumerate(points):
            color = (0, 165, 255) if narrow[section_idx] else colors[key]
            cv2.circle(vis, tuple(point), 3, color, -1, cv2.LINE_AA)
    cervix = tuple(np.round(np.asarray(anchor["cervix"]) * DISPLAY_SCALE).astype(int))
    fundus = tuple(np.round(np.asarray(anchor["fundus"]) * DISPLAY_SCALE).astype(int))
    cv2.circle(vis, cervix, 7, (255, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(vis, fundus, 7, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(vis, (6, 6), (900, 76), (0, 0, 0), -1)
    cv2.putText(
        vis,
        f"{case_id} | {anchor['name']} | frame {anchor['frame_idx']}",
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        "green=ant wall | magenta=post wall | cyan=cervix | yellow=fundus | orange=narrow",
        (16, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return vis


def chronological(local: dict[str, Any], reverse: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in local.items():
        if key == "reasons":
            result[key] = value
            continue
        array = np.asarray(value)
        result[key] = array[::-1].copy() if reverse else array
    return result


def temporal_blend_weights(
    n_frames: int, blend_method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return forward/reverse weights on the chronological frame axis.

    ``rstc`` matches ``DUSTrack.dustrack.lk_opticalflow``: epsilon=0.01,
    forward weight one at the left anchor and zero at the right anchor, with
    the reverse weight following the complementary sigmoid.
    """
    if n_frames < 2:
        raise ValueError("A bounded tracking interval needs at least 2 frames")
    x = np.arange(n_frames, dtype=np.float64)
    if blend_method == "linear":
        reverse_weight = x / float(n_frames - 1)
        forward_weight = 1.0 - reverse_weight
    elif blend_method == "rstc":
        b = 2.0 * np.log(1.0 / RSTC_EPSILON - 1.0) / float(n_frames - 1)
        c = float(n_frames - 1) / 2.0
        forward_weight = (
            (1.0 / (1.0 + np.exp(b * (x - c))) - 0.5)
            / (1.0 - 2.0 * RSTC_EPSILON)
            + 0.5
        )
        reverse_weight = (
            (1.0 / (1.0 + np.exp(-b * (x - c))) - 0.5)
            / (1.0 - 2.0 * RSTC_EPSILON)
            + 0.5
        )
    else:
        raise ValueError(f"Unknown blend method: {blend_method}")
    if not (
        np.allclose(forward_weight + reverse_weight, 1.0, atol=1e-7)
        and np.isclose(forward_weight[0], 1.0, atol=1e-7)
        and np.isclose(reverse_weight[-1], 1.0, atol=1e-7)
    ):
        raise AssertionError("Temporal blend weights do not pin both anchors")
    return forward_weight.astype(np.float32), reverse_weight.astype(np.float32)


def decompose_branch_disagreement(
    forward_tracks: np.ndarray,
    backward_tracks: np.ndarray,
    nodes: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Split branch disagreement into along-line and cross-line components.

    The local long-axis tangent is estimated from the mean forward/backward
    midline. A large tangential component can arise when independently drawn
    anchor curves assign different material tissue to the same arc-length
    index; the normal component more directly measures disagreement in line
    position. Both remain QC diagnostics rather than ground truth.
    """
    section = np.asarray([int(node["section_order"]) for node in nodes])
    width_index = np.asarray([int(node["width_index"]) for node in nodes])
    n_sections = int(np.max(section)) + 1
    midline_ids = np.full(n_sections, -1, dtype=np.int32)
    for point_idx, (section_idx, width_idx) in enumerate(
        zip(section, width_index)
    ):
        if width_idx == 1:
            if midline_ids[section_idx] >= 0:
                raise ValueError(f"Duplicate midline node in section {section_idx}")
            midline_ids[section_idx] = point_idx
    if np.any(midline_ids < 0):
        raise ValueError("Each long-axis section must contain a midline node")

    center = 0.5 * (forward_tracks + backward_tracks)
    midline = center[:, midline_ids, :]
    tangent = np.empty_like(midline)
    tangent[:, 0] = midline[:, 1] - midline[:, 0]
    tangent[:, -1] = midline[:, -1] - midline[:, -2]
    if n_sections > 2:
        tangent[:, 1:-1] = midline[:, 2:] - midline[:, :-2]
    norm = np.linalg.norm(tangent, axis=2, keepdims=True)
    fallback = np.zeros_like(tangent)
    fallback[..., 0] = 1.0
    unit_tangent = np.divide(
        tangent,
        norm,
        out=fallback,
        where=norm > 1e-6,
    )
    point_tangent = unit_tangent[:, section, :]
    delta = forward_tracks - backward_tracks
    tangential = np.abs(np.sum(delta * point_tangent, axis=2))
    normal = np.abs(
        delta[..., 0] * point_tangent[..., 1]
        - delta[..., 1] * point_tangent[..., 0]
    )
    return tangential.astype(np.float32), normal.astype(np.float32)


def symmetric_same_rail_curve_distance(
    forward_tracks: np.ndarray,
    backward_tracks: np.ndarray,
    nodes: list[dict[str, Any]],
) -> np.ndarray:
    """Per-frame symmetric median nearest-curve distance on matching rails."""
    width_index = np.asarray([int(node["width_index"]) for node in nodes])
    frame_values = np.full(len(forward_tracks), np.nan, dtype=np.float32)
    for frame_idx in range(len(forward_tracks)):
        distances: list[np.ndarray] = []
        for rail_idx in (0, 1, 2):
            point_ids = np.flatnonzero(width_index == rail_idx)
            if not len(point_ids):
                continue
            forward = forward_tracks[frame_idx, point_ids]
            backward = backward_tracks[frame_idx, point_ids]
            pairwise = np.linalg.norm(
                forward[:, None, :] - backward[None, :, :], axis=2
            )
            distances.extend(
                (np.min(pairwise, axis=1), np.min(pairwise, axis=0))
            )
        if distances:
            frame_values[frame_idx] = float(np.median(np.concatenate(distances)))
    return frame_values


def track_interval(
    frames: list[np.ndarray],
    left_idx: int,
    right_idx: int,
    left_points: np.ndarray,
    right_points: np.ndarray,
    helper,
    mesh_module,
    adjacency: list[list[int]],
    edges: np.ndarray,
    cells: np.ndarray,
    nodes: list[dict[str, Any]],
    blend_method: str,
) -> tuple[dict[str, Any], dict[str, float], Counter]:
    roi_mask = np.ones_like(frames[0], dtype=np.uint8)
    forward_order = list(range(left_idx, right_idx + 1))
    backward_order = list(range(right_idx, left_idx - 1, -1))
    forward = chronological(
        mesh_module.track_order(
            frames,
            forward_order,
            left_points,
            roi_mask,
            helper,
            adjacency,
            edges,
            cells,
        ),
        reverse=False,
    )
    backward = chronological(
        mesh_module.track_order(
            frames,
            backward_order,
            right_points,
            roi_mask,
            helper,
            adjacency,
            edges,
            cells,
        ),
        reverse=True,
    )
    reasons = Counter()
    reasons.update(forward.pop("reasons"))
    reasons.update(backward.pop("reasons"))

    forward_tracks = np.asarray(forward["tracks"], dtype=np.float32)
    backward_tracks = np.asarray(backward["tracks"], dtype=np.float32)
    disagreement = np.linalg.norm(forward_tracks - backward_tracks, axis=2)
    tangential_disagreement, normal_disagreement = decompose_branch_disagreement(
        forward_tracks, backward_tracks, nodes
    )
    curve_distance = symmetric_same_rail_curve_distance(
        forward_tracks, backward_tracks, nodes
    )
    steps = len(forward_tracks)
    forward_weight, backward_weight = temporal_blend_weights(
        steps, blend_method
    )
    blended = (
        forward_weight[:, None, None] * forward_tracks
        + backward_weight[:, None, None] * backward_tracks
    )
    agree = disagreement <= MAX_BRANCH_DISAGREEMENT_PX
    # Keep the displayed path faithful to RSTC even when the two raw branches
    # disagree. Reliability is represented separately by ``feature_valid``;
    # replacing a failed blend with one raw branch would make this cease to be
    # a reproducible RSTC result and could hide a mid-interval transition.
    tracks = blended.astype(np.float32)
    feature_valid = (
        np.asarray(forward["feature_valid"], dtype=bool)
        & np.asarray(backward["feature_valid"], dtype=bool)
        & agree
    )
    finite = np.all(np.isfinite(tracks), axis=2)
    recovered = finite & ~feature_valid

    forward_reset = np.linalg.norm(forward_tracks[-1] - right_points, axis=1)
    backward_reset = np.linalg.norm(backward_tracks[0] - left_points, axis=1)
    metrics = {
        "forward_to_right_anchor_reset_median_px": float(np.median(forward_reset)),
        "forward_to_right_anchor_reset_p95_px": float(
            np.percentile(forward_reset, 95)
        ),
        "backward_to_left_anchor_reset_median_px": float(np.median(backward_reset)),
        "backward_to_left_anchor_reset_p95_px": float(
            np.percentile(backward_reset, 95)
        ),
        "branch_tangential_disagreement_median_px": float(
            np.median(tangential_disagreement[1:-1])
        ),
        "branch_tangential_disagreement_p95_px": float(
            np.percentile(tangential_disagreement[1:-1], 95)
        ),
        "branch_normal_disagreement_median_px": float(
            np.median(normal_disagreement[1:-1])
        ),
        "branch_normal_disagreement_p95_px": float(
            np.percentile(normal_disagreement[1:-1], 95)
        ),
        "same_rail_curve_distance_median_px": float(
            np.median(curve_distance[1:-1])
        ),
        "same_rail_curve_distance_p95_px": float(
            np.percentile(curve_distance[1:-1], 95)
        ),
    }

    tracks[0] = left_points
    tracks[-1] = right_points
    feature_valid[[0, -1]] = False
    recovered[[0, -1]] = False
    return (
        {
            "tracks": tracks,
            "feature_valid": feature_valid,
            "recovered": recovered,
            "branch_disagreement_px": disagreement.astype(np.float32),
            "branch_tangential_disagreement_px": tangential_disagreement,
            "branch_normal_disagreement_px": normal_disagreement,
            "same_rail_curve_distance_px": curve_distance,
            "raw_forward_tracks": forward_tracks,
            "raw_backward_tracks": backward_tracks,
            "forward_feature_valid": np.asarray(
                forward["feature_valid"], dtype=bool
            ),
            "backward_feature_valid": np.asarray(
                backward["feature_valid"], dtype=bool
            ),
        },
        metrics,
        reasons,
    )


def track_outer_segment(
    frames: list[np.ndarray],
    order: list[int],
    anchor_points: np.ndarray,
    helper,
    mesh_module,
    adjacency: list[list[int]],
    edges: np.ndarray,
    cells: np.ndarray,
) -> tuple[dict[str, Any], Counter]:
    roi_mask = np.ones_like(frames[0], dtype=np.uint8)
    local = mesh_module.track_order(
        frames,
        order,
        anchor_points,
        roi_mask,
        helper,
        adjacency,
        edges,
        cells,
    )
    reverse = len(order) >= 2 and order[1] < order[0]
    local = chronological(local, reverse=reverse)
    reasons = Counter(local.pop("reasons"))
    return local, reasons


def assemble_segmented_tracking(
    frames: list[np.ndarray],
    anchors: list[dict[str, Any]],
    points_by_anchor: dict[str, np.ndarray],
    helper,
    mesh_module,
    adjacency: list[list[int]],
    edges: np.ndarray,
    cells: np.ndarray,
    nodes: list[dict[str, Any]],
    blend_method: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], Counter]:
    total_frames = len(frames)
    n_points = len(points_by_anchor["middle"])
    anchor_indices = [int(anchor["frame_idx"]) for anchor in anchors]
    if (
        len(anchors) < 2
        or anchor_indices != sorted(set(anchor_indices))
        or anchor_indices[0] < 0
        or anchor_indices[-1] >= total_frames
    ):
        raise ValueError(
            f"Anchor order invalid: frames={anchor_indices}, T={total_frames}"
        )

    tracks = np.full((total_frames, n_points, 2), np.nan, dtype=np.float32)
    measurement_valid = np.zeros((total_frames, n_points), dtype=bool)
    recovered = np.zeros((total_frames, n_points), dtype=bool)
    disagreement = np.full((total_frames, n_points), np.nan, dtype=np.float32)
    tangential_disagreement = np.full(
        (total_frames, n_points), np.nan, dtype=np.float32
    )
    normal_disagreement = np.full(
        (total_frames, n_points), np.nan, dtype=np.float32
    )
    curve_distance = np.full(total_frames, np.nan, dtype=np.float32)
    raw_forward_tracks = np.full(
        (total_frames, n_points, 2), np.nan, dtype=np.float32
    )
    raw_backward_tracks = np.full(
        (total_frames, n_points, 2), np.nan, dtype=np.float32
    )
    forward_valid = np.zeros((total_frames, n_points), dtype=bool)
    backward_valid = np.zeros((total_frames, n_points), dtype=bool)
    dual_anchor_frame = np.zeros(total_frames, dtype=bool)
    single_anchor_extrapolation = np.zeros(total_frames, dtype=bool)
    manual_anchor_frame = np.zeros(total_frames, dtype=bool)
    reasons: Counter = Counter()
    interval_rows: list[dict[str, Any]] = []

    first_anchor = anchors[0]
    first_idx = int(first_anchor["frame_idx"])
    first_name = str(first_anchor["name"])
    if first_idx > 0:
        prefix, prefix_reasons = track_outer_segment(
            frames,
            list(range(first_idx, -1, -1)),
            points_by_anchor[first_name],
            helper,
            mesh_module,
            adjacency,
            edges,
            cells,
        )
        reasons.update(prefix_reasons)
        tracks[: first_idx + 1] = prefix["tracks"]
        measurement_valid[: first_idx + 1] = prefix["feature_valid"]
        recovered[: first_idx + 1] = prefix["recovered"]
        single_anchor_extrapolation[:first_idx] = True

    last_anchor = anchors[-1]
    last_idx = int(last_anchor["frame_idx"])
    last_name = str(last_anchor["name"])
    if last_idx < total_frames - 1:
        suffix, suffix_reasons = track_outer_segment(
            frames,
            list(range(last_idx, total_frames)),
            points_by_anchor[last_name],
            helper,
            mesh_module,
            adjacency,
            edges,
            cells,
        )
        reasons.update(suffix_reasons)
        tracks[last_idx:] = suffix["tracks"]
        measurement_valid[last_idx:] = suffix["feature_valid"]
        recovered[last_idx:] = suffix["recovered"]
        single_anchor_extrapolation[last_idx + 1 :] = True

    intervals = [
        (
            f"{left['name']}_to_{right['name']}",
            int(left["frame_idx"]),
            int(right["frame_idx"]),
            str(left["name"]),
            str(right["name"]),
        )
        for left, right in zip(anchors[:-1], anchors[1:])
    ]
    for interval_name, left_idx, right_idx, left_name, right_name in intervals:
        fused, metrics, interval_reasons = track_interval(
            frames,
            left_idx,
            right_idx,
            points_by_anchor[left_name],
            points_by_anchor[right_name],
            helper,
            mesh_module,
            adjacency,
            edges,
            cells,
            nodes,
            blend_method,
        )
        reasons.update(interval_reasons)
        slc = slice(left_idx, right_idx + 1)
        tracks[slc] = fused["tracks"]
        measurement_valid[slc] = fused["feature_valid"]
        recovered[slc] = fused["recovered"]
        disagreement[slc] = fused["branch_disagreement_px"]
        tangential_disagreement[slc] = fused[
            "branch_tangential_disagreement_px"
        ]
        normal_disagreement[slc] = fused["branch_normal_disagreement_px"]
        curve_distance[slc] = fused["same_rail_curve_distance_px"]
        raw_forward_tracks[slc] = fused["raw_forward_tracks"]
        raw_backward_tracks[slc] = fused["raw_backward_tracks"]
        forward_valid[slc] = fused["forward_feature_valid"]
        backward_valid[slc] = fused["backward_feature_valid"]
        dual_anchor_frame[slc] = True
        usable = fused["branch_disagreement_px"][1:-1]
        interval_rows.append(
            {
                "interval_name": interval_name,
                "left_frame_idx": left_idx,
                "right_frame_idx": right_idx,
                "n_frames_including_anchors": right_idx - left_idx + 1,
                "dual_measurement_valid_percent": round(
                    100.0 * float(np.mean(fused["feature_valid"][1:-1])), 4
                ),
                "branch_disagreement_median_px": round(
                    float(np.median(usable)), 4
                ),
                "branch_disagreement_p95_px": round(
                    float(np.percentile(usable, 95)), 4
                ),
                "branch_disagreement_gt2_percent": round(
                    100.0 * float(np.mean(usable > MAX_BRANCH_DISAGREEMENT_PX)), 4
                ),
                "branch_normal_disagreement_gt2_percent": round(
                    100.0
                    * float(
                        np.mean(
                            fused["branch_normal_disagreement_px"][1:-1]
                            > MAX_BRANCH_DISAGREEMENT_PX
                        )
                    ),
                    4,
                ),
                **{key: round(value, 4) for key, value in metrics.items()},
            }
        )

    anchor_points = {
        int(anchor["frame_idx"]): points_by_anchor[str(anchor["name"])]
        for anchor in anchors
    }
    for frame_idx, points in anchor_points.items():
        tracks[frame_idx] = points
        measurement_valid[frame_idx] = False
        recovered[frame_idx] = False
        manual_anchor_frame[frame_idx] = True
        single_anchor_extrapolation[frame_idx] = False

    display_valid = np.all(np.isfinite(tracks), axis=2)
    feature_candidate_valid = (
        measurement_valid
        & dual_anchor_frame[:, None]
        & ~manual_anchor_frame[:, None]
    )
    geometry_candidate_valid = (
        forward_valid
        & backward_valid
        & (normal_disagreement <= MAX_BRANCH_DISAGREEMENT_PX)
        & dual_anchor_frame[:, None]
        & ~manual_anchor_frame[:, None]
    )
    return (
        {
            "tracks": tracks,
            "tracking_measurement_valid": measurement_valid,
            "feature_candidate_valid": feature_candidate_valid,
            "geometry_candidate_valid": geometry_candidate_valid,
            "recovered_display_only": recovered,
            "display_valid_mask": display_valid,
            "branch_disagreement_px": disagreement,
            "branch_tangential_disagreement_px": tangential_disagreement,
            "branch_normal_disagreement_px": normal_disagreement,
            "same_rail_curve_distance_px": curve_distance,
            "raw_forward_tracks": raw_forward_tracks,
            "raw_backward_tracks": raw_backward_tracks,
            "forward_feature_valid": forward_valid,
            "backward_feature_valid": backward_valid,
            "dual_anchor_frame": dual_anchor_frame,
            "single_anchor_extrapolation_frame": single_anchor_extrapolation,
            "manual_anchor_frame": manual_anchor_frame,
        },
        interval_rows,
        reasons,
    )


def draw_tracking_video(
    path: Path,
    frames: list[np.ndarray],
    result: dict[str, np.ndarray],
    nodes: list[dict[str, Any]],
    edges: np.ndarray,
    case_id: str,
    anchor_indices: list[int],
    anchors: list[dict[str, Any]],
    fps: float,
    draw_transverse_rungs: bool = False,
    line_thickness: int = TRACK_LINE_THICKNESS,
    write_video: bool = True,
) -> tuple[list[np.ndarray], int]:
    height, width = frames[0].shape
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * DISPLAY_SCALE, height * DISPLAY_SCALE),
    ) if write_video else None
    if writer is not None and not writer.isOpened():
        raise RuntimeError(path)

    section = np.asarray([int(node["section_order"]) for node in nodes])
    rail = np.asarray([int(node["width_index"]) for node in nodes])
    anchor_by_frame = {
        int(anchor["frame_idx"]): anchor for anchor in anchors
    }
    disagreement = result["branch_disagreement_px"]
    normal_disagreement = result["branch_normal_disagreement_px"]
    disagreement_frames = np.any(np.isfinite(disagreement), axis=1)
    finite_frame_disagreement = np.full(len(frames), -np.inf, dtype=np.float32)
    finite_frame_disagreement[disagreement_frames] = np.nanmedian(
        disagreement[disagreement_frames], axis=1
    )
    normal_frames = np.any(np.isfinite(normal_disagreement), axis=1)
    finite_frame_normal = np.full(len(frames), -np.inf, dtype=np.float32)
    finite_frame_normal[normal_frames] = np.nanmedian(
        normal_disagreement[normal_frames], axis=1
    )
    worst_frame = int(np.argmax(finite_frame_disagreement))
    worst_normal_frame = int(np.argmax(finite_frame_normal))
    sample_indices = sorted(
        {
            0,
            *anchor_indices,
            len(frames) - 1,
            worst_frame,
            worst_normal_frame,
        }
    )
    samples: list[np.ndarray] = []

    for frame_idx, gray in enumerate(frames):
        if writer is None and frame_idx not in sample_indices:
            continue
        vis = cv2.resize(
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE,
        )
        tracking_points = result["tracks"][frame_idx]
        points = tracking_points * DISPLAY_SCALE
        is_manual = bool(result["manual_anchor_frame"][frame_idx])
        is_single = bool(result["single_anchor_extrapolation_frame"][frame_idx])
        candidate_valid = result["feature_candidate_valid"][frame_idx]
        geometry_valid = result["geometry_candidate_valid"][frame_idx]
        display_valid = result["display_valid_mask"][frame_idx]
        frame_disagreement = result["branch_disagreement_px"][frame_idx]
        frame_normal_disagreement = result[
            "branch_normal_disagreement_px"
        ][frame_idx]

        manual_anchor = anchor_by_frame.get(frame_idx)
        if manual_anchor is not None:
            for key in ("ant_raw", "post_raw"):
                cv2.polylines(
                    vis,
                    [
                        np.rint(
                            np.asarray(manual_anchor[key]) * DISPLAY_SCALE
                        ).astype(np.int32)
                    ],
                    False,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )

        anterior_ids = np.flatnonzero(rail == 0)
        anterior_ids = anterior_ids[
            np.argsort(section[anterior_ids])
        ]
        smooth_count = min(SMOOTH_POINT_COUNT, len(anterior_ids))
        smooth_ids = anterior_ids[-smooth_count:]
        can_smooth = (
            len(smooth_ids) >= 3
            and np.all(display_valid[smooth_ids])
        )
        smooth_edge_pairs = {
            tuple(sorted((int(left), int(right))))
            for left, right in zip(smooth_ids[:-1], smooth_ids[1:])
        }

        for left_value, right_value in edges:
            left, right = int(left_value), int(right_value)
            if not (display_valid[left] and display_valid[right]):
                continue
            same_section = section[left] == section[right]
            # Review renderers can choose between the three longitudinal
            # anatomical rails and the complete connected mesh.
            if same_section and not draw_transverse_rungs:
                continue
            same_rail = not same_section and rail[left] == rail[right]
            rail_idx = int(rail[left]) if same_rail else -1
            if (
                can_smooth
                and rail_idx == 0
                and tuple(sorted((left, right))) in smooth_edge_pairs
            ):
                continue
            color = RAIL_COLORS.get(rail_idx, (160, 160, 160))
            cv2.line(
                vis,
                tuple(np.round(points[left]).astype(int)),
                tuple(np.round(points[right]).astype(int)),
                color,
                line_thickness,
                cv2.LINE_AA,
            )

        if can_smooth:
            if manual_anchor is not None:
                smooth_curve = np.asarray(
                    manual_anchor["ant_raw"], dtype=np.float32
                )
            else:
                smooth_curve, _ = smooth_interpolating_curve(
                    tracking_points[smooth_ids]
                )
            cv2.polylines(
                vis,
                [
                    np.rint(smooth_curve * DISPLAY_SCALE).astype(
                        np.int32
                    )
                ],
                False,
                RAIL_COLORS[0],
                line_thickness,
                cv2.LINE_AA,
            )

        for point_idx, point in enumerate(points):
            if not display_valid[point_idx]:
                continue
            color = RAIL_COLORS[int(rail[point_idx])]
            center = tuple(np.round(point).astype(int))
            cv2.circle(
                vis,
                center,
                TRACK_POINT_HALO_RADIUS,
                (0, 0, 0),
                -1,
                cv2.LINE_AA,
            )
            if (
                np.isfinite(frame_normal_disagreement[point_idx])
                and frame_normal_disagreement[point_idx]
                > MAX_BRANCH_DISAGREEMENT_PX
            ):
                cv2.circle(
                    vis,
                    center,
                    TRACK_POINT_RADIUS + 2,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            elif (
                np.isfinite(frame_disagreement[point_idx])
                and frame_disagreement[point_idx]
                > MAX_BRANCH_DISAGREEMENT_PX
            ):
                cv2.circle(
                    vis,
                    center,
                    TRACK_POINT_RADIUS + 2,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.circle(
                vis,
                center,
                TRACK_POINT_RADIUS,
                color,
                -1,
                cv2.LINE_AA,
            )

        if is_manual:
            mode = "MANUAL ANCHOR"
        elif is_single:
            mode = "SINGLE-ANCHOR EXTRAPOLATION / REVIEW ONLY"
        else:
            mode = "DUAL-ANCHOR BIDIRECTIONAL CONSENSUS"
        valid_count = int(np.count_nonzero(candidate_valid))
        geometry_valid_count = int(np.count_nonzero(geometry_valid))
        finite_disagreement = frame_disagreement[np.isfinite(frame_disagreement)]
        disagreement_text = (
            f"{float(np.median(finite_disagreement)):.2f}px"
            if len(finite_disagreement)
            else "n/a"
        )
        finite_normal = frame_normal_disagreement[
            np.isfinite(frame_normal_disagreement)
        ]
        normal_text = (
            f"{float(np.median(finite_normal)):.2f}px"
            if len(finite_normal)
            else "n/a"
        )
        cv2.rectangle(vis, (6, 6), (1180, 132), (0, 0, 0), -1)
        cv2.putText(
            vis,
            f"{case_id} frame {frame_idx} | {mode}",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.61,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"strict point nodes {valid_count}/{len(nodes)} | line-normal nodes {geometry_valid_count}/{len(nodes)}",
            (16, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"median point disagreement {disagreement_text} | median line-normal disagreement {normal_text}",
            (16, 93),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            (
                "small dots=true-wall nodes | all adjacent points connected | "
                "pink ring=branch mismatch | red ring=line-position mismatch"
                if draw_transverse_rungs
                else
                "small dots=true-wall nodes | thin lines=anatomical rails | "
                "pink ring=branch mismatch | red ring=line-position mismatch"
            ),
            (16, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        if writer is not None:
            writer.write(vis)
        if frame_idx in sample_indices:
            samples.append(vis.copy())
    if writer is not None:
        writer.release()
    return samples, worst_frame


def write_contact_sheet(
    path: Path,
    rows: list[tuple[str, list[np.ndarray]]],
    panel_width: int = 300,
) -> None:
    rendered_rows = []
    max_panels = max(len(images) for _, images in rows)
    for case_id, images in rows:
        panels = []
        for image in images:
            height = int(round(panel_width * image.shape[0] / image.shape[1]))
            panels.append(
                cv2.resize(image, (panel_width, height), interpolation=cv2.INTER_AREA)
            )
        while len(panels) < max_panels:
            panels.append(np.zeros_like(panels[0]))
        row = np.hstack(panels)
        cv2.putText(
            row,
            case_id,
            (8, row.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rendered_rows.append(row)
    encoded, buffer = cv2.imencode(".png", np.vstack(rendered_rows))
    if not encoded:
        raise RuntimeError(f"Failed to encode contact sheet: {path}")
    buffer.tofile(str(path))


def safe_percent(values: np.ndarray) -> float:
    return round(100.0 * float(np.mean(values)), 4) if values.size else math.nan


def process_case(
    root: Path,
    output_dir: Path,
    manifest: dict[tuple[str, str], dict[str, str]],
    helper,
    mesh_module,
    case_id: str,
    validate_only: bool,
    blend_method: str,
    run_tag: str,
    include_endpoint_anchors: bool,
    external_label_dir: Path | None = None,
    external_resize_factor: float = 0.5,
    external_n_sections: int = 10,
    skip_videos: bool = False,
) -> tuple[
    dict[str, Any] | None,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[np.ndarray],
    list[np.ndarray],
]:
    if external_label_dir is None:
        video_path, meta_path = input_paths(root, case_id)
        meta = load_json(meta_path)
        resize_factor = float(meta["parameters"]["resize_factor"])
        middle_idx = (
            int(meta["middle_frame_index"])
            if meta.get("middle_frame_index") is not None
            else None
        )
        n_sections = int(meta["n_long_sections"])
    else:
        middle_row = manifest[(case_id, "middle")]
        video_path = Path(middle_row["source_video"])
        meta = None
        resize_factor = float(external_resize_factor)
        middle_idx = int(middle_row["selected_frame_idx"])
        n_sections = int(external_n_sections)
    if not np.isfinite(resize_factor) or resize_factor <= 0.0:
        raise ValueError("resize_factor must be finite and positive")
    if n_sections < 3:
        raise ValueError("n_sections must be at least 3")
    frames, video_info = helper.read_video_gray(video_path, resize_factor)
    if middle_idx is None:
        middle_idx = len(frames) // 2
    if not 0 <= middle_idx < len(frames):
        raise ValueError(f"{case_id}: middle frame is outside the video")
    if external_label_dir is not None:
        for anchor_name in ENDPOINT_ANCHOR_ORDER:
            frame_idx = int(manifest[(case_id, anchor_name)]["selected_frame_idx"])
            if not 0 <= frame_idx < len(frames):
                raise ValueError(
                    f"{case_id}: {anchor_name} frame {frame_idx} is outside "
                    f"the {len(frames)}-frame video"
                )

    manual_anchor_names = (
        ("video_start", "front_half_center", "back_half_center", "video_end")
        if include_endpoint_anchors
        else ("front_half_center", "back_half_center")
    )
    manual_anchors: dict[str, dict[str, Any]] = {}
    for anchor_name in manual_anchor_names:
        row = manifest[(case_id, anchor_name)]
        json_path = Path(row["output_png"]).with_suffix(".json")
        if not json_path.is_absolute():
            json_path = root / json_path
        if not json_path.is_file():
            raise FileNotFoundError(json_path)
        manual_anchors[anchor_name] = load_manual_anchor(
            json_path,
            int(row["selected_frame_idx"]),
            anchor_name,
            resize_factor,
            n_sections,
        )
    if external_label_dir is None:
        middle_anchor = load_middle_anchor(
            root, case_id, meta, middle_idx, resize_factor
        )
    else:
        middle_json = Path(manifest[(case_id, "middle")]["output_png"]).with_suffix(
            ".json"
        )
        middle_anchor = load_manual_anchor(
            middle_json,
            middle_idx,
            "middle",
            resize_factor,
            n_sections,
        )
    # The accepted Step 1.4X geometry rebuilt the middle anchor directly from
    # the original LabelMe wall polylines. Keep that exact behavior instead of
    # reusing the older Step 1.3 section coordinates.
    middle_anchor["ant"] = resample_curve(
        middle_anchor["ant_raw"], n_sections
    )
    middle_anchor["post"] = resample_curve(
        middle_anchor["post_raw"], n_sections
    )
    anchors = [
        manual_anchors["front_half_center"],
        middle_anchor,
        manual_anchors["back_half_center"],
    ]
    expected_order = ANCHOR_ORDER
    if include_endpoint_anchors:
        anchors = [
            manual_anchors["video_start"],
            *anchors,
            manual_anchors["video_end"],
        ]
        expected_order = ENDPOINT_ANCHOR_ORDER
    if [anchor["name"] for anchor in anchors] != list(expected_order):
        raise AssertionError("Anchor ordering bug")

    nodes, points_by_anchor, narrow = build_shared_nodes(anchors, mesh_module)
    edges, cells, adjacency = mesh_module.build_mesh(nodes)
    validation_rows = [
        anchor_validation_row(case_id, anchor, narrow) for anchor in anchors
    ]
    anchor_overlays = [
        draw_anchor_overlay(
            frames[int(anchor["frame_idx"])], anchor, narrow, case_id
        )
        for anchor in anchors
    ]
    if validate_only:
        return None, validation_rows, [], anchor_overlays, []

    result, interval_rows, reasons = assemble_segmented_tracking(
        frames,
        anchors,
        points_by_anchor,
        helper,
        mesh_module,
        adjacency,
        edges,
        cells,
        nodes,
        blend_method,
    )
    for row in interval_rows:
        row["case_id"] = case_id

    review_indices = [
        (
            int(anchors[left]["frame_idx"])
            + int(anchors[left + 1]["frame_idx"])
        )
        // 2
        for left in range(len(anchors) - 1)
    ]
    keep_sections, removed_fundus_sections, spacing_before, spacing_after = (
        fundus_endpoint_keep_sections(
            result["tracks"][review_indices],
            nodes,
            narrow,
        )
    )
    result, nodes, points_by_anchor, kept_point_ids = subset_tracking_points(
        result,
        nodes,
        points_by_anchor,
        keep_sections,
    )
    narrow = narrow[np.asarray(keep_sections, dtype=np.int32)]
    edges, cells, _ = mesh_module.build_mesh(nodes)

    topology_bad_edge, topology_bad_cell = mesh_module.topology_violation_counts(
        result["tracks"],
        result["display_valid_mask"],
        points_by_anchor["middle"],
        edges,
        cells,
    )
    video_output = output_dir / f"{case_id}_{run_tag}_sparse_anchor_overlay.mp4"
    samples, worst_frame = draw_tracking_video(
        video_output,
        frames,
        result,
        nodes,
        edges,
        case_id,
        [int(anchor["frame_idx"]) for anchor in anchors],
        anchors,
        float(video_info["fps"]),
        write_video=not skip_videos,
    )

    np.savez_compressed(
        output_dir / f"{case_id}_{run_tag}_sparse_anchor_tracks.npz",
        **result,
        edges=edges,
        cells=cells,
        shared_narrow_sections=narrow,
        anchor_indices=np.asarray(
            [int(anchor["frame_idx"]) for anchor in anchors], dtype=np.int32
        ),
        anchor_names=np.asarray(
            [str(anchor["name"]) for anchor in anchors]
        ),
        anchor_front_points=points_by_anchor["front_half_center"],
        anchor_middle_points=points_by_anchor["middle"],
        anchor_back_points=points_by_anchor["back_half_center"],
        anchor_video_start_points=points_by_anchor.get(
            "video_start", np.empty((0, 2), dtype=np.float32)
        ),
        anchor_video_end_points=points_by_anchor.get(
            "video_end", np.empty((0, 2), dtype=np.float32)
        ),
        kept_original_point_ids=kept_point_ids,
        removed_fundus_sections=np.asarray(
            removed_fundus_sections, dtype=np.int32
        ),
        fundus_spacing_ratio_before=spacing_before,
        fundus_spacing_ratio_after=spacing_after,
        true_wall_u_levels=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        topology_bad_edge_count=topology_bad_edge,
        topology_bad_cell_count=topology_bad_cell,
    )
    (output_dir / f"{case_id}_{run_tag}_nodes.json").write_text(
        json.dumps(nodes, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    dual_non_anchor = (
        result["dual_anchor_frame"] & ~result["manual_anchor_frame"]
    )
    dual_valid = result["feature_candidate_valid"][dual_non_anchor]
    frame_valid_fraction = np.mean(
        result["feature_candidate_valid"], axis=1
    )
    disagreement = result["branch_disagreement_px"][dual_non_anchor]
    finite_disagreement = disagreement[np.isfinite(disagreement)]
    tangential_disagreement = result[
        "branch_tangential_disagreement_px"
    ][dual_non_anchor]
    finite_tangential_disagreement = tangential_disagreement[
        np.isfinite(tangential_disagreement)
    ]
    normal_disagreement = result[
        "branch_normal_disagreement_px"
    ][dual_non_anchor]
    finite_normal_disagreement = normal_disagreement[
        np.isfinite(normal_disagreement)
    ]
    curve_distance = result["same_rail_curve_distance_px"][dual_non_anchor]
    finite_curve_distance = curve_distance[np.isfinite(curve_distance)]
    prefix = result["single_anchor_extrapolation_frame"].copy()
    prefix[int(anchors[0]["frame_idx"]) :] = False
    suffix = result["single_anchor_extrapolation_frame"].copy()
    suffix[: int(anchors[-1]["frame_idx"]) + 1] = False
    max_anchor_error = 0.0
    for anchor in anchors:
        frame_idx = int(anchor["frame_idx"])
        error = np.linalg.norm(
            result["tracks"][frame_idx] - points_by_anchor[str(anchor["name"])],
            axis=1,
        )
        max_anchor_error = max(max_anchor_error, float(np.max(error)))

    anchor_by_name = {str(anchor["name"]): anchor for anchor in anchors}
    summary = {
        "case_id": case_id,
        "run_tag": run_tag,
        "blend_method": blend_method,
        "rstc_epsilon": RSTC_EPSILON if blend_method == "rstc" else math.nan,
        "n_frames": len(frames),
        "fps": round(float(video_info["fps"]), 6),
        "endpoint_anchors_enabled": include_endpoint_anchors,
        "start_anchor_frame": (
            int(anchor_by_name["video_start"]["frame_idx"])
            if include_endpoint_anchors
            else ""
        ),
        "front_anchor_frame": int(
            anchor_by_name["front_half_center"]["frame_idx"]
        ),
        "middle_anchor_frame": int(anchor_by_name["middle"]["frame_idx"]),
        "back_anchor_frame": int(
            anchor_by_name["back_half_center"]["frame_idx"]
        ),
        "end_anchor_frame": (
            int(anchor_by_name["video_end"]["frame_idx"])
            if include_endpoint_anchors
            else ""
        ),
        "n_sections_before_fundus_dedup": n_sections,
        "n_sections": len(keep_sections),
        "n_shared_narrow_sections": int(np.count_nonzero(narrow)),
        "n_tracking_nodes": len(nodes),
        "true_wall_u_levels": "0.0|0.5|1.0",
        "removed_fundus_sections": json.dumps(
            removed_fundus_sections
        ),
        "fundus_spacing_ratio_before_min": round(
            float(np.nanmin(spacing_before)), 4
        ),
        "fundus_spacing_ratio_after_min": round(
            float(np.nanmin(spacing_after)), 4
        ),
        "dual_anchor_video_frame_percent": safe_percent(dual_non_anchor),
        "single_anchor_extrapolation_frame_percent": safe_percent(
            result["single_anchor_extrapolation_frame"]
        ),
        "dual_anchor_feature_candidate_valid_percent": safe_percent(dual_valid),
        "dual_anchor_frame_ge80pct_valid_percent": safe_percent(
            frame_valid_fraction[dual_non_anchor] >= MIN_FRAME_VALID_FRACTION
        ),
        "branch_disagreement_median_px": round(
            float(np.median(finite_disagreement)), 4
        ),
        "branch_disagreement_p95_px": round(
            float(np.percentile(finite_disagreement, 95)), 4
        ),
        "branch_disagreement_gt2_percent": safe_percent(
            finite_disagreement > MAX_BRANCH_DISAGREEMENT_PX
        ),
        "branch_tangential_disagreement_median_px": round(
            float(np.median(finite_tangential_disagreement)), 4
        ),
        "branch_tangential_disagreement_p95_px": round(
            float(np.percentile(finite_tangential_disagreement, 95)), 4
        ),
        "branch_normal_disagreement_median_px": round(
            float(np.median(finite_normal_disagreement)), 4
        ),
        "branch_normal_disagreement_p95_px": round(
            float(np.percentile(finite_normal_disagreement, 95)), 4
        ),
        "branch_normal_disagreement_gt2_percent": safe_percent(
            finite_normal_disagreement > MAX_BRANCH_DISAGREEMENT_PX
        ),
        "branch_tangential_component_gt_normal_percent": safe_percent(
            finite_tangential_disagreement > finite_normal_disagreement
        ),
        "same_rail_curve_distance_median_px": round(
            float(np.median(finite_curve_distance)), 4
        ),
        "same_rail_curve_distance_p95_px": round(
            float(np.percentile(finite_curve_distance, 95)), 4
        ),
        "dual_anchor_geometry_candidate_valid_percent": safe_percent(
            result["geometry_candidate_valid"][dual_non_anchor]
        ),
        "prefix_single_anchor_measurement_valid_percent": safe_percent(
            result["tracking_measurement_valid"][prefix]
        ),
        "suffix_single_anchor_measurement_valid_percent": safe_percent(
            result["tracking_measurement_valid"][suffix]
        ),
        "topology_bad_edge_frames": int(np.count_nonzero(topology_bad_edge)),
        "topology_bad_cell_frames": int(np.count_nonzero(topology_bad_cell)),
        "max_manual_anchor_coordinate_error_px": round(max_anchor_error, 6),
        "worst_branch_disagreement_frame": worst_frame,
        "reason_counts": json.dumps(
            dict(reasons), ensure_ascii=False, sort_keys=True
        ),
        "video": "" if skip_videos else video_output.name,
        "status": "final_step1_4_tracking_qc_not_downstream_released",
    }
    print(json.dumps(summary, ensure_ascii=False))
    return summary, validation_rows, interval_rows, anchor_overlays, samples


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final sparse-anchor true-wall bidirectional tracking"
    )
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-videos", action="store_true", help="Save numerical tracks and QC stills without encoding MP4.")
    parser.add_argument(
        "--include-endpoint-anchors",
        action="store_true",
        help=(
            "读取人工标注的首帧和末帧，使整段视频都位于两个手工锚点之间；"
            "不加此参数时保持原三锚点结果。"
        ),
    )
    parser.add_argument(
        "--blend",
        choices=("linear", "rstc"),
        default="rstc",
        help="Temporal fusion; rstc matches the official DUSTrack sigmoid.",
    )
    parser.add_argument(
        "--run-tag",
        default="step1_4_final",
        help="Safe filename prefix (default: step1_4_final).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to the final Step 1.4 QC directory.",
    )
    parser.add_argument(
        "--external-label-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing real-case five-anchor JSON/PNG pairs and "
            "<case_id>__five_anchor_manifest.csv files."
        ),
    )
    parser.add_argument(
        "--resize-factor",
        type=float,
        default=0.5,
        help="Frame resize factor for external inputs (default: 0.5).",
    )
    parser.add_argument(
        "--sections",
        type=int,
        default=10,
        help="Long-axis wall sections for external inputs (default: 10).",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    args = parser.parse_args()

    if not args.cases:
        raise ValueError("--cases must contain at least one case_id")
    if len(set(args.cases)) != len(args.cases):
        raise ValueError("--cases contains duplicate case_id values")
    if not args.run_tag.replace("_", "").isalnum():
        raise ValueError("--run-tag may contain only letters, digits, underscores")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else PROJECT_ROOT / "输入" / "03_上游解剖节点"
        / "step1_4_final_sparse_anchor_wall_tracking_5case"
    )
    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not args.allow_overwrite
    ):
        raise FileExistsError(
            f"Refusing to overwrite non-empty Step 1.4 directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    root = module_root_from_script()
    mesh_module = tracking_mesh
    helper = tracking_lk
    external_label_dir = (
        args.external_label_dir.resolve()
        if args.external_label_dir is not None
        else None
    )
    if external_label_dir is None:
        manifest = load_manifest(
            root,
            include_endpoint_anchors=args.include_endpoint_anchors,
            cases=args.cases,
        )
    else:
        manifest = load_external_manifests(external_label_dir, args.cases)

    summaries: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    anchor_contact_rows: list[tuple[str, list[np.ndarray]]] = []
    tracking_contact_rows: list[tuple[str, list[np.ndarray]]] = []
    for case_id in args.cases:
        summary, case_validation, case_intervals, anchors, samples = process_case(
            root,
            output_dir,
            manifest,
            helper,
            mesh_module,
            case_id,
            validate_only=args.validate_only,
            blend_method=args.blend,
            run_tag=args.run_tag,
            include_endpoint_anchors=args.include_endpoint_anchors,
            external_label_dir=external_label_dir,
            external_resize_factor=args.resize_factor,
            external_n_sections=args.sections,
            skip_videos=args.skip_videos,
        )
        validation_rows.extend(case_validation)
        interval_rows.extend(case_intervals)
        anchor_contact_rows.append((case_id, anchors))
        if summary is not None:
            summaries.append(summary)
            tracking_contact_rows.append((case_id, samples))

    write_csv(
        output_dir / f"{args.run_tag}_anchor_annotation_validation.csv",
        validation_rows,
    )
    write_contact_sheet(
        output_dir / f"{args.run_tag}_anchor_annotation_contact_sheet.png",
        anchor_contact_rows,
        panel_width=420,
    )
    if summaries:
        write_csv(output_dir / f"{args.run_tag}_summary.csv", summaries)
        write_csv(
            output_dir / f"{args.run_tag}_interval_summary.csv", interval_rows
        )
        write_contact_sheet(
            output_dir / f"{args.run_tag}_tracking_contact_sheet.png",
            tracking_contact_rows,
            panel_width=300,
        )
    print(f"complete: {output_dir}")


if __name__ == "__main__":
    main()
