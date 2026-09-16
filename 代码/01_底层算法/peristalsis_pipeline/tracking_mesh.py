#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Topology-aware tracking primitives for the maintained Step 1.4 pipeline."""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np


MIN_INDEPENDENT_SPACING_PX = 4.0
MAX_NEIGHBOR_RESIDUAL_PX = 2.5
BLEND_START_RESIDUAL_PX = 1.25
PREDICTION_BLEND = 0.15
MIN_EDGE_RATIO = 0.45
MAX_EDGE_RATIO = 2.50
MIN_CELL_AREA_RATIO = 0.20


def build_mesh(nodes: list[dict]) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    lookup = {
        (int(node["section_order"]), int(node["width_index"])): int(node["point_id"])
        for node in nodes
    }
    n_sections = 1 + max(int(node["section_order"]) for node in nodes)
    edges: list[tuple[int, int]] = []
    cells: list[tuple[int, int, int, int]] = []
    for section_idx in range(n_sections):
        if all((section_idx, width_idx) in lookup for width_idx in (0, 1, 2)):
            edges.extend(
                [
                    (lookup[(section_idx, 0)], lookup[(section_idx, 1)]),
                    (lookup[(section_idx, 1)], lookup[(section_idx, 2)]),
                ]
            )
    for section_idx in range(n_sections - 1):
        for width_idx in (0, 1, 2):
            if (section_idx, width_idx) in lookup and (section_idx + 1, width_idx) in lookup:
                edges.append((lookup[(section_idx, width_idx)], lookup[(section_idx + 1, width_idx)]))
        for width_idx in (0, 1):
            keys = [
                (section_idx, width_idx),
                (section_idx + 1, width_idx),
                (section_idx + 1, width_idx + 1),
                (section_idx, width_idx + 1),
            ]
            if all(key in lookup for key in keys):
                cells.append(tuple(lookup[key] for key in keys))
    edge_array = np.asarray(sorted(set(tuple(sorted(edge)) for edge in edges)), dtype=np.int32)
    cell_array = np.asarray(cells, dtype=np.int32).reshape(-1, 4)
    adjacency = [[] for _ in nodes]
    for left, right in edge_array:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    return edge_array, cell_array, adjacency


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def neighbor_predictions(
    current: np.ndarray,
    candidate: np.ndarray,
    raw_good: np.ndarray,
    adjacency: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    displacement = candidate - current
    predictions = np.full_like(current, np.nan)
    available = np.zeros(len(current), dtype=bool)
    global_indices = np.flatnonzero(raw_good)
    global_displacement = np.median(displacement[global_indices], axis=0) if len(global_indices) >= 3 else None
    for point_idx, neighbors in enumerate(adjacency):
        valid_neighbors = [neighbor for neighbor in neighbors if raw_good[neighbor]]
        if valid_neighbors:
            predictions[point_idx] = current[point_idx] + np.median(displacement[valid_neighbors], axis=0)
            available[point_idx] = True
        elif global_displacement is not None:
            predictions[point_idx] = current[point_idx] + global_displacement
            available[point_idx] = True
    return predictions, available


def repair_topology(
    proposed: np.ndarray,
    current: np.ndarray,
    predictions: np.ndarray,
    feature_valid: np.ndarray,
    recovered: np.ndarray,
    edges: np.ndarray,
    reference_edge_lengths: np.ndarray,
    cells: np.ndarray,
    reference_cell_areas: np.ndarray,
) -> np.ndarray:
    repaired = np.zeros(len(proposed), dtype=bool)

    def restore(indices: np.ndarray) -> None:
        finite_prediction = np.all(np.isfinite(predictions[indices]), axis=1)
        if np.any(finite_prediction):
            delta = np.median(predictions[indices][finite_prediction] - current[indices][finite_prediction], axis=0)
        else:
            delta = np.median(proposed[indices] - current[indices], axis=0)
        proposed[indices] = current[indices] + delta
        feature_valid[indices] = False
        recovered[indices] = True
        repaired[indices] = True

    for edge_idx, (left_value, right_value) in enumerate(edges):
        left, right = int(left_value), int(right_value)
        ratio = float(np.linalg.norm(proposed[right] - proposed[left]) / reference_edge_lengths[edge_idx])
        if ratio < MIN_EDGE_RATIO or ratio > MAX_EDGE_RATIO:
            restore(np.asarray([left, right], dtype=np.int32))

    for cell_idx, cell in enumerate(cells):
        area = polygon_area(proposed[cell])
        reference_area = float(reference_cell_areas[cell_idx])
        if area * reference_area <= 0 or abs(area) < MIN_CELL_AREA_RATIO * abs(reference_area):
            restore(cell)

    # Shared nodes make one-pass edge repairs interfere with each other. A
    # short position-based projection resolves all edge bounds jointly without
    # forcing the rails onto a pre-defined smooth curve.
    for _ in range(16):
        for edge_idx, (left_value, right_value) in enumerate(edges):
            left, right = int(left_value), int(right_value)
            vector = proposed[right] - proposed[left]
            distance = float(np.linalg.norm(vector))
            if distance < 1e-6:
                vector = current[right] - current[left]
                distance = float(np.linalg.norm(vector))
            minimum = MIN_EDGE_RATIO * float(reference_edge_lengths[edge_idx])
            maximum = MAX_EDGE_RATIO * float(reference_edge_lengths[edge_idx])
            target = min(max(distance, minimum), maximum)
            if abs(target - distance) <= 1e-4 or distance < 1e-6:
                continue
            adjustment = 0.5 * (target - distance) * vector / distance
            proposed[left] -= adjustment
            proposed[right] += adjustment
            feature_valid[[left, right]] = False
            recovered[[left, right]] = True
            repaired[[left, right]] = True

    for cell_idx, cell in enumerate(cells):
        area = polygon_area(proposed[cell])
        reference_area = float(reference_cell_areas[cell_idx])
        if area * reference_area <= 0 or abs(area) < MIN_CELL_AREA_RATIO * abs(reference_area):
            restore(cell)
    for _ in range(8):
        for edge_idx, (left_value, right_value) in enumerate(edges):
            left, right = int(left_value), int(right_value)
            vector = proposed[right] - proposed[left]
            distance = float(np.linalg.norm(vector))
            if distance < 1e-6:
                continue
            target = min(
                max(distance, MIN_EDGE_RATIO * float(reference_edge_lengths[edge_idx])),
                MAX_EDGE_RATIO * float(reference_edge_lengths[edge_idx]),
            )
            if abs(target - distance) > 1e-4:
                adjustment = 0.5 * (target - distance) * vector / distance
                proposed[left] -= adjustment
                proposed[right] += adjustment
                feature_valid[[left, right]] = False
                recovered[[left, right]] = True
                repaired[[left, right]] = True
    return repaired


def track_order(
    frames: list[np.ndarray],
    order: list[int],
    init_points: np.ndarray,
    roi_mask: np.ndarray,
    helper,
    adjacency: list[list[int]],
    edges: np.ndarray,
    cells: np.ndarray,
) -> dict[str, np.ndarray | Counter]:
    steps, n_points = len(order), len(init_points)
    arrays = {
        "tracks": np.full((steps, n_points, 2), np.nan, dtype=np.float32),
        "raw_tracks": np.full((steps, n_points, 2), np.nan, dtype=np.float32),
        "feature_valid": np.zeros((steps, n_points), dtype=bool),
        "raw_good": np.zeros((steps, n_points), dtype=bool),
        "recovered": np.zeros((steps, n_points), dtype=bool),
        "topology_repaired": np.zeros((steps, n_points), dtype=bool),
        "inside_roi": np.zeros((steps, n_points), dtype=bool),
        "fb_error": np.full((steps, n_points), np.nan, dtype=np.float32),
        "displacement": np.full((steps, n_points), np.nan, dtype=np.float32),
        "correction_px": np.zeros((steps, n_points), dtype=np.float32),
    }
    reference_edge_lengths = np.linalg.norm(init_points[edges[:, 1]] - init_points[edges[:, 0]], axis=1)
    reference_cell_areas = np.asarray([polygon_area(init_points[cell]) for cell in cells], dtype=np.float32)
    current = init_points.copy()
    arrays["tracks"][0] = current
    arrays["raw_tracks"][0] = current
    arrays["feature_valid"][0] = True
    arrays["raw_good"][0] = True
    arrays["inside_roi"][0] = helper.inside_points(current, roi_mask)
    reasons: Counter = Counter()

    for step_idx in range(1, steps):
        previous_gray = frames[order[step_idx - 1]]
        next_gray = frames[order[step_idx]]
        previous_points = current.reshape(-1, 1, 2).astype(np.float32)
        next_points, status_forward, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, next_gray, previous_points, None, **helper.LK_PARAMS
        )
        if next_points is None or status_forward is None:
            candidate = current.copy()
            raw_good = np.zeros(n_points, dtype=bool)
            fb_error = np.full(n_points, np.nan, dtype=np.float32)
            displacement = np.full(n_points, np.nan, dtype=np.float32)
            reasons["lk_status"] += n_points
        else:
            back_points, status_backward, _ = cv2.calcOpticalFlowPyrLK(
                next_gray, previous_gray, next_points, None, **helper.LK_PARAMS
            )
            if back_points is None or status_backward is None:
                back_points = np.full_like(previous_points, np.nan)
                status_backward = np.zeros_like(status_forward)
            candidate = next_points.reshape(-1, 2).astype(np.float32)
            back = back_points.reshape(-1, 2).astype(np.float32)
            displacement = np.linalg.norm(candidate - current, axis=1).astype(np.float32)
            fb_error = np.linalg.norm(back - current, axis=1).astype(np.float32)
            status_good = status_forward.reshape(-1).astype(bool) & status_backward.reshape(-1).astype(bool)
            raw_good = (
                status_good
                & np.isfinite(fb_error)
                & np.isfinite(displacement)
                & (fb_error <= helper.FB_ERROR_THRESHOLD_PX)
                & (displacement <= helper.MAX_SINGLE_FRAME_DISPLACEMENT_PX)
            )
            reasons["lk_status"] += int(np.count_nonzero(~status_good))
            reasons["fb_error"] += int(np.count_nonzero(status_good & (fb_error > helper.FB_ERROR_THRESHOLD_PX)))
            reasons["displacement"] += int(
                np.count_nonzero(status_good & np.isfinite(fb_error) & (fb_error <= helper.FB_ERROR_THRESHOLD_PX) & (displacement > helper.MAX_SINGLE_FRAME_DISPLACEMENT_PX))
            )

        predictions, prediction_available = neighbor_predictions(current, candidate, raw_good, adjacency)
        residual = np.linalg.norm(candidate - predictions, axis=1)
        feature_valid = raw_good & ((~prediction_available) | (residual <= MAX_NEIGHBOR_RESIDUAL_PX))
        recovered = ~feature_valid & prediction_available
        proposed = current.copy()
        proposed[recovered] = predictions[recovered]
        proposed[feature_valid] = candidate[feature_valid]
        blend = feature_valid & (residual > BLEND_START_RESIDUAL_PX)
        proposed[blend] = (1.0 - PREDICTION_BLEND) * candidate[blend] + PREDICTION_BLEND * predictions[blend]
        topology_repaired = repair_topology(
            proposed, current, predictions, feature_valid, recovered, edges,
            reference_edge_lengths, cells, reference_cell_areas,
        )
        correction = np.linalg.norm(proposed - candidate, axis=1).astype(np.float32)
        correction[~np.all(np.isfinite(candidate), axis=1)] = np.nan

        arrays["tracks"][step_idx] = proposed
        arrays["raw_tracks"][step_idx] = candidate
        arrays["feature_valid"][step_idx] = feature_valid
        arrays["raw_good"][step_idx] = raw_good
        arrays["recovered"][step_idx] = recovered
        arrays["topology_repaired"][step_idx] = topology_repaired
        arrays["inside_roi"][step_idx] = helper.inside_points(proposed, roi_mask)
        arrays["fb_error"][step_idx] = fb_error
        arrays["displacement"][step_idx] = displacement
        arrays["correction_px"][step_idx] = correction
        current = proposed
    arrays["reasons"] = reasons
    return arrays


def topology_violation_counts(
    tracks: np.ndarray,
    valid: np.ndarray,
    init_points: np.ndarray,
    edges: np.ndarray,
    cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    edge_reference = np.linalg.norm(init_points[edges[:, 1]] - init_points[edges[:, 0]], axis=1)
    edge_count = np.zeros(len(tracks), dtype=np.int32)
    cell_count = np.zeros(len(tracks), dtype=np.int32)
    cell_reference = np.asarray([polygon_area(init_points[cell]) for cell in cells])
    for frame_idx in range(len(tracks)):
        distance = np.linalg.norm(tracks[frame_idx, edges[:, 1]] - tracks[frame_idx, edges[:, 0]], axis=1)
        ratio = distance / edge_reference
        both_valid = valid[frame_idx, edges[:, 0]] & valid[frame_idx, edges[:, 1]]
        tolerance = 1e-3
        edge_count[frame_idx] = int(
            np.count_nonzero(
                both_valid
                & ((ratio < MIN_EDGE_RATIO - tolerance) | (ratio > MAX_EDGE_RATIO + tolerance))
            )
        )
        for cell_idx, cell in enumerate(cells):
            if not np.all(valid[frame_idx, cell]):
                continue
            area = polygon_area(tracks[frame_idx, cell])
            if area * cell_reference[cell_idx] <= 0 or abs(area) < MIN_CELL_AREA_RATIO * abs(cell_reference[cell_idx]):
                cell_count[frame_idx] += 1
    return edge_count, cell_count
