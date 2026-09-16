"""Geometry helpers for the Huang-inspired radial-pair preview.

This module only constructs candidate points.  It does not track motion or
calculate radial strain rate.
"""

from __future__ import annotations

import numpy as np


def _curve_tangents(points: np.ndarray) -> np.ndarray:
    tangent = np.empty_like(points)
    tangent[0] = points[1] - points[0]
    tangent[-1] = points[-1] - points[-2]
    if len(points) > 2:
        tangent[1:-1] = points[2:] - points[:-2]
    length = np.linalg.norm(tangent, axis=1, keepdims=True)
    return np.divide(
        tangent,
        length,
        out=np.zeros_like(tangent),
        where=length > 1e-6,
    )


def outward_unit_normals(
    wall_curve: np.ndarray,
    opposite_wall_curve: np.ndarray,
) -> np.ndarray:
    """Return wall normals oriented away from the endometrial cavity."""
    wall = np.asarray(wall_curve, dtype=np.float32)
    opposite = np.asarray(opposite_wall_curve, dtype=np.float32)
    if wall.shape != opposite.shape or wall.ndim != 2 or wall.shape[1] != 2:
        raise ValueError("wall curves must have matching shape (n, 2)")
    if len(wall) < 2:
        raise ValueError("wall curves must contain at least two points")
    if not np.all(np.isfinite(wall)) or not np.all(np.isfinite(opposite)):
        raise ValueError("wall curves must contain finite coordinates")

    tangent = _curve_tangents(wall)
    candidate_normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    toward_cavity = opposite - wall
    points_toward_cavity = np.sum(candidate_normal * toward_cavity, axis=1) >= 0
    inward = np.where(points_toward_cavity[:, None], candidate_normal, -candidate_normal)
    return (-inward).astype(np.float32)


def build_radial_pair_candidates(
    anterior_wall: np.ndarray,
    posterior_wall: np.ndarray,
    offset_px: float,
) -> dict[str, np.ndarray]:
    """Build one inner/outer radial pair at every paired wall section."""
    if not np.isfinite(offset_px) or offset_px <= 0:
        raise ValueError("offset_px must be a positive finite number")
    anterior = np.asarray(anterior_wall, dtype=np.float32)
    posterior = np.asarray(posterior_wall, dtype=np.float32)
    if anterior.shape != posterior.shape:
        raise ValueError("anterior and posterior wall curves must match")

    anterior_outward = outward_unit_normals(anterior, posterior)
    posterior_outward = outward_unit_normals(posterior, anterior)
    return {
        "anterior_inner": anterior.copy(),
        "anterior_outer": anterior + float(offset_px) * anterior_outward,
        "anterior_outward_normal": anterior_outward,
        "posterior_inner": posterior.copy(),
        "posterior_outer": posterior + float(offset_px) * posterior_outward,
        "posterior_outward_normal": posterior_outward,
    }


def radial_strain_rate_from_positions(
    source_inner: np.ndarray,
    source_outer: np.ndarray,
    target_inner: np.ndarray,
    target_outer: np.ndarray,
    fps: float,
) -> np.ndarray:
    """Return signed frame-to-frame radial strain rate from pair lengths."""
    arrays = [
        np.asarray(value, dtype=np.float32)
        for value in (source_inner, source_outer, target_inner, target_outer)
    ]
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("all point arrays must have matching shapes")
    if arrays[0].ndim < 1 or arrays[0].shape[-1] != 2:
        raise ValueError("point arrays must end with coordinate dimension 2")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")

    source_length = np.linalg.norm(arrays[1] - arrays[0], axis=-1)
    target_length = np.linalg.norm(arrays[3] - arrays[2], axis=-1)
    return np.divide(
        (target_length - source_length) * float(fps),
        source_length,
        out=np.full_like(source_length, np.nan, dtype=np.float32),
        where=source_length > 1e-6,
    ).astype(np.float32)
