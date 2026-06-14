"""Differentiable 2D geometry primitives for furniture placement.

All functions accept and return torch tensors so gradients flow through
(x, z, yaw). Static geometry (polygon vertices, other furniture) may be
passed as tensors or python floats — the module lifts them as needed.

Coordinate convention: 2D (x, z) plane. Floor polygon assumed CCW so that
inward normal = rotate(edge, +90°). Yaw in radians, 0 = +x axis.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch

EPS = 1e-8


def _as_tensor(value, dtype=torch.float32, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype, device=device if device is not None else value.device)
    return torch.as_tensor(value, dtype=dtype, device=device)


def smooth_min(values: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Differentiable min via -T * logsumexp(-x/T). Lower temp = sharper."""
    return -temperature * torch.logsumexp(-values / temperature, dim=-1)


def smooth_max(values: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    return temperature * torch.logsumexp(values / temperature, dim=-1)


def point_segment_distance(
    px: torch.Tensor, pz: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Distance from point (px, pz) to segment a→b (each shape (2,))."""
    ax, az = a[0], a[1]
    bx, bz = b[0], b[1]
    dx, dz = bx - ax, bz - az
    seg_len_sq = dx * dx + dz * dz + EPS
    t = ((px - ax) * dx + (pz - az) * dz) / seg_len_sq
    t = torch.clamp(t, 0.0, 1.0)
    cx = ax + t * dx
    cz = az + t * dz
    return torch.sqrt((px - cx) ** 2 + (pz - cz) ** 2 + EPS)


def polygon_edges(
    polygon: Sequence[Sequence[float]],
    device=None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Return [(v_i, v_{i+1}), ...] as tensors, last edge closes the loop."""
    verts = [_as_tensor(v, device=device) for v in polygon]
    n = len(verts)
    return [(verts[i], verts[(i + 1) % n]) for i in range(n)]


def wall_segments_from_polygon(
    polygon: Sequence[Sequence[float]],
    device=None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Alias: treat consecutive polygon vertices as wall segments."""
    return polygon_edges(polygon, device=device)


def edge_yaw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Yaw (radians) of segment a→b, in [-pi, pi]."""
    return torch.atan2(b[1] - a[1], b[0] - a[0])


def min_point_polygon_boundary_distance(
    px: torch.Tensor,
    pz: torch.Tensor,
    polygon: Sequence[Sequence[float]],
    temperature: float = 0.1,
) -> torch.Tensor:
    """Smooth min of point-to-edge distances over all polygon edges."""
    edges = polygon_edges(polygon, device=px.device)
    distances = torch.stack([point_segment_distance(px, pz, a, b) for a, b in edges])
    return smooth_min(distances, temperature=temperature)


def nearest_wall_info(
    px: torch.Tensor,
    pz: torch.Tensor,
    walls: Sequence[Tuple[Sequence[float], Sequence[float]]],
    temperature: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (min_distance, nearest_wall_yaw) using soft argmin weighting.

    `walls` may be raw polygon edges or pre-extracted wall segments as
    (start, end) tuples of 2D points.
    """
    dists = []
    yaws = []
    for a, b in walls:
        at, bt = _as_tensor(a, device=px.device), _as_tensor(b, device=px.device)
        dists.append(point_segment_distance(px, pz, at, bt))
        yaws.append(edge_yaw(at, bt))
    dists_t = torch.stack(dists)
    yaws_t = torch.stack(yaws)
    # Soft argmin: weights proportional to exp(-d/T), sum to 1
    weights = torch.softmax(-dists_t / temperature, dim=0)
    # Average yaw via cos/sin to avoid angular wraparound
    cos_mix = (weights * torch.cos(yaws_t)).sum()
    sin_mix = (weights * torch.sin(yaws_t)).sum()
    nearest_yaw = torch.atan2(sin_mix, cos_mix)
    min_d = smooth_min(dists_t, temperature=temperature)
    return min_d, nearest_yaw


def nearest_wall_yaw(
    px: torch.Tensor,
    pz: torch.Tensor,
    walls: Sequence[Tuple[Sequence[float], Sequence[float]]],
    temperature: float = 0.1,
) -> torch.Tensor:
    _, yaw = nearest_wall_info(px, pz, walls, temperature=temperature)
    return yaw


def signed_distance_to_convex_polygon(
    px: torch.Tensor,
    pz: torch.Tensor,
    polygon: Sequence[Sequence[float]],
    temperature: float = 0.05,
) -> torch.Tensor:
    """Signed distance to polygon boundary, assuming CCW convex polygon.

    Positive = outside, negative = inside. Uses soft max over per-edge
    signed distances to the edge's infinite line (with outward-pointing
    normal), which is exact for convex polygons.
    """
    edges = polygon_edges(polygon, device=px.device)
    signed = []
    for a, b in edges:
        edge = b - a
        norm = torch.sqrt(edge[0] ** 2 + edge[1] ** 2 + EPS)
        # CCW polygon → outward normal is (edge_y, -edge_x) / |edge|
        nx = edge[1] / norm
        nz = -edge[0] / norm
        sd = (px - a[0]) * nx + (pz - a[1]) * nz
        signed.append(sd)
    signed_t = torch.stack(signed)
    return smooth_max(signed_t, temperature=temperature)


def effective_half_extents(
    width: torch.Tensor, depth: torch.Tensor, yaw: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """AABB half-extents of a rotated rectangle (smooth |cos|, |sin|)."""
    cos_y = torch.sqrt(torch.cos(yaw) ** 2 + EPS)
    sin_y = torch.sqrt(torch.sin(yaw) ** 2 + EPS)
    hx = 0.5 * width * cos_y + 0.5 * depth * sin_y
    hz = 0.5 * width * sin_y + 0.5 * depth * cos_y
    return hx, hz


def rect_corners(
    cx: torch.Tensor,
    cz: torch.Tensor,
    width: torch.Tensor,
    depth: torch.Tensor,
    yaw: torch.Tensor,
) -> torch.Tensor:
    """Return the 4 rotated corners of a rectangle. Shape: (4, 2)."""
    hw = width * 0.5
    hd = depth * 0.5
    local = torch.stack(
        [
            torch.stack([hw, hd]),
            torch.stack([-hw, hd]),
            torch.stack([-hw, -hd]),
            torch.stack([hw, -hd]),
        ]
    )  # (4, 2)
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    rot = torch.stack(
        [
            torch.stack([cos_y, -sin_y]),
            torch.stack([sin_y, cos_y]),
        ]
    )  # (2, 2)
    rotated = local @ rot.T  # (4, 2)
    center = torch.stack([cx, cz])
    return rotated + center


def soft_aabb_overlap(
    cx1: torch.Tensor,
    cz1: torch.Tensor,
    w1: torch.Tensor,
    d1: torch.Tensor,
    yaw1: torch.Tensor,
    cx2: torch.Tensor,
    cz2: torch.Tensor,
    w2: torch.Tensor,
    d2: torch.Tensor,
    yaw2: torch.Tensor,
) -> torch.Tensor:
    """Overlap area using axis-aligned bounding boxes of rotated rectangles.

    Conservative upper bound — exact when both yaws are axis-aligned.
    Fully differentiable; zero when AABBs are disjoint.
    """
    hx1, hz1 = effective_half_extents(w1, d1, yaw1)
    hx2, hz2 = effective_half_extents(w2, d2, yaw2)
    dx = torch.abs(cx1 - cx2)
    dz = torch.abs(cz1 - cz2)
    overlap_x = torch.clamp((hx1 + hx2) - dx, min=0.0)
    overlap_z = torch.clamp((hz1 + hz2) - dz, min=0.0)
    return overlap_x * overlap_z


def polygon_centroid(polygon: Sequence[Sequence[float]]) -> Tuple[float, float]:
    xs = [float(p[0]) for p in polygon]
    zs = [float(p[1]) for p in polygon]
    return sum(xs) / len(xs), sum(zs) / len(zs)


def polygon_diagonal(polygon: Sequence[Sequence[float]]) -> float:
    xs = [float(p[0]) for p in polygon]
    zs = [float(p[1]) for p in polygon]
    dx = max(xs) - min(xs)
    dz = max(zs) - min(zs)
    return math.sqrt(dx * dx + dz * dz)


def polygon_area(polygon: Sequence[Sequence[float]]) -> float:
    """Shoelace area (unsigned)."""
    n = len(polygon)
    area = 0.0
    for i in range(n):
        x1, z1 = polygon[i]
        x2, z2 = polygon[(i + 1) % n]
        area += float(x1) * float(z2) - float(x2) * float(z1)
    return abs(area) * 0.5


def polygon_corners_tensor(polygon: Sequence[Sequence[float]], device=None) -> torch.Tensor:
    return torch.stack([_as_tensor(v, device=device) for v in polygon])
