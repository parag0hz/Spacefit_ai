"""2D geometry utilities.

Uses floor-local XZ plane (horizontal) with Y as height.
All polygons are lists of (x, z) tuples unless stated otherwise.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon, box as shapely_box
from shapely.affinity import rotate as shapely_rotate, translate as shapely_translate

Vec2 = Tuple[float, float]
BBox2D = Tuple[float, float, float, float]  # (xmin, zmin, xmax, zmax)


def parse_transform_4x4(flat: Sequence[float]) -> np.ndarray:
    """RoomPlan stores 4x4 matrices as 16 floats column-major."""
    arr = np.asarray(flat, dtype=np.float64)
    if arr.shape != (16,):
        raise ValueError(f"expected 16 floats, got shape {arr.shape}")
    return arr.reshape(4, 4, order="F")


def floor_world_to_local(transform_list: Sequence[float]):
    """Return (R_inv, t, floor_yaw_deg) for world→floor-local conversion."""
    T = parse_transform_4x4(transform_list)
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    floor_yaw_deg = math.degrees(math.atan2(-R[2, 0], R[0, 0]))
    return R_inv, t, floor_yaw_deg


def world_to_local_pose(transform_list, R_inv, t_floor, floor_yaw_deg):
    """Object world transform → (x, y_height, z, yaw_deg) in floor-local frame.

    Floor-local raw axes: p_local[0]=X_horiz, p_local[1]=Y_horiz, p_local[2]=Z_up.
    We permute so the returned y is height and x, z are horizontal, matching
    polygonCorners[i][0] → x and polygonCorners[i][1] → z.
    """
    T = parse_transform_4x4(transform_list)
    p_world = T[:3, 3]
    p_local = R_inv @ (p_world - t_floor)
    x = float(p_local[0])
    z = float(p_local[1])
    y = float(p_local[2])
    world_yaw = math.degrees(math.atan2(T[0, 2], T[2, 2]))
    local_yaw = world_yaw - floor_yaw_deg
    local_yaw = ((local_yaw + 180.0) % 360.0) - 180.0
    return x, y, z, float(local_yaw)


def rotated_bbox_corners(cx: float, cz: float, width: float, depth: float,
                          yaw_deg: float) -> np.ndarray:
    """Return 4 corners (shape (4,2)) of a rotated rectangle centered at (cx,cz).

    `width` runs along local X, `depth` runs along local Z (before rotation).
    yaw_deg rotates counter-clockwise around the Y axis (viewed from above).
    """
    hw, hd = width / 2.0, depth / 2.0
    corners = np.array([[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]])
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    R = np.array([[c, -s], [s, c]])
    return corners @ R.T + np.array([cx, cz])


def rotated_bbox_polygon(cx, cz, width, depth, yaw_deg) -> Polygon:
    corners = rotated_bbox_corners(cx, cz, width, depth, yaw_deg)
    return Polygon(corners.tolist())


def aabb_of(polygon: Polygon) -> BBox2D:
    xmin, zmin, xmax, zmax = polygon.bounds
    return (xmin, zmin, xmax, zmax)


def aabb_overlap(a: BBox2D, b: BBox2D) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def polygons_overlap(a: Polygon, b: Polygon, eps: float = 1e-6) -> bool:
    if not aabb_overlap(aabb_of(a), aabb_of(b)):
        return False
    try:
        return a.intersection(b).area > eps
    except Exception:
        return False


def point_in_polygon(x: float, z: float, polygon: Sequence[Vec2]) -> bool:
    poly = Polygon(polygon)
    from shapely.geometry import Point
    return poly.contains(Point(x, z))


def polygon_area(polygon: Sequence[Vec2]) -> float:
    return Polygon(polygon).area


def polygon_bounds(polygon: Sequence[Vec2]) -> BBox2D:
    xs = [p[0] for p in polygon]
    zs = [p[1] for p in polygon]
    return (min(xs), min(zs), max(xs), max(zs))


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def wall_segments(wall_rects: Iterable[Tuple[float, float, float, float, float]]):
    """Convert (cx, cz, length, thickness, yaw_deg) tuples to (p1, p2) centerlines."""
    segs = []
    for cx, cz, length, _thickness, yaw in wall_rects:
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        hl = length / 2.0
        p1 = (cx - hl * c, cz - hl * s)
        p2 = (cx + hl * c, cz + hl * s)
        segs.append((p1, p2))
    return segs


def closest_wall_distance(cx: float, cz: float, walls: Sequence[dict]) -> Tuple[float, float]:
    """Distance from (cx,cz) to nearest wall centerline + angle of that wall in degrees."""
    best_d = float("inf")
    best_yaw = 0.0
    for w in walls:
        wcx, _, wcz = w["position"]
        length = w["length"]
        yaw = w["yaw"]
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        hl = length / 2.0
        p1 = np.array([wcx - hl * c, wcz - hl * s])
        p2 = np.array([wcx + hl * c, wcz + hl * s])
        p = np.array([cx, cz])
        v = p2 - p1
        vv = float(np.dot(v, v)) or 1e-9
        t = float(np.clip(np.dot(p - p1, v) / vv, 0.0, 1.0))
        proj = p1 + t * v
        d = float(np.linalg.norm(p - proj))
        if d < best_d:
            best_d = d
            best_yaw = yaw
    return best_d, best_yaw


def find_nearest_wall(cx: float, cz: float, walls: Sequence[dict]) -> Tuple[float, float]:
    """Alias used by solver scoring code: returns (distance, wall_yaw_deg)."""
    return closest_wall_distance(cx, cz, walls)


def angle_to_vector(yaw_deg: float) -> Vec2:
    """Return a unit forward vector for the given yaw in the floor XZ plane."""
    rad = math.radians(yaw_deg)
    return (math.sin(rad), math.cos(rad))


def yaw_toward(src: Vec2, dst: Vec2) -> float:
    """Return yaw (degrees) so that +Z (local forward) faces from src to dst.

    Convention: rotated_bbox_corners uses yaw rotating (X,Z) CCW.
    We align the +Z axis of the object with the direction dst-src.
    """
    dx = dst[0] - src[0]
    dz = dst[1] - src[1]
    return math.degrees(math.atan2(dx, dz))


def snap_yaw(yaw_deg: float, candidates: Sequence[float]) -> float:
    """Snap yaw to the nearest candidate (mod 360)."""
    yaw = yaw_deg % 360.0
    best = candidates[0]
    best_err = 1e9
    for c in candidates:
        e = min(abs(yaw - c), 360 - abs(yaw - c))
        if e < best_err:
            best_err = e
            best = c
    return float(best)
