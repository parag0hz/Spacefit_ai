"""2D occupancy grid construction.

Cells are `resolution` meters on a side, indexed as grid[row, col]:
    col ↔ x-axis (index 0 = x_min)
    row ↔ z-axis (index 0 = z_min)
Values:
    0 = free
    1 = existing object
    2 = outside room polygon
    3 = door clearance zone (soft — still 'occupied')
    4 = wall thickness band
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.vectorized import contains as vec_contains

from . import geometry as G

FREE = 0
OCC_OBJECT = 1
OCC_OUTSIDE = 2
OCC_DOOR_CLEARANCE = 3
OCC_WALL = 4


@dataclass
class OccupancyGrid:
    grid: np.ndarray          # shape (H, W), int8
    resolution: float         # meters per cell
    origin: Tuple[float, float]  # (x_min, z_min) world floor-local
    shape: Tuple[int, int]    # (H, W)

    def world_to_cell(self, x: float, z: float) -> Tuple[int, int]:
        col = int(round((x - self.origin[0]) / self.resolution))
        row = int(round((z - self.origin[1]) / self.resolution))
        return row, col

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        return (self.origin[0] + col * self.resolution,
                self.origin[1] + row * self.resolution)

    def is_free(self, row: int, col: int) -> bool:
        if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
            return False
        return bool(self.grid[row, col] == FREE)

    def free_mask(self) -> np.ndarray:
        return (self.grid == FREE)

    def clone(self) -> "OccupancyGrid":
        return OccupancyGrid(self.grid.copy(), self.resolution, self.origin, self.shape)


def _polygon_footprint(cx, cz, width, depth, yaw_deg) -> Polygon:
    return G.rotated_bbox_polygon(cx, cz, width, depth, yaw_deg)


def _rasterize_polygon(grid: np.ndarray, poly: Polygon, origin, resolution,
                       value: int, inplace=True) -> np.ndarray:
    """Set cells whose center lies inside `poly` to `value`."""
    xmin, zmin, xmax, zmax = poly.bounds
    col_lo = max(0, int(np.floor((xmin - origin[0]) / resolution)) - 1)
    col_hi = min(grid.shape[1], int(np.ceil((xmax - origin[0]) / resolution)) + 2)
    row_lo = max(0, int(np.floor((zmin - origin[1]) / resolution)) - 1)
    row_hi = min(grid.shape[0], int(np.ceil((zmax - origin[1]) / resolution)) + 2)
    if col_lo >= col_hi or row_lo >= row_hi:
        return grid
    cols = np.arange(col_lo, col_hi)
    rows = np.arange(row_lo, row_hi)
    xs = origin[0] + cols * resolution
    zs = origin[1] + rows * resolution
    XX, ZZ = np.meshgrid(xs, zs)
    mask = vec_contains(poly, XX, ZZ)
    target = grid[row_lo:row_hi, col_lo:col_hi]
    if inplace:
        np.putmask(target, mask, value)
    else:
        out = target.copy()
        np.putmask(out, mask, value)
        return out
    return grid


def create_occupancy_grid(scene: Dict, resolution: float = 0.1,
                          door_clearance: float = 0.8,
                          wall_thickness: float = 0.05) -> OccupancyGrid:
    xmin, zmin, xmax, zmax = scene["floor"]["bounds"]
    origin = (xmin, zmin)
    width_m = xmax - xmin
    height_m = zmax - zmin
    W = max(1, int(np.ceil(width_m / resolution)) + 1)
    H = max(1, int(np.ceil(height_m / resolution)) + 1)
    grid = np.full((H, W), OCC_OUTSIDE, dtype=np.int8)

    floor_poly = Polygon(scene["floor"]["polygon"])
    _rasterize_polygon(grid, floor_poly, origin, resolution, FREE)

    # Wall thickness band: shrink floor polygon slightly to mark a wall band.
    try:
        interior = floor_poly.buffer(-wall_thickness, join_style=2)
    except Exception:
        interior = None
    if interior is not None and not interior.is_empty:
        band = floor_poly.difference(interior)
        if not band.is_empty:
            geoms = [band] if band.geom_type == "Polygon" else list(band.geoms)
            for g in geoms:
                _rasterize_polygon(grid, g, origin, resolution, OCC_WALL)

    # Existing objects: rotated footprints
    for obj in scene["objects"]:
        cx, _, cz = obj["position"]
        w, _, d = obj["size"]
        poly = _polygon_footprint(cx, cz, w, d, obj["yaw"])
        _rasterize_polygon(grid, poly, origin, resolution, OCC_OBJECT)

    # Door clearance: semicircle-ish of radius `door_clearance` in front of each door.
    # Without the door-swing direction we use a conservative circular buffer.
    for d in scene["doors"]:
        cx, _, cz = d["position"]
        width = d["width"]
        clearance = Point(cx, cz).buffer(max(door_clearance, width / 2.0), resolution=16)
        _rasterize_polygon(grid, clearance, origin, resolution, OCC_DOOR_CLEARANCE)

    return OccupancyGrid(grid=grid, resolution=resolution, origin=origin, shape=grid.shape)


def mark_object(occ: OccupancyGrid, cx, cz, width, depth, yaw_deg,
                value: int = OCC_OBJECT) -> None:
    poly = _polygon_footprint(cx, cz, width, depth, yaw_deg)
    _rasterize_polygon(occ.grid, poly, occ.origin, occ.resolution, value)


def check_footprint_free(occ: OccupancyGrid, cx, cz, width, depth, yaw_deg) -> bool:
    """Return True if the rotated footprint lies entirely on FREE cells."""
    poly = _polygon_footprint(cx, cz, width, depth, yaw_deg)
    xmin, zmin, xmax, zmax = poly.bounds
    col_lo = max(0, int(np.floor((xmin - occ.origin[0]) / occ.resolution)))
    col_hi = min(occ.shape[1], int(np.ceil((xmax - occ.origin[0]) / occ.resolution)) + 1)
    row_lo = max(0, int(np.floor((zmin - occ.origin[1]) / occ.resolution)))
    row_hi = min(occ.shape[0], int(np.ceil((zmax - occ.origin[1]) / occ.resolution)) + 1)
    if col_lo >= col_hi or row_lo >= row_hi:
        return False
    cols = np.arange(col_lo, col_hi)
    rows = np.arange(row_lo, row_hi)
    xs = occ.origin[0] + cols * occ.resolution
    zs = occ.origin[1] + rows * occ.resolution
    XX, ZZ = np.meshgrid(xs, zs)
    mask = vec_contains(poly, XX, ZZ)
    sub = occ.grid[row_lo:row_hi, col_lo:col_hi]
    occupied = sub[mask] != FREE
    return not bool(occupied.any())
