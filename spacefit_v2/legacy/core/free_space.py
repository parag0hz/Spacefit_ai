"""Extract candidate free-space regions from an occupancy grid.

Pipeline:
  1. Binarize free cells.
  2. Connected-component label (scipy.ndimage.label with 8-connectivity).
  3. For each component: area, centroid, bounding box, max inscribed rectangle,
     adjacency to walls / doors / windows.
  4. Filter components smaller than `min_area`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from . import geometry as G
from .occupancy import OccupancyGrid, FREE, OCC_WALL, OCC_DOOR_CLEARANCE


@dataclass
class Candidate:
    id: int
    area: float                         # m²
    centroid: Tuple[float, float]       # (x, z) world-local
    bounds: Tuple[float, float, float, float]  # (xmin, zmin, xmax, zmax)
    max_rect: Tuple[float, float, float, float, float]  # (cx, cz, width, depth, yaw_deg)
    near_wall: bool
    near_door: bool
    near_window: bool
    wall_yaw: Optional[float] = None    # yaw of nearest wall (degrees), if near_wall
    cell_coords: List[Tuple[int, int]] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d.pop("cell_coords", None)
        return d


def _max_inscribed_axis_aligned_rect(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Largest all-True axis-aligned rectangle inside `mask`.

    Uses the histogram / stack algorithm on the 'height' array.
    Returns (row, col, h, w) — top-left indices and size in cells.
    """
    H, W = mask.shape
    heights = np.zeros(W, dtype=np.int32)
    best = (0, 0, 0, 0)  # (row, col, h, w)
    best_area = 0
    for r in range(H):
        for c in range(W):
            heights[c] = heights[c] + 1 if mask[r, c] else 0
        stack: List[Tuple[int, int]] = []  # (start_col, height)
        for c in range(W + 1):
            cur_h = heights[c] if c < W else 0
            start = c
            while stack and stack[-1][1] > cur_h:
                s_c, s_h = stack.pop()
                width = c - s_c
                area = s_h * width
                if area > best_area:
                    best_area = area
                    best = (r - s_h + 1, s_c, s_h, width)
                start = s_c
            if not stack or stack[-1][1] < cur_h:
                stack.append((start, cur_h))
    return best


def _component_adjacency(coords_rows: np.ndarray, coords_cols: np.ndarray,
                          adj_mask: np.ndarray, radius_cells: int) -> bool:
    """Any cell in the component within `radius_cells` of a True in adj_mask?"""
    if not adj_mask.any():
        return False
    H, W = adj_mask.shape
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            if dr * dr + dc * dc > radius_cells * radius_cells:
                continue
            rr = coords_rows + dr
            cc = coords_cols + dc
            valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            if not valid.any():
                continue
            if adj_mask[rr[valid], cc[valid]].any():
                return True
    return False


def _nearest_wall_yaw(cx: float, cz: float, walls: List[dict]) -> Optional[float]:
    if not walls:
        return None
    _, yaw = G.closest_wall_distance(cx, cz, walls)
    return yaw


def extract_candidates(occ: OccupancyGrid, scene: Dict,
                       min_area: float = 0.3,
                       door_adj_radius: float = 0.4,
                       window_adj_radius: float = 0.3,
                       wall_adj_radius: float = 0.2,
                       split_erosion_m: float = 0.4,
                       max_rects_per_component: int = 6,
                       min_rect_side_m: float = 0.3) -> List[Candidate]:
    """Extract candidate regions. Optionally erode the free mask by
    `split_erosion_m` meters before labeling so narrow corridors split into
    distinct rooms/pockets. Each component's cells are then grown back via
    the original free mask intersection (closest-region assignment).
    """
    free = occ.free_mask().astype(np.uint8)

    if split_erosion_m and split_erosion_m > 0:
        erosion_cells = max(1, int(round(split_erosion_m / occ.resolution)))
        struct = np.ones((3, 3), dtype=np.uint8)
        eroded = free.copy()
        for _ in range(erosion_cells):
            eroded = ndimage.binary_erosion(eroded, structure=struct).astype(np.uint8)
        core_lbl, n_core = ndimage.label(eroded, structure=struct)
        if n_core >= 2:
            # Assign every free cell to the nearest core component.
            dist, idx = ndimage.distance_transform_edt(
                core_lbl == 0, return_indices=True
            )
            lbl = np.where(free > 0, core_lbl[idx[0], idx[1]], 0).astype(np.int32)
            n = n_core
        else:
            lbl, n = ndimage.label(free, structure=struct)
    else:
        lbl, n = ndimage.label(free, structure=np.ones((3, 3), dtype=np.uint8))

    if n == 0:
        return []

    wall_mask = (occ.grid == OCC_WALL)
    door_mask = (occ.grid == OCC_DOOR_CLEARANCE)

    # Window adjacency: rasterize a thin disc at each window position
    from shapely.geometry import Point
    from shapely.vectorized import contains as vec_contains
    window_mask = np.zeros_like(free, dtype=bool)
    for w in scene.get("windows", []):
        cx, _, cz = w["position"]
        pt = Point(cx, cz).buffer(max(0.2, w["width"] / 2.0), resolution=16)
        xmin, zmin, xmax, zmax = pt.bounds
        col_lo = max(0, int(np.floor((xmin - occ.origin[0]) / occ.resolution)))
        col_hi = min(occ.shape[1], int(np.ceil((xmax - occ.origin[0]) / occ.resolution)) + 1)
        row_lo = max(0, int(np.floor((zmin - occ.origin[1]) / occ.resolution)))
        row_hi = min(occ.shape[0], int(np.ceil((zmax - occ.origin[1]) / occ.resolution)) + 1)
        if col_lo >= col_hi or row_lo >= row_hi:
            continue
        cols = np.arange(col_lo, col_hi)
        rows = np.arange(row_lo, row_hi)
        xs = occ.origin[0] + cols * occ.resolution
        zs = occ.origin[1] + rows * occ.resolution
        XX, ZZ = np.meshgrid(xs, zs)
        m = vec_contains(pt, XX, ZZ)
        window_mask[row_lo:row_hi, col_lo:col_hi] |= m

    cell_area = occ.resolution ** 2
    door_r = max(1, int(round(door_adj_radius / occ.resolution)))
    window_r = max(1, int(round(window_adj_radius / occ.resolution)))
    wall_r = max(1, int(round(wall_adj_radius / occ.resolution)))

    candidates: List[Candidate] = []
    min_side_cells = max(1, int(round(min_rect_side_m / occ.resolution)))
    min_rect_area = (min_rect_side_m ** 2)

    for comp_id in range(1, n + 1):
        comp_rows, comp_cols = np.where(lbl == comp_id)
        if len(comp_rows) == 0:
            continue
        comp_area = len(comp_rows) * cell_area
        if comp_area < min_area:
            continue

        row_lo, row_hi = int(comp_rows.min()), int(comp_rows.max())
        col_lo, col_hi = int(comp_cols.min()), int(comp_cols.max())
        sub_mask = np.zeros((row_hi - row_lo + 1, col_hi - col_lo + 1), dtype=bool)
        sub_mask[comp_rows - row_lo, comp_cols - col_lo] = True

        for _ in range(max_rects_per_component):
            r0, c0, h_cells, w_cells = _max_inscribed_axis_aligned_rect(sub_mask)
            if h_cells < min_side_cells or w_cells < min_side_cells:
                break
            rect_area = (h_cells * w_cells) * cell_area
            if rect_area < min_rect_area:
                break
            rect_cx = occ.origin[0] + (col_lo + c0 + w_cells / 2.0) * occ.resolution
            rect_cz = occ.origin[1] + (row_lo + r0 + h_cells / 2.0) * occ.resolution
            rect_w = w_cells * occ.resolution
            rect_d = h_cells * occ.resolution

            # World bounds for this sub-rectangle
            xmin = occ.origin[0] + (col_lo + c0) * occ.resolution
            zmin = occ.origin[1] + (row_lo + r0) * occ.resolution
            xmax = xmin + rect_w
            zmax = zmin + rect_d

            # Cells belonging to this rectangle (all True in sub_mask)
            sub_rows = np.arange(r0, r0 + h_cells)
            sub_cols = np.arange(c0, c0 + w_cells)
            RR, CC = np.meshgrid(sub_rows, sub_cols, indexing="ij")
            rect_rows_glob = (RR + row_lo).ravel()
            rect_cols_glob = (CC + col_lo).ravel()

            near_wall = _component_adjacency(rect_rows_glob, rect_cols_glob, wall_mask, wall_r)
            near_door = _component_adjacency(rect_rows_glob, rect_cols_glob, door_mask, door_r)
            near_window = _component_adjacency(rect_rows_glob, rect_cols_glob, window_mask, window_r)
            wall_yaw = _nearest_wall_yaw(rect_cx, rect_cz, scene["walls"]) if near_wall else None

            candidates.append(Candidate(
                id=len(candidates),
                area=rect_area,
                centroid=(rect_cx, rect_cz),
                bounds=(xmin, zmin, xmax, zmax),
                max_rect=(rect_cx, rect_cz, rect_w, rect_d, 0.0),
                near_wall=near_wall,
                near_door=near_door,
                near_window=near_window,
                wall_yaw=wall_yaw,
                cell_coords=list(zip(rect_rows_glob.tolist(), rect_cols_glob.tolist())),
            ))

            sub_mask[r0:r0 + h_cells, c0:c0 + w_cells] = False

    candidates.sort(key=lambda c: -c.area)
    for i, c in enumerate(candidates):
        c.id = i
    return candidates
