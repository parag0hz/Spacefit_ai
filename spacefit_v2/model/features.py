"""Placement feature extraction for the learned scoring MLP.

`extract_placement_features(ctx, furn)` returns a 16-dim torch tensor of
numeric features. The 4-dim learned category embedding is appended inside
`PlacementScorer.forward` to form the final 20-dim input.

All operations are differentiable wrt (x, z, yaw). Other furniture and
the floor polygon are treated as fixed context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from spacefit_v2.model.geom import (
    EPS,
    effective_half_extents,
    min_point_polygon_boundary_distance,
    nearest_wall_info,
    point_segment_distance,
    polygon_area,
    polygon_centroid,
    polygon_diagonal,
    polygon_edges,
    signed_distance_to_convex_polygon,
    smooth_min,
    soft_aabb_overlap,
    wall_segments_from_polygon,
    _as_tensor,
)


# ─────────────────────────────────────────────────────────────────────────────
# Categories — aligned with LayoutGPT/ATISS 3D-FRONT object_types (24 types).
# Loaded from dataset_stats.txt at call time via `load_3dfront_categories`.
# The static list below is used as a fallback / default ordering. The two must
# be kept in sync at training/eval time (callers pass explicit mapping).
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_LIST: List[str] = [
    "armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa",
    "chinese_chair", "children_cabinet", "coffee_table", "console_table",
    "corner_side_table", "desk", "dining_chair", "dining_table", "double_bed",
    "dressing_chair", "dressing_table", "kids_bed", "lounge_chair",
    "l_shaped_sofa", "loveseat_sofa", "multi_seat_sofa", "nightstand",
    "pendant_lamp", "round_end_table", "shelf", "single_bed", "sofa",
    "stool", "table", "tv_stand", "wardrobe", "wine_cabinet",
]
CATEGORY_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CATEGORY_LIST)}
NUM_CATEGORIES_DEFAULT = len(CATEGORY_LIST)

FEATURE_DIM = 16  # numeric features; category embedding concatenated in scorer


# Furniture pairs with meaningful semantic proximity. (symmetric)
RELATED_PAIRS: List[Tuple[str, str]] = [
    ("nightstand", "double_bed"),
    ("nightstand", "single_bed"),
    ("nightstand", "kids_bed"),
    ("dining_chair", "dining_table"),
    ("chinese_chair", "dining_table"),
    ("coffee_table", "sofa"),
    ("coffee_table", "multi_seat_sofa"),
    ("coffee_table", "loveseat_sofa"),
    ("coffee_table", "l_shaped_sofa"),
    ("coffee_table", "chaise_longue_sofa"),
    ("coffee_table", "armchair"),
    ("coffee_table", "lounge_chair"),
    ("tv_stand", "sofa"),
    ("tv_stand", "multi_seat_sofa"),
    ("dressing_chair", "dressing_table"),
    ("table_lamp", "nightstand"),
    ("floor_lamp", "sofa"),
]


def _related_set() -> Dict[str, set]:
    d: Dict[str, set] = {}
    for a, b in RELATED_PAIRS:
        d.setdefault(a, set()).add(b)
        d.setdefault(b, set()).add(a)
    return d


_RELATED = _related_set()


def category_idx(category: str, mapping: Optional[Dict[str, int]] = None) -> int:
    mp = mapping if mapping is not None else CATEGORY_TO_IDX
    if category in mp:
        return mp[category]
    cat = category.lower().replace(" ", "_").replace("-", "_")
    return mp.get(cat, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Input containers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Furniture:
    """Uniform in-memory representation of a furniture item.

    Accepts 3D-FRONT adapter shape or spacefit scene.objects shape via the
    `from_dict` constructor.
    """

    x: float
    z: float
    yaw: float  # radians
    width: float
    depth: float
    height: float
    category: str

    @classmethod
    def from_dict(cls, item: Dict) -> "Furniture":
        # 3D-FRONT adapter: position: {x, z}, size: {width, depth, height}, rotation_y deg
        size = item.get("size")
        if isinstance(size, dict):
            width = float(size.get("width", 0.0))
            depth = float(size.get("depth", 0.0))
            height = float(size.get("height", 0.0))
        elif isinstance(size, (list, tuple)):
            # spacefit: size = (w, h, d)
            width, height, depth = (float(x) for x in size[:3])
        else:
            width = float(item.get("width", 0.0))
            depth = float(item.get("depth", 0.0))
            height = float(item.get("height", 0.0))

        pos = item.get("position")
        if isinstance(pos, dict):
            x = float(pos.get("x", 0.0))
            z = float(pos.get("z", 0.0))
        elif isinstance(pos, (list, tuple)):
            x = float(pos[0])
            z = float(pos[2]) if len(pos) >= 3 else float(pos[1])
        else:
            x = float(item.get("x", 0.0))
            z = float(item.get("z", 0.0))

        if "rotation_y" in item:
            yaw = math_deg_to_rad(float(item["rotation_y"]))
        elif "yaw" in item:
            yaw = float(item["yaw"])  # assume radians unless caller converts
        elif "angle" in item:
            yaw = float(item["angle"])
        else:
            yaw = 0.0

        category = item.get("category", "table")
        return cls(x=x, z=z, yaw=yaw, width=width, depth=depth, height=height, category=category)


def math_deg_to_rad(deg: float) -> float:
    import math
    return deg * math.pi / 180.0


@dataclass
class PlacementContext:
    """Fixed scene context for scoring one furniture placement.

    Fields may be python lists or numpy arrays — geom lifts them to tensors.
    """

    floor_polygon: Sequence[Sequence[float]]
    walls: Sequence[Tuple[Sequence[float], Sequence[float]]] = field(default_factory=list)
    doors: Sequence[Tuple[float, float]] = field(default_factory=list)
    windows: Sequence[Tuple[float, float]] = field(default_factory=list)
    existing_furniture: Sequence[Furniture] = field(default_factory=list)

    @classmethod
    def from_3dfront_scene(cls, scene: Dict) -> "PlacementContext":
        polygon = [(float(p[0]), float(p[1])) for p in scene["floor_plan_vertices"]]
        walls = wall_segments_from_polygon(polygon)
        existing = [Furniture.from_dict(f) for f in scene.get("furniture", [])]
        return cls(floor_polygon=polygon, walls=walls, existing_furniture=existing)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────


_LARGE_DIST = 10.0  # sentinel "no door/window/relation" distance in meters


def extract_placement_features(
    x: torch.Tensor,
    z: torch.Tensor,
    yaw: torch.Tensor,
    width: float,
    depth: float,
    category: str,
    context: PlacementContext,
    also_placed: Sequence[Furniture] = (),
) -> torch.Tensor:
    """Compute a (FEATURE_DIM,) tensor of differentiable placement features.

    `also_placed` are new-furniture items already placed this round.
    They are treated as fixed context (no gradient flows through them).
    """
    if not isinstance(x, torch.Tensor):
        x = _as_tensor(x)
    if not isinstance(z, torch.Tensor):
        z = _as_tensor(z, device=x.device)
    else:
        z = z.to(device=x.device, dtype=torch.float32)
    if not isinstance(yaw, torch.Tensor):
        yaw = _as_tensor(yaw, device=x.device)
    else:
        yaw = yaw.to(device=x.device, dtype=torch.float32)
    w = _as_tensor(width, device=x.device)
    d = _as_tensor(depth, device=x.device)

    polygon = context.floor_polygon
    walls = context.walls if context.walls else wall_segments_from_polygon(polygon, device=x.device)

    # ── Spatial (walls, corners, center) ──
    min_wall_d, near_wall_yaw = nearest_wall_info(x, z, walls, temperature=0.1)
    yaw_diff = yaw - near_wall_yaw
    wall_cos = torch.cos(2.0 * yaw_diff)  # 90° periodic: parallel = 1
    wall_sin = torch.sin(2.0 * yaw_diff)

    corners = [_as_tensor(v, device=x.device) for v in polygon]
    corner_dists = torch.stack(
        [torch.sqrt((x - c[0]) ** 2 + (z - c[1]) ** 2 + EPS) for c in corners]
    )
    min_corner_d = smooth_min(corner_dists, temperature=0.1)

    cx0, cz0 = polygon_centroid(polygon)
    diag = max(polygon_diagonal(polygon), 1e-3)
    center_d = torch.sqrt((x - cx0) ** 2 + (z - cz0) ** 2 + EPS)
    center_d_norm = center_d / diag

    # ── Boundary ──
    sd = signed_distance_to_convex_polygon(x, z, polygon, temperature=0.05)
    # Negative = inside → "boundary margin" is -sd (positive when inside).
    # Clamp for numerical stability of features.
    boundary_margin = torch.clamp(-sd, min=-2.0, max=5.0)

    # ── Collision / nearest other furniture ──
    others = list(context.existing_furniture) + list(also_placed)
    if others:
        other_dists = []
        overlaps = []
        for other in others:
            od = torch.sqrt(
                (x - other.x) ** 2 + (z - other.z) ** 2 + EPS
            )
            other_dists.append(od)
            ov = soft_aabb_overlap(
                x, z, w, d, yaw,
                _as_tensor(other.x, device=x.device), _as_tensor(other.z, device=x.device),
                _as_tensor(other.width, device=x.device), _as_tensor(other.depth, device=x.device),
                _as_tensor(other.yaw, device=x.device),
            )
            overlaps.append(ov)
        other_dists_t = torch.stack(other_dists)
        overlaps_t = torch.stack(overlaps)
        min_furn_d = smooth_min(other_dists_t, temperature=0.1)
        total_overlap = overlaps_t.sum()
    else:
        min_furn_d = _as_tensor(_LARGE_DIST, device=x.device)
        total_overlap = _as_tensor(0.0, device=x.device)

    # ── Doors / windows ──
    def _min_point_distance(points: Sequence[Tuple[float, float]]) -> torch.Tensor:
        if not points:
            return _as_tensor(_LARGE_DIST, device=x.device)
        dists = [
            torch.sqrt((x - float(p[0])) ** 2 + (z - float(p[1])) ** 2 + EPS)
            for p in points
        ]
        return smooth_min(torch.stack(dists), temperature=0.1)

    min_door_d = _min_point_distance(context.doors)
    min_window_d = _min_point_distance(context.windows)

    # ── Related furniture ──
    related_cats = _RELATED.get(category, set())
    related_items = [o for o in others if o.category in related_cats]
    if related_items:
        rel_dists = torch.stack(
            [
                torch.sqrt((x - r.x) ** 2 + (z - r.z) ** 2 + EPS)
                for r in related_items
            ]
        )
        nearest_rel_d = smooth_min(rel_dists, temperature=0.1)
        has_related = _as_tensor(1.0, device=x.device)
    else:
        nearest_rel_d = _as_tensor(_LARGE_DIST, device=x.device)
        has_related = _as_tensor(0.0, device=x.device)

    # ── Size / aspect ──
    room_area = max(polygon_area(polygon), 1e-3)
    furn_area = w * d
    area_ratio = furn_area / room_area
    aspect = w / (d + EPS)

    features = torch.stack(
        [
            min_wall_d,            # 0
            wall_cos,              # 1
            wall_sin,              # 2
            min_corner_d,          # 3
            center_d,              # 4
            center_d_norm,         # 5
            boundary_margin,       # 6
            min_furn_d,            # 7
            total_overlap,         # 8
            min_door_d,            # 9
            min_window_d,          # 10
            nearest_rel_d,         # 11
            has_related,           # 12
            furn_area,             # 13
            area_ratio,            # 14
            aspect,                # 15
        ]
    )
    return features
