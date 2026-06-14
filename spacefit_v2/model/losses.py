"""Differentiable placement losses."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Sequence

import torch
import torch.nn.functional as F

from spacefit_v2 import config
from spacefit_v2.model.geom import (
    EPS,
    nearest_wall_info,
    rect_corners,
    signed_distance_to_convex_polygon,
    soft_aabb_overlap,
    wall_segments_from_polygon,
    _as_tensor,
)


RELATIONS: Dict[tuple[str, str], tuple[float, float]] = {
    ("nightstand", "double_bed"): (0.0, 0.5),
    ("nightstand", "single_bed"): (0.0, 0.5),
    ("nightstand", "kids_bed"): (0.0, 0.5),
    ("chair", "desk"): (0.0, 0.8),
    ("dining_chair", "dining_table"): (0.0, 0.8),
    ("coffee_table", "sofa"): (0.3, 1.2),
    ("tv_stand", "sofa"): (1.0, 3.5),
    ("table_lamp", "nightstand"): (0.0, 0.4),
    ("floor_lamp", "sofa"): (0.3, 1.8),
}

WALL_HUGGING = {
    "wardrobe",
    "bookshelf",
    "shelf",
    "cabinet",
    "children_cabinet",
    "dresser",
    "tv_stand",
}
WALL_PARALLEL = {
    "bed",
    "double_bed",
    "single_bed",
    "kids_bed",
    "desk",
    "sofa",
    "table",
    "dining_table",
    "dressing_table",
}


def _size_to_width_depth(size: Any, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(size, dict):
        return _as_tensor(size["width"], device=device), _as_tensor(size["depth"], device=device)
    if isinstance(size, (list, tuple)):
        if len(size) == 2:
            return _as_tensor(size[0], device=device), _as_tensor(size[1], device=device)
        return _as_tensor(size[0], device=device), _as_tensor(size[-1], device=device)
    raise TypeError(f"unsupported size format: {type(size)!r}")


def _position_xz(item: Dict[str, Any]) -> tuple[float, float]:
    pos = item.get("position")
    if isinstance(pos, dict):
        return float(pos["x"]), float(pos["z"])
    if isinstance(pos, (list, tuple)):
        if len(pos) >= 3:
            return float(pos[0]), float(pos[2])
        return float(pos[0]), float(pos[1])
    return float(item.get("x", 0.0)), float(item.get("z", 0.0))


def _item_width_depth(item: Dict[str, Any]) -> tuple[float, float]:
    size = item.get("size")
    if isinstance(size, dict):
        return float(size["width"]), float(size["depth"])
    if isinstance(size, (list, tuple)):
        if len(size) >= 3:
            return float(size[0]), float(size[2])
        return float(size[0]), float(size[1])
    return float(item["width"]), float(item["depth"])


def _item_yaw_rad(item: Dict[str, Any]) -> float:
    if "yaw" in item:
        yaw = float(item["yaw"])
        return math.radians(yaw) if abs(yaw) > 2.0 * math.pi + 1e-3 else yaw
    if "rotation_y" in item:
        return math.radians(float(item["rotation_y"]))
    return 0.0


def _item_category(item: Dict[str, Any]) -> str:
    return str(item.get("category", "unknown")).lower()


def compute_boundary_loss(
    positions: Sequence[torch.Tensor],
    rotations: Sequence[torch.Tensor],
    sizes: Sequence[Any],
    floor_polygon: Sequence[Sequence[float]],
) -> torch.Tensor:
    device = positions[0].device if positions else None
    total = _as_tensor(0.0, device=device)
    for pos, rot, size in zip(positions, rotations, sizes):
        width, depth = _size_to_width_depth(size, device=device)
        corners = rect_corners(pos[0], pos[1], width, depth, rot)
        signed = torch.stack(
            [
                signed_distance_to_convex_polygon(corner[0], corner[1], floor_polygon)
                for corner in corners
            ]
        )
        total = total + F.relu(signed).sum()
    return total


def compute_collision_loss(
    positions: Sequence[torch.Tensor],
    rotations: Sequence[torch.Tensor],
    sizes: Sequence[Any],
    obstacles: Sequence[Dict[str, Any]],
) -> torch.Tensor:
    device = positions[0].device if positions else None
    total = _as_tensor(0.0, device=device)

    for i in range(len(positions)):
        wi, di = _size_to_width_depth(sizes[i], device=device)
        for j in range(i + 1, len(positions)):
            wj, dj = _size_to_width_depth(sizes[j], device=device)
            total = total + soft_aabb_overlap(
                positions[i][0], positions[i][1], wi, di, rotations[i],
                positions[j][0], positions[j][1], wj, dj, rotations[j],
            )
        for obs in obstacles:
            ox, oz = _position_xz(obs)
            ow, od = _item_width_depth(obs)
            oyaw = _item_yaw_rad(obs)
            total = total + soft_aabb_overlap(
                positions[i][0], positions[i][1], wi, di, rotations[i],
                _as_tensor(ox, device=device), _as_tensor(oz, device=device),
                _as_tensor(ow, device=device), _as_tensor(od, device=device), _as_tensor(oyaw, device=device),
            )
    return total


def compute_door_clearance_loss(
    positions: Sequence[torch.Tensor],
    sizes: Sequence[Any],
    doors: Sequence[Any] | None,
    clearance_radius: float = config.DOOR_CLEARANCE_RADIUS,
) -> torch.Tensor:
    if not doors:
        device = positions[0].device if positions else None
        return _as_tensor(0.0, device=device)

    device = positions[0].device if positions else None
    total = _as_tensor(0.0, device=device)
    for pos, size in zip(positions, sizes):
        width, depth = _size_to_width_depth(size, device=device)
        padding = 0.25 * torch.sqrt(width * width + depth * depth + EPS)
        for door in doors:
            if isinstance(door, dict):
                dpos = door.get("position")
                if isinstance(dpos, dict):
                    dx, dz = float(dpos["x"]), float(dpos["z"])
                else:
                    dx, dz = float(dpos[0]), float(dpos[2] if len(dpos) >= 3 else dpos[1])
            else:
                dx, dz = float(door[0]), float(door[1])
            dist = torch.sqrt((pos[0] - dx) ** 2 + (pos[1] - dz) ** 2 + EPS)
            total = total + F.relu(clearance_radius + padding - dist) ** 2
    return total


def loss_physics(
    positions: Sequence[torch.Tensor],
    rotations: Sequence[torch.Tensor],
    sizes: Sequence[Any],
    floor_polygon: Sequence[Sequence[float]],
    existing_obstacles: Sequence[Dict[str, Any]],
    doors: Sequence[Any] | None = None,
) -> torch.Tensor:
    l_boundary = compute_boundary_loss(positions, rotations, sizes, floor_polygon)
    l_collision = compute_collision_loss(positions, rotations, sizes, existing_obstacles)
    l_door = compute_door_clearance_loss(positions, sizes, doors)
    return l_boundary + l_collision + l_door


def loss_semantic(
    positions: Sequence[torch.Tensor],
    categories: Sequence[str],
    already_placed: Sequence[Dict[str, Any]],
) -> torch.Tensor:
    device = positions[0].device if positions else None
    total = _as_tensor(0.0, device=device)
    for pos, cat in zip(positions, categories):
        cat = str(cat).lower()
        for placed in already_placed:
            other_cat = _item_category(placed)
            target = RELATIONS.get((cat, other_cat)) or RELATIONS.get((other_cat, cat))
            if target is None:
                continue
            px, pz = _position_xz(placed)
            min_d, max_d = target
            dist = torch.sqrt((pos[0] - px) ** 2 + (pos[1] - pz) ** 2 + EPS)
            total = total + F.relu(min_d - dist) ** 2 + F.relu(dist - max_d) ** 2
    return total


def loss_style(
    positions: Sequence[torch.Tensor],
    rotations: Sequence[torch.Tensor],
    categories: Sequence[str],
    walls: Sequence[Any] | None = None,
    floor_polygon: Sequence[Sequence[float]] | None = None,
    wall_distance_threshold: float = config.WALL_DISTANCE_THRESHOLD,
) -> torch.Tensor:
    if not walls and floor_polygon is not None:
        device = positions[0].device if positions else None
        walls = wall_segments_from_polygon(floor_polygon, device=device)
    if not walls:
        device = positions[0].device if positions else None
        return _as_tensor(0.0, device=device)

    device = positions[0].device if positions else None
    total = _as_tensor(0.0, device=device)
    for pos, rot, cat in zip(positions, rotations, categories):
        cat = str(cat).lower()
        wall_dist, wall_yaw = nearest_wall_info(pos[0], pos[1], walls, temperature=0.1)
        angle_diff = rot - wall_yaw
        alignment_loss = 1.0 - torch.cos(2.0 * angle_diff) ** 2

        if cat in WALL_HUGGING:
            total = total + 2.0 * alignment_loss
            total = total + 1.5 * F.relu(wall_dist - wall_distance_threshold)
        elif cat in WALL_PARALLEL:
            total = total + 1.25 * alignment_loss
    return total
