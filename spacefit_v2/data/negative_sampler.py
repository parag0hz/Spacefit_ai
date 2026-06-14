"""Negative placement samplers for scorer training."""
from __future__ import annotations

import math
import random
from typing import Dict, Iterable, Sequence

from shapely.geometry import Point, Polygon

from spacefit_v2.model.features import Furniture
from spacefit_v2.model.geom import wall_segments_from_polygon


def _sample_point_in_polygon(rng: random.Random, floor_polygon: Sequence[Sequence[float]]) -> tuple[float, float]:
    poly = Polygon(floor_polygon)
    xmin, zmin, xmax, zmax = poly.bounds
    for _ in range(256):
        x = rng.uniform(xmin, xmax)
        z = rng.uniform(zmin, zmax)
        if poly.contains(Point(x, z)):
            return x, z
    return (float((xmin + xmax) * 0.5), float((zmin + zmax) * 0.5))


def sample_random_negative(
    item: Dict,
    floor_polygon: Sequence[Sequence[float]],
    rng: random.Random,
) -> Dict:
    x, z = _sample_point_in_polygon(rng, floor_polygon)
    return {
        **item,
        "position": {"x": x, "z": z},
        "rotation_y": rng.uniform(0.0, 360.0),
    }


def sample_collision_negative(
    item: Dict,
    others: Sequence[Dict],
    floor_polygon: Sequence[Sequence[float]],
    rng: random.Random,
) -> Dict:
    if not others:
        return sample_random_negative(item, floor_polygon, rng)
    target = rng.choice(list(others))
    pos = target["position"]
    jitter = 0.05
    return {
        **item,
        "position": {
            "x": float(pos["x"]) + rng.uniform(-jitter, jitter),
            "z": float(pos["z"]) + rng.uniform(-jitter, jitter),
        },
        "rotation_y": float(target.get("rotation_y", rng.uniform(0.0, 360.0))),
    }


def sample_misaligned_negative(
    item: Dict,
    floor_polygon: Sequence[Sequence[float]],
    rng: random.Random,
) -> Dict:
    x, z = _sample_point_in_polygon(rng, floor_polygon)
    walls = wall_segments_from_polygon(floor_polygon)
    best_yaw_deg = 0.0
    best_dist = float("inf")
    for a, b in walls:
        ax, az = float(a[0]), float(a[1])
        bx, bz = float(b[0]), float(b[1])
        vx, vz = bx - ax, bz - az
        vv = vx * vx + vz * vz or 1e-9
        t = max(0.0, min(1.0, ((x - ax) * vx + (z - az) * vz) / vv))
        px, pz = ax + t * vx, az + t * vz
        dist = math.hypot(x - px, z - pz)
        if dist < best_dist:
            best_dist = dist
            best_yaw_deg = math.degrees(math.atan2(vz, vx))
    return {
        **item,
        "position": {"x": x, "z": z},
        "rotation_y": (best_yaw_deg + 45.0) % 360.0,
    }
