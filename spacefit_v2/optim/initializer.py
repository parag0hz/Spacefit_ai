"""Initialization helpers for differentiable placement refinement."""
from __future__ import annotations

from typing import Any, Dict, Sequence

import torch

from spacefit_v2.model.geom import nearest_wall_info, wall_segments_from_polygon


def build_region_box(region: Any, floor_polygon: Sequence[Sequence[float]]) -> Dict[str, float]:
    if region is None:
        xs = [float(v[0]) for v in floor_polygon]
        zs = [float(v[1]) for v in floor_polygon]
        return {
            "xmin": min(xs),
            "zmin": min(zs),
            "xmax": max(xs),
            "zmax": max(zs),
            "centroid": ((min(xs) + max(xs)) / 2.0, (min(zs) + max(zs)) / 2.0),
            "id": 0,
        }

    if hasattr(region, "bounds"):
        xmin, zmin, xmax, zmax = region.bounds
        centroid = tuple(region.centroid)
        return {
            "xmin": float(xmin),
            "zmin": float(zmin),
            "xmax": float(xmax),
            "zmax": float(zmax),
            "centroid": (float(centroid[0]), float(centroid[1])),
            "id": int(getattr(region, "id", 0)),
        }

    bounds = region.get("bounds")
    if bounds is None and "max_rect" in region:
        cx, cz, width, depth, _yaw = region["max_rect"]
        bounds = (cx - width / 2.0, cz - depth / 2.0, cx + width / 2.0, cz + depth / 2.0)
    xmin, zmin, xmax, zmax = bounds
    centroid = region.get("centroid") or ((xmin + xmax) / 2.0, (zmin + zmax) / 2.0)
    return {
        "xmin": float(xmin),
        "zmin": float(zmin),
        "xmax": float(xmax),
        "zmax": float(zmax),
        "centroid": (float(centroid[0]), float(centroid[1])),
        "id": int(region.get("id", 0)),
    }


def initialize_pose(
    region_box: Dict[str, float],
    walls: Sequence[Any] | None = None,
    floor_polygon: Sequence[Sequence[float]] | None = None,
    seed_index: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not walls and floor_polygon is not None:
        walls = wall_segments_from_polygon(floor_polygon)

    xmin, zmin, xmax, zmax = region_box["xmin"], region_box["zmin"], region_box["xmax"], region_box["zmax"]
    cx, cz = region_box["centroid"]
    dx = 0.25 * max(0.0, xmax - xmin)
    dz = 0.25 * max(0.0, zmax - zmin)
    offsets = [
        (0.0, 0.0),
        (-dx, -dz),
        (dx, -dz),
        (-dx, dz),
        (dx, dz),
    ]
    ox, oz = offsets[seed_index % len(offsets)]
    x0 = min(max(cx + ox, xmin), xmax)
    z0 = min(max(cz + oz, zmin), zmax)

    x = torch.tensor(x0, dtype=torch.float32, device=device, requires_grad=True)
    z = torch.tensor(z0, dtype=torch.float32, device=device, requires_grad=True)
    if walls:
        _, wall_yaw = nearest_wall_info(x.detach(), z.detach(), walls)
        yaw0 = float(wall_yaw.detach().cpu())
    else:
        yaw0 = 0.0
    yaw = torch.tensor(yaw0, dtype=torch.float32, device=device, requires_grad=True)
    return x, z, yaw
