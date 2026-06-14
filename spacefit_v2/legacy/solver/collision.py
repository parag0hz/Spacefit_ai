"""Collision / boundary checks for placement refinement."""
from __future__ import annotations

from typing import List, Sequence

from shapely.geometry import Polygon

from ..core import geometry as G


def footprint_polygon(cx, cz, width, depth, yaw_deg) -> Polygon:
    return G.rotated_bbox_polygon(cx, cz, width, depth, yaw_deg)


def is_in_boundary(footprint: Polygon, floor_poly_pts: Sequence,
                    tol: float = 0.01) -> bool:
    """Return True iff footprint ⊂ floor_poly, allowing `tol` m² of spillover
    to absorb numerical/edge-touching cases.
    """
    floor = Polygon(floor_poly_pts)
    try:
        outside = footprint.difference(floor).area
        return outside <= tol
    except Exception:
        return False


def collides_with(footprint: Polygon, obstacle_polys: List[Polygon],
                   eps: float = 1e-4) -> bool:
    for obs in obstacle_polys:
        try:
            if footprint.intersection(obs).area > eps:
                return True
        except Exception:
            continue
    return False


def door_clearance_ok(footprint: Polygon, door_centers: List, door_radius: float = 0.8) -> bool:
    from shapely.geometry import Point
    for (cx, cz) in door_centers:
        if footprint.distance(Point(cx, cz)) < door_radius * 0.5:
            # Strict: if the new piece sits inside the clearance semicircle we
            # disallow placement. Half the full clearance keeps it lenient.
            return False
    return True
