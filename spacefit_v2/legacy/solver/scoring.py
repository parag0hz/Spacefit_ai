"""Score a (cx, cz, yaw) pose for a given furniture within a candidate region."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from shapely.geometry import Polygon, Point

from .. import config
from ..core import geometry as G
from . import collision as col


def _normalize_category(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in (name or "").strip().lower()).strip("_")


def _matches_category(name: str, keywords: Sequence[str]) -> bool:
    cat = _normalize_category(name)
    return any(keyword in cat for keyword in keywords)


WALL_HUGGING_KEYWORDS = (
    "wardrobe", "bookshelf", "shelf", "cabinet", "storage",
    "tv_stand", "dresser", "sideboard", "console",
)

WALL_PARALLEL_KEYWORDS = (
    "bed", "double_bed", "single_bed", "kids_bed", "desk", "sofa",
    "dining_table", "table", "dressing_table",
)

CENTER_FRIENDLY_KEYWORDS = (
    "bed", "double_bed", "single_bed", "kids_bed", "dining_table",
    "coffee_table", "rug", "mat", "pendant_lamp", "ceiling_lamp",
)


@dataclass
class ScoredPose:
    score: float
    cx: float
    cz: float
    yaw: float
    components: Dict[str, float]


def _wall_alignment(yaw_deg: float, wall_yaw: Optional[float]) -> float:
    """Reward when the object yaw is parallel or perpendicular to the nearest wall."""
    if wall_yaw is None:
        return 0.0
    diff = ((yaw_deg - wall_yaw) % 90.0)
    diff = min(diff, 90.0 - diff)
    return float(max(0.0, 1.0 - diff / 15.0))  # full reward within 15°


def _wall_alignment_score(
    yaw_deg: float,
    wall_yaw: Optional[float],
    category: str,
    relax_penalty: bool = False,
) -> float:
    if wall_yaw is None:
        return 0.0

    align = _wall_alignment(yaw_deg, wall_yaw)
    if _matches_category(category, WALL_HUGGING_KEYWORDS):
        if align >= 0.95:
            return 1.8
        if align >= 0.6:
            return 0.9
        if align >= 0.25:
            score = -0.4
        else:
            score = -1.4
        return max(0.0, score) if relax_penalty else score

    if _matches_category(category, WALL_PARALLEL_KEYWORDS):
        if align >= 0.95:
            return 1.2
        if align >= 0.6:
            return 0.5
        if align >= 0.25:
            score = -0.25
        else:
            score = -1.0
        return max(0.0, score) if relax_penalty else score

    if align >= 0.75:
        return 0.2
    return 0.0


def _wall_proximity_score(cx: float, cz: float, walls, category: str) -> float:
    if not walls:
        return 0.0
    min_wall_dist, _ = G.find_nearest_wall(cx, cz, walls)

    if _matches_category(category, WALL_HUGGING_KEYWORDS):
        if min_wall_dist < 0.15:
            return 1.5
        if min_wall_dist < 0.30:
            return 0.8
        if min_wall_dist < 0.60:
            return -0.2
        return -1.2

    if _matches_category(category, CENTER_FRIENDLY_KEYWORDS):
        if min_wall_dist < 0.15:
            return -0.25
        if min_wall_dist < 0.35:
            return -0.1
        if min_wall_dist < 1.20:
            return 0.15
        return 0.0

    if min_wall_dist < 0.20:
        return 1.0
    if min_wall_dist < 0.50:
        return 0.5
    if min_wall_dist < 1.00:
        return 0.0
    return -0.7


def _yaw_error_deg(a: float, b: float) -> float:
    return abs(((a - b + 540.0) % 360.0) - 180.0)


def _semantic_relation_score(cx: float, cz: float, yaw_deg: float,
                             relation: Optional[Dict[str, object]]) -> float:
    """Return roughly [-1,1] relation quality for target-relative layouts."""
    if not relation:
        return 0.0

    anchors = relation.get("anchors") or []
    pos_tol = float(relation.get("position_tolerance", 0.5))
    yaw_tol = float(relation.get("yaw_tolerance", 25.0))
    best = 0.0
    for anchor in anchors:
        ax = float(anchor["x"])
        az = float(anchor["z"])
        pos_dist = math.hypot(cx - ax, cz - az)
        pos_score = max(0.0, 1.0 - pos_dist / max(pos_tol, 1e-6))

        desired_yaw = anchor.get("yaw")
        if desired_yaw is None:
            yaw_score = 1.0
        else:
            yaw_diff = _yaw_error_deg(yaw_deg, float(desired_yaw))
            yaw_score = max(0.0, 1.0 - yaw_diff / max(yaw_tol, 1e-6))

        # Position dominates; yaw fine-tunes within the same semantic relation.
        combined = 0.75 * pos_score + 0.25 * yaw_score
        if combined > best:
            best = combined

    blocked = relation.get("blocked_anchors") or []
    blocked_penalty = 0.0
    for anchor in blocked:
        ax = float(anchor["x"])
        az = float(anchor["z"])
        pos_dist = math.hypot(cx - ax, cz - az)
        blocked_penalty = max(blocked_penalty, max(0.0, 1.0 - pos_dist / max(pos_tol, 1e-6)))

    return best - blocked_penalty


def score_pose(cx: float, cz: float, yaw_deg: float,
               width: float, depth: float,
               floor_poly, obstacle_polys: List[Polygon],
               door_centers: List[Tuple[float, float]],
               wall_yaw: Optional[float],
               walls,
               category: str,
               region_centroid: Tuple[float, float],
               weights: Dict[str, float] = config.WEIGHTS,
               face_target: Optional[Tuple[float, float]] = None,
               desired_yaw: Optional[float] = None,
               relation: Optional[Dict[str, object]] = None,
               relax_wall_alignment_penalty: bool = False) -> Optional[ScoredPose]:
    """Return a ScoredPose or None if the pose is invalid (out of bounds or colliding)."""
    fp = col.footprint_polygon(cx, cz, width, depth, yaw_deg)
    if not col.is_in_boundary(fp, floor_poly):
        return None
    if col.collides_with(fp, obstacle_polys):
        return None

    components = {}
    components["collision_free"] = weights["collision_free"]
    components["in_boundary"] = weights["in_boundary"]

    # Door clearance
    ok = col.door_clearance_ok(fp, door_centers)
    components["door_clearance"] = weights["door_clearance"] if ok else -weights["door_clearance"]

    # Wall alignment
    components["wall_alignment"] = (
        weights["wall_alignment"] * _wall_alignment_score(
            yaw_deg,
            wall_yaw,
            category,
            relax_penalty=relax_wall_alignment_penalty,
        )
    )
    components["wall_proximity"] = (
        weights.get("wall_proximity", 0.0) * _wall_proximity_score(cx, cz, walls, category)
    )

    # Walkway: require footprint edge to be >= WALKWAY_WIDTH from each obstacle
    walk_penalty = 0.0
    for obs in obstacle_polys:
        d = fp.distance(obs)
        if 0 < d < config.WALKWAY_WIDTH:
            walk_penalty += (config.WALKWAY_WIDTH - d) / config.WALKWAY_WIDTH
    components["walkway"] = weights["walkway"] * max(0.0, 1.0 - walk_penalty)

    # Preferred face (yaw points toward a target)
    if face_target is not None:
        target_yaw = G.yaw_toward((cx, cz), face_target)
        diff = abs(((yaw_deg - target_yaw + 540) % 360) - 180)
        face_score = max(0.0, 1.0 - diff / 45.0)
        components["preferred_face"] = weights["preferred_face"] * face_score
    else:
        components["preferred_face"] = 0.0

    relation_score = _semantic_relation_score(cx, cz, yaw_deg, relation)
    components["semantic_relation"] = weights.get("semantic_relation", 0.0) * relation_score

    # Zone-fit proxy: closeness to the candidate centroid (prefer center of region)
    d_cent = math.hypot(cx - region_centroid[0], cz - region_centroid[1])
    components["centroid_distance"] = -weights["centroid_distance"] * d_cent

    score = sum(components.values())
    return ScoredPose(score=score, cx=cx, cz=cz, yaw=yaw_deg, components=components)
