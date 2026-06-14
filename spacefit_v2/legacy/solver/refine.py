"""Grid-search refinement to produce final (x, z, yaw) for each new furniture item.

Runs sequentially: once an item is placed, its footprint becomes a new obstacle
for subsequent items.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon

from .. import config
from ..core import geometry as G
from . import collision as col
from .scoring import score_pose, ScoredPose


def _obstacle_polys(scene: Dict) -> List[Polygon]:
    polys = []
    for o in scene["objects"]:
        cx, _, cz = o["position"]
        w, _, d = o["size"]
        polys.append(G.rotated_bbox_polygon(cx, cz, w, d, o["yaw"]))
    return polys


def _door_centers(scene: Dict) -> List[Tuple[float, float]]:
    return [(d["position"][0], d["position"][2]) for d in scene.get("doors", [])]


def _normalize_category(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")


def _matches_category(name: str, keywords: Sequence[str]) -> bool:
    cat = _normalize_category(name)
    return any(k in cat for k in keywords)


def _is_table_like(name: str) -> bool:
    return _matches_category(name, ("table", "desk"))


def _is_desk_like(name: str) -> bool:
    return _matches_category(name, ("desk",))


def _is_seat_like(name: str) -> bool:
    return _matches_category(name, ("chair", "sofa", "loveseat", "armchair", "bench", "stool", "barstool"))


def _is_chair_like(name: str) -> bool:
    return _matches_category(name, ("chair", "stool", "barstool", "bench"))


def _is_coffee_table(name: str) -> bool:
    return _matches_category(name, ("coffee_table",))


def _is_side_table(name: str) -> bool:
    return _matches_category(name, ("side_table",))


def _is_floor_lamp(name: str) -> bool:
    return _matches_category(name, ("floor_lamp", "lamp"))


def _is_bed_like(name: str) -> bool:
    return _matches_category(name, ("bed",))


def _is_nightstand_like(name: str) -> bool:
    return _matches_category(name, ("nightstand", "side_table"))


def _is_lamp_like(name: str) -> bool:
    return _matches_category(name, ("lamp",))


def _is_rug_like(name: str) -> bool:
    return _matches_category(name, ("rug", "mat"))


def _is_wall_hugging(name: str) -> bool:
    return _matches_category(
        name,
        ("wardrobe", "bookshelf", "shelf", "cabinet", "storage", "tv_stand",
         "dresser", "sideboard", "console"),
    )


def _is_wall_parallel(name: str) -> bool:
    return _matches_category(
        name,
        ("bed", "desk", "sofa", "table", "dining_table", "dressing_table"),
    )


def _is_strict_wall_exempt(name: str) -> bool:
    return _matches_category(
        name,
        ("nightstand", "lamp", "pendant_lamp", "floor_lamp"),
    )


PLACEMENT_PRIORITY = {
    "bed": 100,
    "double_bed": 100,
    "single_bed": 95,
    "kids_bed": 95,
    "sofa": 90,
    "dining_table": 85,
    "table": 80,
    "wardrobe": 75,
    "bookshelf": 70,
    "desk": 70,
    "cabinet": 65,
    "storage": 65,
    "dresser": 65,
    "tv_stand": 60,
    "sideboard": 55,
    "chair": 50,
    "dining_chair": 50,
    "armchair": 50,
    "nightstand": 45,
    "coffee_table": 45,
    "side_table": 40,
    "table_lamp": 30,
    "floor_lamp": 30,
    "pendant_lamp": 25,
    "ceiling_lamp": 25,
    "rug": 20,
    "mat": 20,
    "throw_pillow": 10,
    "plant": 15,
    "vase": 10,
}


def _placement_priority_value(furniture: Dict[str, Any]) -> Tuple[int, float]:
    cat = _normalize_category(furniture.get("category", ""))
    size = furniture.get("size") or {}
    area = float(size.get("width", 0.0)) * float(size.get("depth", 0.0))
    return (PLACEMENT_PRIORITY.get(cat, 35), area)


def _sort_selection_entries(entries: List[Dict[str, Any]], fur_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Priority-first ordering with simple dependency handling for `existing:<id>` links."""
    id_to_entry = {entry["furniture_id"]: dict(entry) for entry in entries}
    deps: Dict[str, set[str]] = {fid: set() for fid in id_to_entry}
    children: Dict[str, set[str]] = {fid: set() for fid in id_to_entry}

    for fid, entry in id_to_entry.items():
        face_toward = str(entry.get("face_toward", "")).lower()
        if not face_toward.startswith("existing:"):
            continue
        query = face_toward.split(":", 1)[1]
        for other_id, other_furniture in fur_by_id.items():
            if other_id == fid:
                continue
            other_cat = _normalize_category(other_furniture.get("category", ""))
            other_low = other_id.lower()
            if other_low.startswith(query) or query in other_low or query == other_cat or query in other_cat:
                deps[fid].add(other_id)
                children[other_id].add(fid)

    scheduled: List[Dict[str, Any]] = []
    ready = [fid for fid, reqs in deps.items() if not reqs]

    def sort_key(fid: str) -> Tuple[int, float, str]:
        priority, area = _placement_priority_value(fur_by_id.get(fid, {}))
        return (-priority, -area, fid)

    while ready:
        ready.sort(key=sort_key)
        fid = ready.pop(0)
        scheduled.append(id_to_entry[fid])
        for child in children[fid]:
            deps[child].discard(fid)
            if not deps[child] and child not in [x["furniture_id"] for x in scheduled] and child not in ready:
                ready.append(child)

    remaining = [fid for fid in id_to_entry if fid not in [x["furniture_id"] for x in scheduled]]
    remaining.sort(key=sort_key)
    scheduled.extend(id_to_entry[fid] for fid in remaining)
    return scheduled


def _relationship_target_keywords(category: str) -> Sequence[str]:
    if _is_nightstand_like(category):
        return ("bed",)
    if _is_lamp_like(category):
        return ("nightstand", "desk", "table")
    if _is_chair_like(category):
        return ("desk", "table", "dining_table")
    if _is_coffee_table(category):
        return ("sofa",)
    if _is_rug_like(category):
        return ("bed", "sofa")
    if _matches_category(category, ("bookshelf",)):
        return ("desk",)
    return ()


def _infer_relationship_target(
    furniture: Dict[str, Any],
    placed_new: Sequence[Dict[str, Any]],
    scene: Dict[str, Any],
    region_center: Tuple[float, float],
) -> Optional[Dict[str, Any]]:
    queries = _relationship_target_keywords(furniture.get("category", ""))
    if not queries:
        return None

    best = None
    best_dist = float("inf")
    cx, cz = region_center

    for pp in placed_new:
        pos = pp.get("position")
        if not pos:
            continue
        cat = pp.get("category", "")
        if not any(_matches_category(cat, (query,)) for query in queries):
            continue
        dist = math.hypot(float(pos["x"]) - cx, float(pos["z"]) - cz)
        if dist < best_dist:
            best_dist = dist
            best = {
                "kind": "object",
                "point": (float(pos["x"]), float(pos["z"])),
                "id": pp.get("furniture_id"),
                "category": cat,
                "yaw": float(pp.get("yaw", 0.0)),
                "size": dict(pp.get("size") or {}),
                "source": "placed_new",
            }

    if best is not None:
        return best

    for obj in scene.get("objects", []):
        cat = obj.get("category", "")
        if not any(_matches_category(cat, (query,)) for query in queries):
            continue
        dist = math.hypot(float(obj["position"][0]) - cx, float(obj["position"][2]) - cz)
        if dist < best_dist:
            best_dist = dist
            best = {
                "kind": "object",
                "point": (float(obj["position"][0]), float(obj["position"][2])),
                "id": obj.get("id"),
                "category": cat,
                "yaw": float(obj.get("yaw", 0.0)),
                "size": {
                    "width": float(obj["size"][0]),
                    "height": float(obj["size"][1]),
                    "depth": float(obj["size"][2]),
                },
                "source": "scene",
            }
    return best


def _rotation_candidates_for(
    furniture: Dict[str, Any],
    region,
    preferred_wall: bool,
    relation: Optional[Dict[str, Any]],
    strict_wall: bool = True,
) -> List[float]:
    category = furniture.get("category", "")
    if region.wall_yaw is None:
        rotations = list(config.ROTATION_CANDIDATES)
    else:
        base = region.wall_yaw % 360.0
        orthogonal = [(base + delta) % 360.0 for delta in (0.0, 90.0, 180.0, 270.0)]
        diagonal = [(angle + 45.0) % 360.0 for angle in orthogonal]
        strict_applicable = strict_wall and not _is_strict_wall_exempt(category)
        if strict_applicable and (_is_wall_hugging(category) or _is_wall_parallel(category) or preferred_wall):
            rotations = orthogonal
        else:
            rotations = orthogonal + diagonal + list(config.ROTATION_CANDIDATES)

    if relation:
        for anchor in relation.get("anchors", []):
            yaw = anchor.get("yaw")
            if yaw is None:
                continue
            for off in (-15.0, 0.0, 15.0):
                rotations.append((float(yaw) + off) % 360.0)

    return sorted({round(rotation % 360.0, 4) for rotation in rotations})


def _basis_from_yaw(yaw_deg: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    rad = math.radians(yaw_deg)
    forward = (math.sin(rad), math.cos(rad))
    right = (math.cos(rad), -math.sin(rad))
    return forward, right


def _point_with_offset(origin: Tuple[float, float],
                       right: Tuple[float, float],
                       forward: Tuple[float, float],
                       side_offset: float,
                       front_offset: float) -> Tuple[float, float]:
    return (
        origin[0] + right[0] * side_offset + forward[0] * front_offset,
        origin[1] + right[1] * side_offset + forward[1] * front_offset,
    )


def _semantic_relation(furniture: Dict, target: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build target-relative anchors for relationship-aware placement."""
    if not target or target.get("kind") != "object":
        return None

    fcat = furniture.get("category", "")
    tcat = target.get("category", "")
    tx, tz = target["point"]
    tyaw = float(target.get("yaw", 0.0))
    tw = float(target.get("size", {}).get("width", 0.0))
    td = float(target.get("size", {}).get("depth", 0.0))
    fw = float(furniture["size"]["width"])
    fd = float(furniture["size"]["depth"])
    front, right = _basis_from_yaw(tyaw)
    center = (tx, tz)

    def seat_yaw(px: float, pz: float) -> float:
        return G.yaw_toward((px, pz), center)

    if _is_chair_like(fcat) and _is_desk_like(tcat):
        gap = 0.08
        front_dist = td / 2.0 + fd / 2.0 + gap
        px, pz = _point_with_offset(center, right, front, 0.0, front_dist)
        return {
            "name": "chair_at_desk",
            "target_key": target.get("id") or f"{target.get('source')}:{target.get('category')}",
            "anchors": [{"x": px, "z": pz, "yaw": seat_yaw(px, pz)}],
            "position_mode": "anchor_neighborhood",
            "position_tolerance": 0.28,
            "yaw_tolerance": 22.5,
        }

    if _is_nightstand_like(fcat) and _is_bed_like(tcat):
        gap = 0.05
        side_dist = tw / 2.0 + fw / 2.0 + gap
        anchors = []
        for side_offset in (side_dist, -side_dist):
            px, pz = _point_with_offset(center, right, front, side_offset, 0.0)
            anchors.append({"x": px, "z": pz, "yaw": tyaw % 360.0})
            anchors.append({"x": px, "z": pz, "yaw": (tyaw + 180.0) % 360.0})
        return {
            "name": "nightstand_beside_bed",
            "target_key": target.get("id") or f"{target.get('source')}:{target.get('category')}",
            "anchors": anchors,
            "position_mode": "anchor_neighborhood",
            "position_tolerance": 0.28,
            "yaw_tolerance": 20.0,
        }

    if _is_lamp_like(fcat) and _is_nightstand_like(tcat):
        anchors = [{"x": tx, "z": tz, "yaw": None}]
        return {
            "name": "lamp_near_nightstand",
            "target_key": target.get("id") or f"{target.get('source')}:{target.get('category')}",
            "anchors": anchors,
            "position_mode": "anchor_neighborhood",
            "position_tolerance": 0.22,
            "yaw_tolerance": 45.0,
        }

    if _is_chair_like(fcat) and _is_table_like(tcat):
        gap = 0.12
        front_dist = td / 2.0 + fd / 2.0 + gap
        side_dist = tw / 2.0 + fd / 2.0 + gap
        anchors = []
        for side_offset, front_offset in (
            (0.0, front_dist),
            (0.0, -front_dist),
            (side_dist, 0.0),
            (-side_dist, 0.0),
        ):
            px, pz = _point_with_offset(center, right, front, side_offset, front_offset)
            anchors.append({"x": px, "z": pz, "yaw": seat_yaw(px, pz)})
        return {
            "name": "chair_around_table",
            "target_key": target.get("id") or f"{target.get('source')}:{target.get('category')}",
            "anchors": anchors,
            "position_mode": "anchor_neighborhood",
            "position_tolerance": 0.35,
            "yaw_tolerance": 25.0,
        }

    if _is_coffee_table(fcat) and _is_seat_like(tcat):
        gap = 0.22
        front_dist = td / 2.0 + fd / 2.0 + gap
        px, pz = _point_with_offset(center, right, front, 0.0, front_dist)
        return {
            "name": "coffee_table_in_front",
            "target_key": target.get("id") or f"{target.get('source')}:{target.get('category')}",
            "anchors": [
                {"x": px, "z": pz, "yaw": tyaw % 360.0},
                {"x": px, "z": pz, "yaw": (tyaw + 180.0) % 360.0},
            ],
            "position_mode": "anchor_neighborhood",
            "position_tolerance": 0.38,
            "yaw_tolerance": 20.0,
        }

    if _is_side_table(fcat) and _is_seat_like(tcat):
        gap = 0.08
        side_dist = tw / 2.0 + fw / 2.0 + gap
        anchors = []
        for side_offset in (side_dist, -side_dist):
            px, pz = _point_with_offset(center, right, front, side_offset, 0.0)
            anchors.append({"x": px, "z": pz, "yaw": tyaw % 360.0})
            anchors.append({"x": px, "z": pz, "yaw": (tyaw + 180.0) % 360.0})
        return {
            "name": "side_table_beside_seat",
            "target_key": target.get("id") or f"{target.get('source')}:{target.get('category')}",
            "anchors": anchors,
            "position_tolerance": 0.30,
            "yaw_tolerance": 25.0,
        }

    if _is_floor_lamp(fcat) and _is_seat_like(tcat):
        gap = 0.18
        side_dist = tw / 2.0 + fw / 2.0 + gap
        anchors = []
        for side_offset in (side_dist, -side_dist):
            for front_offset in (0.0, 0.2):
                px, pz = _point_with_offset(center, right, front, side_offset, front_offset)
                anchors.append({"x": px, "z": pz, "yaw": None})
        return {
            "name": "lamp_beside_seat",
            "target_key": target.get("id") or f"{target.get('source')}:{target.get('category')}",
            "anchors": anchors,
            "position_tolerance": 0.35,
            "yaw_tolerance": 45.0,
        }

    if _is_rug_like(fcat) and (_is_bed_like(tcat) or _is_seat_like(tcat)):
        gap = 0.05
        front_dist = td / 2.0 + fd / 2.0 + gap
        px, pz = _point_with_offset(center, right, front, 0.0, front_dist)
        return {
            "name": "rug_in_front",
            "target_key": target.get("id") or f"{target.get('source')}:{target.get('category')}",
            "anchors": [
                {"x": px, "z": pz, "yaw": tyaw % 360.0},
                {"x": px, "z": pz, "yaw": (tyaw + 180.0) % 360.0},
            ],
            "position_mode": "anchor_neighborhood",
            "position_tolerance": 0.40,
            "yaw_tolerance": 25.0,
        }

    return None


def _prune_used_relation_anchors(relation: Optional[Dict[str, Any]],
                                 placed_new: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Remove anchors already consumed by prior placements of the same relation."""
    if not relation:
        return None

    anchors = list(relation.get("anchors") or [])
    if len(anchors) <= 1:
        return relation

    used = []
    for pp in placed_new:
        if pp.get("semantic_relation") != relation.get("name"):
            continue
        if pp.get("semantic_target_key") != relation.get("target_key"):
            continue
        anchor = pp.get("semantic_anchor")
        if anchor:
            used.append(anchor)

    if not used:
        return relation

    pos_tol = float(relation.get("position_tolerance", 0.35))
    kept = []
    for anchor in anchors:
        ax = float(anchor["x"])
        az = float(anchor["z"])
        occupied = any(
            math.hypot(ax - float(u["x"]), az - float(u["z"])) <= max(0.12, pos_tol * 0.45)
            for u in used
        )
        if not occupied:
            kept.append(anchor)

    if not kept:
        return relation

    out = dict(relation)
    out["anchors"] = kept
    out["blocked_anchors"] = used
    return out


def _closest_relation_anchor(relation: Optional[Dict[str, Any]],
                             cx: float, cz: float, yaw: float) -> Optional[Dict[str, float]]:
    """Return the anchor best matching the chosen pose for bookkeeping."""
    if not relation or not relation.get("anchors"):
        return None

    best = None
    best_score = float("inf")
    for anchor in relation["anchors"]:
        pos_dist = math.hypot(cx - float(anchor["x"]), cz - float(anchor["z"]))
        desired_yaw = anchor.get("yaw")
        yaw_err = 0.0 if desired_yaw is None else abs(((yaw - float(desired_yaw) + 540.0) % 360.0) - 180.0)
        score = pos_dist + yaw_err / 180.0
        if score < best_score:
            best_score = score
            best = {
                "x": float(anchor["x"]),
                "z": float(anchor["z"]),
            }
    return best


def _resolve_face_target(face_toward: str, scene: Dict, region_cx: float, region_cz: float,
                         placed_new: Optional[List[Dict]] = None) -> Optional[Dict[str, Any]]:
    """Resolve a face_toward hint to a semantic target descriptor.

    Returns:
      {"kind": "point"|"object", "point": (x, z), ...}
    """
    face_toward = (face_toward or "").lower()
    if not face_toward or face_toward == "center":
        xmin, zmin, xmax, zmax = scene["floor"]["bounds"]
        return {
            "kind": "point",
            "point": ((xmin + xmax) / 2.0, (zmin + zmax) / 2.0),
            "label": "center",
        }
    if face_toward == "door" and scene.get("doors"):
        d = scene["doors"][0]
        return {
            "kind": "point",
            "point": (d["position"][0], d["position"][2]),
            "label": "door",
        }
    if face_toward == "window" and scene.get("windows"):
        best = None
        best_d = float("inf")
        for w in scene["windows"]:
            dx = w["position"][0] - region_cx
            dz = w["position"][2] - region_cz
            dd = dx * dx + dz * dz
            if dd < best_d:
                best_d = dd
                best = {
                    "kind": "point",
                    "point": (w["position"][0], w["position"][2]),
                    "label": "window",
                }
        return best
    if face_toward == "wall":
        return None
    if face_toward.startswith("existing:"):
        query = face_toward.split(":", 1)[1].lower()
        for pp in (placed_new or []):
            fid = (pp.get("furniture_id") or "").lower()
            cat = (pp.get("category") or "").lower()
            if fid.startswith(query) or query in fid or query in cat:
                pos = pp.get("position")
                if pos:
                    return {
                        "kind": "object",
                        "point": (pos["x"], pos["z"]),
                        "id": pp.get("furniture_id"),
                        "category": pp.get("category"),
                        "yaw": pp.get("yaw", 0.0),
                        "size": dict(pp.get("size") or {}),
                        "source": "placed_new",
                    }
        for o in scene["objects"]:
            oid = (o["id"] or "").lower()
            ocat = (o["category"] or "").lower()
            if oid.startswith(query) or query in oid or query == ocat or query in ocat:
                return {
                    "kind": "object",
                    "point": (o["position"][0], o["position"][2]),
                    "id": o["id"],
                    "category": o["category"],
                    "yaw": o["yaw"],
                    "size": {
                        "width": o["size"][0],
                        "height": o["size"][1],
                        "depth": o["size"][2],
                    },
                    "source": "scene",
                }
    return None


def _search_in_region(region, furniture, floor_poly, obstacle_polys,
                      door_centers, walls, face_target,
                      relation: Optional[Dict[str, Any]] = None,
                      step: float = config.SEARCH_STEP_XY,
                      rotations: Sequence[float] = config.ROTATION_CANDIDATES,
                      margin: float = 0.0,
                      relax_wall_alignment_penalty: bool = False) -> Optional[ScoredPose]:
    """Grid-search over (x, z, yaw). Inset bounds depend on yaw so narrow regions
    with a long axis still get explored (axis-aligned footprints use min(w,d)/2).
    """
    xmin, zmin, xmax, zmax = region.bounds
    w = furniture["size"]["width"]
    d = furniture["size"]["depth"]

    # Augment the search grid with "seats-around-a-target" positions when we
    # have a face_target inside or near the region. For a chair facing a
    # table, we sample points on rings around the target at 8 angles so the
    # solver considers sitting *next to* the table rather than wherever the
    # grid hits.
    extra_xz: List[Tuple[float, float]] = []
    if face_target is not None:
        tx, tz = face_target
        # ring radius: just outside the target's estimated half-extent + our own half-extent
        for radius in (max(w, d) / 2.0 + 0.35, max(w, d) / 2.0 + 0.60):
            for ang_deg in range(0, 360, 45):
                a = math.radians(ang_deg)
                px = tx + radius * math.sin(a)
                pz = tz + radius * math.cos(a)
                if xmin <= px <= xmax and zmin <= pz <= zmax:
                    extra_xz.append((px, pz))
    if relation:
        for anchor in relation.get("anchors", []):
            px = float(anchor["x"])
            pz = float(anchor["z"])
            if xmin <= px <= xmax and zmin <= pz <= zmax:
                extra_xz.append((px, pz))
                for dx, dz in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)):
                    qx = px + dx
                    qz = pz + dz
                    if xmin <= qx <= xmax and zmin <= qz <= zmax:
                        extra_xz.append((qx, qz))

    best: Optional[ScoredPose] = None
    for yaw in rotations:
        # footprint half-extents in world frame for 0/90/180/270 rotations
        # (for non-axis-aligned yaw this is an under-estimate; boundary check
        # catches anything that actually sticks out)
        if int(round(yaw)) % 180 == 0:
            hx, hz = w / 2.0, d / 2.0
        else:
            hx, hz = d / 2.0, w / 2.0
        x_lo = xmin + hx - margin
        x_hi = xmax - hx + margin
        z_lo = zmin + hz - margin
        z_hi = zmax - hz + margin
        if x_lo > x_hi:
            x_lo = x_hi = region.centroid[0]
        if z_lo > z_hi:
            z_lo = z_hi = region.centroid[1]

        xs = np.arange(x_lo, x_hi + 1e-9, step)
        zs = np.arange(z_lo, z_hi + 1e-9, step)
        if xs.size == 0:
            xs = np.array([region.centroid[0]])
        if zs.size == 0:
            zs = np.array([region.centroid[1]])

        positions = [(float(x), float(z)) for x in xs for z in zs]
        positions.extend(extra_xz)
        if relation and relation.get("position_mode") == "anchor_neighborhood" and extra_xz:
            positions = list(extra_xz)

        for (x, z) in positions:
            sp = score_pose(
                cx=float(x), cz=float(z), yaw_deg=float(yaw),
                width=w, depth=d,
                floor_poly=floor_poly,
                obstacle_polys=obstacle_polys,
                door_centers=door_centers,
                wall_yaw=region.wall_yaw,
                walls=walls,
                category=furniture.get("category", ""),
                region_centroid=region.centroid,
                face_target=face_target,
                relation=relation,
                relax_wall_alignment_penalty=relax_wall_alignment_penalty,
            )
            if sp is None:
                continue
            if best is None or sp.score > best.score:
                best = sp
    return best


def refine_placements(scene: Dict, candidates: List, selections: Dict,
                      new_furniture: List[Dict]) -> List[Dict]:
    """Return a list of placement dicts:
         [{furniture_id, category, position {x,y,z}, rotation_y, size {width,depth,height},
           region_id, score, reason}]
    """
    floor_poly = scene["floor"]["polygon"]
    obstacle_polys = _obstacle_polys(scene)
    door_centers = _door_centers(scene)
    walls = scene.get("walls", [])

    cand_by_id = {c.id: c for c in candidates}
    fur_by_id = {f["id"]: f for f in new_furniture}

    placements: List[Dict] = []
    # Running list of successfully-placed new items, used to resolve
    # face_toward references like "existing:chair-new-0" that point to items
    # placed earlier in THIS pipeline run.
    placed_new: List[Dict] = []
    ordered_entries = _sort_selection_entries(list(selections["placements"]), fur_by_id)
    for entry in ordered_entries:
        fid = entry["furniture_id"]
        furniture = fur_by_id.get(fid)
        if furniture is None:
            continue

        tries = []
        if entry.get("selected_region") is not None:
            tries.append(entry["selected_region"])
        if entry.get("backup_region") is not None:
            tries.append(entry["backup_region"])
        # Last-resort: all candidates sorted by area desc.
        for c in sorted(candidates, key=lambda x: -x.area):
            if c.id not in tries:
                tries.append(c.id)

        chosen: Optional[ScoredPose] = None
        chosen_region_id: Optional[int] = None
        chosen_relation: Optional[Dict[str, Any]] = None

        def try_mode(strict_wall: bool) -> Tuple[Optional[ScoredPose], Optional[int], Optional[Dict[str, Any]]]:
            for rid in tries:
                region = cand_by_id.get(rid)
                if region is None:
                    continue
                target_spec = _resolve_face_target(
                    entry.get("face_toward", "center"), scene,
                    region.centroid[0], region.centroid[1],
                    placed_new=placed_new,
                )
                if (
                    (target_spec is None or target_spec.get("kind") != "object")
                    and str(entry.get("face_toward", "center")).lower() in {"", "center"}
                ):
                    inferred = _infer_relationship_target(
                        furniture,
                        placed_new,
                        scene,
                        region.centroid,
                    )
                    if inferred is not None:
                        target_spec = inferred
                face_target = target_spec["point"] if target_spec else None
                relation = _semantic_relation(furniture, target_spec)
                relation = _prune_used_relation_anchors(relation, placed_new)
                rotations = _rotation_candidates_for(
                    furniture,
                    region,
                    bool(entry.get("preferred_wall")),
                    relation,
                    strict_wall=strict_wall,
                )
                sp = _search_in_region(
                    region,
                    furniture,
                    floor_poly,
                    obstacle_polys,
                    door_centers,
                    walls,
                    face_target,
                    relation=relation,
                    rotations=rotations,
                    relax_wall_alignment_penalty=not strict_wall,
                )
                if sp is not None:
                    return sp, rid, relation
            return None, None, None

        chosen, chosen_region_id, chosen_relation = try_mode(strict_wall=True)
        if chosen is None:
            chosen, chosen_region_id, chosen_relation = try_mode(strict_wall=False)

        if chosen is None:
            placements.append({
                "furniture_id": fid,
                "category": furniture["category"],
                "status": "unplaced",
                "reason": entry.get("reason", "") + " | no valid pose in any region",
                "size": dict(furniture["size"]),
            })
            continue

        # Add the new footprint to obstacles so subsequent items avoid it.
        fp_poly = col.footprint_polygon(chosen.cx, chosen.cz, furniture["size"]["width"],
                                        furniture["size"]["depth"], chosen.yaw)
        obstacle_polys.append(fp_poly)

        placed_new.append({
            "furniture_id": fid,
            "category": furniture.get("category", ""),
            "position": {"x": chosen.cx, "z": chosen.cz},
            "yaw": chosen.yaw,
            "size": dict(furniture["size"]),
            "semantic_relation": chosen_relation.get("name") if chosen_relation else None,
            "semantic_target_key": chosen_relation.get("target_key") if chosen_relation else None,
            "semantic_anchor": _closest_relation_anchor(chosen_relation, chosen.cx, chosen.cz, chosen.yaw),
        })

        placements.append({
            "furniture_id": fid,
            "category": furniture["category"],
            "status": "placed",
            "position": {"x": round(chosen.cx, 4),
                         "y": round(furniture["size"]["height"] / 2.0, 4),
                         "z": round(chosen.cz, 4)},
            "rotation_y": round(chosen.yaw, 2),
            "size": {"width": furniture["size"]["width"],
                     "depth": furniture["size"]["depth"],
                     "height": furniture["size"]["height"]},
            "selected_model": furniture.get("_selected_model"),
            "region_id": chosen_region_id,
            "score": round(chosen.score, 3),
            "score_components": {k: round(v, 3) for k, v in chosen.components.items()},
            "reason": entry.get("reason", ""),
        })

    return placements
