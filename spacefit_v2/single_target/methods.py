"""Single-target method variants built on top of existing SpaceFit modules."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.optim as optim
from shapely.geometry import Point, Polygon as SPolygon

from spacefit_v2.legacy.core import geometry as G
from spacefit_v2.legacy.core.free_space import Candidate, extract_candidates
from spacefit_v2.legacy.core.occupancy import create_occupancy_grid
from spacefit_v2.legacy.solver.collision import collides_with, footprint_polygon, is_in_boundary
from spacefit_v2.legacy.solver.refine import (
    _door_centers,
    _obstacle_polys,
    _resolve_face_target,
    _rotation_candidates_for,
    _search_in_region,
    refine_placements,
)
from spacefit_v2.legacy.solver.scoring import score_pose
from spacefit_v2.model.geom import EPS, nearest_wall_info, wall_segments_from_polygon
from spacefit_v2.model.losses import loss_physics
from spacefit_v2.optim.initializer import build_region_box, initialize_pose

from .intent_grounder import IntentGrounder

# ── Experiment A tuning constants ────────────────────────────────────────────
_PHYSICS_WEIGHT = 40.0      # raised from 20.0 to penalise CF/IB violations harder
_ANCHOR_WEIGHT = 1.0        # local-motion regulariser: keeps diffopt near feasible seed
_REGION_WEIGHT = 0.1        # centroid regulariser (unchanged)


def _target_size(case: Dict[str, Any]) -> Dict[str, float]:
    size = case["target_asset"]["size"]
    return {
        "width": float(size["width"]),
        "height": float(size["height"]),
        "depth": float(size["depth"]),
    }


def _target_item(case: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": case["target_asset"]["id"],
        "category": case["target_asset"]["category"],
        "size": _target_size(case),
    }


def _scene_for_method(case: Dict[str, Any]) -> Dict[str, Any]:
    scene = case["scene"]
    floor = scene["floor"]
    return {
        "floor": floor,
        "walls": list(scene.get("walls", [])),
        "doors": list(scene.get("doors", [])),
        "windows": list(scene.get("windows", [])),
        "objects": list(scene.get("objects", [])),
        "room_type": scene.get("room_type"),
        "entrance_point": scene.get("entrance_point"),
    }


def _normalize_constraints(grounded: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(item) for item in grounded.get("constraints", [])]


def propose_candidates(case: Dict[str, Any]) -> List[Candidate]:
    scene = _scene_for_method(case)
    occ = create_occupancy_grid(scene)
    return extract_candidates(occ, scene)


def _resolve_target_point(scene: Dict[str, Any], constraint: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    kind = constraint.get("target_kind")
    if kind == "window" and scene.get("windows"):
        window = scene["windows"][0]
        return (float(window["position"][0]), float(window["position"][2]))
    if kind == "door" and scene.get("doors"):
        door = scene["doors"][0]
        return (float(door["position"][0]), float(door["position"][2]))
    if kind == "center":
        xmin, zmin, xmax, zmax = scene["floor"]["bounds"]
        return ((xmin + xmax) * 0.5, (zmin + zmax) * 0.5)

    target_id = constraint.get("target_id")
    if target_id:
        for obj in scene.get("objects", []):
            if str(obj["id"]) == str(target_id):
                return (float(obj["position"][0]), float(obj["position"][2]))

    target_category = _slug(constraint.get("target_category", "") or "")
    if target_category:
        matches = [obj for obj in scene.get("objects", []) if _slug(obj.get("category", "")) == target_category]
        if matches:
            obj = matches[0]
            return (float(obj["position"][0]), float(obj["position"][2]))
    return None


def _face_toward_string(constraints: Sequence[Dict[str, Any]]) -> str:
    facing = next((c for c in constraints if c.get("constraint_type") == "facing"), None)
    if facing is None:
        return "center"
    if facing.get("target_kind") in {"window", "door", "center"}:
        return str(facing["target_kind"])
    if facing.get("target_category"):
        return f"existing:{facing['target_category']}"
    if facing.get("target_id"):
        return f"existing:{facing['target_id']}"
    return "center"


def _preferred_wall(constraints: Sequence[Dict[str, Any]]) -> bool:
    return any(c.get("constraint_type") == "against_wall" for c in constraints)


def _region_bias(region: Candidate, scene: Dict[str, Any], constraints: Sequence[Dict[str, Any]]) -> float:
    score = float(region.area)
    cx, cz = region.centroid
    for constraint in constraints:
        ctype = constraint.get("constraint_type")
        if ctype == "against_wall" and region.near_wall:
            score += 2.0
        if ctype == "keep_window_clear" and not region.near_window:
            score += 1.0
        if ctype == "not_block_door" and not region.near_door:
            score += 1.0
        if ctype == "near":
            point = _resolve_target_point(scene, constraint)
            if point is not None:
                score -= 0.5 * math.hypot(cx - point[0], cz - point[1])
        if ctype == "facing":
            point = _resolve_target_point(scene, constraint)
            if point is not None:
                score -= 0.1 * math.hypot(cx - point[0], cz - point[1])
    return score


def _candidate_ranking(case: Dict[str, Any], constraints: Sequence[Dict[str, Any]]) -> List[Candidate]:
    scene = _scene_for_method(case)
    candidates = propose_candidates(case)
    return sorted(candidates, key=lambda region: _region_bias(region, scene, constraints), reverse=True)


def _existing_polys(scene: Dict[str, Any]):
    return _obstacle_polys(scene)


def _placement_from_pose(case: Dict[str, Any], region: Candidate, pose: Any) -> Dict[str, Any]:
    size = _target_size(case)
    return {
        "furniture_id": case["target_asset"]["id"],
        "category": case["target_asset"]["category"],
        "position": {"x": float(pose.cx), "y": 0.0, "z": float(pose.cz)},
        "rotation_y": float(pose.yaw),
        "size": size,
        "region_id": int(region.id),
        "score": float(pose.score),
        "status": "placed",
    }


def _invalid_prediction(case: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "furniture_id": case["target_asset"]["id"],
        "category": case["target_asset"]["category"],
        "status": "unplaced",
        "reason": reason,
    }


def run_heuristic_baseline(case: Dict[str, Any], grounded: Dict[str, Any], topk: int = 5) -> List[Dict[str, Any]]:
    scene = _scene_for_method(case)
    furniture = _target_item(case)
    floor_poly = scene["floor"]["polygon"]
    obstacle_polys = _existing_polys(scene)
    door_centers = _door_centers(scene)
    constraints = _normalize_constraints(grounded)
    ranked_regions = _candidate_ranking(case, constraints)
    face_toward = _face_toward_string(constraints)

    predictions: List[Dict[str, Any]] = []
    for region in ranked_regions:
        rotations = _rotation_candidates_for(furniture, region, _preferred_wall(constraints), relation=None, strict_wall=False)
        face_target_spec = _resolve_face_target(face_toward, scene, region.centroid[0], region.centroid[1], placed_new=[])
        face_target = face_target_spec["point"] if face_target_spec else None
        best = None
        for yaw in rotations:
            pose = score_pose(
                cx=float(region.centroid[0]),
                cz=float(region.centroid[1]),
                yaw_deg=float(yaw),
                width=float(furniture["size"]["width"]),
                depth=float(furniture["size"]["depth"]),
                floor_poly=floor_poly,
                obstacle_polys=obstacle_polys,
                door_centers=door_centers,
                wall_yaw=region.wall_yaw,
                walls=scene.get("walls", []),
                category=furniture["category"],
                region_centroid=region.centroid,
                face_target=face_target,
            )
            if pose is not None and (best is None or pose.score > best.score):
                best = pose
        if best is not None:
            predictions.append(_placement_from_pose(case, region, best))
        if len(predictions) >= topk:
            break
    return predictions or [_invalid_prediction(case, "no feasible centroid placement")]


def run_heuristic_refinement(case: Dict[str, Any], grounded: Dict[str, Any], topk: int = 5) -> List[Dict[str, Any]]:
    constraints = _normalize_constraints(grounded)
    ranked_regions = _candidate_ranking(case, constraints)
    furniture = _target_item(case)
    face_toward = _face_toward_string(constraints)
    predictions: List[Dict[str, Any]] = []

    for region in ranked_regions[: max(topk, 1)]:
        selections = {
            "placements": [
                {
                    "furniture_id": furniture["id"],
                    "selected_region": int(region.id),
                    "preferred_wall": _preferred_wall(constraints),
                    "face_toward": face_toward,
                }
            ]
        }
        placements = refine_placements(_scene_for_method(case), propose_candidates(case), selections, [furniture])
        if placements and placements[0].get("status") == "placed":
            predictions.append(placements[0])
        if len(predictions) >= topk:
            break
    return predictions or [_invalid_prediction(case, "no feasible heuristic refinement")]


def _constraint_loss(
    pos: torch.Tensor,
    yaw: torch.Tensor,
    scene: Dict[str, Any],
    size: Dict[str, float],
    constraints: Sequence[Dict[str, Any]],
) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float32, device=pos.device)
    walls = wall_segments_from_polygon(scene["floor"]["polygon"], device=pos.device)
    wall_dist, wall_yaw = nearest_wall_info(pos[0], pos[1], walls)

    for constraint in constraints:
        ctype = constraint.get("constraint_type")
        point = _resolve_target_point(scene, constraint)
        if ctype == "against_wall":
            total = total + torch.relu(wall_dist - 0.18) ** 2
            total = total + 0.25 * (1.0 - torch.cos(2.0 * (yaw - wall_yaw)) ** 2)
        elif ctype == "near" and point is not None:
            tx = torch.tensor(float(point[0]), device=pos.device)
            tz = torch.tensor(float(point[1]), device=pos.device)
            dist = torch.sqrt((pos[0] - tx) ** 2 + (pos[1] - tz) ** 2 + EPS)
            max_distance = float(constraint.get("max_distance", 1.4))
            min_distance = float(constraint.get("min_distance", 0.0))
            total = total + torch.relu(dist - max_distance) ** 2 + torch.relu(min_distance - dist) ** 2
        elif ctype == "facing" and point is not None:
            tx = torch.tensor(float(point[0]), device=pos.device)
            tz = torch.tensor(float(point[1]), device=pos.device)
            desired = torch.atan2(tz - pos[1], tx - pos[0])
            total = total + 0.75 * (1.0 - torch.cos(yaw - desired))
        elif ctype == "not_block_door":
            for door in scene.get("doors", []):
                dx = torch.tensor(float(door["position"][0]), device=pos.device)
                dz = torch.tensor(float(door["position"][2]), device=pos.device)
                dist = torch.sqrt((pos[0] - dx) ** 2 + (pos[1] - dz) ** 2 + EPS)
                radius = max(0.8, float(door.get("width", 0.9)) * 0.5)
                total = total + torch.relu(radius - dist) ** 2
        elif ctype == "keep_window_clear":
            for window in scene.get("windows", []):
                wx = torch.tensor(float(window["position"][0]), device=pos.device)
                wz = torch.tensor(float(window["position"][2]), device=pos.device)
                dist = torch.sqrt((pos[0] - wx) ** 2 + (pos[1] - wz) ** 2 + EPS)
                radius = max(0.45, float(window.get("width", 1.0)) * 0.4)
                total = total + 0.5 * torch.relu(radius - dist) ** 2
        elif ctype == "access_zone":
            access_margin = 0.30 if str(casefold(scene.get("room_type", ""))) == "library" else 0.40
            total = total + 0.5 * torch.relu(access_margin - wall_dist) ** 2
        elif ctype == "beside":
            ref = _resolve_ref_object(scene, constraint)
            if ref is not None:
                rx = torch.tensor(float(ref["position"][0]), device=pos.device)
                rz = torch.tensor(float(ref["position"][2]), device=pos.device)
                dist = torch.sqrt((pos[0] - rx) ** 2 + (pos[1] - rz) ** 2 + EPS)
                max_distance = float(constraint.get("max_distance", 0.9))
                min_distance = float(constraint.get("min_distance", 0.25))
                total = total + torch.relu(dist - max_distance) ** 2 + 0.5 * torch.relu(min_distance - dist) ** 2
        elif ctype in ("in_front_of", "behind", "left_of", "right_of"):
            ref = _resolve_ref_object(scene, constraint)
            if ref is not None:
                rx = torch.tensor(float(ref["position"][0]), device=pos.device)
                rz = torch.tensor(float(ref["position"][2]), device=pos.device)
                ref_yaw = torch.tensor(math.radians(float(ref.get("yaw", 0.0))), device=pos.device)
                fwd_x = torch.sin(ref_yaw)
                fwd_z = torch.cos(ref_yaw)
                right_x = torch.cos(ref_yaw)
                right_z = -torch.sin(ref_yaw)
                dx = pos[0] - rx
                dz = pos[1] - rz
                dot_fwd = dx * fwd_x + dz * fwd_z
                dot_right = dx * right_x + dz * right_z
                signed = {
                    "in_front_of": dot_fwd,
                    "behind": -dot_fwd,
                    "left_of": -dot_right,
                    "right_of": dot_right,
                }[str(ctype)]
                cross_axis = dot_right if ctype in ("in_front_of", "behind") else dot_fwd
                margin = float(constraint.get("min_distance", 0.25))
                max_distance = float(constraint.get("max_distance", 2.0))
                total = total + torch.relu(margin - signed) ** 2
                total = total + 0.05 * torch.relu(torch.abs(cross_axis) - max_distance) ** 2
    return total


def casefold(text: str) -> str:
    return str(text or "").strip().lower()


def _placement_score(scene: Dict[str, Any], placement: Dict[str, Any], constraints: Sequence[Dict[str, Any]]) -> float:
    if placement.get("status") != "placed":
        return -1e9
    size = placement["size"]
    poly = footprint_polygon(
        float(placement["position"]["x"]),
        float(placement["position"]["z"]),
        float(size["width"]),
        float(size["depth"]),
        float(placement["rotation_y"]),
    )
    if not is_in_boundary(poly, scene["floor"]["polygon"]):
        return -1e9
    if collides_with(poly, _existing_polys(scene)):
        return -1e9

    score = float(placement.get("score", 0.0))
    cx = float(placement["position"]["x"])
    cz = float(placement["position"]["z"])
    for constraint in constraints:
        point = _resolve_target_point(scene, constraint)
        if constraint.get("constraint_type") == "near" and point is not None:
            score -= 0.5 * math.hypot(cx - point[0], cz - point[1])
        elif constraint.get("constraint_type") == "against_wall":
            dist, _ = G.closest_wall_distance(cx, cz, scene.get("walls", []))
            score -= dist
    return score


def _get_heuristic_seed_pose(
    case: Dict[str, Any],
    region: Any,
    scene: Dict[str, Any],
    constraints: Sequence[Dict[str, Any]],
) -> Optional[Tuple[float, float, float]]:
    """Quick heuristic score at the region centroid; returns (x, z, yaw_rad) or None."""
    if region is None or not hasattr(region, "centroid"):
        return None
    furniture = _target_item(case)
    floor_poly = scene["floor"]["polygon"]
    obstacle_polys = _existing_polys(scene)
    door_centers = _door_centers(scene)
    face_toward = _face_toward_string(constraints)
    rotations = _rotation_candidates_for(furniture, region, _preferred_wall(constraints), relation=None, strict_wall=False)
    face_target_spec = _resolve_face_target(face_toward, scene, region.centroid[0], region.centroid[1], placed_new=[])
    face_target = face_target_spec["point"] if face_target_spec else None

    best_pose = None
    for yaw in rotations:
        pose = score_pose(
            cx=float(region.centroid[0]),
            cz=float(region.centroid[1]),
            yaw_deg=float(yaw),
            width=float(furniture["size"]["width"]),
            depth=float(furniture["size"]["depth"]),
            floor_poly=floor_poly,
            obstacle_polys=obstacle_polys,
            door_centers=door_centers,
            wall_yaw=region.wall_yaw,
            walls=scene.get("walls", []),
            category=furniture["category"],
            region_centroid=region.centroid,
            face_target=face_target,
        )
        if pose is not None and (best_pose is None or pose.score > best_pose.score):
            best_pose = pose
    if best_pose is not None:
        return (float(best_pose.cx), float(best_pose.cz), math.radians(float(best_pose.yaw)))
    return None


def _post_snap_feasible(
    x_val: float,
    z_val: float,
    yaw_val: float,
    scene: Dict[str, Any],
    size: Dict[str, float],
    region_box: Dict[str, float],
) -> Tuple[float, float, float]:
    """Push placement to nearest collision-free, in-boundary position if needed."""
    floor_poly = scene["floor"]["polygon"]
    obstacle_polys = _existing_polys(scene)
    w, d = size["width"], size["depth"]
    yaw_deg = math.degrees(yaw_val)

    try:
        poly = footprint_polygon(x_val, z_val, w, d, yaw_deg)
        if is_in_boundary(poly, floor_poly) and not collides_with(poly, obstacle_polys):
            return x_val, z_val, yaw_val
    except Exception:
        return x_val, z_val, yaw_val

    # Expanding radial search for nearest feasible position
    for radius in (0.08, 0.15, 0.25, 0.40, 0.60):
        for k in range(8):
            angle = math.pi * 2.0 * k / 8.0
            nx = min(max(x_val + radius * math.cos(angle), region_box["xmin"]), region_box["xmax"])
            nz = min(max(z_val + radius * math.sin(angle), region_box["zmin"]), region_box["zmax"])
            try:
                poly = footprint_polygon(nx, nz, w, d, yaw_deg)
                if is_in_boundary(poly, floor_poly) and not collides_with(poly, obstacle_polys):
                    return nx, nz, yaw_val
            except Exception:
                continue
    return x_val, z_val, yaw_val


def _diffopt_single_region(
    case: Dict[str, Any],
    region: Any,
    constraints: Sequence[Dict[str, Any]],
    device: str = "cpu",
    n_iters: int = 100,
    lr: float = 1e-2,
    n_restarts: int = 4,
    use_constraints: bool = False,
    heuristic_seed: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    scene = _scene_for_method(case)
    size = _target_size(case)
    placed_so_far = list(scene.get("objects", []))
    floor_polygon = scene["floor"]["polygon"]
    region_box = build_region_box(region, floor_polygon)
    wall_segments = wall_segments_from_polygon(floor_polygon)

    # Anchor tensors: keep optimisation near the known-feasible heuristic seed
    x_anchor: Optional[torch.Tensor] = None
    z_anchor: Optional[torch.Tensor] = None
    if heuristic_seed is not None:
        x_anchor = torch.tensor(heuristic_seed[0], dtype=torch.float32, device=device)
        z_anchor = torch.tensor(heuristic_seed[1], dtype=torch.float32, device=device)

    best_state = None
    best_loss = float("inf")

    for restart_idx in range(max(1, int(n_restarts))):
        # Restart 0: initialise from heuristic seed when available
        if restart_idx == 0 and heuristic_seed is not None:
            x = torch.tensor(heuristic_seed[0], dtype=torch.float32, device=device, requires_grad=True)
            z = torch.tensor(heuristic_seed[1], dtype=torch.float32, device=device, requires_grad=True)
            yaw = torch.tensor(heuristic_seed[2], dtype=torch.float32, device=device, requires_grad=True)
        else:
            x, z, yaw = initialize_pose(region_box, walls=wall_segments, floor_polygon=floor_polygon, seed_index=restart_idx, device=device)

        optimizer = optim.Adam([x, z, yaw], lr=lr)

        for _ in range(int(n_iters)):
            optimizer.zero_grad()
            pos = torch.stack([x, z])
            # Stronger physics weight prevents violations of CF/IB
            total = _PHYSICS_WEIGHT * loss_physics(
                positions=[pos],
                rotations=[yaw],
                sizes=[size],
                floor_polygon=floor_polygon,
                existing_obstacles=placed_so_far,
                doors=scene.get("doors", []),
            )
            rcx, rcz = region_box["centroid"]
            total = total + _REGION_WEIGHT * ((x - rcx) ** 2 + (z - rcz) ** 2)
            # Local-motion regulariser: anchors to feasible heuristic seed
            if x_anchor is not None:
                total = total + _ANCHOR_WEIGHT * ((x - x_anchor) ** 2 + (z - z_anchor) ** 2)
            if use_constraints:
                total = total + 8.0 * _constraint_loss(pos, yaw, scene, size, constraints)

            total.backward()
            optimizer.step()
            with torch.no_grad():
                x.clamp_(region_box["xmin"], region_box["xmax"])
                z.clamp_(region_box["zmin"], region_box["zmax"])
                yaw.copy_(torch.atan2(torch.sin(yaw), torch.cos(yaw)))

        loss_value = float(total.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = (float(x.detach().cpu()), float(z.detach().cpu()), float(yaw.detach().cpu()))

    if best_state is None:
        return _invalid_prediction(case, "diffopt failed")

    # Post-optimisation feasibility snap: resolve any residual CF/IB violations
    snap_x, snap_z, snap_yaw = _post_snap_feasible(
        best_state[0], best_state[1], best_state[2], scene, size, region_box
    )

    return {
        "furniture_id": case["target_asset"]["id"],
        "category": case["target_asset"]["category"],
        "position": {"x": snap_x, "y": 0.0, "z": snap_z},
        "rotation_y": float(math.degrees(snap_yaw) % 360.0),
        "size": size,
        "region_id": int(region_box["id"]),
        "score": -best_loss,
        "status": "placed",
    }


def run_diffopt(
    case: Dict[str, Any],
    grounded: Dict[str, Any],
    topk: int = 5,
    device: str = "cpu",
    n_iters: int = 100,
    lr: float = 1e-2,
    use_constraints: bool = False,
    use_heuristic_seed: bool = True,
) -> List[Dict[str, Any]]:
    constraints = _normalize_constraints(grounded)
    ranked_regions = _candidate_ranking(case, constraints)
    predictions = []
    scene = _scene_for_method(case)
    for region in ranked_regions[: max(topk, 1)]:
        seed = _get_heuristic_seed_pose(case, region, scene, constraints) if use_heuristic_seed else None
        placement = _diffopt_single_region(
            case=case,
            region=region,
            constraints=constraints,
            device=device,
            n_iters=n_iters,
            lr=lr,
            use_constraints=use_constraints,
            heuristic_seed=seed,
        )
        placement["score"] = _placement_score(scene, placement, constraints)
        predictions.append(placement)
        if len(predictions) >= topk:
            break
    predictions = [p for p in predictions if p.get("status") == "placed"]
    predictions.sort(key=lambda item: float(item.get("score", -1e9)), reverse=True)
    return predictions or [_invalid_prediction(case, "no feasible diffopt placement")]


def run_diffopt_no_proposal(
    case: Dict[str, Any],
    grounded: Dict[str, Any],
    topk: int = 5,
    device: str = "cpu",
    n_iters: int = 100,
    lr: float = 1e-2,
    use_constraints: bool = False,
) -> List[Dict[str, Any]]:
    """DiffOpt without proposal stage — uses full floor bounding box as single region."""
    constraints = _normalize_constraints(grounded)
    scene = _scene_for_method(case)
    # region=None tells build_region_box to use the full floor bounding box
    placement = _diffopt_single_region(
        case=case,
        region=None,
        constraints=constraints,
        device=device,
        n_iters=n_iters,
        lr=lr,
        n_restarts=max(topk, 5),  # more restarts to cover the larger unbounded domain
        use_constraints=use_constraints,
        heuristic_seed=None,
    )
    placement["score"] = _placement_score(scene, placement, constraints)
    predictions = [placement] if placement.get("status") == "placed" else []
    return predictions or [_invalid_prediction(case, "no feasible no-proposal diffopt")]


def _angle_diff(a_deg: float, b_deg: float) -> float:
    """Smallest absolute angle difference in [0, 180]."""
    return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)


def _slug(text: str) -> str:
    """Normalize category name: lowercase, non-alphanumeric → underscore."""
    import re
    return re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_")


def _nearest_wall_dist_yaw(cx: float, cz: float, floor_polygon: list) -> Tuple[float, float]:
    """Return (distance, yaw_deg) to the nearest wall segment of the floor polygon."""
    best_dist = float("inf")
    best_yaw = 0.0
    pts = floor_polygon
    n = len(pts)
    for i in range(n):
        x1, z1 = float(pts[i][0]), float(pts[i][1])
        x2, z2 = float(pts[(i + 1) % n][0]), float(pts[(i + 1) % n][1])
        # Project (cx, cz) onto segment
        dx, dz = x2 - x1, z2 - z1
        seg_len_sq = dx * dx + dz * dz
        if seg_len_sq < 1e-10:
            continue
        t = max(0.0, min(1.0, ((cx - x1) * dx + (cz - z1) * dz) / seg_len_sq))
        px, pz = x1 + t * dx, z1 + t * dz
        dist = math.hypot(cx - px, cz - pz)
        if dist < best_dist:
            best_dist = dist
            best_yaw = math.degrees(math.atan2(dx, dz)) % 180.0
    return best_dist, best_yaw


def _resolve_ref_object(scene: Dict[str, Any], constraint: Dict[str, Any]) -> Optional[Dict]:
    """Find the referenced object in the scene by target_id or target_category."""
    target_id = constraint.get("target_id")
    target_cat = _slug(constraint.get("target_category", "") or "")
    for obj in scene.get("objects", []):
        if target_id and str(obj["id"]) == str(target_id):
            return obj
        if target_cat and _slug(obj.get("category", "")) == target_cat:
            return obj
    return None


def _score_single_constraint(
    cx: float,
    cz: float,
    yaw_deg: float,
    size: Dict[str, float],
    scene: Dict[str, Any],
    constraint: Dict[str, Any],
    floor_polygon: list,
) -> float:
    """Continuous constraint score in [0, 1] for a single (cx, cz, yaw) pose."""
    ctype = constraint.get("constraint_type", "")

    if ctype == "against_wall":
        dist, wall_yaw = _nearest_wall_dist_yaw(cx, cz, floor_polygon)
        size_ref = max(size["width"], size["depth"])
        thresh = max(0.18, 0.15 * size_ref)
        if dist > thresh:
            return 0.0
        aligned = _angle_diff(yaw_deg, wall_yaw) % 90.0
        aligned = min(aligned, 90.0 - aligned)
        return 1.0 if aligned <= 25.0 else 0.5

    if ctype == "facing":
        target = _resolve_target_point(scene, constraint)
        if target is None:
            return 0.0
        desired = math.degrees(math.atan2(target[0] - cx, target[1] - cz))
        diff = _angle_diff(yaw_deg, desired)
        return max(0.0, 1.0 - diff / 90.0)

    if ctype == "near":
        target = _resolve_target_point(scene, constraint)
        if target is None:
            return 0.0
        dist = math.hypot(cx - target[0], cz - target[1])
        max_dist = float(constraint.get("max_distance") or 1.25)
        min_dist = float(constraint.get("min_distance") or 0.0)
        if dist < min_dist - 1e-6:
            return max(0.0, 1.0 - (min_dist - dist) / max(min_dist, 0.5))
        if dist <= max_dist + 1e-6:
            return 1.0
        return max(0.0, 1.0 - (dist - max_dist) / max(max_dist, 1.0))

    if ctype == "beside":
        ref = _resolve_ref_object(scene, constraint)
        if ref is None:
            return 0.0
        rx, rz = float(ref["position"][0]), float(ref["position"][2])
        dist = math.hypot(cx - rx, cz - rz)
        return max(0.0, 1.0 - dist / 0.6)

    if ctype in ("in_front_of", "behind", "left_of", "right_of"):
        ref = _resolve_ref_object(scene, constraint)
        if ref is None:
            return 0.0
        rx, rz = float(ref["position"][0]), float(ref["position"][2])
        ref_yaw = float(ref.get("yaw", 0.0))
        rad = math.radians(ref_yaw)
        fwd = (math.sin(rad), math.cos(rad))
        rgt = (math.cos(rad), -math.sin(rad))
        dx, dz = cx - rx, cz - rz
        dot_fwd = dx * fwd[0] + dz * fwd[1]
        dot_rgt = dx * rgt[0] + dz * rgt[1]
        denom = abs(dot_fwd) + abs(dot_rgt) + 1e-8
        if ctype == "in_front_of":
            return max(0.0, min(1.0, dot_fwd / denom))
        if ctype == "behind":
            return max(0.0, min(1.0, -dot_fwd / denom))
        if ctype == "left_of":
            return max(0.0, min(1.0, -dot_rgt / denom))
        if ctype == "right_of":
            return max(0.0, min(1.0, dot_rgt / denom))

    if ctype == "not_block_door":
        for door in scene.get("doors", []):
            dx = float(door["position"][0])
            dz = float(door["position"][2])
            clearance = max(0.8, float(door.get("width", 0.9)) * 0.5)
            if math.hypot(cx - dx, cz - dz) < clearance:
                return 0.0
        return 1.0

    if ctype == "keep_window_clear":
        for win in scene.get("windows", []):
            wx = float(win["position"][0])
            wz = float(win["position"][2])
            clearance = max(0.45, float(win.get("width", 1.0)) * 0.4)
            if math.hypot(cx - wx, cz - wz) < clearance:
                return 0.0
        return 1.0

    return 1.0  # unknown constraint → neutral


def _score_constraints(
    cx: float,
    cz: float,
    yaw_deg: float,
    size: Dict[str, float],
    scene: Dict[str, Any],
    constraints: Sequence[Dict[str, Any]],
    floor_polygon: list,
) -> float:
    """Average constraint score across all constraints."""
    if not constraints:
        return 1.0
    scores = [
        _score_single_constraint(cx, cz, yaw_deg, size, scene, c, floor_polygon)
        for c in constraints
    ]
    return sum(scores) / len(scores)


def run_constraint_solver(
    case: Dict[str, Any],
    grounded: Dict[str, Any],
    topk: int = 5,
    n_rotations: int = 16,
    grid_step: float = 0.18,
) -> List[Dict[str, Any]]:
    """Constraint-satisfying exhaustive solver.

    LLM provides structured constraints; this solver finds the best feasible
    placements by grid-searching (x, z, yaw) over all candidate free-space
    regions. CF + IB are guaranteed by construction. Returns top-k by
    constraint satisfaction score.
    """
    scene = _scene_for_method(case)
    size = _target_size(case)
    constraints = _normalize_constraints(grounded)
    floor_polygon = scene["floor"]["polygon"]

    occ = create_occupancy_grid(scene)
    candidates = extract_candidates(occ, scene)
    if not candidates:
        return [_invalid_prediction(case, "constraint_solver: no free-space candidates")]

    obstacle_polys = _existing_polys(scene)
    floor_spoly = SPolygon(floor_polygon)

    # Build door keepout zones matching evaluator (0.45m buffer around door center)
    keepout_polys: List[SPolygon] = []
    for door in scene.get("doors", []):
        seg = door.get("segment")
        width = float(door.get("width", 0.9))
        depth = 0.45
        if seg:
            from shapely.geometry import LineString
            keepout_polys.append(LineString(seg).buffer(depth, cap_style=2, join_style=2))
        else:
            dx, dz = float(door["position"][0]), float(door["position"][2])
            keepout_polys.append(Point(dx, dz).buffer(max(depth, width * 0.5)))

    # Rotation candidates: uniform + wall-aligned angles
    base_yaws = [360.0 * i / n_rotations for i in range(n_rotations)]

    w, d = size["width"], size["depth"]

    all_poses: List[Tuple[float, float, float, float, int]] = []  # (score, cx, cz, yaw, region_id)

    for region in candidates:
        xmin, zmin, xmax, zmax = region.bounds

        # Add wall-aligned yaw for near-wall regions
        extra_yaws: List[float] = []
        if region.near_wall and region.wall_yaw is not None:
            for offset in (0.0, 90.0, 180.0, 270.0):
                extra_yaws.append((float(region.wall_yaw) + offset) % 360.0)
        yaws = base_yaws + extra_yaws

        xs = np.arange(xmin, xmax + grid_step, grid_step)
        zs = np.arange(zmin, zmax + grid_step, grid_step)

        for cx in xs:
            for cz in zs:
                for yaw_deg in yaws:
                    try:
                        poly = footprint_polygon(cx, cz, w, d, yaw_deg)
                        # Strict IB: match evaluator threshold (1e-4 m²)
                        if poly.difference(floor_spoly).area > 1e-4:
                            continue
                        if collides_with(poly, obstacle_polys):
                            continue
                        # Door keepout: match evaluator check
                        if any(poly.intersection(kp).area > 1e-4 for kp in keepout_polys):
                            continue
                    except Exception:
                        continue

                    c_score = _score_constraints(cx, cz, yaw_deg, size, scene, constraints, floor_polygon)
                    all_poses.append((c_score, float(cx), float(cz), float(yaw_deg), int(region.id)))

    if not all_poses:
        return [_invalid_prediction(case, "constraint_solver: no feasible placement found")]

    # Sort by constraint score descending
    all_poses.sort(key=lambda p: -p[0])

    # Return top-k, preferring diversity across regions
    results: List[Dict] = []
    seen_regions: set = set()
    remaining: List = []

    for score, cx, cz, yaw_deg, region_id in all_poses:
        if len(results) >= topk:
            break
        if region_id not in seen_regions:
            seen_regions.add(region_id)
            results.append({
                "furniture_id": case["target_asset"]["id"],
                "category": case["target_asset"]["category"],
                "position": {"x": cx, "y": 0.0, "z": cz},
                "rotation_y": yaw_deg % 360.0,
                "size": size,
                "region_id": region_id,
                "score": score,
                "status": "placed",
                "method": "constraint_solver",
            })
        else:
            remaining.append((score, cx, cz, yaw_deg, region_id))

    # Fill remaining slots from best-scoring poses (any region)
    for score, cx, cz, yaw_deg, region_id in remaining:
        if len(results) >= topk:
            break
        results.append({
            "furniture_id": case["target_asset"]["id"],
            "category": case["target_asset"]["category"],
            "position": {"x": cx, "y": 0.0, "z": cz},
            "rotation_y": yaw_deg % 360.0,
            "size": size,
            "region_id": region_id,
            "score": score,
            "status": "placed",
            "method": "constraint_solver",
        })

    return results or [_invalid_prediction(case, "constraint_solver: all regions failed")]


def _scene_text_for_gpt(case: Dict[str, Any]) -> str:
    """Render scene as a concise text description for GPT prompts."""
    scene = case["scene"]
    floor = scene["floor"]
    poly = floor["polygon"]
    bounds = floor["bounds"]  # (xmin, zmin, xmax, zmax)
    lines = [
        f"Room type: {scene.get('room_type', 'unknown')}",
        f"Floor bounds: x=[{bounds[0]:.2f}, {bounds[2]:.2f}]m, z=[{bounds[1]:.2f}, {bounds[3]:.2f}]m",
        f"Floor polygon (x,z vertices): {[(round(x,2), round(z,2)) for x,z in poly]}",
    ]
    if scene.get("doors"):
        for d in scene["doors"]:
            p = d["position"]
            lines.append(f"Door at ({p[0]:.2f}, {p[2]:.2f}), width={d.get('width', 0.9):.2f}m")
    if scene.get("windows"):
        for w in scene["windows"]:
            p = w["position"]
            lines.append(f"Window at ({p[0]:.2f}, {p[2]:.2f}), width={w.get('width', 1.0):.2f}m")
    lines.append("Existing furniture (category, center_x, center_z, width, depth, yaw_deg):")
    for obj in scene.get("objects", []):
        pos = obj["position"]
        sz = obj["size"]
        lines.append(
            f"  {obj['category']}: pos=({pos[0]:.2f},{pos[2]:.2f})"
            f" size=({sz[0]:.2f}x{sz[2]:.2f}m) yaw={obj.get('yaw', 0.0):.0f}°"
        )
    return "\n".join(lines)


def run_layoutgpt_direct(case: Dict[str, Any], openai_client: Any, model: str = "gpt-4o") -> List[Dict[str, Any]]:
    """M2: LayoutGPT-style — GPT outputs (x, z, rotation_y) directly."""
    import json as _json

    scene_text = _scene_text_for_gpt(case)
    target = case["target_asset"]
    intent_text = case["intent"]["text"]
    size = target["size"]
    floor = case["scene"]["floor"]
    bounds = floor["bounds"]

    system_prompt = (
        "You are a furniture placement assistant. Given a room layout and a user instruction, "
        "output the exact position and rotation for a single furniture item.\n"
        "Respond with ONLY a JSON object: {\"x\": <float>, \"z\": <float>, \"rotation_y\": <float>}\n"
        "- x and z are the item's CENTER position in meters within the room coordinate system shown.\n"
        "- rotation_y is yaw in degrees (0–360, CCW from +x axis viewed from above).\n"
        "- The item must be fully inside the floor polygon and must not overlap existing furniture."
    )
    user_prompt = (
        f"{scene_text}\n\n"
        f"Target item to place:\n"
        f"  category: {target['category']}\n"
        f"  size: {size['width']:.2f}m wide × {size['depth']:.2f}m deep × {size['height']:.2f}m tall\n\n"
        f"User instruction: {intent_text}\n\n"
        "Output JSON only."
    )

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = _json.loads(response.choices[0].message.content)
        x = float(raw["x"])
        z = float(raw["z"])
        rot = float(raw.get("rotation_y", raw.get("rotation", 0.0)))
    except Exception as exc:
        return [_invalid_prediction(case, f"layoutgpt_direct GPT error: {exc}")]

    return [{
        "furniture_id": target["id"],
        "category": target["category"],
        "position": {"x": x, "y": 0.0, "z": z},
        "rotation_y": rot % 360.0,
        "size": size,
        "status": "placed",
        "method": "layoutgpt_direct",
    }]


def run_spacefit_gpt_text(
    case: Dict[str, Any],
    openai_client: Any,
    model: str = "gpt-4o",
    topk: int = 5,
) -> List[Dict[str, Any]]:
    """M3: SpaceFit + GPT-Text — GPT picks a region ID from pre-computed candidates."""
    import json as _json

    scene = _scene_for_method(case)
    occ = create_occupancy_grid(scene)
    candidates = extract_candidates(occ, scene)
    if not candidates:
        return [_invalid_prediction(case, "no candidates extracted")]

    target = case["target_asset"]
    intent_text = case["intent"]["text"]
    size = target["size"]

    cand_desc = []
    for c in candidates:
        cand_desc.append({
            "id": int(c.id),
            "area_m2": round(float(c.area), 2),
            "centroid_x": round(float(c.centroid[0]), 2),
            "centroid_z": round(float(c.centroid[1]), 2),
            "max_rect_w": round(float(c.max_rect[2]), 2),
            "max_rect_d": round(float(c.max_rect[3]), 2),
            "near_wall": bool(c.near_wall),
            "near_door": bool(c.near_door),
            "near_window": bool(c.near_window),
        })

    system_prompt = (
        "You are a furniture placement assistant. Pre-computed free-space candidate regions are provided. "
        "Rank the TOP 5 best region_ids for placing the target item. Do NOT generate coordinates.\n"
        "For each region also specify:\n"
        "  face_toward: one of 'center', 'wall', 'door', 'window', or 'existing:<category>'\n"
        "  preferred_wall: true/false\n"
        "Respond with ONLY compact JSON. Do not include explanations or markdown:\n"
        "{\"candidates\": [{\"region_id\": <int>, \"face_toward\": <str>, \"preferred_wall\": <bool>}, ...]}"
    )

    scene_text = _scene_text_for_gpt(case)
    user_prompt = (
        f"{scene_text}\n\n"
        f"Target item: {target['category']} ({size['width']:.2f}m × {size['depth']:.2f}m)\n"
        f"User instruction: {intent_text}\n\n"
        f"Available free-space regions:\n{_json.dumps(cand_desc, indent=2)}\n\n"
        f"Rank the best {min(topk, len(cand_desc))} region_ids. Output JSON only."
    )

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = _json.loads(response.choices[0].message.content)
        ranked = raw.get("candidates") or raw.get("regions") or []
        if not ranked and "region_id" in raw:
            ranked = [raw]  # backwards-compat: single selection
    except Exception as exc:
        return [_invalid_prediction(case, f"spacefit_gpt_text GPT error: {exc}")]

    valid_ids = {int(c.id) for c in candidates}
    # Fallback: append largest regions if GPT gave fewer than topk
    fallback_ids = [int(c.id) for c in sorted(candidates, key=lambda c: -c.area)]

    results: List[Dict] = []
    seen_regions: set = set()
    furniture_item = _target_item(case)

    for entry in ranked:
        if len(results) >= topk:
            break
        try:
            rid = int(entry["region_id"])
        except Exception:
            continue
        if rid not in valid_ids or rid in seen_regions:
            continue
        seen_regions.add(rid)
        face_toward = str(entry.get("face_toward", "center"))
        preferred_wall = bool(entry.get("preferred_wall", False))

        constraints = [
            *([{"constraint_type": "against_wall", "subject_id": target["id"]}] if preferred_wall else []),
            {"constraint_type": "facing", "subject_id": target["id"],
             "target_kind": face_toward if face_toward in {"center", "wall", "door", "window"} else None,
             "target_category": face_toward.replace("existing:", "") if face_toward.startswith("existing:") else None},
        ]
        constraints = [c for c in constraints if c.get("target_kind") or c.get("target_category") or c.get("constraint_type") == "against_wall"]

        selections = {
            "placements": [{
                "furniture_id": furniture_item["id"],
                "selected_region": rid,
                "preferred_wall": preferred_wall,
                "face_toward": face_toward,
                "backup_region": None,
            }]
        }
        try:
            placements = refine_placements(scene, candidates, selections, [furniture_item])
        except Exception:
            continue
        if placements and placements[0].get("status") == "placed":
            placements[0]["method"] = "spacefit_gpt_text"
            placements[0]["region_id"] = rid
            results.append(placements[0])

    # Fill remaining slots with fallback regions
    for rid in fallback_ids:
        if len(results) >= topk:
            break
        if rid in seen_regions:
            continue
        seen_regions.add(rid)
        selections = {
            "placements": [{
                "furniture_id": furniture_item["id"],
                "selected_region": rid,
                "preferred_wall": False,
                "face_toward": "center",
                "backup_region": None,
            }]
        }
        try:
            placements = refine_placements(scene, candidates, selections, [furniture_item])
        except Exception:
            continue
        if placements and placements[0].get("status") == "placed":
            placements[0]["method"] = "spacefit_gpt_text"
            placements[0]["region_id"] = rid
            results.append(placements[0])

    if results:
        return results
    return [_invalid_prediction(case, "spacefit_gpt_text: all regions failed")]


def run_method(
    method: str,
    case: Dict[str, Any],
    grounder: Optional[IntentGrounder] = None,
    topk: int = 5,
    constraint_grid_step: float = 0.18,
    device: str = "cpu",
    diffopt_iters: int = 100,
    diffopt_lr: float = 1e-2,
    openai_client: Optional[Any] = None,
    openai_model: str = "gpt-4o",
) -> List[Dict[str, Any]]:
    if grounder is None:
        grounder = IntentGrounder()
    if method == "heuristic_baseline":
        grounded = {"constraints": []}
        return run_heuristic_baseline(case, grounded, topk=topk)
    if method == "proposal_heuristic":
        grounded = {"constraints": list(case["intent"]["constraints"])}
        return run_heuristic_refinement(case, grounded, topk=topk)
    if method == "proposal_diffopt_basic":
        grounded = {"constraints": []}
        return run_diffopt(case, grounded, topk=topk, device=device, n_iters=diffopt_iters, lr=diffopt_lr, use_constraints=False)
    if method == "proposal_diffopt_constraint":
        grounded = {"constraints": list(case["intent"]["constraints"])}
        return run_diffopt(case, grounded, topk=topk, device=device, n_iters=diffopt_iters, lr=diffopt_lr, use_constraints=True)
    if method == "proposal_llm_grounded_diffopt":
        grounded = grounder.ground(case["target_asset"], intent=case["intent"])
        return run_diffopt(case, grounded, topk=topk, device=device, n_iters=diffopt_iters, lr=diffopt_lr, use_constraints=True)
    if method == "constraint_solver":
        grounded = {"constraints": list(case["intent"]["constraints"])}
        return run_constraint_solver(case, grounded, topk=topk, grid_step=constraint_grid_step)
    # ── GPT API methods ───────────────────────────────────────────────────────
    if method == "layoutgpt_direct":
        if openai_client is None:
            raise ValueError("layoutgpt_direct requires openai_client")
        return run_layoutgpt_direct(case, openai_client, model=openai_model)
    if method == "spacefit_gpt_text":
        if openai_client is None:
            raise ValueError("spacefit_gpt_text requires openai_client")
        return run_spacefit_gpt_text(case, openai_client, model=openai_model, topk=topk)
    # ── Experiment B: no-proposal ablation ───────────────────────────────────
    if method == "no_proposal_diffopt_basic":
        grounded = {"constraints": []}
        return run_diffopt_no_proposal(case, grounded, topk=topk, device=device, n_iters=diffopt_iters, lr=diffopt_lr, use_constraints=False)
    if method == "no_proposal_diffopt_constraint":
        grounded = {"constraints": list(case["intent"]["constraints"])}
        return run_diffopt_no_proposal(case, grounded, topk=topk, device=device, n_iters=diffopt_iters, lr=diffopt_lr, use_constraints=True)
    raise ValueError(f"unsupported method '{method}'")
