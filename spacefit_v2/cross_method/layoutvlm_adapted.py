"""Adapt LayoutVLM's optimization backend to the single-target benchmark.

This module intentionally bypasses LayoutVLM's OpenAI-based prompt-generation
front-end and reuses only the local differentiable constraint optimizer.
The resulting baseline must therefore be reported as an adapted backend
comparison, not as a fair direct rerun of the original LayoutVLM system.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import random
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Point, Polygon


ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = ROOT.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from dd0nw.LayoutVLM.src.layoutvlm.constraints import Constraint, against_wall, distance_constraint, point_towards
from dd0nw.LayoutVLM.src.layoutvlm.grad_solver import GradSolver
from dd0nw.LayoutVLM.src.layoutvlm.scene import AssetInstance, Wall


SUPPORTED_CTYPES = {"against_wall", "near", "facing"}

warnings.filterwarnings(
    "ignore",
    message="Error using oriented_iou_loss: vertices must be a CUDA tensor. Using simplified fallback.",
)


@dataclass
class AdaptedRun:
    prediction: Dict[str, Any]
    objective: float
    supported_constraints: List[str]
    unsupported_constraints: List[str]
    wall_id: Optional[str]


def load_benchmark_cases(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def _solver_size(size: Mapping[str, Any]) -> List[float]:
    return [
        float(size["width"]),
        float(size["depth"]),
        float(size["height"]),
    ]


def _solver_size_from_scene_object(obj: Mapping[str, Any]) -> List[float]:
    return [
        float(obj["size"][0]),
        float(obj["size"][2]),
        float(obj["size"][1]),
    ]


def _solver_position_from_scene_object(obj: Mapping[str, Any]) -> List[float]:
    size = _solver_size_from_scene_object(obj)
    return [float(obj["position"][0]), float(obj["position"][2]), size[2] / 2.0]


def _floor_polygon(case: Mapping[str, Any]) -> List[Tuple[float, float]]:
    return [(float(x), float(z)) for x, z in case["scene"]["floor"]["polygon"]]


def _room_polygon(case: Mapping[str, Any]) -> Polygon:
    return Polygon(_floor_polygon(case))


def _room_centroid(case: Mapping[str, Any]) -> Tuple[float, float]:
    centroid = _room_polygon(case).centroid
    return float(centroid.x), float(centroid.y)


def _wall_assets(case: Mapping[str, Any]) -> Dict[str, Wall]:
    polygon = _floor_polygon(case)
    walls: Dict[str, Wall] = {}
    for idx, start in enumerate(polygon):
        end = polygon[(idx + 1) % len(polygon)]
        walls[f"walls_{idx}"] = Wall(
            wall_id=f"walls_{idx}",
            vertices=[[float(start[0]), float(start[1]), 0.0], [float(end[0]), float(end[1]), 0.0]],
        )
    return walls


def _project_inside(polygon: Polygon, point: Tuple[float, float], margin: float = 0.03) -> Tuple[float, float]:
    p = Point(point)
    if polygon.buffer(-margin).contains(p):
        return float(point[0]), float(point[1])
    boundary_point = polygon.exterior.interpolate(polygon.exterior.project(p))
    cx, cz = polygon.centroid.x, polygon.centroid.y
    dx = cx - boundary_point.x
    dz = cz - boundary_point.y
    norm = math.hypot(dx, dz)
    if norm < 1e-6:
        return float(boundary_point.x), float(boundary_point.y)
    scale = margin / norm
    return float(boundary_point.x + dx * scale), float(boundary_point.y + dz * scale)


def _normalize_angle_deg(angle: float) -> float:
    value = float(angle)
    while value <= -180.0:
        value += 360.0
    while value > 180.0:
        value -= 360.0
    return value


def _pick_anchor_object(case: Mapping[str, Any], constraint: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    target_id = constraint.get("target_id")
    if target_id is not None:
        for obj in case["scene"].get("objects", []):
            if str(obj["id"]) == str(target_id):
                return obj

    target_category = str(constraint.get("target_category", "") or "").lower()
    if target_category:
        for obj in case["scene"].get("objects", []):
            if str(obj.get("category", "")).lower() == target_category:
                return obj
    return None


def _resolve_target_point(case: Mapping[str, Any], constraint: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    obj = _pick_anchor_object(case, constraint)
    if obj is not None:
        return float(obj["position"][0]), float(obj["position"][2])

    target_kind = str(constraint.get("target_kind", "") or "")
    if target_kind == "window" and case["scene"].get("windows"):
        win = case["scene"]["windows"][0]
        return float(win["position"][0]), float(win["position"][2])
    if target_kind == "door" and case["scene"].get("doors"):
        door = case["scene"]["doors"][0]
        return float(door["position"][0]), float(door["position"][2])
    if target_kind == "center":
        return _room_centroid(case)
    return None


def _seed_near_anchor(
    case: Mapping[str, Any],
    target_size: Sequence[float],
    anchor: Tuple[float, float],
    restart_idx: int,
    num_restarts: int,
    max_distance: Optional[float],
) -> Tuple[Tuple[float, float], float]:
    polygon = _room_polygon(case)
    base_angle = 2.0 * math.pi * (restart_idx / max(num_restarts, 1))
    min_clearance = 0.55 * max(float(target_size[0]), float(target_size[1]))
    radius = float(max_distance) * 0.75 if max_distance is not None else 1.1
    radius = max(radius, min_clearance + 0.15)
    candidate = (
        float(anchor[0] + radius * math.cos(base_angle)),
        float(anchor[1] + radius * math.sin(base_angle)),
    )
    projected = _project_inside(polygon, candidate)
    yaw = _normalize_angle_deg(math.degrees(math.atan2(anchor[1] - projected[1], anchor[0] - projected[0])))
    return projected, yaw


def _seed_against_wall(
    case: Mapping[str, Any],
    target_size: Sequence[float],
    restart_idx: int,
) -> Tuple[Tuple[float, float], float, str]:
    polygon_pts = _floor_polygon(case)
    polygon = _room_polygon(case)
    centroid = _room_centroid(case)
    wall_idx = restart_idx % len(polygon_pts)
    start = polygon_pts[wall_idx]
    end = polygon_pts[(wall_idx + 1) % len(polygon_pts)]
    wall_vec = (float(end[0] - start[0]), float(end[1] - start[1]))
    wall_len = math.hypot(wall_vec[0], wall_vec[1]) or 1.0
    tangent = (wall_vec[0] / wall_len, wall_vec[1] / wall_len)
    midpoint = ((float(start[0]) + float(end[0])) * 0.5, (float(start[1]) + float(end[1])) * 0.5)
    normals = [(-tangent[1], tangent[0]), (tangent[1], -tangent[0])]
    inward = max(normals, key=lambda normal: (centroid[0] - midpoint[0]) * normal[0] + (centroid[1] - midpoint[1]) * normal[1])
    offset = max(0.15, 0.5 * min(float(target_size[0]), float(target_size[1])) + 0.08)
    candidate = (midpoint[0] + inward[0] * offset, midpoint[1] + inward[1] * offset)
    projected = _project_inside(polygon, candidate)
    yaw = _normalize_angle_deg(math.degrees(math.atan2(tangent[1], tangent[0])) + 90.0)
    return projected, yaw, f"walls_{wall_idx}"


def _seed_default(case: Mapping[str, Any], restart_idx: int, num_restarts: int) -> Tuple[Tuple[float, float], float]:
    polygon = _room_polygon(case)
    centroid = _room_centroid(case)
    radius = 0.45 + 0.12 * restart_idx
    angle = 2.0 * math.pi * (restart_idx / max(num_restarts, 1))
    candidate = (centroid[0] + radius * math.cos(angle), centroid[1] + radius * math.sin(angle))
    projected = _project_inside(polygon, candidate)
    return projected, _normalize_angle_deg(math.degrees(angle))


def _initial_pose(case: Mapping[str, Any], restart_idx: int, num_restarts: int) -> Tuple[Tuple[float, float], float, Optional[str]]:
    constraints = list(case["intent"].get("constraints", []))
    target_size = _solver_size(case["target_asset"]["size"])
    wall_constraint = next((c for c in constraints if c.get("constraint_type") == "against_wall"), None)
    near_constraint = next((c for c in constraints if c.get("constraint_type") == "near"), None)
    facing_constraint = next((c for c in constraints if c.get("constraint_type") == "facing"), None)

    wall_id = None
    if wall_constraint is not None:
        position, yaw, wall_id = _seed_against_wall(case, target_size, restart_idx)
    elif near_constraint is not None:
        anchor = _resolve_target_point(case, near_constraint)
        if anchor is not None:
            position, yaw = _seed_near_anchor(
                case,
                target_size,
                anchor=anchor,
                restart_idx=restart_idx,
                num_restarts=num_restarts,
                max_distance=near_constraint.get("max_distance"),
            )
        else:
            position, yaw = _seed_default(case, restart_idx, num_restarts)
    else:
        position, yaw = _seed_default(case, restart_idx, num_restarts)

    if facing_constraint is not None:
        face_point = _resolve_target_point(case, facing_constraint)
        if face_point is not None:
            yaw = _normalize_angle_deg(math.degrees(math.atan2(face_point[1] - position[1], face_point[0] - position[0])))
    return position, yaw, wall_id


def _build_assets(
    case: Mapping[str, Any],
    init_position: Tuple[float, float],
    init_yaw: float,
) -> Tuple[Dict[str, Any], str, Dict[str, str]]:
    assets: Dict[str, Any] = {}
    key_by_scene_id: Dict[str, str] = {}

    for idx, obj in enumerate(case["scene"].get("objects", [])):
        asset_key = f"existing_{idx}"
        key_by_scene_id[str(obj["id"])] = asset_key
        assets[asset_key] = AssetInstance(
            id=asset_key,
            position=_solver_position_from_scene_object(obj),
            rotation=[0.0, 0.0, math.radians(float(obj.get("yaw", 0.0)))],
            size=_solver_size_from_scene_object(obj),
            onCeiling=False,
            optimize=0,
        )

    target_size = _solver_size(case["target_asset"]["size"])
    target_key = "target_0"
    assets[target_key] = AssetInstance(
        id=target_key,
        position=[float(init_position[0]), float(init_position[1]), float(target_size[2] / 2.0)],
        rotation=[0.0, 0.0, math.radians(init_yaw)],
        size=target_size,
        onCeiling=False,
        optimize=1,
    )
    return assets, target_key, key_by_scene_id


def _wall_distance(point: Tuple[float, float], wall: Wall) -> float:
    start = np.array([float(wall.corner1[0]), float(wall.corner1[1])], dtype=np.float32)
    end = np.array([float(wall.corner2[0]), float(wall.corner2[1])], dtype=np.float32)
    target = np.array([float(point[0]), float(point[1])], dtype=np.float32)
    seg = end - start
    denom = float(np.dot(seg, seg))
    if denom < 1e-8:
        return float(np.linalg.norm(target - start))
    t = float(np.clip(np.dot(target - start, seg) / denom, 0.0, 1.0))
    closest = start + t * seg
    return float(np.linalg.norm(target - closest))


def _nearest_wall_id(point: Tuple[float, float], wall_assets: Mapping[str, Wall]) -> Optional[str]:
    if not wall_assets:
        return None
    return min(wall_assets.keys(), key=lambda wall_id: _wall_distance(point, wall_assets[wall_id]))


def _build_constraints(
    case: Mapping[str, Any],
    assets: Dict[str, Any],
    target_key: str,
    key_by_scene_id: Mapping[str, str],
    wall_assets: Mapping[str, Wall],
    init_position: Tuple[float, float],
    preferred_wall_id: Optional[str],
) -> Tuple[List[Tuple[Constraint, List[str]]], List[str], List[str]]:
    constraints: List[Tuple[Constraint, List[str]]] = []
    supported: List[str] = []
    unsupported: List[str] = []
    fixed_point_idx = 0

    for raw in case["intent"].get("constraints", []):
        ctype = str(raw.get("constraint_type"))
        if ctype not in SUPPORTED_CTYPES:
            unsupported.append(ctype)
            continue

        if ctype == "against_wall":
            wall_id = preferred_wall_id or _nearest_wall_id(init_position, wall_assets)
            if wall_id is None:
                unsupported.append(ctype)
                continue
            constraints.append((Constraint("against_wall", against_wall), [target_key, wall_id]))
            supported.append(ctype)
            continue

        if ctype == "near":
            target_obj = _pick_anchor_object(case, raw)
            if target_obj is None:
                unsupported.append(ctype)
                continue
            anchor_key = key_by_scene_id.get(str(target_obj["id"]))
            if anchor_key is None:
                unsupported.append(ctype)
                continue
            min_distance = raw.get("min_distance")
            max_distance = raw.get("max_distance")
            constraints.append(
                (
                    Constraint(
                        "distance_constraint",
                        distance_constraint,
                        min_distance=float(min_distance) if min_distance is not None else 0.0,
                        max_distance=float(max_distance) if max_distance is not None else 10000.0,
                        weight=1.0,
                    ),
                    [target_key, anchor_key],
                )
            )
            supported.append(ctype)
            continue

        if ctype == "facing":
            target_obj = _pick_anchor_object(case, raw)
            if target_obj is not None:
                anchor_key = key_by_scene_id.get(str(target_obj["id"]))
                if anchor_key is None:
                    unsupported.append(ctype)
                    continue
                constraints.append((Constraint("point_towards", point_towards, angle=0.0), [target_key, anchor_key]))
                supported.append(ctype)
                continue

            point = _resolve_target_point(case, raw)
            if point is None:
                unsupported.append(ctype)
                continue
            fixed_key = f"fixed_point_{fixed_point_idx}"
            fixed_point_idx += 1
            assets[fixed_key] = AssetInstance(
                id=fixed_key,
                position=[float(point[0]), float(point[1]), 0.0],
                rotation=[0.0, 0.0, 0.0],
                size=[0.01, 0.01, 0.01],
                onCeiling=False,
                optimize=0,
            )
            constraints.append((Constraint("point_towards", point_towards, angle=0.0), [target_key, fixed_key]))
            supported.append(ctype)
            continue

    return constraints, supported, unsupported


def _prediction_from_result(
    case: Mapping[str, Any],
    result: Mapping[str, Any],
    objective: float,
    supported: Sequence[str],
    unsupported: Sequence[str],
    restart_idx: int,
) -> Dict[str, Any]:
    position = result["position"]
    yaw_deg = _normalize_angle_deg(math.degrees(float(result["rotation"][-1])))
    return {
        "furniture_id": str(case["target_asset"]["id"]),
        "category": str(case["target_asset"]["category"]),
        "position": {
            "x": float(position[0]),
            "y": 0.0,
            "z": float(position[1]),
        },
        "rotation_y": float(yaw_deg),
        "size": dict(case["target_asset"]["size"]),
        "region_id": int(restart_idx),
        "score": float(-objective),
        "status": "placed",
        "metadata": {
            "objective": float(objective),
            "supported_constraints": list(supported),
            "unsupported_constraints": list(unsupported),
            "runner": "layoutvlm_adapted_backend",
        },
    }


def _dedupe_predictions(predictions: Sequence[Dict[str, Any]], max_keep: int = 5) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for pred in predictions:
        if pred.get("status") != "placed":
            continue
        px = float(pred["position"]["x"])
        pz = float(pred["position"]["z"])
        pyaw = float(pred.get("rotation_y", 0.0))
        duplicate = False
        for existing in kept:
            ex = float(existing["position"]["x"])
            ez = float(existing["position"]["z"])
            eyaw = float(existing.get("rotation_y", 0.0))
            if math.hypot(px - ex, pz - ez) < 0.12 and abs(_normalize_angle_deg(pyaw - eyaw)) < 12.0:
                duplicate = True
                break
        if not duplicate:
            kept.append(pred)
        if len(kept) >= max_keep:
            break
    return kept


def run_layoutvlm_adapted_case(
    case: Mapping[str, Any],
    num_restarts: int = 6,
    iterations: int = 220,
    learning_rate: float = 0.03,
    seed: int = 7,
) -> List[Dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    runs: List[AdaptedRun] = []

    for restart_idx in range(num_restarts):
        try:
            init_position, init_yaw, preferred_wall_id = _initial_pose(case, restart_idx=restart_idx, num_restarts=num_restarts)
            assets, target_key, key_by_scene_id = _build_assets(case, init_position=init_position, init_yaw=init_yaw)
            wall_assets = _wall_assets(case)
            assets.update(wall_assets)
            constraints, supported, unsupported = _build_constraints(
                case,
                assets=assets,
                target_key=target_key,
                key_by_scene_id=key_by_scene_id,
                wall_assets=wall_assets,
                init_position=init_position,
                preferred_wall_id=preferred_wall_id,
            )

            solver = GradSolver(_floor_polygon(case))
            with tempfile.TemporaryDirectory(prefix="layoutvlm_adapted_") as temp_dir:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    results = solver.optimize(
                        assets=assets,
                        existing_constraints=[],
                        new_constraints=constraints,
                        iterations=iterations,
                        learning_rate=learning_rate,
                        temp_dir=temp_dir,
                        output_gif_path=None,
                    )
            if target_key not in results:
                continue
            overlap_loss, existing_constraint_loss, new_constraint_loss = solver.calc_loss([], constraints)
            objective = float(
                (overlap_loss.item() if hasattr(overlap_loss, "item") else overlap_loss)
                + (existing_constraint_loss.item() if hasattr(existing_constraint_loss, "item") else existing_constraint_loss)
                + (new_constraint_loss.item() if hasattr(new_constraint_loss, "item") else new_constraint_loss)
            )
            prediction = _prediction_from_result(
                case,
                result=results[target_key],
                objective=objective,
                supported=supported,
                unsupported=unsupported,
                restart_idx=restart_idx,
            )
            runs.append(
                AdaptedRun(
                    prediction=prediction,
                    objective=objective,
                    supported_constraints=supported,
                    unsupported_constraints=unsupported,
                    wall_id=preferred_wall_id,
                )
            )
        except Exception:
            continue

    ordered = sorted(runs, key=lambda item: (item.objective, -len(item.supported_constraints), len(item.unsupported_constraints)))
    predictions = _dedupe_predictions([item.prediction for item in ordered], max_keep=5)
    if predictions:
        return predictions
    return [
        {
            "furniture_id": str(case["target_asset"]["id"]),
            "category": str(case["target_asset"]["category"]),
            "status": "unplaced",
            "reason": "layoutvlm_adapted_backend_failed",
            "size": dict(case["target_asset"]["size"]),
        }
    ]


def run_layoutvlm_adapted_benchmark(
    cases: Sequence[Mapping[str, Any]],
    num_restarts: int = 6,
    iterations: int = 220,
    learning_rate: float = 0.03,
    seed: int = 7,
) -> Dict[str, List[Dict[str, Any]]]:
    predictions: Dict[str, List[Dict[str, Any]]] = {}
    for idx, case in enumerate(cases):
        case_seed = seed + idx * 97
        predictions[str(case["id"])] = run_layoutvlm_adapted_case(
            case,
            num_restarts=num_restarts,
            iterations=iterations,
            learning_rate=learning_rate,
            seed=case_seed,
        )
    return predictions
