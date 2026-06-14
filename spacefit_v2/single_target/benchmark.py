"""Benchmark generation for single-target asset placement."""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from shapely.geometry import Point

from spacefit_v2.legacy.core import geometry as G

from .raw_3dfront import TARGET_CATEGORIES, iter_room_scenes


RELATION_TARGETS = {
    "nightstand": {"bed"},
    "dining_chair": {"table", "desk"},
    "coffee_table": {"sofa", "armchair"},
    "tv_stand": {"sofa", "armchair"},
    "armchair": {"coffee_table", "sofa"},
}


WALL_FAVORING = {"tv_stand", "wardrobe", "bookshelf", "nightstand", "desk", "bed", "sofa"}


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text or "").lower()).strip("_")


def _placement_poly(item: Dict[str, Any]):
    px, _, pz = item["position"]
    width, _, depth = item["size"]
    w = max(float(width), 0.05)
    d = max(float(depth), 0.05)
    return G.rotated_bbox_polygon(float(px), float(pz), w, d, float(item.get("yaw", 0.0)))


def _distance_to_walls(item: Dict[str, Any], scene: Dict[str, Any]) -> tuple[float, float]:
    px, _, pz = item["position"]
    return G.closest_wall_distance(float(px), float(pz), scene.get("walls", []))


def _orientation_toward(source_xy: tuple[float, float], target_xy: tuple[float, float]) -> float:
    return G.yaw_toward(source_xy, target_xy)


def _angle_diff_deg(a: float, b: float) -> float:
    return abs(((float(a) - float(b) + 540.0) % 360.0) - 180.0)


def _target_category(item: Dict[str, Any]) -> str:
    return _slug(item.get("category", ""))


def _room_scene_hash(scene: Dict[str, Any]) -> str:
    return f"{scene['scene_id']}__{_slug(scene['room_id'])}"


def _split_for_case(case_id: str) -> str:
    bucket = int(hashlib.md5(case_id.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def _choose_target(scene: Dict[str, Any], rng: random.Random) -> Optional[Dict[str, Any]]:
    eligible = [obj for obj in scene["objects"] if _target_category(obj) in TARGET_CATEGORIES]
    if not eligible:
        return None
    eligible = sorted(eligible, key=lambda item: (item["category"], item["id"]))
    return dict(rng.choice(eligible))


def _nearest_anchor(target: Dict[str, Any], existing: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    target_cat = _target_category(target)
    expected = RELATION_TARGETS.get(target_cat)
    if not expected:
        return None

    try:
        target_poly = _placement_poly(target)
    except Exception:
        return None
    best = None
    best_dist = float("inf")
    for obj in existing:
        if _target_category(obj) not in expected:
            continue
        try:
            dist = target_poly.distance(_placement_poly(obj))
        except Exception:
            continue
        if dist < best_dist:
            best_dist = dist
            best = obj
    if best is None or best_dist > 1.8:
        return None
    return dict(best)


def _infer_constraints(scene: Dict[str, Any], target: Dict[str, Any], existing: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    constraints: List[Dict[str, Any]] = []
    subject_id = str(target["id"])
    category = _target_category(target)

    if scene.get("doors"):
        constraints.append({"constraint_type": "not_block_door", "subject_id": subject_id})
    if scene.get("windows"):
        constraints.append({"constraint_type": "keep_window_clear", "subject_id": subject_id})
    constraints.append({"constraint_type": "access_zone", "subject_id": subject_id})

    wall_dist, wall_yaw = _distance_to_walls(target, scene)
    major_dim = max(float(target["size"][0]), float(target["size"][2]))
    if category in WALL_FAVORING or wall_dist <= max(0.18, 0.12 * major_dim):
        constraints.append({"constraint_type": "against_wall", "subject_id": subject_id})

    anchor = _nearest_anchor(target, existing)
    if anchor is not None:
        constraints.append(
            {
                "constraint_type": "near",
                "subject_id": subject_id,
                "target_id": str(anchor["id"]),
                "target_category": anchor["category"],
                "max_distance": 1.4,
            }
        )
        source_xy = (float(target["position"][0]), float(target["position"][2]))
        target_xy = (float(anchor["position"][0]), float(anchor["position"][2]))
        desired_yaw = _orientation_toward(source_xy, target_xy)
        if _angle_diff_deg(float(target.get("yaw", 0.0)), desired_yaw) <= 40.0:
            constraints.append(
                {
                    "constraint_type": "facing",
                    "subject_id": subject_id,
                    "target_id": str(anchor["id"]),
                    "target_category": anchor["category"],
                }
            )
    elif scene.get("windows"):
        nearest_window = min(
            scene["windows"],
            key=lambda entry: (float(entry["position"][0]) - float(target["position"][0])) ** 2
            + (float(entry["position"][2]) - float(target["position"][2])) ** 2,
        )
        source_xy = (float(target["position"][0]), float(target["position"][2]))
        target_xy = (float(nearest_window["position"][0]), float(nearest_window["position"][2]))
        desired_yaw = _orientation_toward(source_xy, target_xy)
        if _angle_diff_deg(float(target.get("yaw", 0.0)), desired_yaw) <= 35.0:
            constraints.append(
                {
                    "constraint_type": "facing",
                    "subject_id": subject_id,
                    "target_kind": "window",
                }
            )

    return constraints


def _intent_text(target: Dict[str, Any], constraints: Sequence[Dict[str, Any]]) -> str:
    category = _target_category(target).replace("_", " ")
    phrases = [f"place the {category}"]
    if any(c["constraint_type"] == "against_wall" for c in constraints):
        phrases.append("against the wall")
    near = next((c for c in constraints if c["constraint_type"] == "near"), None)
    if near is not None:
        phrases.append(f"near the {str(near.get('target_category', 'anchor')).replace('_', ' ')}")
    facing = next((c for c in constraints if c["constraint_type"] == "facing"), None)
    if facing is not None:
        target = facing.get("target_kind") or facing.get("target_category") or "anchor"
        phrases.append(f"facing the {str(target).replace('_', ' ')}")
    if any(c["constraint_type"] == "not_block_door" for c in constraints):
        phrases.append("without blocking the door")
    if any(c["constraint_type"] == "keep_window_clear" for c in constraints):
        phrases.append("while keeping the window clear")
    if any(c["constraint_type"] == "access_zone" for c in constraints):
        phrases.append("and preserving access around it")
    return ", ".join(phrases)


def build_case(scene: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    existing = [dict(obj) for obj in scene["objects"] if obj["id"] != target["id"]]
    constraints = _infer_constraints(scene, target, existing)
    case_id = f"{_room_scene_hash(scene)}__{_slug(target['id'])}"
    return {
        "id": case_id,
        "split": _split_for_case(case_id),
        "scene": {
            "scene_id": scene["scene_id"],
            "room_id": scene["room_id"],
            "room_type": scene["room_type"],
            "source_path": scene["source_path"],
            "floor": scene["floor"],
            "walls": scene["walls"],
            "doors": scene["doors"],
            "windows": scene["windows"],
            "openings": scene.get("openings", []),
            "objects": existing,
            "entrance_point": scene.get("entrance_point"),
        },
        "target_asset": {
            "id": str(target["id"]),
            "category": str(target["category"]),
            "size": {
                "width": float(target["size"][0]),
                "height": float(target["size"][1]),
                "depth": float(target["size"][2]),
            },
        },
        "reference_pose": {
            "position": {"x": float(target["position"][0]), "y": float(target["position"][1]), "z": float(target["position"][2])},
            "rotation_y": float(target.get("yaw", 0.0)),
        },
        "intent": {
            "type": "structured_template",
            "text": _intent_text(target, constraints),
            "constraints": constraints,
        },
    }


def generate_benchmark(
    data_dir: str | Path,
    out_dir: str | Path,
    seed: int = 0,
    max_scenes: Optional[int] = None,
    max_cases: Optional[int] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    out_path = Path(out_dir)
    cases_dir = out_path / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    cases: List[Dict[str, Any]] = []
    for scene in iter_room_scenes(Path(data_dir), max_scenes=max_scenes):
        target = _choose_target(scene, rng)
        if target is None:
            continue
        case = build_case(scene, target)
        cases.append(case)
        if max_cases is not None and len(cases) >= int(max_cases):
            break

    for case in cases:
        with open(cases_dir / f"{case['id']}.json", "w") as f:
            json.dump(case, f, indent=2)

    splits = {
        split: [case["id"] for case in cases if case["split"] == split]
        for split in ("train", "val", "test")
    }
    manifest = {
        "seed": seed,
        "data_dir": str(data_dir),
        "num_cases": len(cases),
        "cases": [{"id": case["id"], "split": case["split"], "file": f"cases/{case['id']}.json"} for case in cases],
        "splits": splits,
    }
    with open(out_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(out_path / "cases.json", "w") as f:
        json.dump(cases, f, indent=2)

    category_counts = Counter(_target_category({"category": case["target_asset"]["category"]}) for case in cases)
    summary = {
        "num_cases": len(cases),
        "splits": {key: len(value) for key, value in splits.items()},
        "target_categories": dict(sorted(category_counts.items())),
    }
    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Single-Target Benchmark Generation",
        "",
        f"- Source data: `{data_dir}`",
        f"- Random seed: `{seed}`",
        f"- Number of generated cases: `{len(cases)}`",
        "- Room types: bedroom, living_room, library (from raw 3D-FRONT room labels)",
        "- One target object is removed per room and treated as the target asset.",
        "- Structured constraints are inferred from the removed object's reference pose plus room openings.",
        "",
        "## Target Categories",
        "",
    ]
    for name, count in sorted(category_counts.items()):
        lines.append(f"- `{name}`: {count}")
    with open(out_path / "BENCHMARK_GENERATION_NOTE.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    return summary
