"""Run differentiable optimization on 3D-FRONT scenes."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.adapters.threedfront_adapter import load_3dfront_scenes
from spacefit_v2.legacy.core.free_space import extract_candidates
from spacefit_v2.legacy.core.occupancy import create_occupancy_grid
from spacefit_v2 import config
from spacefit_v2.device import resolve_torch_device
from spacefit_v2.model.geom import wall_segments_from_polygon
from spacefit_v2.optim.diff_refine import DifferentiableRefiner, _normalize_category, _size_dict
from spacefit_v2.pipeline_v2 import heuristic_select


def _polygon_bounds(polygon: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    xs = [float(v[0]) for v in polygon]
    zs = [float(v[1]) for v in polygon]
    return min(xs), min(zs), max(xs), max(zs)


def _walls_from_polygon(polygon: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
    walls = []
    for idx in range(len(polygon)):
        a = polygon[idx]
        b = polygon[(idx + 1) % len(polygon)]
        ax, az = float(a[0]), float(a[1])
        bx, bz = float(b[0]), float(b[1])
        cx = 0.5 * (ax + bx)
        cz = 0.5 * (az + bz)
        length = ((bx - ax) ** 2 + (bz - az) ** 2) ** 0.5
        import math
        yaw = math.degrees(math.atan2(bz - az, bx - ax))
        walls.append({
            "id": f"wall-{idx}",
            "length": length,
            "height": 2.7,
            "position": (cx, 1.35, cz),
            "yaw": yaw,
        })
    return walls


def _scene_for_occupancy(scene: Dict[str, Any], existing_furniture: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    polygon = scene["floor_plan_vertices"]
    bounds = _polygon_bounds(polygon)
    objects = []
    for item in existing_furniture:
        size = _size_dict(item)
        pos = item["position"]
        objects.append({
            "id": item["id"],
            "category": item["category"],
            "size": (size["width"], size["height"], size["depth"]),
            "position": (float(pos["x"]), 0.0, float(pos["z"])),
            "yaw": float(item.get("rotation_y", 0.0)),
        })
    return {
        "floor": {"polygon": polygon, "bounds": bounds, "area_m2": (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])},
        "walls": _walls_from_polygon(polygon),
        "doors": [],
        "windows": [],
        "objects": objects,
        "room_type": scene["room_type"],
    }


def _generation_inputs(scene: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    existing: List[Dict[str, Any]] = []
    new_items: List[Dict[str, Any]] = []
    for idx, furn in enumerate(scene["furniture"]):
        size = _size_dict(furn)
        new_items.append({
            "id": f"{furn['category']}-new-{idx}",
            "category": _normalize_category(furn["category"]),
            "size": size,
        })
    return existing, new_items


def _placements_to_prediction(scene: Dict[str, Any], placements: Sequence[Dict[str, Any]], elapsed_sec: float, num_candidates: int) -> Dict[str, Any]:
    object_list = []
    class_labels = []
    sizes = []
    translations = []
    angles = []

    for placement in placements:
        if placement.get("status") != "placed":
            continue
        size = placement["size"]
        pos = placement["position"]
        yaw = float(placement["rotation_y"])
        category = placement["category"]
        box = {
            "left": float(pos["x"]),
            "top": float(pos["z"]),
            "length": float(size["width"]),
            "width": float(size["depth"]),
            "orientation": yaw,
        }
        object_list.append((category, box))
        class_labels.append(category)
        sizes.append([size["width"] / 2.0, size.get("height", 0.5) / 2.0, size["depth"] / 2.0])
        translations.append([pos["x"], 0.0, pos["z"]])
        angles.append([yaw])

    return {
        "id": scene["id"],
        "query_id": scene["id"],
        "room_type": scene["room_type"],
        "floor_plan_vertices": scene["floor_plan_vertices"],
        "object_list": object_list,
        "class_labels": class_labels,
        "sizes": sizes,
        "translations": translations,
        "angles": angles,
        "time": elapsed_sec,
        "num_candidates": num_candidates,
        "num_placed": len(object_list),
        "num_unplaced": max(0, len(placements) - len(object_list)),
        "llm_usage": None,
    }


def run(args: argparse.Namespace) -> List[Dict[str, Any]]:
    device = resolve_torch_device(args.device)
    scenes = load_3dfront_scenes(
        args.data_dir,
        room_type=args.room,
        split=args.split,
        limit=args.max_scenes,
        rect_only=not args.all_test,
    )
    print(f"Running diff-opt on {device}")
    refiner = DifferentiableRefiner(scorer_path=args.scorer_path, device=device)
    outputs = []

    for idx, scene in enumerate(scenes, start=1):
        existing, new_furniture = _generation_inputs(scene)
        occ_scene = _scene_for_occupancy(scene, existing)
        occ = create_occupancy_grid(occ_scene)
        candidates = extract_candidates(occ, occ_scene)
        selections = heuristic_select(candidates, new_furniture)

        t0 = time.time()
        placements = refiner.refine(
            scene={
                "floor_plan_vertices": scene["floor_plan_vertices"],
                "walls": wall_segments_from_polygon(scene["floor_plan_vertices"]),
                "doors": [],
                "existing_furniture": existing,
            },
            new_furniture=new_furniture,
            candidate_regions=candidates,
            selected_regions=selections,
            n_iters=args.iters,
            lr=args.lr,
            n_restarts=args.restarts,
        )
        elapsed = time.time() - t0
        outputs.append(_placements_to_prediction(scene, placements, elapsed, len(candidates)))
        print(f"[{idx}/{len(scenes)}] {scene['id']} placed={outputs[-1]['num_placed']} time={elapsed:.2f}s")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"Saved {len(outputs)} predictions to {out_path}")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(config.DATA_DIR))
    parser.add_argument("--room", default="bedroom")
    parser.add_argument("--split", default="test")
    parser.add_argument("--all_test", action="store_true")
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--iters", type=int, default=config.DIFFOPT_ITERS)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--lr", type=float, default=config.DIFFOPT_LR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scorer_path", default=None)
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "3dfront_bedroom_diffopt.json"))
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
