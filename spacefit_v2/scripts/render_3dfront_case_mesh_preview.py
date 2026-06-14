"""Render 3D-FRONT/3D-FUTURE case previews without Blender.

This is a lightweight fallback renderer for machines without Blender. It uses
the real 3D-FUTURE OBJ meshes, simplified by face sampling, and saves PNG
preview figures with matplotlib.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "spacefit_v2" / "data" / "single_target_benchmark" / "cases"
FRONT_DIR = ROOT / "dataset" / "3D-FRONT"
FUTURE_DIRS = [
    ROOT / "dataset" / "3D-FUTURE-model",
    ROOT / "dataset" / "3D-FUTURE-model-part1",
]
DEFAULT_PREDICTIONS = (
    ROOT
    / "spacefit_v2"
    / "results"
    / "final_constraint_solver_human_rerank"
    / "test_gpt_intent"
    / "raw_predictions_human_reranked.json"
)
DEFAULT_OUTPUT = ROOT / "spacefit_v2" / "results" / "3d_qualitative_renders"

CATEGORY_COLORS = {
    "sofa": "#a9c5df",
    "coffee_table": "#d5b58a",
    "tv_stand": "#8fb98c",
    "armchair": "#b8add9",
    "dining_chair": "#c4a484",
    "dining_table": "#d2b48c",
    "cabinet": "#b6b6b6",
    "desk": "#9eb6cf",
    "bed": "#d5a6a6",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def future_model_path(jid: str) -> Path | None:
    for base in FUTURE_DIRS:
        path = base / jid / "normalized_model.obj"
        if path.exists():
            return path
    return None


def parse_obj(path: Path, max_faces: int = 900) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    vertices: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                raw = [int(part.split("/")[0]) - 1 for part in line.split()[1:]]
                if len(raw) >= 3:
                    for i in range(1, len(raw) - 1):
                        faces.append((raw[0], raw[i], raw[i + 1]))
    if len(faces) > max_faces:
        step = max(1, len(faces) // max_faces)
        faces = faces[::step][:max_faces]
    return np.asarray(vertices, dtype=float), faces


def yaw_from_quat_y_up(rot: list[float] | tuple[float, ...] | None) -> float:
    if not rot or len(rot) < 4:
        return 0.0
    x, y, z, w = [float(v) for v in rot[:4]]
    siny = 2.0 * (w * y + x * z)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny, cosy))


def find_room(raw_scene: dict[str, Any], room_id: str) -> dict[str, Any]:
    for room in raw_scene.get("scene", {}).get("room", []):
        if str(room.get("instanceid")) == room_id:
            return room
    raise ValueError(f"Room not found: {room_id}")


def furniture_id_to_child(room: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(child.get("instanceid")): child
        for child in room.get("children", [])
        if str(child.get("instanceid", "")).startswith("furniture/")
    }


def furniture_uid_to_jid(raw_scene: dict[str, Any]) -> dict[str, str]:
    return {
        str(item.get("uid")): str(item.get("jid"))
        for item in raw_scene.get("furniture", [])
        if item.get("uid") and item.get("jid")
    }


def prediction_for_case(predictions: dict[str, Any], method: str, case_id: str) -> dict[str, Any]:
    entries = (predictions.get(method) or {}).get(case_id) or []
    placed = [entry for entry in entries if entry.get("status", "placed") == "placed"]
    if not placed and not entries:
        raise ValueError(f"No prediction for {case_id}")
    return dict(placed[0] if placed else entries[0])


def transform_vertices(
    vertices: np.ndarray,
    position_xz: tuple[float, float],
    yaw_deg: float,
    size: tuple[float, float, float],
) -> np.ndarray:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center_x = (mins[0] + maxs[0]) / 2.0
    center_z = (mins[2] + maxs[2]) / 2.0
    width = max(maxs[0] - mins[0], 1e-6)
    height = max(maxs[1] - mins[1], 1e-6)
    depth = max(maxs[2] - mins[2], 1e-6)

    local_x = (vertices[:, 0] - center_x) * (size[0] / width)
    local_y = (vertices[:, 2] - center_z) * (size[2] / depth)
    local_z = (vertices[:, 1] - mins[1]) * (size[1] / height)

    yaw = math.radians(yaw_deg)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    world_x = position_xz[0] + cos_y * local_x - sin_y * local_y
    world_y = position_xz[1] + sin_y * local_x + cos_y * local_y
    return np.column_stack([world_x, world_y, local_z])


def add_mesh(
    ax: Any,
    obj_path: Path,
    name: str,
    category: str,
    position_xz: tuple[float, float],
    yaw_deg: float,
    size: tuple[float, float, float],
    target: bool,
) -> None:
    vertices, faces = parse_obj(obj_path, max_faces=1100 if target else 700)
    transformed = transform_vertices(vertices, position_xz, yaw_deg, size)
    tris = [transformed[list(face)] for face in faces]
    color = "#35a853" if target else CATEGORY_COLORS.get(category, "#c8c8c8")
    collection = Poly3DCollection(
        tris,
        facecolors=color,
        edgecolors="#374151" if target else "#555555",
        linewidths=0.10 if not target else 0.22,
        alpha=1.0 if target else 0.96,
        shade=True,
    )
    ax.add_collection3d(collection)

    if target:
        ax.text(
            position_xz[0],
            position_xz[1],
            size[1] + 0.18,
            f"TARGET\n{name}",
            color="#087f3f",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )


def add_floor_and_walls(ax: Any, polygon: list[list[float]]) -> tuple[float, float, float, float]:
    pts = [(float(x), float(y)) for x, y in polygon]
    min_x = min(x for x, _ in pts)
    max_x = max(x for x, _ in pts)
    min_y = min(y for _, y in pts)
    max_y = max(y for _, y in pts)
    floor = [[(x, y, 0.0) for x, y in pts]]
    ax.add_collection3d(
        Poly3DCollection(floor, facecolors="#f0ece2", edgecolors="#333333", linewidths=1.1, alpha=0.30)
    )
    wall_h = 2.5
    walls = []
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(i + 1) % len(pts)]
        if math.hypot(x2 - x1, y2 - y1) >= 0.12:
            walls.append([(x1, y1, 0.0), (x2, y2, 0.0), (x2, y2, wall_h), (x1, y1, wall_h)])
    ax.add_collection3d(
        Poly3DCollection(walls, facecolors="#f7f7f4", edgecolors="#c6c6c6", linewidths=0.35, alpha=0.08)
    )
    return min_x, max_x, min_y, max_y


def set_axes_equal(ax: Any, bounds: tuple[float, float, float, float]) -> None:
    min_x, max_x, min_y, max_y = bounds
    span = max(max_x - min_x, max_y - min_y, 3.0)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2, cy + span / 2)
    ax.set_zlim(0, span * 0.45)
    ax.set_box_aspect((1, 1, 0.45))


def render_case(case_id: str, predictions: dict[str, Any], method: str, output_dir: Path) -> Path:
    case = load_json(CASE_DIR / f"{case_id}.json")
    raw_scene = load_json(FRONT_DIR / f"{case['scene']['scene_id']}.json")
    room = find_room(raw_scene, case["scene"]["room_id"])
    child_by_id = furniture_id_to_child(room)
    jid_by_uid = furniture_uid_to_jid(raw_scene)
    pred = prediction_for_case(predictions, method, case_id)
    target_id = str(case["target_asset"]["id"])
    existing_ids = {str(obj["id"]) for obj in case["scene"].get("objects", [])}
    ids_to_render = sorted(existing_ids | {target_id})

    fig = plt.figure(figsize=(14, 9), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    bounds = add_floor_and_walls(ax, case["scene"]["floor"]["polygon"])

    for instance_id in ids_to_render:
        child = child_by_id.get(instance_id)
        if child is None:
            continue
        jid = child.get("replace_jid") or jid_by_uid.get(str(child.get("ref")))
        obj_path = future_model_path(str(jid)) if jid else None
        if obj_path is None:
            continue

        is_target = instance_id == target_id
        if is_target:
            pos = pred["position"]
            position_xz = (float(pos["x"]), float(pos["z"]))
            yaw = float(pred.get("rotation_y", case["reference_pose"]["rotation_y"]))
            size_dict = pred.get("size") or case["target_asset"]["size"]
            size = (float(size_dict["width"]), float(size_dict["height"]), float(size_dict["depth"]))
            category = str(case["target_asset"]["category"])
            name = category.replace("_", " ")
        else:
            case_obj = next(obj for obj in case["scene"]["objects"] if str(obj["id"]) == instance_id)
            pos = child.get("pos") or case_obj["position"]
            position_xz = (float(pos[0]), float(pos[2]))
            yaw = yaw_from_quat_y_up(child.get("rot"))
            s = case_obj["size"]
            size = (float(s[0]), float(s[1]), float(s[2]))
            category = str(case_obj["category"])
            name = category.replace("_", " ")

        add_mesh(ax, obj_path, name, category, position_xz, yaw, size, is_target)

    set_axes_equal(ax, bounds)
    ax.view_init(elev=34, azim=-58)
    ax.set_axis_off()
    title = f"3D mesh preview: {case['target_asset']['category']} placement"
    subtitle = case["intent"]["text"][:130]
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, pad=12, loc="left")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{case_id}__mesh_preview.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--method", default="constraint_solver")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    case_ids = args.case_id or [
        "01a90e65-5653-4b48-88fa-4aa780db0621__livingroom_515__furniture_107",
        "0dd9e55c-dac2-4727-b8a1-f266fd11c987__livingdiningroom_12142__furniture_665",
    ]
    predictions = load_json(args.predictions)
    for case_id in case_ids:
        print(render_case(case_id, predictions, args.method, args.output_dir))


if __name__ == "__main__":
    main()
