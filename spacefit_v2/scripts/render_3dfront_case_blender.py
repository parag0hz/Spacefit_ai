"""Render SpaceFit single-target cases with 3D-FRONT/3D-FUTURE assets in Blender.

Run with Blender, for example:
blender -b --python spacefit_v2/scripts/render_3dfront_case_blender.py -- --case-id CASE_ID
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


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


def parse_args(argv: list[str]) -> argparse.Namespace:
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Case id without .json. Can be passed multiple times.",
    )
    parser.add_argument("--method", default="constraint_solver")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=int, default=1600)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--camera", choices=["perspective", "topdown"], default="perspective")
    return parser.parse_args(argv)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_room(raw_scene: dict[str, Any], room_id: str) -> dict[str, Any]:
    for room in raw_scene.get("scene", {}).get("room", []):
        if str(room.get("instanceid")) == room_id:
            return room
    raise ValueError(f"Room not found in raw scene: {room_id}")


def prediction_for_case(predictions: dict[str, Any], method: str, case_id: str) -> dict[str, Any]:
    method_predictions = predictions.get(method)
    if not isinstance(method_predictions, dict):
        raise ValueError(f"Method not found in predictions: {method}")
    entries = method_predictions.get(case_id) or []
    if not entries:
        raise ValueError(f"No predictions for case: {case_id}")
    placed = [entry for entry in entries if entry.get("status", "placed") == "placed"]
    return dict(placed[0] if placed else entries[0])


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


def future_model_path(jid: str) -> Path | None:
    for base in FUTURE_DIRS:
        candidate = base / jid / "normalized_model.obj"
        if candidate.exists():
            return candidate
    return None


def yaw_from_quat_y_up(rot: list[float] | tuple[float, ...] | None) -> float:
    if not rot or len(rot) < 4:
        return 0.0
    x, y, z, w = [float(v) for v in rot[:4]]
    # 3D-FRONT uses Y-up quaternions. This extracts yaw around Y.
    siny = 2.0 * (w * y + x * z)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny, cosy))


def clear_scene(bpy: Any) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_material(bpy: Any, name: str, color: tuple[float, float, float, float], roughness: float = 0.65) -> Any:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
    return mat


def create_floor_and_walls(bpy: Any, case: dict[str, Any]) -> tuple[float, float, float, float]:
    floor_mat = make_material(bpy, "warm_light_floor", (0.86, 0.82, 0.74, 1.0), 0.8)
    wall_mat = make_material(bpy, "soft_white_walls", (0.92, 0.92, 0.90, 1.0), 0.75)
    polygon = [(float(x), float(z)) for x, z in case["scene"]["floor"]["polygon"]]
    min_x = min(x for x, _ in polygon)
    max_x = max(x for x, _ in polygon)
    min_y = min(z for _, z in polygon)
    max_y = max(z for _, z in polygon)

    verts = [(x, z, 0.0) for x, z in polygon]
    mesh = bpy.data.meshes.new("room_floor_mesh")
    mesh.from_pydata(verts, [], [list(range(len(verts)))])
    mesh.update()
    floor_obj = bpy.data.objects.new("room_floor", mesh)
    bpy.context.collection.objects.link(floor_obj)
    floor_obj.data.materials.append(floor_mat)

    wall_height = 2.8
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if math.hypot(x2 - x1, y2 - y1) < 0.08:
            continue
        mesh = bpy.data.meshes.new(f"wall_{i:03d}_mesh")
        verts = [(x1, y1, 0.0), (x2, y2, 0.0), (x2, y2, wall_height), (x1, y1, wall_height)]
        mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
        mesh.update()
        obj = bpy.data.objects.new(f"wall_{i:03d}", mesh)
        bpy.context.collection.objects.link(obj)
        obj.data.materials.append(wall_mat)

    return min_x, max_x, min_y, max_y


def object_bounds(obj: Any) -> tuple[float, float, float]:
    xs = [v[0] for v in obj.bound_box]
    ys = [v[1] for v in obj.bound_box]
    zs = [v[2] for v in obj.bound_box]
    return max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)


def import_obj_asset(
    bpy: Any,
    obj_path: Path,
    name: str,
    position_xz: tuple[float, float],
    yaw_deg: float,
    size: tuple[float, float, float],
    target: bool = False,
) -> list[Any]:
    before = set(bpy.context.scene.objects)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(obj_path))
    else:
        bpy.ops.import_scene.obj(filepath=str(obj_path))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    if not imported:
        return []

    parent = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(parent)
    parent.location = (float(position_xz[0]), float(position_xz[1]), 0.0)
    parent.rotation_euler = (math.radians(90.0), 0.0, math.radians(-yaw_deg))

    width, height, depth = [max(float(v), 0.001) for v in size]
    for obj in imported:
        obj.name = f"{name}__{obj.name}"
        obj.parent = parent
        bx, by, bz = object_bounds(obj)
        # The normalized 3D-FUTURE OBJ is Y-up: x=width, y=height, z=depth.
        sx = width / bx if bx > 0 else 1.0
        sy = height / by if by > 0 else 1.0
        sz = depth / bz if bz > 0 else 1.0
        obj.scale = (sx, sy, sz)
        if target:
            obj.show_name = True
    return imported + [parent]


def add_target_marker(bpy: Any, position_xz: tuple[float, float], size: tuple[float, float, float], yaw_deg: float) -> None:
    marker_mat = make_material(bpy, "target_green_marker", (0.08, 0.65, 0.32, 1.0), 0.45)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(position_xz[0], position_xz[1], size[1] + 0.03))
    marker = bpy.context.object
    marker.name = "target_highlight_bbox"
    marker.dimensions = (size[0] + 0.10, size[2] + 0.10, 0.04)
    marker.rotation_euler[2] = math.radians(-yaw_deg)
    marker.data.materials.append(marker_mat)


def add_scene_label(bpy: Any, text: str, bounds: tuple[float, float, float, float]) -> None:
    min_x, max_x, min_y, max_y = bounds
    bpy.ops.object.text_add(location=(min_x, min_y - 0.55, 0.03), rotation=(math.radians(90), 0, 0))
    label = bpy.context.object
    label.name = "case_label"
    label.data.body = text
    label.data.align_x = "LEFT"
    label.data.size = 0.22
    label.data.extrude = 0.002


def setup_camera_and_lights(bpy: Any, bounds: tuple[float, float, float, float], mode: str) -> None:
    min_x, max_x, min_y, max_y = bounds
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    span = max(max_x - min_x, max_y - min_y)

    bpy.ops.object.light_add(type="AREA", location=(cx, cy, 5.8))
    light = bpy.context.object
    light.name = "large_softbox"
    light.data.energy = 550.0
    light.data.size = max(5.0, span * 0.9)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    if mode == "topdown":
        camera.location = (cx, cy, span * 1.25)
        camera.rotation_euler = (0.0, 0.0, 0.0)
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = span * 1.15
    else:
        camera.location = (cx - span * 0.55, cy - span * 0.75, span * 0.72)
        direction = (cx - camera.location.x, cy - camera.location.y, 0.7 - camera.location.z)
        camera.rotation_euler = direction_to_euler(direction)
        camera.data.lens = 30
    bpy.context.scene.camera = camera


def direction_to_euler(direction: tuple[float, float, float]) -> tuple[float, float, float]:
    import mathutils

    vec = mathutils.Vector(direction)
    return vec.to_track_quat("-Z", "Y").to_euler()


def set_render_options(bpy: Any, resolution: int, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = resolution
    scene.render.resolution_y = int(resolution * 0.72)
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.world.color = (1.0, 1.0, 1.0)


def render_case(bpy: Any, case_id: str, prediction: dict[str, Any], output_dir: Path, resolution: int, samples: int, camera: str) -> Path:
    case = load_json(CASE_DIR / f"{case_id}.json")
    raw_scene = load_json(FRONT_DIR / f"{case['scene']['scene_id']}.json")
    room = find_room(raw_scene, case["scene"]["room_id"])
    children_by_id = furniture_id_to_child(room)
    jid_by_uid = furniture_uid_to_jid(raw_scene)
    target_id = str(case["target_asset"]["id"])
    ids_to_render = {str(obj["id"]) for obj in case["scene"].get("objects", [])}
    ids_to_render.add(target_id)

    clear_scene(bpy)
    bounds = create_floor_and_walls(bpy, case)

    missing: list[str] = []
    for instance_id in sorted(ids_to_render):
        child = children_by_id.get(instance_id)
        if child is None:
            missing.append(instance_id)
            continue
        jid = child.get("replace_jid") or jid_by_uid.get(str(child.get("ref")))
        if not jid:
            missing.append(instance_id)
            continue
        model_path = future_model_path(str(jid))
        if model_path is None:
            missing.append(instance_id)
            continue

        is_target = instance_id == target_id
        if is_target:
            pos = prediction.get("position") or case["reference_pose"]["position"]
            position_xz = (float(pos["x"]), float(pos["z"]))
            yaw = float(prediction.get("rotation_y", case["reference_pose"]["rotation_y"]))
            size_dict = prediction.get("size") or case["target_asset"]["size"]
            size = (float(size_dict["width"]), float(size_dict["height"]), float(size_dict["depth"]))
            name = f"TARGET_{case['target_asset']['category']}"
        else:
            pos = child.get("pos") or [0.0, 0.0, 0.0]
            position_xz = (float(pos[0]), float(pos[2]))
            yaw = yaw_from_quat_y_up(child.get("rot"))
            case_obj = next((obj for obj in case["scene"].get("objects", []) if str(obj["id"]) == instance_id), None)
            if case_obj is not None:
                size_raw = case_obj["size"]
                size = (float(size_raw[0]), float(size_raw[1]), float(size_raw[2]))
                name = f"{case_obj['category']}_{instance_id.split('/')[-1]}"
            else:
                bbox = child.get("replace_bbox") or {}
                size = (
                    float(bbox.get("xLen", 100.0)) / 100.0,
                    float(bbox.get("zLen", 100.0)) / 100.0,
                    float(bbox.get("yLen", 100.0)) / 100.0,
                )
                name = instance_id.replace("/", "_")

        import_obj_asset(bpy, model_path, name, position_xz, yaw, size, target=is_target)
        if is_target:
            add_target_marker(bpy, position_xz, size, yaw)

    label = f"{case_id}\\n{case['target_asset']['category']} | {case.get('intent', {}).get('text', '')[:100]}"
    add_scene_label(bpy, label, bounds)
    setup_camera_and_lights(bpy, bounds, camera)
    set_render_options(bpy, resolution, samples)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{case_id}__3d_{camera}.png"
    bpy.context.scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)

    if missing:
        (output_dir / f"{case_id}__missing_assets.txt").write_text("\n".join(missing), encoding="utf-8")
    return out_path


def main() -> None:
    import bpy

    args = parse_args(sys.argv)
    case_ids = args.case_id or [
        "01a90e65-5653-4b48-88fa-4aa780db0621__livingroom_515__furniture_107",
        "0dd9e55c-dac2-4727-b8a1-f266fd11c987__livingdiningroom_12142__furniture_665",
    ]
    predictions = load_json(args.predictions)
    rendered = []
    for case_id in case_ids:
        pred = prediction_for_case(predictions, args.method, case_id)
        rendered.append(render_case(bpy, case_id, pred, args.output_dir, args.resolution, args.samples, args.camera))
    print("Rendered files:")
    for path in rendered:
        print(path)


if __name__ == "__main__":
    main()
