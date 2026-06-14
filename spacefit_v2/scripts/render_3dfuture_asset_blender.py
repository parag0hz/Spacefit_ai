"""Render a single 3D-FUTURE asset with Blender.

Run with Blender, not regular Python:

    blender --background --python spacefit_v2/scripts/render_3dfuture_asset_blender.py -- \
      --asset_dir dataset/3D-FUTURE-model/0a0f0cf2-3a34-4ba2-b24f-34f361c36b3e \
      --out spacefit_v2/results/3d_qualitative_renders/asset_0a0f0cf2_blender.png
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--use_raw", action="store_true", help="Use raw_model.obj instead of normalized_model.obj.")
    parser.add_argument("--resolution", type=int, default=1400)
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_asset(asset_dir: Path, use_raw: bool) -> list[bpy.types.Object]:
    obj_path = asset_dir / ("raw_model.obj" if use_raw else "normalized_model.obj")
    if not obj_path.exists():
        raise FileNotFoundError(obj_path)

    bpy.ops.wm.obj_import(filepath=str(obj_path))
    objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError(f"No mesh imported from {obj_path}")
    return objects


def normalize_objects(objects: list[bpy.types.Object]) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]

    min_v = Vector((float("inf"), float("inf"), float("inf")))
    max_v = Vector((float("-inf"), float("-inf"), float("-inf")))
    for obj in objects:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            min_v.x = min(min_v.x, world.x)
            min_v.y = min(min_v.y, world.y)
            min_v.z = min(min_v.z, world.z)
            max_v.x = max(max_v.x, world.x)
            max_v.y = max(max_v.y, world.y)
            max_v.z = max(max_v.z, world.z)

    center = (min_v + max_v) * 0.5
    size = max(max_v.x - min_v.x, max_v.y - min_v.y, max_v.z - min_v.z)
    scale = 2.4 / max(size, 1e-6)
    for obj in objects:
        obj.location -= center
        obj.scale *= scale


def setup_world() -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.color = (1.0, 1.0, 1.0)

    bpy.ops.mesh.primitive_plane_add(size=5.5, location=(0, 0, -1.21))
    floor = bpy.context.object
    floor.name = "matte_floor"
    mat = bpy.data.materials.new("matte_floor")
    mat.diffuse_color = (0.86, 0.88, 0.90, 1)
    floor.data.materials.append(mat)

    bpy.ops.object.light_add(type="AREA", location=(0, -3.5, 4.5))
    key = bpy.context.object
    key.name = "large_softbox"
    key.data.energy = 450
    key.data.size = 4.0

    bpy.ops.object.light_add(type="POINT", location=(-3, 3, 3))
    fill = bpy.context.object
    fill.name = "fill_light"
    fill.data.energy = 80


def setup_camera() -> None:
    bpy.ops.object.camera_add(location=(3.3, -4.2, 2.7), rotation=(math.radians(61), 0, math.radians(39)))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    direction = Vector((0, 0, 0)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = 55


def setup_render(out_path: Path, resolution: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 96
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.render.film_transparent = False
    scene.render.filepath = str(out_path)


def main() -> None:
    args = parse_args()
    asset_dir = Path(args.asset_dir).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    clear_scene()
    objects = import_asset(asset_dir, args.use_raw)
    normalize_objects(objects)
    setup_world()
    setup_camera()
    setup_render(out, args.resolution)
    bpy.ops.wm.save_as_mainfile(filepath=str(out.with_suffix(".blend")))
    bpy.ops.render.render(write_still=True)
    print(out)


if __name__ == "__main__":
    main()
