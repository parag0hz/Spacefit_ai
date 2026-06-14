"""Parse RoomPlan Room.json into a structured scene dict.

Output scene dict schema:
{
  "source_path": str,
  "floor": {
      "polygon": [(x, z), ...],          # floor-local, 2D horizontal, offset so min=0
      "bounds": (xmin, zmin, xmax, zmax),
      "offset": (x_off, z_off),           # subtracted from polygon corners
      "R_inv": np.ndarray (3,3),          # world→floor-local rotation
      "t": np.ndarray (3,),               # floor world position
      "yaw_deg": float,                   # floor rotation in world frame
      "area_m2": float
  },
  "walls": [{"id","length","height","yaw","position": (x,y,z)}],
  "doors":   [{"id","width","height","position": (x,y,z), "yaw", "wall_id"}],
  "windows": [{"id","width","height","position": (x,y,z), "yaw", "wall_id"}],
  "objects": [{"id","category","subtype","size": (w,h,d),
               "position": (x,y,z), "yaw", "attributes"}],
  "room_type": str | None
}

Coordinate convention (floor-local):
  x  — horizontal axis 1 (matches polygonCorners[i][0])
  z  — horizontal axis 2 (matches polygonCorners[i][1])
  y  — vertical (height above floor); up is +y

All offsets make polygon start at (0,0) so grid indexing is trivial.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .geometry import (
    floor_world_to_local,
    parse_transform_4x4,
    polygon_area,
    polygon_bounds,
    world_to_local_pose,
)


def _first_key(d: Dict[str, Any]) -> str:
    return next(iter(d.keys())) if d else "unknown"


def _extract_category_and_subtype(obj: Dict[str, Any]) -> Tuple[str, str]:
    cat_field = obj.get("category", {}) or {}
    if isinstance(cat_field, dict):
        cat = _first_key(cat_field)
    else:
        cat = str(cat_field)
    attrs = obj.get("attributes", {}) or {}
    subtype = ""
    if cat == "chair":
        subtype = attrs.get("ChairType", "")
    elif cat == "table":
        subtype = attrs.get("TableType", "") or attrs.get("TableShapeType", "")
    elif cat == "storage":
        subtype = attrs.get("StorageType", "")
    elif cat == "sofa":
        subtype = attrs.get("SofaType", "")
    return cat, subtype


def _parse_floor(floor_dict: Dict[str, Any]):
    R_inv, t, floor_yaw_deg = floor_world_to_local(floor_dict["transform"])

    raw_corners = np.asarray(floor_dict.get("polygonCorners", []), dtype=np.float64)
    if raw_corners.size == 0:
        raise ValueError("floor has no polygonCorners")
    if raw_corners.shape[1] == 3:
        pts2d = raw_corners[:, :2]
    else:
        pts2d = raw_corners
    # polygonCorners[:, 0] = polygon X (local-X, horizontal axis 1)
    # polygonCorners[:, 1] = polygon Y (local-Y, horizontal axis 2) — we call this z
    x_min, z_min = float(pts2d[:, 0].min()), float(pts2d[:, 1].min())
    polygon = [(float(p[0] - x_min), float(p[1] - z_min)) for p in pts2d]
    bounds = polygon_bounds(polygon)
    return {
        "polygon": polygon,
        "bounds": bounds,
        "offset": (x_min, z_min),
        "R_inv": R_inv,
        "t": t,
        "yaw_deg": floor_yaw_deg,
        "area_m2": polygon_area(polygon),
    }


def _transform_world_point_to_local_offset(transform_list, R_inv, t_floor,
                                           floor_yaw_deg, offset):
    """World transform → local (x, y_height, z, yaw_deg), offset applied to horizontal axes."""
    x, y, z, yaw = world_to_local_pose(transform_list, R_inv, t_floor, floor_yaw_deg)
    ox, oz = offset
    return (x - ox, y, z - oz, yaw)


def _parse_objects(objects, floor):
    out = []
    R_inv, t_floor, floor_yaw = floor["R_inv"], floor["t"], floor["yaw_deg"]
    for i, obj in enumerate(objects or []):
        dims = obj.get("dimensions") or [0.5, 0.5, 0.5]
        category, subtype = _extract_category_and_subtype(obj)
        px, py, pz, yaw = _transform_world_point_to_local_offset(
            obj["transform"], R_inv, t_floor, floor_yaw, floor["offset"]
        )
        out.append({
            "id": obj.get("identifier") or f"obj-{i}",
            "category": category,
            "subtype": subtype,
            "size": (float(dims[0]), float(dims[1]), float(dims[2])),
            "position": (px, py, pz),
            "yaw": yaw,
            "attributes": obj.get("attributes") or {},
            "confidence": _first_key(obj.get("confidence", {}) or {}),
        })
    return out


def _parse_walls(walls, floor):
    out = []
    R_inv, t_floor, floor_yaw = floor["R_inv"], floor["t"], floor["yaw_deg"]
    for i, w in enumerate(walls or []):
        dims = w.get("dimensions") or [0.0, 0.0, 0.0]
        px, py, pz, yaw = _transform_world_point_to_local_offset(
            w["transform"], R_inv, t_floor, floor_yaw, floor["offset"]
        )
        out.append({
            "id": w.get("identifier") or f"wall-{i}",
            "length": float(dims[0]),
            "height": float(dims[1]),
            "position": (px, py, pz),
            "yaw": yaw,
        })
    return out


def _parse_openings(items, floor, kind):
    out = []
    R_inv, t_floor, floor_yaw = floor["R_inv"], floor["t"], floor["yaw_deg"]
    for i, d in enumerate(items or []):
        dims = d.get("dimensions") or [0.8, 2.0, 0.0]
        px, py, pz, yaw = _transform_world_point_to_local_offset(
            d["transform"], R_inv, t_floor, floor_yaw, floor["offset"]
        )
        out.append({
            "id": d.get("identifier") or f"{kind}-{i}",
            "width": float(dims[0]),
            "height": float(dims[1]),
            "position": (px, py, pz),
            "yaw": yaw,
            "wall_id": d.get("parentIdentifier"),
        })
    return out


def _infer_room_type(objects: List[dict]) -> str:
    counts: Dict[str, int] = {}
    for o in objects:
        counts[o["category"]] = counts.get(o["category"], 0) + 1
    has = lambda k: counts.get(k, 0) > 0
    if has("bed"):
        return "bedroom"
    if has("bathtub") or has("toilet"):
        return "bathroom"
    if has("refrigerator") or has("oven") or has("stove") or has("sink"):
        return "kitchen"
    if counts.get("chair", 0) >= 4 and has("table"):
        return "dining_room"
    if has("sofa") and has("television"):
        return "living_room"
    if counts.get("chair", 0) >= 1 and counts.get("table", 0) >= 1:
        return "office"
    return "unknown"


def parse_room(json_path) -> Dict[str, Any]:
    json_path = str(json_path)
    with open(json_path, "r") as f:
        raw = json.load(f)

    if not raw.get("floors"):
        raise ValueError(f"no floors in {json_path}")
    floor = _parse_floor(raw["floors"][0])

    objects = _parse_objects(raw.get("objects"), floor)
    walls = _parse_walls(raw.get("walls"), floor)
    doors = _parse_openings(raw.get("doors"), floor, "door")
    windows = _parse_openings(raw.get("windows"), floor, "window")
    room_type = _infer_room_type(objects)

    return {
        "source_path": json_path,
        "floor": floor,
        "walls": walls,
        "doors": doors,
        "windows": windows,
        "objects": objects,
        "room_type": room_type,
    }


def describe_scene(scene: Dict[str, Any]) -> str:
    f = scene["floor"]
    xmin, zmin, xmax, zmax = f["bounds"]
    lines = [
        f"Scene: {Path(scene['source_path']).name}",
        f"  room_type={scene['room_type']} area={f['area_m2']:.2f} m² size≈{xmax-xmin:.2f}x{zmax-zmin:.2f}",
        f"  walls={len(scene['walls'])} doors={len(scene['doors'])} windows={len(scene['windows'])} objects={len(scene['objects'])}",
    ]
    cat_counts: Dict[str, int] = {}
    for o in scene["objects"]:
        cat_counts[o["category"]] = cat_counts.get(o["category"], 0) + 1
    if cat_counts:
        lines.append("  objects by category: " + ", ".join(f"{k}:{v}" for k, v in sorted(cat_counts.items())))
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset/roomplan/Room.json"
    scene = parse_room(path)
    print(describe_scene(scene))
    print(f"Floor polygon: {len(scene['floor']['polygon'])} corners, bounds={scene['floor']['bounds']}")
    print("First 3 objects:")
    for o in scene["objects"][:3]:
        print(f"  {o['category']}/{o['subtype']} size={o['size']} pos={o['position']} yaw={o['yaw']:.1f}")
