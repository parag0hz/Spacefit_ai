"""Raw 3D-FRONT room adapter for single-target placement experiments."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


ALLOWED_ROOM_TYPES = {
    "Bedroom",
    "MasterBedroom",
    "SecondBedroom",
    "LivingRoom",
    "LivingDiningRoom",
    "Library",
}


ROOM_TYPE_MAP = {
    "Bedroom": "bedroom",
    "MasterBedroom": "bedroom",
    "SecondBedroom": "bedroom",
    "LivingRoom": "living_room",
    "LivingDiningRoom": "living_room",
    "Library": "library",
}


TARGET_CATEGORIES = {
    "sofa",
    "armchair",
    "desk",
    "coffee_table",
    "dining_chair",
    "tv_stand",
    "nightstand",
    "bookshelf",
    "wardrobe",
    "bed",
}


def _flattened_tris(values: Sequence[int]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int32)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    return arr.reshape(-1, 3)


def _signed_area(points: Sequence[Sequence[float]]) -> float:
    area = 0.0
    for idx in range(len(points)):
        x1, z1 = points[idx]
        x2, z2 = points[(idx + 1) % len(points)]
        area += x1 * z2 - x2 * z1
    return area * 0.5


def _polygon_area(points: Sequence[Sequence[float]]) -> float:
    return abs(_signed_area(points))


def _polygon_bounds(points: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in points]
    zs = [float(p[1]) for p in points]
    return (min(xs), min(zs), max(xs), max(zs))


def _mesh_vertices_xz(mesh: Dict[str, Any]) -> np.ndarray:
    xyz = np.asarray(mesh.get("xyz", []), dtype=np.float64).reshape(-1, 3)
    if xyz.size == 0:
        raise ValueError("mesh has no vertices")
    return xyz[:, [0, 2]]


def _ordered_polygon_from_mesh(mesh: Dict[str, Any]) -> List[Tuple[float, float]]:
    verts = _mesh_vertices_xz(mesh)
    faces = _flattened_tris(mesh.get("faces", []))
    if len(verts) < 3:
        raise ValueError("floor mesh has too few vertices")

    tri_polys = []
    for face in faces:
        tri = [tuple(float(v) for v in verts[int(idx)]) for idx in face]
        poly = Polygon(tri)
        if poly.is_valid and poly.area > 1e-6:
            tri_polys.append(poly)
    if tri_polys:
        merged = unary_union(tri_polys)
        if merged.geom_type == "MultiPolygon":
            merged = max(list(merged.geoms), key=lambda poly: poly.area)
        if merged.geom_type == "Polygon" and merged.area > 1e-6:
            polygon = [(float(x), float(z)) for x, z in list(merged.exterior.coords)[:-1]]
            if len(polygon) >= 3:
                if _signed_area(polygon) < 0:
                    polygon = list(reversed(polygon))
                return polygon

    unique_vertices: List[Tuple[float, float]] = []
    key_to_uid: Dict[Tuple[float, float], int] = {}
    original_to_unique: Dict[int, int] = {}
    for idx, v in enumerate(verts):
        key = (round(float(v[0]), 6), round(float(v[1]), 6))
        uid = key_to_uid.get(key)
        if uid is None:
            uid = len(unique_vertices)
            key_to_uid[key] = uid
            unique_vertices.append(key)
        original_to_unique[idx] = uid

    edge_counts: Counter[Tuple[int, int]] = Counter()
    for face in faces:
        tri = [original_to_unique[int(face[0])], original_to_unique[int(face[1])], original_to_unique[int(face[2])]]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if a != b:
                edge_counts[tuple(sorted((a, b)))] += 1

    adjacency: Dict[int, List[int]] = defaultdict(list)
    for (a, b), count in edge_counts.items():
        if count != 1:
            continue
        adjacency[a].append(b)
        adjacency[b].append(a)

    if not adjacency:
        polygon = [(float(x), float(z)) for x, z in unique_vertices]
    else:
        start = min(adjacency, key=lambda idx: (unique_vertices[idx][0], unique_vertices[idx][1], idx))
        cycle = [start]
        prev = None
        current = start
        while True:
            neighbors = sorted(adjacency[current], key=lambda idx: (unique_vertices[idx][0], unique_vertices[idx][1], idx))
            next_idx = None
            for cand in neighbors:
                if cand != prev:
                    next_idx = cand
                    break
            if next_idx is None or next_idx == start:
                break
            if next_idx in cycle:
                break
            cycle.append(next_idx)
            prev, current = current, next_idx
        polygon = [(float(unique_vertices[idx][0]), float(unique_vertices[idx][1])) for idx in cycle]

    if len(polygon) < 3:
        raise ValueError("failed to recover floor polygon")
    if _signed_area(polygon) < 0:
        polygon = list(reversed(polygon))
    return polygon


def _merged_floor_polygon(meshes: Sequence[Dict[str, Any]]) -> List[Tuple[float, float]]:
    polys = []
    for mesh in meshes:
        try:
            poly = Polygon(_ordered_polygon_from_mesh(mesh))
        except Exception:
            continue
        if poly.is_valid and poly.area > 1e-6:
            polys.append(poly)
    if not polys:
        raise ValueError("no valid floor polygons")
    merged = unary_union(polys)
    if merged.geom_type == "MultiPolygon":
        merged = max(list(merged.geoms), key=lambda poly: poly.area)
    polygon = [(float(x), float(z)) for x, z in list(merged.exterior.coords)[:-1]]
    if _signed_area(polygon) < 0:
        polygon = list(reversed(polygon))
    return polygon


def _quat_to_yaw_deg(quat: Sequence[float]) -> float:
    if len(quat) != 4:
        return 0.0
    x, y, z, w = (float(v) for v in quat)
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def _size_from_entry(child: Dict[str, Any], meta: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    scale = child.get("scale") or [1.0, 1.0, 1.0]
    sx, sy, sz = (abs(float(v)) for v in scale[:3])

    replace_bbox = child.get("replace_bbox")
    if isinstance(replace_bbox, dict):
        width = float(replace_bbox.get("xLen", 0.0)) / 100.0
        height = float(replace_bbox.get("yLen", 0.0)) / 100.0
        depth = float(replace_bbox.get("zLen", 0.0)) / 100.0
    else:
        size = meta.get("size") or meta.get("bbox")
        if isinstance(size, list) and size and isinstance(size[0], list):
            size = size[0]
        if not isinstance(size, list) or len(size) < 3:
            return None
        width, height, depth = (float(v) for v in size[:3])

    width *= sx
    height *= sy
    depth *= sz
    if min(width, height, depth) <= 0.05:
        return None
    if max(width, height, depth) > 8.0:
        return None
    return (width, height, depth)


def _normalize_category(meta: Dict[str, Any]) -> Optional[str]:
    title = str(meta.get("title", "") or "").lower()
    category = str(meta.get("category", "") or "").lower()
    text = " ".join(part for part in (title, category) if part).strip()
    if not text:
        return None

    if "nightstand" in text or "bedside table" in text:
        return "nightstand"
    if "tv stand" in text or "media unit" in text:
        return "tv_stand"
    if "wardrobe" in text or "closet" in text:
        return "wardrobe"
    if "bookcase" in text or "bookshelf" in text or "jewelry armoire" in text:
        return "bookshelf"
    if "coffee table" in text or "tea table" in text:
        return "coffee_table"
    if "dining chair" in text or "chinese chair" in text:
        return "dining_chair"
    if "armchair" in text or "lounge chair" in text or "book-chair" in text:
        return "armchair"
    if "desk" in text or "writing table" in text:
        return "desk"
    if "sofa" in text:
        return "sofa"
    if " bed" in f" {text} " or text.startswith("bed") or "double bed" in text or "single bed" in text:
        return "bed"
    if "chair" in text:
        return "armchair"
    if "cabinet/shelf/desk" in text and "desk" in text:
        return "desk"
    return None


def _walls_from_polygon(polygon: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
    walls = []
    for idx in range(len(polygon)):
        ax, az = (float(v) for v in polygon[idx])
        bx, bz = (float(v) for v in polygon[(idx + 1) % len(polygon)])
        cx = 0.5 * (ax + bx)
        cz = 0.5 * (az + bz)
        length = math.hypot(bx - ax, bz - az)
        yaw = math.degrees(math.atan2(bz - az, bx - ax))
        walls.append(
            {
                "id": f"wall-{idx}",
                "length": length,
                "height": 2.8,
                "position": (cx, 1.4, cz),
                "yaw": yaw,
            }
        )
    return walls


def _mesh_opening(mesh: Dict[str, Any], kind: str, idx: int) -> Dict[str, Any]:
    verts = np.asarray(mesh.get("xyz", []), dtype=np.float64).reshape(-1, 3)
    xs = verts[:, 0]
    ys = verts[:, 1]
    zs = verts[:, 2]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    zmin, zmax = float(zs.min()), float(zs.max())
    dx = xmax - xmin
    dz = zmax - zmin
    if dx >= dz:
        segment = [(xmin, 0.5 * (zmin + zmax)), (xmax, 0.5 * (zmin + zmax))]
        yaw = 0.0
        width = dx
    else:
        segment = [(0.5 * (xmin + xmax), zmin), (0.5 * (xmin + xmax), zmax)]
        yaw = 90.0
        width = dz
    return {
        "id": f"{kind}-{idx}",
        "width": float(max(width, 0.4)),
        "height": float(max(ymax - ymin, 1.0)),
        "position": (0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)),
        "yaw": yaw,
        "segment": [(float(a), float(b)) for a, b in segment],
    }


def _object_polygon(obj: Dict[str, Any]) -> Tuple[float, float, float, float]:
    px, _, pz = obj["position"]
    width, _, depth = obj["size"]
    return (float(px), float(pz), float(width), float(depth))


def _room_scene(raw: Dict[str, Any], room: Dict[str, Any], source_path: Path) -> Optional[Dict[str, Any]]:
    room_type_raw = str(room.get("type", ""))
    if room_type_raw not in ALLOWED_ROOM_TYPES:
        return None

    furniture_by_uid = {entry["uid"]: entry for entry in raw.get("furniture", [])}
    mesh_by_uid = {entry["uid"]: entry for entry in raw.get("mesh", [])}

    floor_meshes = []
    for child in room.get("children", []):
        mesh = mesh_by_uid.get(child.get("ref"))
        if mesh and mesh.get("type") == "Floor":
            floor_meshes.append(mesh)
    if not floor_meshes:
        return None

    polygon = _merged_floor_polygon(floor_meshes)
    bounds = _polygon_bounds(polygon)
    walls = _walls_from_polygon(polygon)

    doors: List[Dict[str, Any]] = []
    windows: List[Dict[str, Any]] = []
    openings: List[Dict[str, Any]] = []
    objects: List[Dict[str, Any]] = []

    for child in room.get("children", []):
        ref = child.get("ref")
        mesh = mesh_by_uid.get(ref)
        if mesh:
            mesh_type = str(mesh.get("type", ""))
            if mesh_type == "Door":
                door = _mesh_opening(mesh, "door", len(doors))
                doors.append(door)
                openings.append(dict(door, kind="door"))
            elif mesh_type in {"Window", "BayWindow"}:
                window = _mesh_opening(mesh, "window", len(windows))
                windows.append(window)
                openings.append(dict(window, kind="window"))
            elif mesh_type == "Hole":
                openings.append(dict(_mesh_opening(mesh, "opening", len(openings)), kind="opening"))
            continue

        meta = furniture_by_uid.get(ref)
        if meta is None:
            continue
        category = _normalize_category(meta)
        if category is None:
            continue
        size = _size_from_entry(child, meta)
        if size is None:
            continue
        pos = child.get("pos") or [0.0, 0.0, 0.0]
        if len(pos) < 3:
            continue
        yaw = _quat_to_yaw_deg(child.get("rot") or [0.0, 0.0, 0.0, 1.0])
        objects.append(
            {
                "id": str(child.get("instanceid") or ref),
                "category": category,
                "subtype": str(meta.get("title", "") or ""),
                "size": (float(size[0]), float(size[1]), float(size[2])),
                "position": (float(pos[0]), float(pos[1]), float(pos[2])),
                "yaw": float(yaw),
                "attributes": {
                    "raw_ref": ref,
                    "raw_title": meta.get("title"),
                    "raw_category": meta.get("category"),
                    "raw_room_type": room_type_raw,
                },
            }
        )

    if len(objects) < 2:
        return None

    room_type = ROOM_TYPE_MAP.get(room_type_raw, "unknown")
    floor_area = _polygon_area(polygon)
    entrance_point = None
    if doors:
        entrance_point = (float(doors[0]["position"][0]), float(doors[0]["position"][2]))

    return {
        "source_path": str(source_path),
        "scene_id": str(raw.get("uid", source_path.stem)),
        "room_id": str(room.get("instanceid", room_type_raw)),
        "room_type": room_type,
        "room_type_raw": room_type_raw,
        "floor": {
            "polygon": [(float(x), float(z)) for x, z in polygon],
            "bounds": bounds,
            "area_m2": float(floor_area),
        },
        "walls": walls,
        "doors": doors,
        "windows": windows,
        "openings": openings,
        "objects": objects,
        "entrance_point": entrance_point,
    }


def iter_room_scenes(
    data_dir: str | Path,
    max_scenes: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    scene_paths = sorted(Path(data_dir).glob("*.json"))
    if max_scenes is not None:
        scene_paths = scene_paths[: int(max_scenes)]

    for path in scene_paths:
        with open(path, "r") as f:
            raw = json.load(f)
        for room in raw.get("scene", {}).get("room", []):
            scene = _room_scene(raw, room, path)
            if scene is not None:
                yield scene
