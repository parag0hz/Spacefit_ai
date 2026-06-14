"""3D rendering of a SpaceFit placement using real 3D-FUTURE meshes.

- New items (with `selected_model.model_id`) load `raw_model.obj` + texture
  from `3D-FUTURE-model/<model_id>/` — full textured geometry at real scale.
- Existing RoomPlan objects render as labeled colored boxes (no mesh data in
  RoomPlan scans).
- Floor: a flat polygon plane at y=0.

Rendering uses pyrender's offscreen EGL/OSMesa path. If that fails (no display,
no EGL), we fall back to a matplotlib 3D wireframe.

Usage:
    from spacefit_v2.legacy.viz.render3d import render_result
    render_result(result, save_path="scene.png", view="iso")
"""
from __future__ import annotations

import os
# OSMesa is our reliable CPU-only headless backend; EGL often fails on servers.
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

MODEL_ROOT = "dataset/3D-FUTURE-model"

_CAT_COLORS = {
    "sofa": (0.85, 0.46, 0.34),
    "chair": (0.66, 0.63, 0.83),
    "table": (0.48, 0.63, 0.43),
    "bed": (0.79, 0.48, 0.54),
    "storage": (0.75, 0.64, 0.43),
    "refrigerator": (0.42, 0.61, 0.71),
    "television": (0.33, 0.33, 0.33),
    "bathtub": (0.51, 0.69, 0.76),
    "toilet": (0.79, 0.71, 0.83),
    "oven": (0.55, 0.43, 0.31),
    "sink": (0.54, 0.70, 0.68),
    "washer": (0.56, 0.56, 0.69),
    "dishwasher": (0.49, 0.62, 0.57),
    "stairs": (0.4, 0.4, 0.4),
    "fireplace": (0.64, 0.36, 0.30),
    "stove": (0.55, 0.43, 0.31),
}


def _cat_color(cat: str):
    return _CAT_COLORS.get(cat, (0.55, 0.55, 0.55))


def _yaw_to_matrix(yaw_deg: float, translate=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Return a 4x4 homogeneous transform: translate then rotate around world +Y."""
    c = math.cos(math.radians(yaw_deg))
    s = math.sin(math.radians(yaw_deg))
    R = np.array([[c, 0, s, 0],
                  [0, 1, 0, 0],
                  [-s, 0, c, 0],
                  [0, 0, 0, 1]], dtype=np.float64)
    T = np.eye(4)
    T[:3, 3] = translate
    return T @ R


def _box_mesh(width, height, depth, color=(0.7, 0.7, 0.7), alpha=1.0):
    import trimesh
    box = trimesh.creation.box(extents=[width, height, depth])
    rgba = np.array([*color, alpha], dtype=np.float32)
    mat = np.tile(rgba, (len(box.faces), 1))
    box.visual.face_colors = (mat * 255).astype(np.uint8)
    return box


def _load_model_mesh(model_id: str):
    """Load 3D-FUTURE raw_model.obj with its texture (via .mtl resolve)."""
    import trimesh
    obj_path = Path(MODEL_ROOT) / model_id / "raw_model.obj"
    if not obj_path.exists():
        return None
    try:
        # process=True and skip_materials=False so trimesh picks up the
        # model.mtl + texture.png that sits next to the OBJ.
        loaded = trimesh.load(str(obj_path), process=True)
        # trimesh returns a Scene if the OBJ has multiple materials
        if isinstance(loaded, trimesh.Scene):
            meshes = list(loaded.geometry.values())
            if not meshes:
                return None
            # Combine into a single mesh while preserving each sub's visual
            mesh = trimesh.util.concatenate([m for m in meshes])
        else:
            mesh = loaded
        return mesh
    except Exception:
        return None


def _floor_mesh(polygon: List[Tuple[float, float]], color=(0.93, 0.93, 0.93)):
    """Triangulate the 2D floor polygon and return a trimesh placed at y=0."""
    import trimesh
    from shapely.geometry import Polygon
    poly = Polygon(polygon)
    try:
        verts_2d, faces = trimesh.creation.triangulate_polygon(poly, engine="earcut")
    except Exception:
        from shapely import geometry as sg
        import numpy as np
        # fall back to triangle-fan
        pts = np.asarray(polygon, dtype=np.float64)
        cx, cz = pts.mean(axis=0)
        verts_2d = np.vstack([[cx, cz], pts])
        n = len(pts)
        faces = np.array([[0, i + 1, ((i + 1) % n) + 1] for i in range(n)])
    verts_3d = np.zeros((len(verts_2d), 3))
    verts_3d[:, 0] = verts_2d[:, 0]
    verts_3d[:, 2] = verts_2d[:, 1]  # our z → world Z
    m = trimesh.Trimesh(vertices=verts_3d, faces=faces, process=False)
    col = (np.array([*color, 1.0]) * 255).astype(np.uint8)
    m.visual.face_colors = np.tile(col, (len(m.faces), 1))
    return m


def _wall_outlines(polygon: List[Tuple[float, float]], height: float = 2.4,
                    thickness: float = 0.04):
    """Create thin wall boxes along the polygon perimeter."""
    import trimesh
    walls = []
    n = len(polygon)
    for i in range(n):
        p1 = np.array(polygon[i], dtype=float)
        p2 = np.array(polygon[(i + 1) % n], dtype=float)
        mid = (p1 + p2) / 2.0
        length = float(np.linalg.norm(p2 - p1))
        if length < 1e-3:
            continue
        yaw = math.degrees(math.atan2(p2[0] - p1[0], p2[1] - p1[1]))
        box = trimesh.creation.box(extents=[length, height, thickness])
        col = (np.array([0.82, 0.82, 0.82, 1.0]) * 255).astype(np.uint8)
        box.visual.face_colors = np.tile(col, (len(box.faces), 1))
        T = _yaw_to_matrix(yaw, translate=(mid[0], height / 2.0, mid[1]))
        box.apply_transform(T)
        walls.append(box)
    return walls


def _scene_obstacle_box(obj: Dict, alpha: float = 0.35):
    cat = obj.get("category", "")
    color = _cat_color(cat)
    w, h, d = obj["size"]
    box = _box_mesh(w, h, d, color=color, alpha=alpha)
    cx, cy, cz = obj["position"]
    # obj y is center height
    T = _yaw_to_matrix(obj["yaw"], translate=(cx, cy, cz))
    box.apply_transform(T)
    return box


def _placement_mesh(p: Dict, fallback_highlight=True):
    """Prefer loaded 3D-FUTURE mesh; fall back to a colored highlight box."""
    import trimesh
    sel = p.get("selected_model") or {}
    model_id = sel.get("model_id")
    w = p["size"]["width"]; h = p["size"]["height"]; d = p["size"]["depth"]
    cx = p["position"]["x"]; cz = p["position"]["z"]
    yaw = p["rotation_y"]

    mesh = None
    if model_id:
        mesh = _load_model_mesh(model_id)
        if mesh is not None:
            # 3D-FUTURE raw_model.obj is in real meters, Y-up, with its origin
            # typically at the mesh's centroid in XZ and floor at y=min.
            # Shift so y-min sits at 0, then translate to (cx, 0, cz) and yaw.
            try:
                mesh = mesh.copy()
                b = mesh.bounds
                # center in XZ, floor at y=0
                center_x = (b[0, 0] + b[1, 0]) / 2.0
                center_z = (b[0, 2] + b[1, 2]) / 2.0
                y_min = b[0, 1]
                mesh.apply_translation([-center_x, -y_min, -center_z])
                T = _yaw_to_matrix(yaw, translate=(cx, 0.0, cz))
                mesh.apply_transform(T)
                return mesh, "textured"
            except Exception:
                mesh = None

    # Fallback: highlight box tinted orange
    if fallback_highlight:
        box = _box_mesh(w, h, d, color=(0.88, 0.63, 0.25), alpha=1.0)
        T = _yaw_to_matrix(yaw, translate=(cx, h / 2.0, cz))
        box.apply_transform(T)
        return box, "box"
    return None, None


def _build_scene(result: Dict):
    """Return a list of trimesh objects forming the full scene."""
    items = []
    floor_polygon = result["scene"]["floor"]["polygon"]
    items.append(("floor", _floor_mesh(floor_polygon)))
    # Short wall "curb" (20cm) — visual outline without blocking the camera.
    for w in _wall_outlines(floor_polygon, height=0.2, thickness=0.08):
        items.append(("wall", w))
    for obj in result["scene"]["objects"]:
        # Keep existing furniture translucent so textured new meshes remain visible.
        items.append(("existing", _scene_obstacle_box(obj, alpha=0.30)))
    for p in result["placements"]:
        if p.get("status") != "placed" or "position" not in p:
            continue
        mesh, kind = _placement_mesh(p)
        if mesh is not None:
            items.append((f"new/{kind}", mesh))
    return items, floor_polygon


def _look_at(eye, target, up=(0, 1, 0)) -> np.ndarray:
    """Camera-to-world pose that looks from `eye` toward `target`."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-9
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-9
    true_up = np.cross(right, forward)
    M = np.eye(4)
    M[:3, 0] = right
    M[:3, 1] = true_up
    M[:3, 2] = -forward
    M[:3, 3] = eye
    return M


def _camera_pose(view: str, polygon) -> np.ndarray:
    pts = np.asarray(polygon)
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    zmin, zmax = pts[:, 1].min(), pts[:, 1].max()
    cx = (xmin + xmax) / 2.0
    cz = (zmin + zmax) / 2.0
    size = max(xmax - xmin, zmax - zmin)

    if view == "top":
        # top-down, camera high above, Y-up convention
        eye = np.array([cx, size * 1.2 + 3.0, cz + 0.01])
        target = np.array([cx, 0, cz])
        return _look_at(eye, target, up=(0, 0, -1))

    # 3/4 isometric from "outside" the room, looking in
    eye = np.array([cx - size * 0.8, size * 0.7 + 2.0, cz - size * 1.2])
    target = np.array([cx, 0.5, cz])
    return _look_at(eye, target, up=(0, 1, 0))


def render_result(result: Dict, save_path: str, view: str = "iso",
                    width: int = 1280, height: int = 900,
                    bg_color=(1.0, 1.0, 1.0)) -> str:
    import pyrender
    import trimesh

    items, polygon = _build_scene(result)
    scene = pyrender.Scene(bg_color=list(bg_color) + [1.0],
                            ambient_light=[0.35, 0.35, 0.35])

    for tag, tri in items:
        try:
            mesh = pyrender.Mesh.from_trimesh(tri, smooth=False)
            scene.add(mesh)
        except Exception as e:
            # Try once more with smooth=True (some meshes don't have face colors)
            try:
                mesh = pyrender.Mesh.from_trimesh(tri, smooth=True)
                scene.add(mesh)
            except Exception:
                continue

    # Camera
    cam = pyrender.PerspectiveCamera(yfov=math.radians(40.0), aspectRatio=width / height)
    cam_pose = _camera_pose(view, polygon)
    scene.add(cam, pose=cam_pose)

    # Lights: one key + one fill + hemisphere
    key = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.5)
    key_pose = _camera_pose("iso", polygon).copy()
    key_pose[:3, 3] += np.array([2, 5, 2])
    scene.add(key, pose=key_pose)
    fill = pyrender.DirectionalLight(color=[1.0, 0.95, 0.85], intensity=1.5)
    fill_pose = np.eye(4); fill_pose[:3, 3] = np.array([0, 5, 0])
    scene.add(fill, pose=fill_pose)

    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color, _depth = r.render(scene)
    finally:
        r.delete()

    import imageio.v2 as imageio
    imageio.imwrite(save_path, color)
    return save_path


def render_result_matplotlib(result: Dict, save_path: str, view: str = "iso"):
    """Matplotlib 3D fallback — no textures, wireframe + patches only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import trimesh

    items, polygon = _build_scene(result)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    for tag, tri in items:
        if tri is None:
            continue
        verts = tri.vertices
        faces = tri.faces
        cols = None
        if tri.visual is not None and getattr(tri.visual, "face_colors", None) is not None:
            fc = tri.visual.face_colors
            cols = fc[:, :3] / 255.0
        tri_verts = verts[faces]
        pc = Poly3DCollection(tri_verts, alpha=0.9, linewidths=0.15,
                                edgecolor=(0.2, 0.2, 0.2, 0.4))
        if cols is not None:
            pc.set_facecolor(cols)
        ax.add_collection3d(pc)

    pts = np.asarray(polygon)
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    zmin, zmax = pts[:, 1].min(), pts[:, 1].max()
    size = max(xmax - xmin, zmax - zmin) + 0.5
    cx = (xmin + xmax) / 2.0
    cz = (zmin + zmax) / 2.0
    ax.set_xlim(cx - size / 2, cx + size / 2)
    ax.set_zlim(0, size * 0.7)
    ax.set_ylim(cz - size / 2, cz + size / 2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_zlabel("y (m)")
    if view == "top":
        ax.view_init(elev=90, azim=-90)
    else:
        ax.view_init(elev=28, azim=-55)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def render_auto(result: Dict, save_path: str, view: str = "iso"):
    """Try pyrender first; fall back to matplotlib if OpenGL/EGL unavailable."""
    try:
        return render_result(result, save_path, view=view)
    except Exception as e:
        print(f"[render3d] pyrender failed ({e.__class__.__name__}: {e}) — "
              f"falling back to matplotlib 3D")
        return render_result_matplotlib(result, save_path, view=view)
