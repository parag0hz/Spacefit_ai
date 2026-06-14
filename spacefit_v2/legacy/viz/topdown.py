"""Top-down 2D visualization of a scene + analysis results."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from ..core import geometry as G

_CAT_COLORS = {
    "sofa": "#d97757",
    "chair": "#a8a1d4",
    "table": "#7ba06e",
    "bed": "#c97b89",
    "storage": "#bfa36e",
    "refrigerator": "#6a9ab5",
    "television": "#555555",
    "bathtub": "#83b0c3",
    "toilet": "#c9b6d4",
    "oven": "#8b6d4f",
    "sink": "#89b2ae",
    "washer": "#9090b0",
    "dishwasher": "#7d9d92",
    "stairs": "#666666",
    "fireplace": "#a45d4d",
    "stove": "#8b6d4f",
}


def _cat_color(cat: str) -> str:
    return _CAT_COLORS.get(cat, "#888888")


def _draw_polygon(ax, polygon, **kwargs):
    xs = [p[0] for p in polygon] + [polygon[0][0]]
    ys = [p[1] for p in polygon] + [polygon[0][1]]
    ax.plot(xs, ys, **kwargs)


def _draw_rotated_box(ax, cx, cz, w, d, yaw_deg, label=None, color="#444", alpha=0.6,
                       edge="black", linewidth=1.2):
    corners = G.rotated_bbox_corners(cx, cz, w, d, yaw_deg)
    poly = plt.Polygon(corners, closed=True, facecolor=color, edgecolor=edge,
                       alpha=alpha, linewidth=linewidth)
    ax.add_patch(poly)
    # Indicate forward direction (local +Z) with a small tick
    import math
    cy, cs = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    fx, fz = -cs * (d * 0.4), cy * (d * 0.4)
    ax.plot([cx, cx + fx], [cz, cz + fz], color=edge, linewidth=1.0)
    if label:
        ax.text(cx, cz, label, ha="center", va="center", fontsize=7, color=edge)


def _draw_existing(ax, scene):
    for o in scene["objects"]:
        cx, _, cz = o["position"]
        w, _, d = o["size"]
        _draw_rotated_box(ax, cx, cz, w, d, o["yaw"],
                          label=o["category"][:4], color=_cat_color(o["category"]),
                          alpha=0.5, edge="#333")


def _draw_doors_windows(ax, scene):
    for d in scene.get("doors", []):
        cx, _, cz = d["position"]
        ax.plot(cx, cz, marker="s", color="red", markersize=10, markeredgecolor="black")
        ax.annotate("D", (cx, cz), color="white", ha="center", va="center", fontsize=7,
                    fontweight="bold")
    for w in scene.get("windows", []):
        cx, _, cz = w["position"]
        ax.plot(cx, cz, marker="s", color="#3a7fbf", markersize=10, markeredgecolor="black")
        ax.annotate("W", (cx, cz), color="white", ha="center", va="center", fontsize=7,
                    fontweight="bold")


def draw_scene(scene: Dict, ax=None, title: Optional[str] = None,
               show_object_labels: bool = True, figsize=(9, 9)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    _draw_polygon(ax, scene["floor"]["polygon"], color="black", linewidth=2.0)
    _draw_existing(ax, scene)
    _draw_doors_windows(ax, scene)
    ax.set_aspect("equal")
    xmin, zmin, xmax, zmax = scene["floor"]["bounds"]
    ax.set_xlim(xmin - 0.5, xmax + 0.5)
    ax.set_ylim(zmin - 0.5, zmax + 0.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    if title:
        ax.set_title(title)
    return ax


def draw_occupancy(occ, scene: Dict, save_path: Optional[str] = None,
                   title: str = "Occupancy grid", figsize=(9, 9)):
    from ..core.occupancy import FREE, OCC_OBJECT, OCC_OUTSIDE, OCC_WALL, OCC_DOOR_CLEARANCE
    cmap = {
        FREE: "#f5f5f5",
        OCC_OBJECT: "#d97757",
        OCC_OUTSIDE: "#dddddd",
        OCC_WALL: "#000000",
        OCC_DOOR_CLEARANCE: "#ffe0a0",
    }
    H, W = occ.shape
    img = np.zeros((H, W, 3), dtype=np.float32)
    import matplotlib.colors as mc
    for val, hex_ in cmap.items():
        rgb = np.array(mc.to_rgb(hex_), dtype=np.float32)
        mask = occ.grid == val
        img[mask] = rgb
    fig, ax = plt.subplots(figsize=figsize)
    xmin = occ.origin[0]
    zmin = occ.origin[1]
    extent = (xmin, xmin + W * occ.resolution, zmin + H * occ.resolution, zmin)
    ax.imshow(img, extent=extent, origin="upper")
    ax.invert_yaxis()
    _draw_polygon(ax, scene["floor"]["polygon"], color="black", linewidth=1.5)
    _draw_doors_windows(ax, scene)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    if save_path:
        plt.tight_layout()
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    return fig, ax


def draw_candidates(scene: Dict, candidates: List, save_path: Optional[str] = None,
                    title: str = "Candidate regions", figsize=(9, 9)):
    fig, ax = plt.subplots(figsize=figsize)
    draw_scene(scene, ax=ax, title=title)
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(candidates):
        cx, cz, w, d, yaw = c.max_rect
        color = cmap(i % 20)
        rect = plt.Rectangle((cx - w / 2, cz - d / 2), w, d, angle=yaw,
                              facecolor=color, edgecolor="black", alpha=0.35, linewidth=1.0)
        ax.add_patch(rect)
        ax.text(c.centroid[0], c.centroid[1], f"#{c.id}\n{c.area:.1f}m²",
                ha="center", va="center", fontsize=7, fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.75, edgecolor="none"))
    if save_path:
        plt.tight_layout()
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    return fig, ax


def draw_placements(scene: Dict, placements: List[Dict], candidates: Optional[List] = None,
                    save_path: Optional[str] = None, title: str = "SpaceFit placements",
                    figsize=(9, 9)):
    fig, ax = plt.subplots(figsize=figsize)
    draw_scene(scene, ax=ax, title=title)
    if candidates:
        for c in candidates:
            cx, cz, w, d, yaw = c.max_rect
            rect = plt.Rectangle((cx - w / 2, cz - d / 2), w, d, angle=yaw,
                                  facecolor="#b9e6c4", edgecolor="#5aa672",
                                  alpha=0.15, linewidth=0.8)
            ax.add_patch(rect)
    for p in placements:
        if p.get("status") != "placed" or "position" not in p:
            continue
        cx = p["position"]["x"]
        cz = p["position"]["z"]
        w = p["size"]["width"]
        d = p["size"]["depth"]
        yaw = p["rotation_y"]
        model = p.get("selected_model") or {}
        model_cat = model.get("category", "") if isinstance(model, dict) else ""
        label = (model_cat[:10] if model_cat else p.get("category", "new"))[:12]
        _draw_rotated_box(ax, cx, cz, w, d, yaw,
                          label=label,
                          color="#e1a13f", alpha=0.8, edge="red", linewidth=1.8)
    # Annotate unplaced items in the corner
    unplaced = [p for p in placements if p.get("status") != "placed"]
    if unplaced:
        lines = ["UNPLACED:"] + [f" • {p.get('category','?')} ({p.get('furniture_id','')})" for p in unplaced]
        ax.text(0.02, 0.02, "\n".join(lines), transform=ax.transAxes,
                fontsize=8, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeaea", edgecolor="red"))
    if save_path:
        plt.tight_layout()
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    return fig, ax
