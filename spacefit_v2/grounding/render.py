"""Top-down 2D room renderer for VLM grounding experiments.

Renders a floor plan showing:
  - room outline (floor polygon)
  - existing furniture (labelled rectangles)
  - doors (marked with D)
  - windows (marked with W)

Output: a PNG image suitable for GPT-4V / claude-3-vision input.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def render_topdown(
    scene: Dict[str, Any],
    target_asset: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
    dpi: int = 120,
    fig_size_m: float = 6.0,
) -> Optional[bytes]:
    """Render a top-down floor plan. Returns PNG bytes or saves to save_path.

    Args:
        scene: scene dict with floor, walls, doors, windows, objects
        target_asset: optional target asset to annotate on the image
        save_path: if provided, write PNG to this path and return None
        dpi: render resolution
        fig_size_m: figure size in inches

    Returns:
        PNG bytes if save_path is None, else None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
        import numpy as np
    except ImportError:
        return None

    floor_poly = scene.get("floor", {}).get("polygon", [])
    if not floor_poly:
        return None

    xs = [float(v[0]) for v in floor_poly]
    zs = [float(v[1]) for v in floor_poly]
    xmin, xmax = min(xs), max(xs)
    zmin, zmax = min(zs), max(zs)
    room_w = xmax - xmin
    room_h = zmax - zmin
    margin = max(room_w, room_h) * 0.07

    aspect = room_w / room_h if room_h > 0 else 1.0
    fig_w = fig_size_m
    fig_h = fig_size_m / aspect
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_aspect("equal")

    # Draw floor polygon
    from matplotlib.patches import Polygon as MplPoly
    floor_pts = [(float(v[0]), float(v[1])) for v in floor_poly]
    floor_patch = MplPoly(floor_pts, closed=True, facecolor="#f5f0e8", edgecolor="#333333", linewidth=2.5, zorder=1)
    ax.add_patch(floor_patch)

    # Draw existing furniture
    for obj in scene.get("objects", []):
        _draw_furniture(ax, obj)

    # Draw doors
    for door in scene.get("doors", []):
        dx, dz = float(door["position"][0]), float(door["position"][2])
        w = float(door.get("width", 0.9))
        ax.plot(dx, dz, "s", color="#d44", markersize=max(6, w * 10), zorder=4)
        ax.annotate("D", (dx, dz), ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold", zorder=5)

    # Draw windows
    for win in scene.get("windows", []):
        wx, wz = float(win["position"][0]), float(win["position"][2])
        w = float(win.get("width", 1.0))
        ax.plot(wx, wz, "s", color="#47a", markersize=max(6, w * 8), zorder=4)
        ax.annotate("W", (wx, wz), ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold", zorder=5)

    # Entrance point
    ep = scene.get("entrance_point")
    if ep:
        ax.plot(float(ep[0]), float(ep[1]), "^", color="#2a2", markersize=10, zorder=5)

    # Target asset annotation (dashed box at room center if no pose known)
    if target_asset:
        cat = str(target_asset.get("category", "?")).replace("_", " ")
        cx = (xmin + xmax) / 2
        cz = (zmin + zmax) / 2
        s = target_asset.get("size", {})
        w = float(s.get("width", 0.6))
        d = float(s.get("depth", 0.6))
        rect = patches.FancyBboxPatch(
            (cx - w / 2, cz - d / 2), w, d,
            boxstyle="round,pad=0.02",
            linestyle="--", linewidth=1.5, edgecolor="#e55", facecolor="#fee", alpha=0.5, zorder=6,
        )
        ax.add_patch(rect)
        ax.annotate(f"[{cat}?]", (cx, cz), ha="center", va="center",
                    fontsize=7, color="#c00", zorder=7)

    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(zmin - margin, zmax + margin)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    room_type = scene.get("room_type", "")
    ax.set_title(f"Room layout — {room_type}", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.3)
    plt.tight_layout()

    import io
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()


def _draw_furniture(ax: Any, obj: Dict[str, Any]) -> None:
    """Draw a single furniture item as a labelled rectangle."""
    import matplotlib.patches as patches
    import numpy as np

    pos = obj.get("position", [0, 0, 0])
    cx = float(pos[0] if isinstance(pos, (list, tuple)) else pos.get("x", 0))
    cz = float(pos[2] if isinstance(pos, (list, tuple)) else pos.get("z", 0))
    size = obj.get("size", [0.6, 0.8, 0.6])
    if isinstance(size, (list, tuple)):
        w, d = float(size[0]), float(size[2] if len(size) >= 3 else size[-1])
    else:
        w = float(size.get("width", 0.6))
        d = float(size.get("depth", 0.6))
    yaw_raw = obj.get("yaw", 0.0)
    yaw_deg = float(yaw_raw) if abs(float(yaw_raw)) > 2 * math.pi else math.degrees(float(yaw_raw))

    # Build rotated rectangle corners
    corners = _rect_corners(cx, cz, w, d, yaw_deg)
    cat = str(obj.get("category", "")).replace("_", "\n")

    poly = patches.Polygon(corners, closed=True, facecolor="#cde", edgecolor="#555",
                           linewidth=0.8, alpha=0.75, zorder=2)
    ax.add_patch(poly)
    ax.annotate(cat[:12], (cx, cz), ha="center", va="center",
                fontsize=5.5, color="#222", zorder=3, wrap=True)


def _rect_corners(cx: float, cz: float, w: float, d: float, yaw_deg: float) -> List[Tuple[float, float]]:
    yaw = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    hw, hd = w / 2, d / 2
    local = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
    return [(cx + lx * cos_y - lz * sin_y, cz + lx * sin_y + lz * cos_y) for lx, lz in local]
