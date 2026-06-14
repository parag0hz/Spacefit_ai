"""Top-down visualization with real 3D-FUTURE thumbnails stamped at each
placement's location. Much more readable than plain rectangles because the
viewer sees the actual chosen model.

Each 3D-FUTURE model directory contains an `image.jpg` — a rendered preview of
that specific model from a standard viewpoint. We:
  1. Load the thumbnail, remove the mostly-white background.
  2. Rotate it to match the placement yaw.
  3. Scale it to the footprint size in room coordinates.
  4. Paste it onto the 2D top-down floor plan at the placement centroid.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image, ImageDraw

MODEL_ROOT = "dataset/3D-FUTURE-model"


def _load_thumbnail(model_id: str, bg_tolerance: int = 8) -> Optional[Image.Image]:
    """Load thumbnail and mask out near-white background to RGBA."""
    path = Path(MODEL_ROOT) / model_id / "image.jpg"
    if not path.exists():
        return None
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    # Corners are almost certainly background — sample them to get the
    # actual background color instead of assuming white.
    corners = np.array([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]])
    bg = corners.mean(axis=0)[:3]
    diff = np.abs(arr[:, :, :3].astype(int) - bg.astype(int)).max(axis=-1)
    alpha = (diff > bg_tolerance).astype(np.uint8) * 255
    # Blur alpha edges a bit
    from scipy.ndimage import gaussian_filter
    alpha = gaussian_filter(alpha.astype(float), sigma=0.8).clip(0, 255).astype(np.uint8)
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, mode="RGBA")


def _bbox_of_alpha(img: Image.Image) -> Tuple[int, int, int, int]:
    a = np.array(img.split()[-1])
    ys, xs = np.where(a > 10)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, img.width, img.height)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _stamp_thumbnail(ax, thumb: Image.Image, cx: float, cz: float,
                      width: float, depth: float, yaw_deg: float):
    """Rotate + scale + place a thumbnail onto ax at the given world coords."""
    # crop to visible content
    l, t, r, b = _bbox_of_alpha(thumb)
    thumb = thumb.crop((l, t, r + 1, b + 1))

    # 3D-FUTURE thumbnails are rendered from a ~45° front-view, not top-down.
    # For a top-down 2D plan we can't get a true top view from the JPG.
    # Compromise: rotate the thumbnail by yaw and scale it so its on-floor
    # width × depth covers the footprint. The viewer sees "this specific model"
    # rather than "a generic box" even if the viewing angle isn't orthographic.
    rotated = thumb.rotate(-yaw_deg, expand=True, resample=Image.BICUBIC)
    # Scale so that the rotated footprint approximately matches
    # width x depth on the plot (in data units)
    max_side_m = max(width, depth) * 1.05
    px_per_m = 220 / max_side_m  # resolution scale
    new_w = max(20, int(rotated.width  * (max_side_m / max(rotated.width, rotated.height))
                         * px_per_m / 10) )  # unused side path (simplify below)
    # Simpler: just use imshow with extent in data units
    half_w, half_d = width / 2.0, depth / 2.0
    # Rotate extent box corners to find rotated aabb to set extent
    c = np.cos(np.radians(yaw_deg)); s = np.sin(np.radians(yaw_deg))
    corners = np.array([[-half_w, -half_d], [half_w, -half_d],
                          [half_w, half_d], [-half_w, half_d]])
    R = np.array([[c, -s], [s, c]])
    rot_corners = corners @ R.T
    xrange = rot_corners[:, 0].max() - rot_corners[:, 0].min()
    zrange = rot_corners[:, 1].max() - rot_corners[:, 1].min()
    extent = (cx - xrange / 2, cx + xrange / 2,
               cz - zrange / 2, cz + zrange / 2)
    ax.imshow(np.array(rotated), extent=extent, origin="upper", zorder=5,
              interpolation="bilinear")
    # add a thin border outline
    rect = plt.Rectangle((cx - half_w, cz - half_d), width, depth, angle=yaw_deg,
                          fill=False, edgecolor="red", linewidth=1.2, zorder=6,
                          rotation_point="center")
    ax.add_patch(rect)


def render_with_thumbnails(result: Dict, save_path: str,
                             figsize: Tuple[float, float] = (11, 9),
                             title: Optional[str] = None):
    """2D top-down plan with 3D-FUTURE thumbnails for new placements + boxes
    for existing RoomPlan furniture."""
    from ..viz.topdown import _draw_polygon, _draw_existing, _draw_doors_windows

    fig, ax = plt.subplots(figsize=figsize)
    scene = result["scene"]
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

    for p in result["placements"]:
        if p.get("status") != "placed" or "position" not in p:
            continue
        sel = p.get("selected_model") or {}
        model_id = sel.get("model_id")
        w = p["size"]["width"]; d = p["size"]["depth"]
        cx = p["position"]["x"]; cz = p["position"]["z"]
        yaw = p["rotation_y"]

        thumb = _load_thumbnail(model_id) if model_id else None
        if thumb is not None:
            _stamp_thumbnail(ax, thumb, cx, cz, w, d, yaw)
            # Small label below
            ax.annotate(sel.get("category", "")[:22], xy=(cx, cz),
                        xytext=(0, -d * 20), textcoords="offset pixels",
                        ha="center", va="top", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="red", alpha=0.85), zorder=7)
        else:
            rect = plt.Rectangle((cx - w / 2, cz - d / 2), w, d, angle=yaw,
                                  facecolor="#e1a13f", edgecolor="red",
                                  alpha=0.85, linewidth=1.5, zorder=5,
                                  rotation_point="center")
            ax.add_patch(rect)

    unplaced = [p for p in result["placements"] if p.get("status") != "placed"]
    if unplaced:
        lines = ["UNPLACED:"] + [f" • {p.get('category','?')} ({p.get('furniture_id','')})"
                                   for p in unplaced]
        ax.text(0.02, 0.02, "\n".join(lines), transform=ax.transAxes,
                fontsize=8, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeaea", edgecolor="red"))

    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return save_path
