"""Create a scene-centric SpaceFit stage pipeline figure.

The figure is designed for paper and slide use. It shows how the same room
scene evolves through empty-space extraction, candidate generation, intent
constraints, filtering, reranking, and final placement.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


OUT_DIR = Path("spacefit_v2/results/presentation_figures")
PNG_PATH = OUT_DIR / "spacefit_stage_pipeline.png"
SVG_PATH = OUT_DIR / "spacefit_stage_pipeline.svg"


ROOM = {
    "x": 0.9,
    "y": 0.72,
    "w": 8.2,
    "h": 4.85,
}

FURNITURE = {
    "bed": {"cx": 2.1, "cy": 3.55, "w": 2.15, "h": 1.25, "label": "Bed"},
    "desk": {"cx": 6.25, "cy": 4.0, "w": 1.55, "h": 0.62, "label": "Desk"},
    "cabinet": {"cx": 6.75, "cy": 1.25, "w": 1.45, "h": 0.72, "label": "Cabinet"},
}

CANDIDATES = [
    {"cx": 1.4, "cy": 1.25, "angle": 90, "valid": False, "reason": "door"},
    {"cx": 3.25, "cy": 1.35, "angle": 20, "valid": True, "rank": 3, "score": 0.62},
    {"cx": 4.35, "cy": 2.25, "angle": 25, "valid": True, "rank": 2, "score": 0.74},
    {"cx": 5.35, "cy": 2.65, "angle": 165, "valid": True, "rank": 1, "score": 0.91},
    {"cx": 6.15, "cy": 3.35, "angle": 90, "valid": False, "reason": "collision"},
    {"cx": 7.55, "cy": 4.55, "angle": 0, "valid": False, "reason": "boundary"},
    {"cx": 7.05, "cy": 2.35, "angle": -45, "valid": True, "rank": 4, "score": 0.52},
    {"cx": 4.8, "cy": 4.2, "angle": -20, "valid": True, "rank": 5, "score": 0.45},
]


def set_figure_font() -> None:
    candidates = ["Arial", "DejaVu Sans", "Malgun Gothic", "Noto Sans", "Liberation Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"


def room_xy(x: float, y: float) -> Tuple[float, float]:
    return ROOM["x"] + x, ROOM["y"] + y


def rotate_rect(cx: float, cy: float, w: float, h: float, angle_deg: float) -> list[Tuple[float, float]]:
    hw, hh = w / 2.0, h / 2.0
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    points = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [(cx + px * c - py * s, cy + px * s + py * c) for px, py in points]


def draw_stage_box(ax: Any, title: str, subtitle: str, color: str) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.12, 0.15),
            9.76,
            6.7,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            facecolor="#ffffff",
            edgecolor=color,
            linewidth=1.8,
            zorder=-10,
        )
    )
    ax.text(0.42, 6.52, title, ha="left", va="center", fontsize=12.3, weight="bold", color=color)
    ax.text(0.42, 6.18, subtitle, ha="left", va="center", fontsize=8.4, color="#4b5563")


def draw_room(ax: Any, *, free_space: bool = False) -> None:
    ax.add_patch(
        patches.Rectangle(
            (ROOM["x"], ROOM["y"]),
            ROOM["w"],
            ROOM["h"],
            facecolor="#fbfdff",
            edgecolor="#2f343b",
            linewidth=1.65,
            zorder=0,
        )
    )
    if free_space:
        draw_free_space(ax)
    draw_window(ax)
    draw_door(ax)
    draw_existing_furniture(ax)


def draw_door(ax: Any) -> None:
    x0, y0 = room_xy(0.72, 0)
    x1, _ = room_xy(1.82, 0)
    ax.plot([x0, x1], [y0, y0], color="#ffffff", linewidth=5, solid_capstyle="butt", zorder=7)
    ax.plot([x0, x1], [y0, y0], color="#b98549", linewidth=2.2, solid_capstyle="butt", zorder=8)
    ax.add_patch(patches.Arc((x0, y0), 1.22, 1.22, theta1=0, theta2=90, color="#c79a65", linewidth=1.0, zorder=6))
    ax.text((x0 + x1) / 2, y0 + 0.22, "Door", ha="center", va="bottom", fontsize=6.8, color="#8a5526", weight="bold")


def draw_window(ax: Any) -> None:
    x0, y = room_xy(3.25, ROOM["h"])
    x1, _ = room_xy(5.55, ROOM["h"])
    ax.plot([x0, x1], [y, y], color="#ffffff", linewidth=5, solid_capstyle="butt", zorder=7)
    ax.plot([x0, x1], [y, y], color="#2f80c1", linewidth=2.4, solid_capstyle="butt", zorder=8)
    ax.text((x0 + x1) / 2, y - 0.22, "Window", ha="center", va="top", fontsize=6.8, color="#1f6698", weight="bold")


def draw_furniture(
    ax: Any,
    cx: float,
    cy: float,
    w: float,
    h: float,
    label: str,
    *,
    angle: float = 0.0,
    face: str = "#dfeaf6",
    edge: str = "#607f9d",
    text: str = "#111827",
    alpha: float = 0.97,
    linewidth: float = 1.25,
    zorder: int = 12,
) -> None:
    px, py = room_xy(cx, cy)
    poly = patches.Polygon(
        rotate_rect(px, py, w, h, angle),
        closed=True,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(poly)
    ax.text(px, py, label, ha="center", va="center", fontsize=6.9, weight="bold", color=text, zorder=zorder + 1)


def draw_existing_furniture(ax: Any, highlights: Mapping[str, str] | None = None) -> None:
    highlights = highlights or {}
    for key, item in FURNITURE.items():
        edge = highlights.get(key, "#607f9d")
        face = "#e6eef8" if key != "cabinet" else "#e4e8ed"
        lw = 2.2 if key in highlights else 1.25
        draw_furniture(ax, item["cx"], item["cy"], item["w"], item["h"], item["label"], face=face, edge=edge, linewidth=lw)


def draw_occupied_overlay(ax: Any) -> None:
    for item in FURNITURE.values():
        px, py = room_xy(item["cx"] - item["w"] / 2, item["cy"] - item["h"] / 2)
        ax.add_patch(
            patches.Rectangle(
                (px, py),
                item["w"],
                item["h"],
                facecolor="#6b7280",
                edgecolor="none",
                alpha=0.18,
                zorder=3,
            )
        )


def draw_free_space(ax: Any) -> None:
    ax.add_patch(
        patches.Rectangle(
            room_xy(0.32, 0.42),
            7.55,
            4.0,
            facecolor="#d9fbe5",
            edgecolor="#86cfa2",
            linewidth=0.7,
            alpha=0.24,
            zorder=1,
        )
    )
    for x in [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2]:
        for y in [1.0, 1.8, 2.6, 3.4, 4.2]:
            ax.scatter(*room_xy(x, y), s=5, color="#71b989", alpha=0.42, zorder=2)


def draw_door_clearance(ax: Any) -> None:
    ax.add_patch(
        patches.Rectangle(
            room_xy(0.55, 0.06),
            1.85,
            0.96,
            facecolor="#f97316",
            edgecolor="#ea580c",
            linewidth=0.8,
            alpha=0.18,
            linestyle="--",
            zorder=4,
        )
    )


def draw_target_candidate(
    ax: Any,
    cx: float,
    cy: float,
    angle: float,
    *,
    label: str = "",
    face: str = "#f6d365",
    edge: str = "#d97706",
    alpha: float = 0.9,
    linewidth: float = 1.25,
    arrow_color: str | None = None,
    zorder: int = 18,
) -> None:
    px, py = room_xy(cx, cy)
    poly = patches.Polygon(
        rotate_rect(px, py, 0.58, 0.42, angle),
        closed=True,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(poly)
    if label:
        ax.text(px, py, label, ha="center", va="center", fontsize=6.3, color=edge, weight="bold", zorder=zorder + 2)
    color = arrow_color or edge
    theta = math.radians(angle)
    ax.annotate(
        "",
        xy=(px + math.cos(theta) * 0.42, py + math.sin(theta) * 0.42),
        xytext=(px, py),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.25, shrinkA=0, shrinkB=0),
        zorder=zorder + 1,
    )


def draw_invalid_x(ax: Any, cx: float, cy: float, label: str) -> None:
    px, py = room_xy(cx, cy)
    ax.text(px, py + 0.42, "X", ha="center", va="center", fontsize=10.5, color="#dc2626", weight="bold", zorder=30)
    ax.text(px, py + 0.72, label, ha="center", va="center", fontsize=5.4, color="#dc2626", zorder=31)


def draw_constraints_box(ax: Any) -> None:
    instruction = "User intent\n\"Place the chair near the desk,\n face the bed, and do not\n block the door.\""
    constraints = "Placement constraints\n- near(desk)\n- facing(bed)\n- not_block(door)"
    ax.add_patch(
        patches.FancyBboxPatch((0.62, 4.82), 3.95, 1.0, boxstyle="round,pad=0.08", fc="#f8fafc", ec="#94a3b8", lw=1.1, zorder=40)
    )
    ax.text(0.78, 5.32, instruction, ha="left", va="center", fontsize=5.8, color="#111827", zorder=41)
    ax.add_patch(
        patches.FancyBboxPatch((5.05, 4.82), 3.95, 1.0, boxstyle="round,pad=0.08", fc="#ecfdf5", ec="#16a34a", lw=1.1, zorder=40)
    )
    ax.text(5.22, 5.32, constraints, ha="left", va="center", fontsize=5.8, color="#14532d", zorder=41)
    ax.annotate(
        "",
        xy=(4.94, 5.32),
        xytext=(4.62, 5.32),
        arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2),
        zorder=42,
    )


def draw_topk_panel(ax: Any) -> None:
    x0, y0 = 6.2, 3.18
    ax.add_patch(
        patches.FancyBboxPatch((x0, y0), 2.72, 1.95, boxstyle="round,pad=0.08", fc="#ffffff", ec="#9ca3af", lw=1.0, zorder=40)
    )
    ax.text(x0 + 0.14, y0 + 1.72, "Top-k reranking", ha="left", va="center", fontsize=6.9, weight="bold", color="#111827", zorder=41)
    rows = [("Rank 1", 0.91, "#16a34a"), ("Rank 2", 0.74, "#65a30d"), ("Rank 3", 0.62, "#ca8a04")]
    for i, (name, score, color) in enumerate(rows):
        yy = y0 + 1.28 - i * 0.45
        ax.text(x0 + 0.15, yy, name, ha="left", va="center", fontsize=5.8, color="#374151", zorder=42)
        ax.add_patch(patches.Rectangle((x0 + 0.85, yy - 0.08), 1.45, 0.16, fc="#e5e7eb", ec="none", zorder=41))
        ax.add_patch(patches.Rectangle((x0 + 0.85, yy - 0.08), 1.45 * score, 0.16, fc=color, ec="none", zorder=42))
        ax.text(x0 + 2.43, yy, f"{score:.2f}", ha="center", va="center", fontsize=5.5, color="#374151", zorder=42)


def draw_stage1(ax: Any) -> None:
    draw_room(ax, free_space=True)
    draw_occupied_overlay(ax)
    draw_door_clearance(ax)
    ax.text(1.0, 5.82, "Input: furnished room + target chair + user intent", fontsize=6.5, color="#334155", zorder=45)
    ax.text(1.15, 1.25, "Free space", fontsize=6.2, color="#15803d", weight="bold")
    ax.text(6.95, 5.0, "Occupied area", fontsize=5.7, color="#4b5563")
    ax.text(1.15, 1.58, "Door clearance", fontsize=5.7, color="#c2410c")


def draw_stage2(ax: Any) -> None:
    draw_room(ax, free_space=True)
    for idx, cand in enumerate(CANDIDATES, start=1):
        draw_target_candidate(ax, cand["cx"], cand["cy"], cand["angle"], label=str(idx), face="#fde68a", edge="#d97706", alpha=0.82)
    ax.text(1.08, 5.82, "Sample many positions and orientations", fontsize=6.5, color="#334155")


def draw_stage3(ax: Any) -> None:
    draw_room(ax, free_space=False)
    draw_constraints_box(ax)
    draw_existing_furniture(ax, highlights={"desk": "#2563eb", "bed": "#7c3aed"})
    px, py = room_xy(1.15, 0.5)
    ax.add_patch(patches.Rectangle((px, py), 1.86, 0.86, fill=False, ec="#dc2626", lw=1.3, linestyle="--", zorder=30))
    ax.text(px + 0.93, py + 0.43, "door", ha="center", va="center", fontsize=5.8, color="#dc2626", weight="bold", zorder=31)
    ax.text(1.05, 0.3, "Anchor objects are highlighted", fontsize=5.6, color="#64748b")


def draw_stage4(ax: Any) -> None:
    draw_room(ax, free_space=True)
    draw_door_clearance(ax)
    for cand in CANDIDATES:
        if cand["valid"]:
            draw_target_candidate(ax, cand["cx"], cand["cy"], cand["angle"], face="#bbf7d0", edge="#16a34a", alpha=0.9)
        else:
            draw_target_candidate(ax, cand["cx"], cand["cy"], cand["angle"], face="#fecaca", edge="#dc2626", alpha=0.55)
            reason = {"door": "door blocking", "collision": "collision", "boundary": "out of boundary"}.get(cand["reason"], "invalid")
            draw_invalid_x(ax, cand["cx"], cand["cy"], reason)
    ax.text(1.0, 5.82, "Remove physically invalid candidates", fontsize=6.5, color="#334155")


def draw_stage5(ax: Any) -> None:
    draw_room(ax, free_space=False)
    for cand in CANDIDATES:
        if cand["valid"] and cand.get("rank", 9) <= 3:
            color = "#15803d" if cand["rank"] == 1 else "#65a30d"
            draw_target_candidate(ax, cand["cx"], cand["cy"], cand["angle"], label=f"#{cand['rank']}", face="#dcfce7", edge=color, alpha=0.88, linewidth=1.5)
    draw_topk_panel(ax)
    ax.annotate(
        "preferred\nafter reranking",
        xy=room_xy(5.35, 2.65),
        xytext=(3.3, 5.25),
        ha="center",
        va="center",
        fontsize=5.8,
        color="#15803d",
        bbox=dict(boxstyle="round,pad=0.24", fc="#ffffff", ec="#15803d", lw=1.0),
        arrowprops=dict(arrowstyle="->", color="#15803d", lw=1.1),
        zorder=50,
    )


def draw_stage6(ax: Any) -> None:
    draw_room(ax, free_space=False)
    draw_target_candidate(ax, 5.35, 2.65, 165, label="Chair", face="#86efac", edge="#15803d", alpha=0.96, linewidth=1.9)
    callouts = [
        ("near desk", room_xy(5.65, 2.72), (6.9, 3.6)),
        ("facing bed", room_xy(5.1, 2.75), (3.55, 3.0)),
        ("door clear", room_xy(1.35, 0.45), (3.25, 1.05)),
    ]
    for text, xy, xytext in callouts:
        ax.annotate(
            text,
            xy=xy,
            xytext=xytext,
            ha="center",
            va="center",
            fontsize=6.0,
            color="#15803d",
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#15803d", lw=1.0),
            arrowprops=dict(arrowstyle="->", color="#15803d", lw=1.0),
            zorder=50,
        )
    ax.text(1.0, 5.82, "Single final placement with pose", fontsize=6.5, color="#334155")


def add_flow_arrows(fig: Any, axes: Sequence[Any]) -> None:
    for left, right in zip(axes[:-1], axes[1:]):
        lbox = left.get_position()
        rbox = right.get_position()
        y = lbox.y0 + 0.5 * lbox.height
        if rbox.x0 > lbox.x1:
            x0 = lbox.x1 + 0.004
            x1 = rbox.x0 - 0.004
        else:
            mid = (lbox.x1 + rbox.x0) / 2.0
            x0 = mid - 0.006
            x1 = mid + 0.006
        arrow = FancyArrowPatch(
            (x0, y),
            (x1, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.15,
            color="#64748b",
            zorder=100,
        )
        fig.add_artist(arrow)


def main() -> None:
    set_figure_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 6, figsize=(21.5, 6.4), dpi=180, facecolor="#ffffff")
    stages = [
        ("1. Room Analysis", "empty-space extraction", "#2563eb", draw_stage1),
        ("2. Candidates", "position + orientation", "#d97706", draw_stage2),
        ("3. Intent Parsing", "text to constraints", "#7c3aed", draw_stage3),
        ("4. Physical Filter", "collision / boundary / door", "#dc2626", draw_stage4),
        ("5. Top-k Rerank", "constraint + human-aligned score", "#0f766e", draw_stage5),
        ("6. Final Placement", "selected target pose", "#15803d", draw_stage6),
    ]
    for ax, (title, subtitle, color, draw_fn) in zip(axes, stages):
        draw_stage_box(ax, title, subtitle, color)
        draw_fn(ax)

    fig.suptitle(
        "SpaceFit: Scene-Centric Pipeline for Single-Target Furniture Placement",
        fontsize=18,
        weight="bold",
        color="#111827",
        y=0.985,
    )
    fig.text(
        0.5,
        0.035,
        "The same room is preserved across stages; target candidates are generated, filtered, reranked, and reduced to one final placement.",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    fig.tight_layout(rect=(0.012, 0.07, 0.988, 0.93), w_pad=0.65)
    add_flow_arrows(fig, axes)
    fig.savefig(PNG_PATH, bbox_inches="tight", facecolor="#ffffff")
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor="#ffffff")
    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    main()
