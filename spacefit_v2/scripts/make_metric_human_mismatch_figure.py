"""Create a slide-ready figure for metric-human mismatch in placement."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt


OUT_DIR = Path("spacefit_v2/results/presentation_figures")
PNG_PATH = OUT_DIR / "metric_possible_but_awkward.png"
SVG_PATH = OUT_DIR / "metric_possible_but_awkward.svg"


def set_korean_font() -> None:
    """Pick a Korean-capable system font when available."""
    candidates = [
        "Malgun Gothic",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "맑은 고딕",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"


def rotate_rect(cx: float, cy: float, w: float, h: float, angle_deg: float) -> list[Tuple[float, float]]:
    hw, hh = w / 2, h / 2
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    pts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts]


def draw_room(ax: Any) -> None:
    ax.set_xlim(-0.35, 8.35)
    ax.set_ylim(-0.35, 5.35)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            8,
            5,
            facecolor="#ffffff",
            edgecolor="#30343b",
            linewidth=2.4,
            zorder=1,
        )
    )
    ax.text(4, -0.27, "Door", ha="center", va="top", fontsize=10, color="#6b7280")


def draw_door(ax: Any) -> None:
    ax.plot([0.8, 2.0], [0, 0], color="#ffffff", linewidth=5, solid_capstyle="butt", zorder=3)
    ax.plot([0.8, 2.0], [0, 0], color="#b77a40", linewidth=3.2, solid_capstyle="butt", zorder=4)
    ax.add_patch(patches.Arc((0.8, 0), 1.8, 1.8, theta1=0, theta2=90, color="#c99a65", linewidth=1.3, zorder=3))
    ax.text(1.4, 0.24, "문", ha="center", va="bottom", fontsize=11, color="#8a5526", weight="bold")


def draw_window(ax: Any) -> None:
    ax.plot([3.05, 5.35], [5, 5], color="#ffffff", linewidth=5, solid_capstyle="butt", zorder=3)
    ax.plot([3.05, 5.35], [5, 5], color="#2f80c1", linewidth=3.2, solid_capstyle="butt", zorder=4)
    ax.text(4.2, 4.76, "창문", ha="center", va="top", fontsize=11, color="#1f6698", weight="bold")


def draw_access_area(ax: Any, xy: Tuple[float, float], w: float, h: float, color: str, label: str | None = None) -> None:
    ax.add_patch(
        patches.Rectangle(
            xy,
            w,
            h,
            facecolor=color,
            edgecolor=color,
            alpha=0.13,
            linestyle="--",
            linewidth=1.2,
            zorder=0,
        )
    )
    if label:
        ax.text(xy[0] + w / 2, xy[1] + h / 2, label, ha="center", va="center", fontsize=10, color=color)


def draw_furniture(
    ax: Any,
    cx: float,
    cy: float,
    w: float,
    h: float,
    angle: float,
    label: str,
    face: str,
    edge: str,
    text_color: str = "#111827",
    linewidth: float = 1.5,
    alpha: float = 0.95,
    zorder: int = 5,
) -> None:
    poly = patches.Polygon(
        rotate_rect(cx, cy, w, h, angle),
        closed=True,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(poly)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=11, color=text_color, weight="bold", zorder=zorder + 1)


def draw_direction_arrow(ax: Any, cx: float, cy: float, angle: float, color: str) -> None:
    length = 0.72
    theta = math.radians(angle)
    dx, dy = math.cos(theta) * length, math.sin(theta) * length
    ax.annotate(
        "",
        xy=(cx + dx, cy + dy),
        xytext=(cx, cy),
        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=2.4, shrinkA=0, shrinkB=0),
        zorder=9,
    )


def draw_existing_furniture(ax: Any) -> None:
    draw_furniture(ax, 1.85, 3.52, 2.25, 1.35, 0, "Bed", "#dce8f7", "#5b7fa6")
    draw_furniture(ax, 6.0, 3.95, 1.65, 0.62, 0, "Desk", "#d8e6ef", "#53788f")
    draw_furniture(ax, 6.0, 3.02, 0.76, 0.72, 0, "Chair", "#e8edf3", "#7a8998")
    draw_furniture(ax, 6.82, 1.15, 1.45, 0.78, 0, "Cabinet", "#e1e5ea", "#7b828a")


def draw_callout(ax: Any, text: str, xy: Tuple[float, float], xytext: Tuple[float, float], color: str) -> None:
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        color=color,
        bbox=dict(boxstyle="round,pad=0.28", fc="#ffffff", ec=color, lw=1.5),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.6),
        zorder=20,
    )


def draw_badges(ax: Any, labels: Iterable[str], color: str, x: float, y: float) -> None:
    for i, label in enumerate(labels):
        prefix = "OK" if color.startswith("#2") or color.startswith("#1") else "!"
        ax.text(
            x,
            y - i * 0.36,
            f"{prefix} {label}",
            ha="left",
            va="center",
            fontsize=11,
            color=color,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.24", fc="#ffffff", ec=color, lw=1.0, alpha=0.95),
            zorder=30,
        )


def draw_panel(ax: Any, kind: str) -> None:
    draw_room(ax)
    draw_access_area(ax, (0.8, 0.1), 5.8, 1.0, "#6b7280", "이동/접근 공간")
    draw_door(ax)
    draw_window(ax)
    draw_existing_furniture(ax)

    if kind == "awkward":
        ax.set_title("수치상 가능하지만 어색한 배치", fontsize=18, weight="bold", color="#b54708", pad=14)
        draw_access_area(ax, (0.25, 0.65), 1.35, 0.92, "#d97706")
        draw_furniture(ax, 0.92, 1.08, 1.22, 0.78, 92, "Target", "#f7b267", "#c2410c", text_color="#7c2d12", linewidth=2.2)
        draw_direction_arrow(ax, 0.92, 1.08, 92, "#c2410c")
        draw_callout(ax, "방향 어색", (0.92, 1.54), (2.25, 1.92), "#c2410c")
        draw_callout(ax, "가구군과 떨어짐", (0.95, 1.08), (3.0, 0.68), "#b45309")
        draw_callout(ax, "접근성 부족", (1.16, 0.70), (2.24, 0.25), "#dc2626")
        draw_badges(ax, ["충돌 없음", "경계 안 배치"], "#4b5563", 5.35, 0.42)
    else:
        ax.set_title("사람이 보기에도 자연스러운 배치", fontsize=18, weight="bold", color="#166534", pad=14)
        draw_access_area(ax, (4.55, 2.15), 2.15, 1.12, "#16a34a")
        draw_furniture(ax, 5.18, 2.58, 1.22, 0.78, 18, "Target", "#86efac", "#15803d", text_color="#14532d", linewidth=2.2)
        draw_direction_arrow(ax, 5.18, 2.58, 18, "#15803d")
        draw_callout(ax, "기존 가구와 어울림", (5.18, 2.58), (3.55, 2.3), "#15803d")
        draw_callout(ax, "접근 가능", (5.55, 2.25), (4.22, 1.28), "#16a34a")
        draw_callout(ax, "방향 자연스러움", (5.78, 2.77), (6.78, 3.45), "#15803d")
        draw_badges(ax, ["충돌 없음", "접근 가능", "방향 자연스러움"], "#166534", 0.38, 0.72)


def main() -> None:
    set_korean_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 8.1), dpi=180, facecolor="#ffffff")
    draw_panel(axes[0], "awkward")
    draw_panel(axes[1], "natural")
    fig.suptitle(
        "자동 지표와 사람 판단의 차이: 가능한 배치와 자연스러운 배치는 다를 수 있음",
        fontsize=20,
        weight="bold",
        y=0.985,
        color="#111827",
    )
    fig.text(
        0.5,
        0.035,
        "두 패널은 동일한 방 구조와 기존 가구를 사용하며, Target Furniture의 위치와 방향만 다릅니다.",
        ha="center",
        fontsize=12,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.94), w_pad=2.4)
    fig.savefig(PNG_PATH, bbox_inches="tight", facecolor="#ffffff")
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor="#ffffff")
    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    main()
