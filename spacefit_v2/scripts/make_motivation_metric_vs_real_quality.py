"""Create a slide-ready motivation figure for metric vs. perceived quality."""
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
PNG_PATH = OUT_DIR / "motivation_metric_vs_real_quality.png"
SVG_PATH = OUT_DIR / "motivation_metric_vs_real_quality.svg"


def setup_korean_font() -> None:
    """Pick a Korean-capable system font when available."""
    candidates = [
        "Malgun Gothic",
        "맑은 고딕",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"


def rotated_rect(cx: float, cy: float, w: float, h: float, angle_deg: float) -> list[Tuple[float, float]]:
    hw, hh = w / 2, h / 2
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    points = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in points]


def draw_room(ax: Any) -> None:
    ax.set_xlim(-0.55, 8.55)
    ax.set_ylim(-0.65, 5.65)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            8,
            5,
            facecolor="#fffdf8",
            edgecolor="#30343b",
            linewidth=2.5,
            zorder=1,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (0.85, 0.12),
            5.75,
            1.05,
            facecolor="#dbeafe",
            edgecolor="#60a5fa",
            alpha=0.16,
            linestyle="--",
            linewidth=1.3,
            zorder=0,
        )
    )
    ax.text(4.0, 0.64, "이동/접근 공간", ha="center", va="center", fontsize=10.5, color="#64748b")


def draw_door(ax: Any, x: float = 0.85, width: float = 1.15) -> None:
    ax.plot([x, x + width], [0, 0], color="#fffdf8", linewidth=6, solid_capstyle="butt", zorder=4)
    ax.plot([x, x + width], [0, 0], color="#a16207", linewidth=3.3, solid_capstyle="butt", zorder=5)
    ax.add_patch(
        patches.Arc((x, 0), width * 1.45, width * 1.45, theta1=0, theta2=90, color="#ca8a04", linewidth=1.5, zorder=5)
    )
    ax.text(x + width / 2, 0.22, "문", ha="center", va="bottom", fontsize=11, color="#854d0e", weight="bold")


def draw_window(ax: Any, x: float = 3.0, width: float = 2.2) -> None:
    ax.plot([x, x + width], [5, 5], color="#fffdf8", linewidth=6, solid_capstyle="butt", zorder=4)
    ax.plot([x, x + width], [5, 5], color="#0284c7", linewidth=3.4, solid_capstyle="butt", zorder=5)
    ax.text(x + width / 2, 4.74, "창문", ha="center", va="top", fontsize=11, color="#0369a1", weight="bold")


def draw_furniture(
    ax: Any,
    cx: float,
    cy: float,
    w: float,
    h: float,
    label: str,
    angle: float = 0,
    face: str = "#e5e7eb",
    edge: str = "#6b7280",
    text_color: str = "#111827",
    linewidth: float = 1.5,
    alpha: float = 0.96,
    zorder: int = 8,
) -> None:
    ax.add_patch(
        patches.Polygon(
            rotated_rect(cx, cy, w, h, angle),
            closed=True,
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
    )
    ax.text(cx, cy, label, ha="center", va="center", fontsize=11.2, color=text_color, weight="bold", zorder=zorder + 1)


def draw_chair_with_yaw(
    ax: Any,
    cx: float,
    cy: float,
    yaw: float,
    label: str = "의자",
    face: str = "#f97316",
    edge: str = "#c2410c",
) -> None:
    draw_furniture(ax, cx, cy, 0.9, 0.72, label, yaw, face, edge, text_color="#ffffff", linewidth=2.4, zorder=14)
    theta = math.radians(yaw)
    length = 0.75
    ax.annotate(
        "",
        xy=(cx + math.cos(theta) * length, cy + math.sin(theta) * length),
        xytext=(cx, cy),
        arrowprops=dict(arrowstyle="-|>", color=edge, linewidth=2.7, shrinkA=0, shrinkB=0),
        zorder=18,
    )


def draw_badge(ax: Any, labels: Iterable[str], color: str, x: float, y: float) -> None:
    text = "\n".join(labels)
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="bottom",
        fontsize=11,
        color=color,
        weight="bold",
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.36", facecolor="#ffffff", edgecolor=color, linewidth=1.5, alpha=0.97),
        zorder=30,
    )


def draw_callout(ax: Any, text: str, xy: Tuple[float, float], xytext: Tuple[float, float], color: str) -> None:
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
        color=color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor=color, linewidth=1.5),
        arrowprops=dict(arrowstyle="->", color=color, linewidth=1.7),
        zorder=28,
    )


def draw_existing_layout(ax: Any) -> None:
    draw_furniture(ax, 1.85, 3.55, 2.35, 1.35, "침대", face="#e7dfd2", edge="#8b7d6b")
    draw_furniture(ax, 6.05, 3.88, 1.65, 0.65, "책상", face="#dbeafe", edge="#64748b")
    draw_furniture(ax, 6.85, 1.45, 1.35, 0.75, "수납장", face="#e2e8f0", edge="#64748b")


def draw_panel(ax: Any, kind: str) -> None:
    draw_room(ax)
    draw_door(ax)
    draw_window(ax)
    draw_existing_layout(ax)

    if kind == "awkward":
        ax.set_title("수치상 성공했지만 어색한 배치", fontsize=18, weight="bold", color="#b45309", pad=16)
        ax.add_patch(
            patches.Rectangle((0.25, 0.55), 1.45, 1.05, facecolor="#fed7aa", edgecolor="#f97316", alpha=0.28, linestyle="--", zorder=2)
        )
        draw_chair_with_yaw(ax, 0.98, 1.07, 102, face="#fb923c", edge="#c2410c")
        draw_callout(ax, "방향 어색", (1.0, 1.52), (2.35, 2.15), "#dc2626")
        draw_callout(ax, "가구군과 떨어짐", (1.0, 1.07), (3.15, 0.52), "#b45309")
        draw_callout(ax, "접근성 부족", (1.28, 0.74), (2.38, 0.16), "#dc2626")
        draw_badge(ax, ["CF=1", "IB=1", "조건 통과"], "#475569", 5.55, 0.25)
    else:
        ax.set_title("실사용 관점에서 자연스러운 배치", fontsize=18, weight="bold", color="#166534", pad=16)
        ax.add_patch(
            patches.Rectangle((4.35, 2.25), 2.35, 1.05, facecolor="#bbf7d0", edgecolor="#22c55e", alpha=0.24, linestyle="--", zorder=2)
        )
        draw_chair_with_yaw(ax, 5.22, 2.72, 20, face="#22c55e", edge="#15803d")
        draw_callout(ax, "기존 가구와 어울림", (5.22, 2.72), (3.5, 2.38), "#15803d")
        draw_callout(ax, "접근 가능", (5.45, 2.28), (4.12, 1.35), "#16a34a")
        draw_callout(ax, "방향 자연스러움", (5.95, 2.98), (6.86, 3.43), "#15803d")
        draw_badge(ax, ["CF=1", "IB=1", "조건 통과", "체감 품질 높음"], "#166534", 0.35, 0.25)


def main() -> None:
    setup_korean_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 8.1), dpi=180, facecolor="#ffffff")
    draw_panel(axes[0], "awkward")
    draw_panel(axes[1], "natural")
    fig.suptitle("개선이 필요한 이유와 한계점", fontsize=22, weight="bold", y=0.982, color="#111827")
    fig.text(
        0.5,
        0.045,
        "수치상 가능한 배치가 항상 좋은 배치는 아니다.",
        ha="center",
        va="bottom",
        fontsize=18,
        weight="bold",
        color="#1f2937",
    )
    fig.text(
        0.5,
        0.018,
        "방향, 접근성, 가구 간 어울림 같은 체감 품질을 반영해 top-k 후보 중 더 자연스러운 배치를 선택할 필요가 있다.",
        ha="center",
        va="bottom",
        fontsize=12.5,
        color="#475569",
    )
    fig.tight_layout(rect=(0.02, 0.075, 0.98, 0.93), w_pad=2.8)
    fig.savefig(PNG_PATH, bbox_inches="tight", facecolor="#ffffff")
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor="#ffffff")
    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    main()
