"""Create a stage-by-stage SpaceFit output visualization.

This figure focuses on what is produced at each stage of the current method:
scene normalization, intent grounding, free-space extraction, candidate
generation, filtering, scoring, human-aligned reranking, and final evaluation.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


OUT_DIR = Path("spacefit_v2/results/presentation_figures")
PNG_PATH = OUT_DIR / "spacefit_stage_outputs.png"
SVG_PATH = OUT_DIR / "spacefit_stage_outputs.svg"
MD_PATH = Path("spacefit_v2/results/SPACEFIT_STAGE_OUTPUTS.md")


ROOM = {"x": 0.55, "y": 0.78, "w": 4.65, "h": 3.15}
FURNITURE = {
    "bed": (1.35, 2.85, 1.35, 0.62, "Bed"),
    "desk": (3.75, 2.95, 0.82, 0.42, "Desk"),
    "cabinet": (4.02, 1.22, 0.78, 0.42, "Cab."),
}
CANDIDATES = [
    (1.05, 1.20, 20, False),
    (1.80, 1.18, 90, True),
    (2.35, 1.55, 40, True),
    (2.85, 2.05, 170, True),
    (3.58, 2.35, -20, False),
    (4.55, 3.55, 0, False),
    (3.90, 1.75, -50, True),
]


def setup_font() -> None:
    candidates = ["Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "Arial", "DejaVu Sans"]
    available = {font.name for font in fm.fontManager.ttflist}
    for family in candidates:
        if family in available:
            plt.rcParams["font.family"] = family
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"


def room_xy(x: float, y: float) -> Tuple[float, float]:
    return ROOM["x"] + x, ROOM["y"] + y


def rotated_rect(cx: float, cy: float, w: float, h: float, angle: float) -> list[Tuple[float, float]]:
    hw, hh = w / 2, h / 2
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return [(cx + dx * c - dy * s, cy + dx * s + dy * c) for dx, dy in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]]


def draw_stage_frame(ax: Any, idx: int, title: str, output: str, color: str) -> None:
    ax.set_xlim(0, 6.1)
    ax.set_ylim(0, 5.05)
    ax.axis("off")
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.08, 0.08),
            5.94,
            4.88,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            facecolor="#ffffff",
            edgecolor=color,
            linewidth=1.8,
            zorder=-20,
        )
    )
    ax.text(0.28, 4.66, f"{idx}. {title}", ha="left", va="center", fontsize=12, weight="bold", color=color)
    ax.text(
        0.28,
        0.30,
        f"Output: {output}",
        ha="left",
        va="center",
        fontsize=8.6,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#f8fafc", edgecolor="#e2e8f0", linewidth=0.8),
    )


def draw_room_base(ax: Any, *, target: bool = False, free: bool = False, occupied: bool = False) -> None:
    ax.add_patch(
        patches.Rectangle(
            (ROOM["x"], ROOM["y"]),
            ROOM["w"],
            ROOM["h"],
            facecolor="#fbfdff",
            edgecolor="#334155",
            linewidth=1.5,
            zorder=0,
        )
    )
    if free:
        ax.add_patch(
            patches.Rectangle(
                room_xy(0.28, 0.42),
                4.0,
                2.28,
                facecolor="#dcfce7",
                edgecolor="#86efac",
                linewidth=0.8,
                alpha=0.48,
                zorder=1,
            )
        )
        for gx in [0.9, 1.5, 2.1, 2.7, 3.3, 3.9]:
            for gy in [1.0, 1.6, 2.2, 2.8]:
                ax.scatter(*room_xy(gx, gy), s=6, color="#16a34a", alpha=0.35, zorder=2)
    draw_door_window(ax)
    for key, (cx, cy, w, h, label) in FURNITURE.items():
        face = "#dbeafe" if key != "cabinet" else "#e2e8f0"
        ax.add_patch(
            patches.Rectangle(
                room_xy(cx - w / 2, cy - h / 2),
                w,
                h,
                facecolor=face,
                edgecolor="#64748b",
                linewidth=1.05,
                alpha=0.9,
                zorder=5,
            )
        )
        ax.text(*room_xy(cx, cy), label, ha="center", va="center", fontsize=6.4, color="#0f172a", zorder=6)
        if occupied:
            ax.add_patch(
                patches.Rectangle(
                    room_xy(cx - w / 2 - 0.05, cy - h / 2 - 0.05),
                    w + 0.10,
                    h + 0.10,
                    facecolor="#ef4444",
                    edgecolor="none",
                    alpha=0.16,
                    zorder=4,
                )
            )
    if occupied:
        ax.add_patch(
            patches.Rectangle(
                room_xy(0.28, 0.02),
                1.15,
                0.62,
                facecolor="#f97316",
                edgecolor="#ea580c",
                linewidth=0.8,
                linestyle="--",
                alpha=0.22,
                zorder=4,
            )
        )
    if target:
        draw_candidate(ax, 2.65, 1.75, 35, label="T", color="#22c55e", edge="#166534", alpha=0.95)


def draw_door_window(ax: Any) -> None:
    x0, y0 = room_xy(0.35, 0.0)
    x1, _ = room_xy(1.05, 0.0)
    ax.plot([x0, x1], [y0, y0], color="#b45309", linewidth=2.2, zorder=8)
    ax.text((x0 + x1) / 2, y0 + 0.16, "Door", ha="center", va="bottom", fontsize=5.8, color="#92400e")
    wx0, wy = room_xy(1.85, ROOM["h"])
    wx1, _ = room_xy(3.25, ROOM["h"])
    ax.plot([wx0, wx1], [wy, wy], color="#0284c7", linewidth=2.4, zorder=8)
    ax.text((wx0 + wx1) / 2, wy - 0.16, "Window", ha="center", va="top", fontsize=5.8, color="#0369a1")


def draw_candidate(ax: Any, cx: float, cy: float, yaw: float, *, label: str = "", color: str = "#fbbf24", edge: str = "#d97706", alpha: float = 0.82, invalid: bool = False, rank: str = "") -> None:
    px, py = room_xy(cx, cy)
    ax.add_patch(
        patches.Polygon(
            rotated_rect(px, py, 0.42, 0.30, yaw),
            closed=True,
            facecolor=color,
            edgecolor=edge,
            linewidth=1.2,
            alpha=alpha,
            zorder=15,
        )
    )
    rad = math.radians(yaw)
    ax.annotate(
        "",
        xy=(px + math.cos(rad) * 0.30, py + math.sin(rad) * 0.30),
        xytext=(px, py),
        arrowprops=dict(arrowstyle="-|>", color=edge, lw=1.05, shrinkA=0, shrinkB=0),
        zorder=16,
    )
    if label or rank:
        ax.text(px, py, label or rank, ha="center", va="center", fontsize=6.2, color="#111827", weight="bold", zorder=17)
    if invalid:
        ax.text(px, py + 0.36, "X", ha="center", va="center", fontsize=11, color="#dc2626", weight="bold", zorder=18)


def draw_arrow_between(fig: Any, ax_from: Any, ax_to: Any) -> None:
    b1 = ax_from.get_position()
    b2 = ax_to.get_position()
    start = (b1.x1 + 0.004, (b1.y0 + b1.y1) / 2)
    end = (b2.x0 - 0.004, (b2.y0 + b2.y1) / 2)
    fig.add_artist(
        FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color="#94a3b8",
            zorder=30,
        )
    )


def draw_down_arrow(fig: Any, ax_from: Any, ax_to: Any) -> None:
    b1 = ax_from.get_position()
    b2 = ax_to.get_position()
    start = ((b1.x0 + b1.x1) / 2, b1.y0 - 0.008)
    end = ((b2.x0 + b2.x1) / 2, b2.y1 + 0.008)
    fig.add_artist(
        FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color="#94a3b8",
            zorder=30,
        )
    )


def draw_text_block(ax: Any, lines: Iterable[str], x: float, y: float, *, color: str = "#0f172a", face: str = "#f8fafc", size: float = 8.5) -> None:
    ax.text(
        x,
        y,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=size,
        color=color,
        linespacing=1.38,
        bbox=dict(boxstyle="round,pad=0.38", facecolor=face, edgecolor="#cbd5e1", linewidth=0.9),
    )


def build_figure() -> None:
    setup_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(18.5, 13.2))
    fig.patch.set_facecolor("white")
    axes = axes.reshape(3, 3)

    colors = ["#2563eb", "#7c3aed", "#059669", "#0f766e", "#f97316", "#dc2626", "#9333ea", "#16a34a", "#334155"]

    # 0 Input
    ax = axes[0, 0]
    draw_stage_frame(ax, 0, "Input package", "room + fixed furniture + target + user intent", colors[0])
    draw_room_base(ax)
    draw_candidate(ax, -0.08, 1.70, 0, label="T", color="#22c55e", edge="#166534")
    draw_text_block(ax, ["Target: chair", "Intent: near desk", "face bed", "do not block door"], 3.95, 4.02, size=7.6)

    # 1 Scene normalization
    ax = axes[0, 1]
    draw_stage_frame(ax, 1, "Scene normalization", "target removed, fixed scene representation", colors[1])
    draw_room_base(ax)
    ax.text(3.05, 4.08, "target is handled\nas a separate asset", ha="center", va="center", fontsize=8.3, color="#475569")

    # 2 Intent grounding
    ax = axes[0, 2]
    draw_stage_frame(ax, 2, "Intent grounding", "structured constraints", colors[2])
    draw_room_base(ax)
    draw_text_block(ax, ['near("desk")', 'facing("bed")', 'not_block("door")', 'access_zone()'], 3.45, 4.10, color="#064e3b", face="#ecfdf5", size=8.2)

    # 3 Free-space extraction
    ax = axes[1, 0]
    draw_stage_frame(ax, 3, "Free-space extraction", "occupancy map + candidate regions", colors[3])
    draw_room_base(ax, free=True, occupied=True)
    ax.text(3.15, 1.05, "free regions", ha="center", va="center", fontsize=9, weight="bold", color="#166534")

    # 4 Candidate generation
    ax = axes[1, 1]
    draw_stage_frame(ax, 4, "Candidate generation", "many (x, z, yaw) poses", colors[4])
    draw_room_base(ax, free=True)
    for i, (cx, cy, yaw, valid) in enumerate(CANDIDATES, 1):
        draw_candidate(ax, cx, cy, yaw, label="", color="#fed7aa", edge="#ea580c", alpha=0.72)
    draw_text_block(ax, ["grid step: 0.18m", "yaw candidates: 16", "plus wall-aligned yaw"], 3.50, 4.10, color="#7c2d12", face="#fff7ed", size=7.7)

    # 5 Physical filtering
    ax = axes[1, 2]
    draw_stage_frame(ax, 5, "Physical filtering", "feasible candidate set", colors[5])
    draw_room_base(ax, free=True, occupied=True)
    for cx, cy, yaw, valid in CANDIDATES:
        draw_candidate(
            ax,
            cx,
            cy,
            yaw,
            color="#bbf7d0" if valid else "#fecaca",
            edge="#16a34a" if valid else "#dc2626",
            alpha=0.75,
            invalid=not valid,
        )

    # 6 Constraint scoring
    ax = axes[2, 0]
    draw_stage_frame(ax, 6, "Constraint scoring + Top-k", "ranked top-k candidate list", colors[6])
    draw_room_base(ax)
    ranked = [(2.85, 2.05, 170, "1"), (2.35, 1.55, 40, "2"), (1.80, 1.18, 90, "3"), (3.90, 1.75, -50, "4")]
    for cx, cy, yaw, rank in ranked:
        draw_candidate(ax, cx, cy, yaw, rank=rank, color="#ddd6fe", edge="#7c3aed", alpha=0.88)
    draw_text_block(ax, ["1: score 0.91", "2: score 0.74", "3: score 0.62", "4: score 0.52"], 3.78, 4.10, color="#581c87", face="#faf5ff", size=7.9)

    # 7 Rerank
    ax = axes[2, 1]
    draw_stage_frame(ax, 7, "Human-aligned rerank", "RF quality score + reordered top-k", colors[7])
    draw_room_base(ax)
    draw_candidate(ax, 2.35, 1.55, 40, rank="1", color="#22c55e", edge="#166534", alpha=0.94)
    draw_candidate(ax, 2.85, 2.05, 170, rank="2", color="#bbf7d0", edge="#16a34a", alpha=0.72)
    draw_text_block(ax, ["features:", "CF / IB / CA", "distance / access", "human label pattern", "", "RF -> P(overall_ok)"], 3.48, 4.18, color="#14532d", face="#f0fdf4", size=7.3)

    # 8 Final & judge
    ax = axes[2, 2]
    draw_stage_frame(ax, 8, "Final output + evaluation", "final pose + metrics + VLM quality report", colors[8])
    draw_room_base(ax, target=True)
    draw_text_block(
        ax,
        ['position: (x, y, z)', "yaw: rotation_y", "status: placed", "", "CF / IB / CPS", "VLM naturalness"],
        3.45,
        4.15,
        color="#0f172a",
        face="#f8fafc",
        size=7.5,
    )

    # Arrows
    for c in range(2):
        draw_arrow_between(fig, axes[0, c], axes[0, c + 1])
        draw_arrow_between(fig, axes[1, c], axes[1, c + 1])
        draw_arrow_between(fig, axes[2, c], axes[2, c + 1])
    draw_down_arrow(fig, axes[0, 2], axes[1, 2])
    draw_down_arrow(fig, axes[1, 0], axes[2, 0])
    # Add a subtle note for row transition reading order.
    fig.text(0.5, 0.042, "Reading order: input scene -> grounded constraints -> free-space/candidates -> filtering/scoring -> rerank -> final placement", ha="center", fontsize=12, color="#475569")

    fig.suptitle("SpaceFit Method Flow: Stage-wise Outputs", fontsize=24, weight="bold", y=0.984)
    fig.tight_layout(rect=[0.015, 0.055, 0.985, 0.955], h_pad=1.7, w_pad=1.0)
    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_markdown() -> None:
    lines = [
        "# SpaceFit Stage-wise Outputs",
        "",
        "이 문서는 현재 SpaceFit 방법론을 발표에서 설명하기 쉽도록 stage별 입력/출력 중심으로 정리한 것이다.",
        "",
        "## 핵심 흐름",
        "",
        "| Stage | 처리 내용 | 출력물 |",
        "|---:|---|---|",
        "| 0 | 방 구조, 기존 가구, target furniture, user intent 입력 | `input package` |",
        "| 1 | target을 기존 장면에서 분리하고 fixed scene으로 정규화 | `normalized scene` |",
        "| 2 | 자연어 의도를 solver가 읽을 수 있는 구조화 제약으로 변환 | `constraint list` |",
        "| 3 | 기존 가구, 문, 창문, 방 경계를 기반으로 점유/빈 공간 계산 | `occupancy map`, `free-space regions` |",
        "| 4 | 빈 공간에서 여러 x/z/yaw 후보 생성 | `candidate poses` |",
        "| 5 | 충돌, 경계 이탈, 문 차단 등 물리적으로 불가능한 후보 제거 | `feasible candidate set` |",
        "| 6 | 각 후보에 대해 near/facing/access 등의 제약 점수 계산 후 top-k 선택 | `ranked top-k candidates` |",
        "| 7 | human visual label로 학습한 Random Forest scorer가 top-k를 재정렬 | `human-aligned top-k` |",
        "| 8 | 최종 top-1 pose를 출력하고 CF/IB/CPS/VLM judge로 평가 | `final placement`, `evaluation report` |",
        "",
        "## 발표용 설명",
        "",
        "> SpaceFit은 LLM이 직접 좌표를 찍는 방식이 아니라, 방 안의 가능한 빈 공간에서 여러 후보 pose를 만들고, 물리 필터와 의도 제약 점수로 top-k를 고른 뒤, 사람 정성평가 기준을 학습한 scorer로 다시 정렬해 최종 배치를 선택한다.",
        "",
        "## 생성 파일",
        "",
        f"- `{PNG_PATH}`",
        f"- `{SVG_PATH}`",
    ]
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    build_figure()
    write_markdown()
    print(PNG_PATH)
    print(SVG_PATH)
    print(MD_PATH)


if __name__ == "__main__":
    main()
