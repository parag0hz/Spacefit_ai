"""Visualize original LayoutGPT 3D JSON outputs as top-down layout figures."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import font_manager, rcParams


COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
]


def setup_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR", "DejaVu Sans"]:
        if name in available:
            rcParams["font.family"] = name
            break
    rcParams["axes.unicode_minus"] = False


def parse_room_size(prompt: str) -> tuple[float, float]:
    match = re.search(r"max length\s+([0-9.]+)px,\s+max width\s+([0-9.]+)px", prompt)
    if not match:
        return 256.0, 256.0
    return float(match.group(1)), float(match.group(2))


def draw_rotated_furniture(
    ax: plt.Axes,
    label: str,
    box: Mapping[str, Any],
    color: str,
) -> None:
    left = float(box.get("left", 0.0))
    top = float(box.get("top", 0.0))
    length = max(1.0, float(box.get("length", 8.0)))
    width = max(1.0, float(box.get("width", 8.0)))
    angle = float(box.get("orientation", 0.0))

    rect = patches.Rectangle(
        (left - length / 2, top - width / 2),
        length,
        width,
        angle=angle,
        rotation_point="center",
        linewidth=1.4,
        edgecolor=color,
        facecolor=color,
        alpha=0.30,
    )
    ax.add_patch(rect)

    rad = math.radians(angle)
    arrow_len = max(12.0, min(length, width) * 0.55)
    ax.arrow(
        left,
        top,
        math.cos(rad) * arrow_len,
        math.sin(rad) * arrow_len,
        width=0.6,
        head_width=5.0,
        head_length=6.0,
        length_includes_head=True,
        color=color,
        alpha=0.95,
    )
    ax.text(
        left,
        top,
        label.replace("_", " ")[:18],
        ha="center",
        va="center",
        fontsize=7.5,
        color="#1f2933",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=color, alpha=0.84),
    )


def draw_layout(ax: plt.Axes, item: Mapping[str, Any], title: str) -> None:
    room_len, room_wid = parse_room_size(str(item.get("prompt", "")))
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            room_len,
            room_wid,
            linewidth=2.0,
            edgecolor="#222222",
            facecolor="#fbfbfb",
        )
    )
    for idx, entry in enumerate(item.get("object_list") or []):
        if not isinstance(entry, Sequence) or len(entry) != 2:
            continue
        label, box = entry
        if not isinstance(box, Mapping):
            continue
        draw_rotated_furniture(ax, str(label), box, COLORS[idx % len(COLORS)])

    ax.set_title(title, fontsize=13, weight="bold", pad=8)
    ax.set_xlim(-8, room_len + 8)
    ax.set_ylim(room_wid + 8, -8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0,
        room_wid + 13,
        f"{item.get('query_id', '')}",
        fontsize=8,
        color="#555",
        ha="left",
        va="top",
    )


def make_grid(items: Sequence[Mapping[str, Any]], output: Path, title: str) -> None:
    cols = min(3, len(items))
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.1, rows * 4.2), facecolor="white")
    if rows == 1 and cols == 1:
        axes_list = [axes]
    else:
        axes_list = list(getattr(axes, "flat", [axes]))
    for idx, ax in enumerate(axes_list):
        if idx < len(items):
            draw_layout(ax, items[idx], f"LayoutGPT Output #{idx + 1}")
        else:
            ax.axis("off")
    fig.suptitle(title, fontsize=17, weight="bold", y=0.99)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--title", default="Original LayoutGPT 3D Outputs")
    return parser.parse_args()


def main() -> None:
    setup_font()
    args = parse_args()
    data = json.loads(args.pred.read_text(encoding="utf-8"))
    items = data[args.offset : args.offset + args.n]
    if not items:
        raise RuntimeError(f"No layouts found in {args.pred}")
    make_grid(items, args.out, args.title)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
