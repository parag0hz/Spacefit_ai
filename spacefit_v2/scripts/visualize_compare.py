"""Render side-by-side top-down comparisons for two prediction files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spacefit_v2.legacy.viz.topdown import _draw_polygon, _draw_rotated_box
from spacefit_v2.viz_labels import category_label_ko, setup_korean_matplotlib


def _draw_prediction(ax, polygon: List[Tuple[float, float]], object_list: List[List[Any]], title: str, color: str, edge: str) -> None:
    _draw_polygon(ax, polygon, color="black", linewidth=2.0)
    for category, box in object_list:
        _draw_rotated_box(
            ax,
            float(box["left"]),
            float(box["top"]),
            float(box["length"]),
            float(box["width"]),
            float(box["orientation"]),
            label=category_label_ko(str(category), max_chars=6),
            color=color,
            alpha=0.68,
            edge=edge,
            linewidth=1.3,
        )
    xs = [p[0] for p in polygon]
    zs = [p[1] for p in polygon]
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    ax.set_ylim(min(zs) - 0.5, max(zs) + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title)


def _load_rows(path: str | Path) -> Dict[str, Dict[str, Any]]:
    with open(path, "r") as f:
        rows = [row for row in json.load(f) if not row.get("error")]
    return {row["query_id"]: row for row in rows}


def _make_contact_sheet(image_paths: List[Path], out_path: Path, cols: int = 3) -> None:
    if not image_paths:
        return
    rows = (len(image_paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.5))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, img_path in zip(axes_list, image_paths):
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.set_title(img_path.stem, fontsize=8)
        ax.axis("off")
    for ax in axes_list[len(image_paths):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_korean_matplotlib()
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_file", required=True)
    parser.add_argument("--right_file", required=True)
    parser.add_argument("--left_label", default="v3")
    parser.add_argument("--right_label", default="v2")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--out_dir", default="spacefit_v2/results/viz/compare")
    parser.add_argument("--contact_sheet", action="store_true")
    args = parser.parse_args()

    left_rows = _load_rows(args.left_file)
    right_rows = _load_rows(args.right_file)
    common_ids = [qid for qid in left_rows if qid in right_rows][: args.n]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths: List[Path] = []
    for qid in common_ids:
        left = left_rows[qid]
        right = right_rows[qid]
        polygon = [(float(x), float(z)) for x, z in left["floor_plan_vertices"]]

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        _draw_prediction(
            axes[0],
            polygon,
            left["object_list"],
            f"{args.left_label} | placed={left.get('num_placed')} time={left.get('time', 0):.2f}s",
            color="#efb366",
            edge="#8a3f00",
        )
        _draw_prediction(
            axes[1],
            polygon,
            right["object_list"],
            f"{args.right_label} | placed={right.get('num_placed')} time={right.get('time', 0):.2f}s",
            color="#8ecae6",
            edge="#005f73",
        )
        fig.suptitle(qid, fontsize=12)
        fig.tight_layout()
        out_path = out_dir / f"{qid}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        image_paths.append(out_path)

    if args.contact_sheet:
        _make_contact_sheet(image_paths, out_dir / "contact_sheet.png")

    print(out_dir)


if __name__ == "__main__":
    main()
