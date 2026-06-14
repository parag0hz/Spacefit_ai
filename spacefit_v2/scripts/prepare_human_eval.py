"""Prepare blinded pairwise human-evaluation assets for layout comparison.

Creates:
- randomized A/B comparison images
- participant-facing manifest CSV
- answer key CSV
- instructions markdown

Example:
  python -m spacefit_v2.scripts.prepare_human_eval \
    --left_file experiments/ours/3dfront_bedroom_generation_v3.json \
    --right_file spacefit_v2/results/3dfront_bedroom_diffopt_full_e100gpu_balcat_b8192_r3.json \
    --sample_mode random --n 24 --seed 42 \
    --study_name v3_vs_v2_full24 \
    --out_dir spacefit_v2/results/human_eval
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spacefit_v2.legacy.viz.topdown import _draw_polygon, _draw_rotated_box
from spacefit_v2.viz_labels import category_label_ko, setup_korean_matplotlib


def _load_rows(path: str | Path) -> Dict[str, Dict[str, Any]]:
    with open(path, "r") as f:
        rows = [row for row in json.load(f) if not row.get("error")]
    return {row["query_id"]: row for row in rows}


def _draw_prediction(
    ax,
    polygon: List[Tuple[float, float]],
    object_list: List[List[Any]],
    title: str,
    color: str,
    edge: str,
) -> None:
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


def _sample_qids(
    left_rows: Dict[str, Dict[str, Any]],
    right_rows: Dict[str, Dict[str, Any]],
    sample_mode: str,
    n: int,
    seed: int,
    qid_file: str | None,
) -> List[str]:
    common = sorted(set(left_rows) & set(right_rows))
    if qid_file:
        with open(qid_file, "r") as f:
            qids = [line.strip() for line in f if line.strip()]
        return [qid for qid in qids if qid in common]

    rng = random.Random(seed)
    if sample_mode == "first":
        return common[:n]
    if sample_mode == "random":
        chosen = common[:]
        rng.shuffle(chosen)
        return chosen[:n]
    raise ValueError(f"unsupported sample_mode: {sample_mode}")


def _write_protocol(study_dir: Path, study_name: str, manifest_name: str) -> None:
    protocol = f"""# Human Preference Study: {study_name}

## Goal

Compare two room layouts for the same scene and choose which one is better from a human point of view.

## Instructions for Raters

For each scene, look at layout `A` and layout `B` and answer:

1. Which layout looks more natural?
2. Which layout looks more functional / easier to use?
3. Which layout do you prefer overall?

If they look equally good, choose `tie`.

## Rating Guidelines

- Focus on whether the room feels plausible and pleasant to use.
- Consider furniture spacing, wall usage, openness, and visual balance.
- Do not try to guess which system produced which image.
- Ignore tiny numerical differences if both layouts look effectively the same.

## Files

- Panels: `images/`
- Participant sheet: `{manifest_name}`
- Answer key: `answer_key.csv` (for researchers only)

## Suggested Setup

- 3 to 5 raters
- 20 to 30 scenes
- Report majority preference rate for each question
"""
    (study_dir / "protocol.md").write_text(protocol)


def _write_participant_template(study_dir: Path) -> None:
    header = [
        "participant_id",
        "scene_id",
        "image_file",
        "more_natural",
        "more_functional",
        "overall_preference",
        "comments",
    ]
    with open(study_dir / "participant_template.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def main() -> None:
    setup_korean_matplotlib()
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_file", required=True)
    parser.add_argument("--right_file", required=True)
    parser.add_argument("--left_label", default="left_system")
    parser.add_argument("--right_label", default="right_system")
    parser.add_argument("--sample_mode", default="random", choices=["random", "first"])
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qid_file", default=None)
    parser.add_argument("--study_name", default="v3_vs_v2_human_eval")
    parser.add_argument("--out_dir", default="spacefit_v2/results/human_eval")
    args = parser.parse_args()

    left_rows = _load_rows(args.left_file)
    right_rows = _load_rows(args.right_file)
    qids = _sample_qids(
        left_rows,
        right_rows,
        sample_mode=args.sample_mode,
        n=args.n,
        seed=args.seed,
        qid_file=args.qid_file,
    )
    rng = random.Random(args.seed)

    study_dir = Path(args.out_dir) / args.study_name
    image_dir = study_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, str]] = []
    answer_rows: List[Dict[str, str]] = []

    for idx, qid in enumerate(qids, start=1):
        left = left_rows[qid]
        right = right_rows[qid]
        polygon = [(float(x), float(z)) for x, z in left["floor_plan_vertices"]]

        if rng.random() < 0.5:
            a_row, b_row = left, right
            a_system, b_system = args.left_label, args.right_label
        else:
            a_row, b_row = right, left
            a_system, b_system = args.right_label, args.left_label

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        _draw_prediction(
            axes[0],
            polygon,
            a_row["object_list"],
            "Layout A",
            color="#efb366",
            edge="#8a3f00",
        )
        _draw_prediction(
            axes[1],
            polygon,
            b_row["object_list"],
            "Layout B",
            color="#8ecae6",
            edge="#005f73",
        )
        fig.suptitle(f"Scene {idx:02d}", fontsize=12)
        fig.tight_layout()

        file_name = f"{idx:02d}_{qid}.png"
        fig.savefig(image_dir / file_name, dpi=140, bbox_inches="tight")
        plt.close(fig)

        manifest_rows.append(
            {
                "scene_id": qid,
                "display_id": f"scene_{idx:02d}",
                "image_file": f"images/{file_name}",
            }
        )
        answer_rows.append(
            {
                "scene_id": qid,
                "display_id": f"scene_{idx:02d}",
                "image_file": f"images/{file_name}",
                "A_system": a_system,
                "B_system": b_system,
            }
        )

    manifest_name = "manifest.csv"
    with open(study_dir / manifest_name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "display_id", "image_file"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    with open(study_dir / "answer_key.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene_id", "display_id", "image_file", "A_system", "B_system"],
        )
        writer.writeheader()
        writer.writerows(answer_rows)

    _write_participant_template(study_dir)
    _write_protocol(study_dir, args.study_name, manifest_name)
    print(study_dir)


if __name__ == "__main__":
    main()
