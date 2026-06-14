"""Build the free-form grounding benchmark from the existing single-target benchmark.

For each benchmark case, generates 5 paraphrase styles:
  formal / casual / descriptive / indirect / richer

Output:
  spacefit_v2/data/grounding_benchmark/
    manifest.json
    cases.json       — flat list of paraphrase items (one per style per case)

Also builds the VLM grounding subset (15 test cases × visual-grounding instructions):
  spacefit_v2/data/vlm_subset/
    manifest.json
    cases.json
    renders/  (top-down PNG per case)

Usage:
    python -m spacefit_v2.scripts.build_grounding_benchmark [--splits test] [--vlm_n 15]
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from spacefit_v2.grounding.paraphraser import STYLES, generate_all_paraphrases, generate_paraphrase
from spacefit_v2.grounding.render import render_topdown


# ── VLM visual-grounding instructions ────────────────────────────────────────
# These instructions require understanding the room layout from an image.
# They are created per case based on which visual cues are available.

def _vlm_instructions(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate 2–3 visual-grounding instructions for a case."""
    scene = case["scene"]
    cat = case["target_asset"]["category"].replace("_", " ")
    instrs: List[Dict[str, Any]] = []

    n_windows = len(scene.get("windows", []))
    n_objects = len(scene.get("objects", []))
    room_type = scene.get("room_type", "room")

    # Always add a spatial-reasoning instruction
    instrs.append({
        "id": f"{case['id']}__vlm_spatial",
        "style": "vlm_spatial",
        "text": f"Looking at the room layout, place the {cat} against the most open wall where it doesn't crowd the entrance or block circulation.",
        "requires_visual": True,
        "visual_cue": "most_open_wall",
    })

    # Window-based if windows exist
    if n_windows > 0:
        instrs.append({
            "id": f"{case['id']}__vlm_window",
            "style": "vlm_window",
            "text": f"Place the {cat} near the largest window so it gets good natural light, while keeping the path from the door clear.",
            "requires_visual": True,
            "visual_cue": "largest_window",
        })

    # Ensemble/scene-level if room has many objects
    if n_objects >= 3:
        instrs.append({
            "id": f"{case['id']}__vlm_scene",
            "style": "vlm_scene",
            "text": f"Based on the overall room arrangement, place the {cat} in the spot that looks most natural and leaves comfortable movement space.",
            "requires_visual": True,
            "visual_cue": "scene_composition",
        })

    # Add GT constraints as structured reference
    for instr in instrs:
        instr["case_id"] = case["id"]
        instr["split"] = case.get("split", "test")
        instr["target_category"] = case["target_asset"]["category"]
        instr["room_type"] = room_type
        instr["gt_constraints"] = [dict(c) for c in case["intent"].get("constraints", [])]
        instr["gt_constraint_types"] = [c["constraint_type"] for c in case["intent"].get("constraints", [])]
        instr["n_windows"] = n_windows
        instr["n_objects"] = n_objects

    return instrs


# ── main builder ──────────────────────────────────────────────────────────────

def build_grounding_benchmark(
    benchmark_dir: Path,
    out_dir: Path,
    splits: List[str],
) -> None:
    with open(benchmark_dir / "cases.json") as f:
        all_cases = json.load(f)

    selected = [c for c in all_cases if c["split"] in splits]
    print(f"Building grounding benchmark: {len(selected)} cases × 5 styles = {len(selected)*5} instances")

    paraphrase_items: List[Dict[str, Any]] = []
    for case in selected:
        paraphrase_items.extend(generate_all_paraphrases(case))

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_benchmark": str(benchmark_dir),
        "splits": splits,
        "n_cases": len(selected),
        "n_paraphrases": len(paraphrase_items),
        "styles": list(STYLES),
        "constraint_types": ["against_wall", "near", "facing", "not_block_door", "keep_window_clear", "access_zone"],
        "note": (
            "5 paraphrase styles per case. "
            "'formal' uses canonical vocabulary; 'casual'/'descriptive'/'indirect'/'richer' use vocabulary "
            "that the simple rule parser systematically misses, creating a meaningful gap for LLM evaluation."
        ),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(out_dir / "cases.json", "w") as f:
        json.dump(paraphrase_items, f, indent=2)

    print(f"  Saved {len(paraphrase_items)} paraphrase cases to {out_dir}/cases.json")


def build_vlm_subset(
    benchmark_dir: Path,
    out_dir: Path,
    n_cases: int = 15,
) -> None:
    with open(benchmark_dir / "cases.json") as f:
        all_cases = json.load(f)

    # Select test cases; prefer variety: bedroom / living / library
    test_cases = [c for c in all_cases if c["split"] == "test"]
    # Sort by room_type variety then take up to n_cases
    by_type: Dict[str, List] = {}
    for c in test_cases:
        rt = c["scene"].get("room_type", "other")
        by_type.setdefault(rt, []).append(c)
    selected: List[Dict[str, Any]] = []
    # Round-robin by room type until we have n_cases
    types = sorted(by_type.keys())
    idx = {t: 0 for t in types}
    while len(selected) < n_cases:
        added = False
        for t in types:
            if idx[t] < len(by_type[t]) and len(selected) < n_cases:
                selected.append(by_type[t][idx[t]])
                idx[t] += 1
                added = True
        if not added:
            break

    print(f"Building VLM subset: {len(selected)} cases")
    render_dir = out_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    vlm_items: List[Dict[str, Any]] = []
    for case in selected:
        instrs = _vlm_instructions(case)
        for instr in instrs:
            # Render top-down image
            png_path = render_dir / f"{case['id']}.png"
            if not png_path.exists():
                render_topdown(case["scene"], case["target_asset"], save_path=str(png_path))
            instr["render_path"] = str(png_path) if png_path.exists() else None
        vlm_items.extend(instrs)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_benchmark": str(benchmark_dir),
        "n_cases": len(selected),
        "n_instructions": len(vlm_items),
        "visual_cue_types": ["most_open_wall", "largest_window", "scene_composition"],
        "note": (
            "VLM grounding subset. Each instruction requires visual scene understanding. "
            "Top-down renders are stored in renders/. "
            "For VLM inference, pass the render image + instruction to a vision-language model."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(out_dir / "cases.json", "w") as f:
        json.dump(vlm_items, f, indent=2)

    print(f"  Saved {len(vlm_items)} VLM instructions to {out_dir}/cases.json")
    print(f"  Renders saved to {render_dir}/")


def main(args: argparse.Namespace) -> None:
    benchmark_dir = Path(args.benchmark_dir)
    splits = args.splits

    build_grounding_benchmark(
        benchmark_dir=benchmark_dir,
        out_dir=Path(args.out_dir),
        splits=splits,
    )
    build_vlm_subset(
        benchmark_dir=benchmark_dir,
        out_dir=Path(args.vlm_dir),
        n_cases=args.vlm_n,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_dir", default="spacefit_v2/data/single_target_benchmark")
    p.add_argument("--out_dir", default="spacefit_v2/data/grounding_benchmark")
    p.add_argument("--vlm_dir", default="spacefit_v2/data/vlm_subset")
    p.add_argument("--splits", nargs="+", default=["test"])
    p.add_argument("--vlm_n", type=int, default=15)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
