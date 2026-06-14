"""Visualize single-target benchmark placement results.

Each figure shows one room: existing furniture (gray), doors/windows,
predicted placements per method (colored by success), and the intent text.

Usage:
    python -m spacefit_v2.scripts.visualize_results \
        --cases spacefit_v2/data/single_target_benchmark/gpt_intent_cases_test.json \
        --predictions spacefit_v2/results/experiment_gpt_intent/test_gpt_intent/raw_predictions.json \
        --out_dir spacefit_v2/results/visualizations \
        --n 20 \
        --mode interesting     # interesting | random | all | <case_id>
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ── colour scheme ─────────────────────────────────────────────────────────────

METHOD_ORDER = [
    "heuristic_baseline",
    "proposal_heuristic",
    "proposal_diffopt_basic",
    "proposal_diffopt_constraint",
    "constraint_solver",
    "layoutgpt_direct",
    "spacefit_gpt_text",
]
METHOD_LABELS = {
    "heuristic_baseline":         "Heuristic",
    "proposal_heuristic":         "Proposal+Heuristic",
    "proposal_diffopt_basic":     "DiffOpt-Basic",
    "proposal_diffopt_constraint":"DiffOpt-Constraint",
    "constraint_solver":          "Constraint Solver",
    "layoutgpt_direct":           "LayoutGPT",
    "spacefit_gpt_text":          "SpaceFit+GPT",
}

FIXED_COLOR   = "#C8C8C8"   # existing furniture
TARGET_OK     = "#2ECC71"   # placed + CF + IB
TARGET_BAD_CF = "#E74C3C"   # collision
TARGET_BAD_IB = "#E67E22"   # out of boundary
TARGET_FAIL   = "#95A5A6"   # not placed / error
DOOR_COLOR    = "#8B4513"
WINDOW_COLOR  = "#5DADE2"
GROUND_TRUTH  = "#F1C40F"   # dashed outline


# ── geometry helpers ──────────────────────────────────────────────────────────

def _rotated_corners(cx: float, cz: float, w: float, d: float, yaw_deg: float):
    hw, hd = max(w, 0.05) / 2, max(d, 0.05) / 2
    rad = math.radians(yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    local = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
    return [(cx + dx * c - dz * s, cz + dx * s + dz * c) for dx, dz in local]


def _add_box(ax, cx, cz, w, d, yaw_deg, color, alpha=0.75, lw=1.0,
             edgecolor=None, linestyle="-", zorder=2, label=None):
    pts = _rotated_corners(cx, cz, w, d, yaw_deg)
    patch = mpatches.Polygon(pts, closed=True,
                             facecolor=color, edgecolor=edgecolor or "white",
                             linewidth=lw, alpha=alpha, linestyle=linestyle,
                             zorder=zorder, label=label)
    ax.add_patch(patch)
    return patch


def _add_arrow(ax, cx, cz, yaw_deg, length=0.15, color="white", zorder=3):
    rad = math.radians(yaw_deg)
    dx, dz = math.cos(rad) * length, math.sin(rad) * length
    ax.annotate("", xy=(cx + dx, cz + dz), xytext=(cx, cz),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                zorder=zorder)


# ── collision / boundary helpers ──────────────────────────────────────────────

def _shapely_poly(cx, cz, w, d, yaw_deg):
    from shapely.geometry import Polygon
    return Polygon(_rotated_corners(cx, cz, w, d, yaw_deg))


def _check_placement(pred: Dict, case: Dict) -> Tuple[bool, bool]:
    """Returns (collision_free, in_boundary)."""
    pos = pred.get("position")
    size = pred.get("size")
    if not pos or not size:
        return False, False
    try:
        from shapely.geometry import Polygon as SPoly
        px, pz = float(pos["x"]), float(pos["z"])
        pw, pd = max(float(size["width"]), 0.05), max(float(size["depth"]), 0.05)
        yaw = float(pred.get("rotation_y", 0.0))
        pred_poly = _shapely_poly(px, pz, pw, pd, yaw)

        # in-boundary
        floor_pts = [(float(x), float(z)) for x, z in case["scene"]["floor"]["polygon"]]
        floor_poly = SPoly(floor_pts)
        ib = floor_poly.contains(pred_poly) or floor_poly.intersection(pred_poly).area / pred_poly.area > 0.9

        # collision-free vs existing furniture
        cf = True
        for obj in case["scene"].get("objects", []):
            op = obj["position"]; os = obj["size"]
            ox, oz = float(op[0]), float(op[2])
            ow, od = max(float(os[0]), 0.05), max(float(os[2]), 0.05)
            oyaw = float(obj.get("yaw", 0.0))
            fixed_poly = _shapely_poly(ox, oz, ow, od, oyaw)
            if pred_poly.intersection(fixed_poly).area > 1e-4:
                cf = False
                break
        return cf, ib
    except Exception:
        return False, False


# ── single-case drawing ───────────────────────────────────────────────────────

def _draw_room_base(ax, case: Dict):
    scene = case["scene"]
    floor_pts = [(float(x), float(z)) for x, z in scene["floor"]["polygon"]]

    # Floor outline
    floor_patch = mpatches.Polygon(floor_pts, closed=True,
                                   facecolor="#F5F5F0", edgecolor="#555", linewidth=1.5, zorder=0)
    ax.add_patch(floor_patch)

    # Existing furniture
    for obj in scene.get("objects", []):
        px, _, pz = obj["position"]
        w, _, d = obj["size"]
        yaw = float(obj.get("yaw", 0.0))
        _add_box(ax, float(px), float(pz), float(w), float(d), yaw,
                 color=FIXED_COLOR, edgecolor="#888", alpha=0.9, zorder=1)
        cat = str(obj.get("category", "")).replace("_", " ")[:8]
        cx2 = float(px); cz2 = float(pz)
        ax.text(cx2, cz2, cat, fontsize=3.5, ha="center", va="center",
                color="#333", zorder=4, wrap=False)

    # Doors
    for door in scene.get("doors", []):
        p = door.get("position", [0, 0, 0])
        ax.plot(float(p[0]), float(p[2]), "s", color=DOOR_COLOR, ms=5, zorder=3)

    # Windows
    for win in scene.get("windows", []):
        p = win.get("position", [0, 0, 0])
        ax.plot(float(p[0]), float(p[2]), "D", color=WINDOW_COLOR, ms=4, zorder=3)

    # Ground-truth reference (dashed outline)
    ref = case.get("reference_pose")
    if ref:
        rpos = ref["position"]
        rsize = case["target_asset"]["size"]
        _add_box(ax, float(rpos["x"]), float(rpos["z"]),
                 float(rsize["width"]), float(rsize["depth"]), 0.0,
                 color="none", edgecolor=GROUND_TRUTH, lw=1.5,
                 linestyle="--", alpha=1.0, zorder=5, label="GT")

    xs = [p[0] for p in floor_pts]; zs = [p[1] for p in floor_pts]
    margin = 0.4
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(zs) - margin, max(zs) + margin)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_prediction(ax, pred: Dict, case: Dict, color: str):
    if pred.get("status") != "placed":
        ax.text(0.5, 0.5, "not placed\n" + str(pred.get("reason", ""))[:30],
                transform=ax.transAxes, ha="center", va="center",
                fontsize=6, color="#888")
        return
    pos = pred.get("position")
    size = pred.get("size")
    if not pos or not size:
        return
    px, pz = float(pos["x"]), float(pos["z"])
    pw, pd = float(size["width"]), float(size["depth"])
    yaw = float(pred.get("rotation_y", 0.0))
    _add_box(ax, px, pz, pw, pd, yaw, color=color, edgecolor="white",
             lw=1.5, alpha=0.9, zorder=6)
    _add_arrow(ax, px, pz, yaw, length=min(pw, pd) * 0.35, zorder=7)


def _pred_color(pred: Dict, case: Dict) -> str:
    if pred.get("status") != "placed":
        return TARGET_FAIL
    cf, ib = _check_placement(pred, case)
    if cf and ib:
        return TARGET_OK
    if not cf:
        return TARGET_BAD_CF
    return TARGET_BAD_IB


def visualize_case(
    case: Dict,
    predictions_by_method: Dict[str, Dict[str, List[Dict]]],
    out_path: Optional[Path] = None,
    methods: Optional[List[str]] = None,
) -> plt.Figure:
    methods = [m for m in (methods or METHOD_ORDER) if m in predictions_by_method]
    n_methods = len(methods)
    ncols = 3
    nrows = math.ceil(n_methods / ncols)
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 6.5 * nrows))
    axes = []
    for r in range(nrows):
        for c in range(ncols):
            ax = axes_grid[r, c] if nrows > 1 else axes_grid[c]
            idx = r * ncols + c
            if idx < n_methods:
                axes.append(ax)
            else:
                ax.set_visible(False)

    case_id = case["id"]
    target_cat = str(case["target_asset"]["category"]).replace("_", " ")
    intent_text = case.get("intent", {}).get("text", "")
    # Wrap intent text at ~55 chars
    words = intent_text.split()
    lines, cur = [], []
    for w in words:
        cur.append(w)
        if len(" ".join(cur)) > 55:
            lines.append(" ".join(cur))
            cur = []
    if cur:
        lines.append(" ".join(cur))
    intent_wrapped = "\n".join(lines[:3])

    fig.suptitle(
        f"Target: {target_cat}   |   {case_id[:50]}\n\"{intent_wrapped}\"",
        fontsize=7, y=1.01, ha="center", wrap=True
    )

    for ax, method in zip(axes, methods):
        _draw_room_base(ax, case)
        preds = predictions_by_method.get(method, {}).get(case_id, [])
        top_pred = preds[0] if preds else {"status": "error", "reason": "no prediction"}
        color = _pred_color(top_pred, case)
        _draw_prediction(ax, top_pred, case, color)
        cf, ib = (False, False)
        if top_pred.get("status") == "placed":
            cf, ib = _check_placement(top_pred, case)
        status_str = ("✓CF ✓IB" if cf and ib else
                      "✗CF" if not cf else "✗IB")
        ax.set_title(f"{METHOD_LABELS.get(method, method)}\n{status_str}",
                     fontsize=6.5, pad=3)

    # Legend
    legend_patches = [
        mpatches.Patch(color=TARGET_OK,     label="CF + IB ✓"),
        mpatches.Patch(color=TARGET_BAD_CF, label="Collision"),
        mpatches.Patch(color=TARGET_BAD_IB, label="Out-of-bound"),
        mpatches.Patch(color=TARGET_FAIL,   label="Not placed"),
        mpatches.Patch(color=FIXED_COLOR,   label="Fixed furniture"),
        mpatches.Patch(color="none", label="GT (dashed)",
                       edgecolor=GROUND_TRUTH, linestyle="--", linewidth=1.5),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=6,
               fontsize=5.5, frameon=False,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ── case selection ────────────────────────────────────────────────────────────

def _method_fails(case_id: str, preds_by_method: Dict, methods: List[str]) -> Dict[str, bool]:
    result = {}
    for m in methods:
        p = (preds_by_method.get(m, {}).get(case_id) or [{}])[0]
        result[m] = p.get("status") != "placed"
    return result


def select_interesting(
    cases: List[Dict],
    preds_by_method: Dict,
    n: int,
    methods: List[str],
) -> List[Dict]:
    """Pick cases that show method disagreement — most informative for diagnosis."""
    scored = []
    for case in cases:
        cid = case["id"]
        placed = [m for m in methods
                  if (preds_by_method.get(m, {}).get(cid) or [{}])[0].get("status") == "placed"]
        # Disagreement score: methods don't agree (some placed, some not)
        disagreement = len(placed) * (len(methods) - len(placed))
        scored.append((disagreement, case))
    scored.sort(key=lambda x: -x[0])
    # Take top half by disagreement, shuffle for variety
    top = [c for _, c in scored[:max(n * 3, 30)]]
    random.shuffle(top)
    return top[:n]


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--cases",
                   default="spacefit_v2/data/single_target_benchmark/gpt_intent_cases_test.json")
    p.add_argument("--predictions",
                   default="spacefit_v2/results/experiment_gpt_intent/test_gpt_intent/raw_predictions.json")
    p.add_argument("--out_dir", default="spacefit_v2/results/visualizations")
    p.add_argument("--n", type=int, default=20, help="Number of cases to visualize")
    p.add_argument("--mode", default="interesting",
                   help="interesting | random | all | <case_id>")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(args: argparse.Namespace):
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.cases) as f:
        cases = json.load(f)
    with open(args.predictions) as f:
        preds_by_method = json.load(f)

    available_methods = [m for m in METHOD_ORDER if m in preds_by_method]

    if args.mode == "interesting":
        selected = select_interesting(cases, preds_by_method, args.n, available_methods)
    elif args.mode == "random":
        selected = random.sample(cases, min(args.n, len(cases)))
    elif args.mode == "all":
        selected = cases
    else:
        # treat as case_id
        selected = [c for c in cases if args.mode in c["id"]]
        if not selected:
            raise SystemExit(f"No case matching '{args.mode}'")

    print(f"Visualizing {len(selected)} cases → {out_dir}")
    for idx, case in enumerate(selected, 1):
        safe_id = case["id"].replace("/", "_")[:80]
        out_path = out_dir / f"{idx:03d}_{safe_id}.png"
        visualize_case(case, preds_by_method, out_path=out_path,
                       methods=available_methods)
        print(f"  [{idx}/{len(selected)}] {out_path.name}")

    print(f"\nDone. {len(selected)} PNGs saved to {out_dir}/")


if __name__ == "__main__":
    main(build_parser().parse_args())
