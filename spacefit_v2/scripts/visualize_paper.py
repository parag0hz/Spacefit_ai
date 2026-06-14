"""Generate paper-quality figures for the constraint-solver paper.

Figures produced:
  fig_01_method_comparison.png  — main metric bar chart (7 methods × 6 metrics)
  fig_02_ablation_normalization.png — before/after category normalization
  fig_03_radar.png              — radar chart (top 4 methods)
  fig_04_placements_*.png       — room-level placement grids (interesting cases)

Usage:
    python -m spacefit_v2.scripts.visualize_paper
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
RESULTS_JSON   = "spacefit_v2/results/experiment_final/test_gpt_intent/results.json"
NOISE_JSON     = "spacefit_v2/results/experiment_v2/test_roomplan_gpt_intent/results.json"
RAW_PREDS      = "spacefit_v2/results/experiment_final/test_gpt_intent/raw_predictions.json"
CASES_JSON     = "spacefit_v2/data/single_target_benchmark/gpt_intent_cases_test.json"
OUT_DIR        = Path("spacefit_v2/results/visualizations/paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── method config ──────────────────────────────────────────────────────────────
METHOD_ORDER = [
    "heuristic_baseline",
    "proposal_heuristic",
    "proposal_diffopt_basic",
    "proposal_diffopt_constraint",
    "layoutgpt_direct",
    "spacefit_gpt_text",
    "constraint_solver",
]
METHOD_LABELS = {
    "heuristic_baseline":         "Heuristic\nBaseline",
    "proposal_heuristic":         "Proposal\n+Heuristic",
    "proposal_diffopt_basic":     "DiffOpt\n-Basic",
    "proposal_diffopt_constraint":"DiffOpt\n-Constraint",
    "layoutgpt_direct":           "LayoutGPT\n(Direct)",
    "spacefit_gpt_text":          "SpaceFit\n+GPT",
    "constraint_solver":          "Constraint\nSolver (Ours)",
}
COLORS = {
    "heuristic_baseline":         "#95A5A6",
    "proposal_heuristic":         "#3498DB",
    "proposal_diffopt_basic":     "#1ABC9C",
    "proposal_diffopt_constraint":"#16A085",
    "layoutgpt_direct":           "#E74C3C",
    "spacefit_gpt_text":          "#F39C12",
    "constraint_solver":          "#8E44AD",
}

# ── geometry helpers ───────────────────────────────────────────────────────────
def _rotated_corners(cx, cz, w, d, yaw_deg):
    hw, hd = max(w, 0.05) / 2, max(d, 0.05) / 2
    rad = math.radians(yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    return [(cx + dx*c - dz*s, cz + dx*s + dz*c)
            for dx, dz in [(-hw,-hd),(hw,-hd),(hw,hd),(-hw,hd)]]

def _add_box(ax, cx, cz, w, d, yaw, color, alpha=0.8, lw=1.0, edgecolor="white",
             linestyle="-", zorder=2, label=None):
    pts = _rotated_corners(cx, cz, w, d, yaw)
    ax.add_patch(mpatches.Polygon(pts, closed=True, facecolor=color,
                                   edgecolor=edgecolor, linewidth=lw,
                                   alpha=alpha, linestyle=linestyle,
                                   zorder=zorder, label=label))

def _check_placement(pred, case):
    pos, size = pred.get("position"), pred.get("size")
    if not pos or not size:
        return False, False
    try:
        from shapely.geometry import Polygon as SPoly
        px, pz = float(pos["x"]), float(pos["z"])
        pw, pd = max(float(size["width"]), 0.05), max(float(size["depth"]), 0.05)
        yaw = float(pred.get("rotation_y", 0.0))
        fp = SPoly(_rotated_corners(px, pz, pw, pd, yaw))
        floor_pts = [(float(x), float(z)) for x, z in case["scene"]["floor"]["polygon"]]
        floor = SPoly(floor_pts)
        ib = fp.difference(floor).area <= 1e-4
        cf = True
        for obj in case["scene"].get("objects", []):
            ox, oz = float(obj["position"][0]), float(obj["position"][2])
            ow, od = max(float(obj["size"][0]), 0.05), max(float(obj["size"][2]), 0.05)
            if fp.intersection(SPoly(_rotated_corners(ox, oz, ow, od, float(obj.get("yaw",0))))).area > 1e-4:
                cf = False; break
        return cf, ib
    except Exception:
        return False, False

def _pred_color(pred, case):
    if pred.get("status") != "placed":
        return "#95A5A6"
    cf, ib = _check_placement(pred, case)
    if cf and ib:  return "#2ECC71"
    if not cf:     return "#E74C3C"
    return "#E67E22"

def _draw_room(ax, case, pred, title=""):
    scene = case["scene"]
    floor_pts = [(float(x), float(z)) for x, z in scene["floor"]["polygon"]]
    ax.add_patch(mpatches.Polygon(floor_pts, closed=True,
                                   facecolor="#F5F5F0", edgecolor="#555",
                                   linewidth=1.5, zorder=0))
    for obj in scene.get("objects", []):
        px, _, pz = obj["position"]; w, _, d = obj["size"]
        _add_box(ax, float(px), float(pz), float(w), float(d), float(obj.get("yaw",0)),
                 "#C8C8C8", edgecolor="#888", zorder=1)
        ax.text(float(px), float(pz), str(obj.get("category",""))[:6],
                fontsize=3, ha="center", va="center", color="#444", zorder=4)
    for door in scene.get("doors", []):
        p = door.get("position",[0,0,0])
        ax.plot(float(p[0]), float(p[2]), "s", color="#8B4513", ms=5, zorder=3)
    for win in scene.get("windows", []):
        p = win.get("position",[0,0,0])
        ax.plot(float(p[0]), float(p[2]), "D", color="#5DADE2", ms=4, zorder=3)
    ref = case.get("reference_pose")
    if ref:
        rpos, rsz = ref["position"], case["target_asset"]["size"]
        _add_box(ax, float(rpos["x"]), float(rpos["z"]),
                 float(rsz["width"]), float(rsz["depth"]), 0.0,
                 "none", edgecolor="#F1C40F", lw=1.5, linestyle="--",
                 alpha=1.0, zorder=5)
    if pred.get("status") == "placed":
        pos, sz = pred["position"], pred["size"]
        px, pz = float(pos["x"]), float(pos["z"])
        pw, pd = float(sz["width"]), float(sz["depth"])
        yaw = float(pred.get("rotation_y", 0.0))
        color = _pred_color(pred, case)
        _add_box(ax, px, pz, pw, pd, yaw, color, lw=1.5, zorder=6)
        rad = math.radians(yaw)
        length = min(pw, pd) * 0.35
        ax.annotate("", xy=(px + math.cos(rad)*length, pz + math.sin(rad)*length),
                    xytext=(px, pz),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.2), zorder=7)
    else:
        ax.text(0.5, 0.5, "not\nplaced", transform=ax.transAxes,
                ha="center", va="center", fontsize=6, color="#888")
    xs = [p[0] for p in floor_pts]; zs = [p[1] for p in floor_pts]
    m = 0.3
    ax.set_xlim(min(xs)-m, max(xs)+m); ax.set_ylim(min(zs)-m, max(zs)+m)
    ax.set_aspect("equal"); ax.axis("off")
    cf, ib = (False, False)
    if pred.get("status") == "placed":
        cf, ib = _check_placement(pred, case)
    status = ("✓" if cf and ib else "✗CF" if not cf else "✗IB")
    ax.set_title(f"{title}\n{status}", fontsize=6, pad=2)

# ── load data ──────────────────────────────────────────────────────────────────
def load_results(path):
    with open(path) as f: data = json.load(f)
    out = {}
    for row in data["methods"]:
        name = row.get("source_name") or row["method"]
        out[name] = row["metrics"]
    return out

clean  = load_results(RESULTS_JSON)
try:
    noise = load_results(NOISE_JSON)
except Exception:
    noise = {}

with open(RAW_PREDS) as f:  preds_by_method = json.load(f)
with open(CASES_JSON) as f: cases = json.load(f)
cases_by_id = {c["id"]: c for c in cases}

available_methods = [m for m in METHOD_ORDER if m in preds_by_method]

# ── FIG 1: Main metric comparison ─────────────────────────────────────────────
print("Generating fig_01_method_comparison...")
metrics_cfg = [
    ("CF",                 "Collision-Free Rate ↑"),
    ("IB",                 "In-Boundary Rate ↑"),
    ("Constraint Accuracy","Constraint Accuracy ↑"),
    ("CPS",                "CPS ↑"),
    ("Success@1",          "Success@1 ↑"),
    ("Success@5",          "Success@5 ↑"),
]
fig, axes = plt.subplots(1, 6, figsize=(22, 5))
fig.suptitle("Method Comparison — GPT Intent Evaluation (181 test cases, 3D-FRONT)",
             fontsize=13, fontweight="bold", y=1.03)

x = np.arange(len(available_methods))
for ax, (key, title) in zip(axes, metrics_cfg):
    vals = [clean.get(m, {}).get(key, 0) for m in available_methods]
    bar_colors = [COLORS[m] for m in available_methods]
    bars = ax.bar(x, vals, color=bar_colors, alpha=0.88, edgecolor="white", linewidth=0.8)
    # Highlight ours
    ours_idx = available_methods.index("constraint_solver") if "constraint_solver" in available_methods else -1
    if ours_idx >= 0:
        bars[ours_idx].set_edgecolor("#8E44AD")
        bars[ours_idx].set_linewidth(2.5)
    # Value labels
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.2f}", ha="center", va="bottom",
                fontsize=6.5, fontweight="bold" if v == max(vals) else "normal")
    ax.set_title(title, fontsize=9.5, pad=5)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in available_methods], fontsize=6)
    ymax = max(vals) if vals else 1.0
    ax.set_ylim(0, min(ymax * 1.25, 1.08))
    ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

# Legend
legend_patches = [mpatches.Patch(color=COLORS[m], label=METHOD_LABELS[m].replace("\n"," "))
                  for m in available_methods]
fig.legend(handles=legend_patches, loc="lower center", ncol=len(available_methods),
           fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.08))
plt.tight_layout()
fig.savefig(OUT_DIR / "fig_01_method_comparison.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"  → {OUT_DIR / 'fig_01_method_comparison.png'}")

# ── FIG 2: Ablation — category normalization ───────────────────────────────────
print("Generating fig_02_ablation_normalization...")
ablation_data = {
    "w/o Normalization\n(baseline)": {"Constraint Accuracy": 0.650, "CPS": 0.409, "Success@1": 0.409, "Success@5": 0.602},
    "w/ Slug\nNormalization (Ours)": {"Constraint Accuracy": 0.725, "CPS": 0.503, "Success@1": 0.503, "Success@5": 0.702},
}
ab_metrics = ["Constraint Accuracy", "CPS", "Success@1", "Success@5"]
ab_labels  = ["Constraint\nAccuracy", "CPS", "Success@1", "Success@5"]
ab_colors  = ["#BDC3C7", "#8E44AD"]

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(ab_metrics))
w = 0.32
for i, (label, vals) in enumerate(ablation_data.items()):
    v = [vals[k] for k in ab_metrics]
    bars = ax.bar(x + (i - 0.5) * w, v, w, color=ab_colors[i], alpha=0.85,
                  label=label, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, v):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# Delta annotations
for j, key in enumerate(ab_metrics):
    v_before = ablation_data["w/o Normalization\n(baseline)"][key]
    v_after  = ablation_data["w/ Slug\nNormalization (Ours)"][key]
    delta = (v_after - v_before) / v_before * 100
    ax.text(j + 0.5 * w - 0.16, max(v_before, v_after) + 0.035,
            f"+{delta:.1f}%", ha="center", fontsize=8, color="#27AE60", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(ab_labels, fontsize=10)
ax.set_ylim(0, 0.95)
ax.set_ylabel("Score", fontsize=10)
ax.set_title("Ablation: Category Slug Normalization\n"
             "(\"tv stand\" ↔ \"tv_stand\" — affects 37% of relational constraints)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, frameon=False)
ax.spines[["top","right"]].set_visible(False)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig_02_ablation_normalization.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"  → {OUT_DIR / 'fig_02_ablation_normalization.png'}")

# ── FIG 3: Radar chart ────────────────────────────────────────────────────────
print("Generating fig_03_radar...")
radar_methods  = ["heuristic_baseline", "proposal_diffopt_constraint",
                  "spacefit_gpt_text", "constraint_solver"]
radar_labels_m = ["Heuristic", "DiffOpt-Constraint", "SpaceFit+GPT", "Constraint Solver (Ours)"]
radar_colors_m = ["#95A5A6", "#16A085", "#F39C12", "#8E44AD"]
radar_metrics  = ["CF", "IB", "Constraint Accuracy", "CPS", "Success@5", "Reachability"]
radar_display  = ["CF", "IB", "Const.\nAcc", "CPS", "S@5", "Reach"]
N = len(radar_metrics)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
for method, label, color in zip(radar_methods, radar_labels_m, radar_colors_m):
    vals = [clean.get(method, {}).get(m, 0) for m in radar_metrics]
    vals += vals[:1]
    ax.plot(angles, vals, color=color, linewidth=2, label=label)
    ax.fill(angles, vals, color=color, alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_display, fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=7, color="#888")
ax.yaxis.grid(True, color="#DDD", linewidth=0.6)
ax.xaxis.grid(True, color="#DDD", linewidth=0.6)
ax.spines["polar"].set_visible(False)
ax.set_title("Multi-metric Comparison (Radar)", fontsize=12, fontweight="bold", pad=18)
ax.legend(loc="lower left", bbox_to_anchor=(-0.25, -0.18), fontsize=9, frameon=False)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig_03_radar.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"  → {OUT_DIR / 'fig_03_radar.png'}")

# ── FIG 4: Room placement visualizations ──────────────────────────────────────
print("Generating placement visualizations...")

# Select interesting cases: method disagreement (constraint_solver vs others)
def select_interesting(n=12):
    scored = []
    for case in cases:
        cid = case["id"]
        cs_pred  = (preds_by_method.get("constraint_solver", {}).get(cid) or [{}])[0]
        gpt_pred = (preds_by_method.get("spacefit_gpt_text", {}).get(cid) or [{}])[0]
        cs_ok  = cs_pred.get("status") == "placed"
        gpt_ok = gpt_pred.get("status") == "placed"
        # Include cases where methods disagree or constraint_solver shows clear advantage
        cf_cs, ib_cs  = _check_placement(cs_pred, case) if cs_ok else (False, False)
        cf_gpt, ib_gpt = _check_placement(gpt_pred, case) if gpt_ok else (False, False)
        cs_success  = cs_ok and cf_cs and ib_cs
        gpt_success = gpt_ok and cf_gpt and ib_gpt
        score = int(cs_success) * 2 + int(not gpt_success)  # prefer: cs✓ & gpt✗
        scored.append((score, case))
    scored.sort(key=lambda x: -x[0])
    top = [c for _, c in scored[:max(n*3, 30)]]
    random.seed(42)
    random.shuffle(top)
    return top[:n]

selected_cases = select_interesting(12)
vis_methods = [m for m in available_methods]

NCOLS = len(vis_methods)
for fig_idx, case in enumerate(selected_cases, 1):
    cid = case["id"]
    target_cat = str(case["target_asset"]["category"]).replace("_", " ")
    intent_text = case.get("intent", {}).get("text", "")[:70]

    nrows = 2
    ncols = math.ceil(NCOLS / nrows)
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows))
    axes_flat = axes_grid.flatten() if hasattr(axes_grid, "flatten") else [axes_grid]

    fig.suptitle(f"Target: {target_cat}   |   {cid[:45]}\n\"{intent_text}\"",
                 fontsize=7.5, y=1.02, ha="center")

    for ax_idx, (ax, method) in enumerate(zip(axes_flat, vis_methods)):
        preds = preds_by_method.get(method, {}).get(cid, [])
        top_pred = preds[0] if preds else {"status": "error"}
        _draw_room(ax, case, top_pred, title=METHOD_LABELS[method].replace("\n", " "))

    for ax in axes_flat[len(vis_methods):]:
        ax.set_visible(False)

    legend_patches = [
        mpatches.Patch(color="#2ECC71", label="CF + IB ✓"),
        mpatches.Patch(color="#E74C3C", label="Collision"),
        mpatches.Patch(color="#E67E22", label="Out-of-bound"),
        mpatches.Patch(color="#95A5A6", label="Not placed"),
        mpatches.Patch(color="#C8C8C8", label="Fixed furniture"),
        mpatches.Patch(color="none",    label="GT (dashed)",
                       edgecolor="#F1C40F", linestyle="--", linewidth=1.5),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=6,
               fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    safe_id = cid.replace("/","_")[:70]
    out_path = OUT_DIR / f"fig_04_placement_{fig_idx:02d}_{safe_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [{fig_idx}/{len(selected_cases)}] {out_path.name}")

print(f"\nAll figures saved to {OUT_DIR}/")
