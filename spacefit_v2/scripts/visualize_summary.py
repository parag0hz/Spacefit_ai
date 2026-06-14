"""Generate summary charts comparing all methods across metrics."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = Path("spacefit_v2/results/visualizations/summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = [
    "heuristic_baseline",
    "proposal_heuristic",
    "proposal_diffopt_basic",
    "proposal_diffopt_constraint",
    "layoutgpt_direct",
    "spacefit_gpt_text",
]
METHOD_LABELS = [
    "Heuristic",
    "Proposal\n+Heuristic",
    "DiffOpt\n-Basic",
    "DiffOpt\n-Constraint",
    "LayoutGPT\n(Direct)",
    "SpaceFit\n+GPT",
]
COLORS = ["#95A5A6", "#3498DB", "#2ECC71", "#27AE60", "#E74C3C", "#F39C12"]


def load_results(results_json: str) -> dict:
    with open(results_json) as f:
        data = json.load(f)
    out = {}
    for row in data["methods"]:
        name = row.get("source_name") or row["method"]
        out[name] = row["metrics"]
    return out


def load_vlm(vlm_json: str) -> dict:
    with open(vlm_json) as f:
        data = json.load(f)
    return data["summary"]


# ── 1. 메인 메트릭 비교 바 차트 ──────────────────────────────────────────────

def plot_metric_bars(clean: dict, noise: dict, vlm: dict):
    metrics = [
        ("CF",                "Collision-Free ↑",         0.3, 1.0),
        ("IB",                "In-Boundary ↑",            0.3, 1.0),
        ("Constraint Accuracy","Constraint Accuracy ↑",   0.3, 1.0),
        ("Success@1",         "Success@1 ↑",              0.0, 0.7),
        ("Success@5",         "Success@5 ↑",              0.0, 0.7),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    fig.suptitle("Method Comparison — GPT Intent Evaluation (181 test cases)",
                 fontsize=13, fontweight="bold", y=1.02)

    x = np.arange(len(METHOD_ORDER))
    w = 0.38

    for ax, (key, title, ymin, ymax) in zip(axes, metrics):
        clean_vals = [clean.get(m, {}).get(key, 0) for m in METHOD_ORDER]
        noise_vals = [noise.get(m, {}).get(key, 0) for m in METHOD_ORDER]

        bars_c = ax.bar(x - w/2, clean_vals, w, color=COLORS, alpha=0.9, label="Clean")
        bars_n = ax.bar(x + w/2, noise_vals, w, color=COLORS, alpha=0.45,
                        edgecolor=[c for c in COLORS], linewidth=1.2, label="Noise")

        for bar, v in zip(bars_c, clean_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
        for bar, v in zip(bars_n, noise_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=5.5, color="#555")

        ax.set_title(title, fontsize=10, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_LABELS, fontsize=7)
        ax.set_ylim(ymin, ymax + 0.08)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    # Legend
    legend_patches = [
        mpatches.Patch(color="#888", alpha=0.9, label="Clean (synthetic)"),
        mpatches.Patch(color="#888", alpha=0.4, label="Noise (sim-to-real)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    path = OUT_DIR / "01_metric_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── 2. VLM Judge 점수 + 물리/의도 분해 ───────────────────────────────────────

def plot_vlm_scores(vlm: dict):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("VLM-as-Judge Results (GPT-4V, 181 cases)",
                 fontsize=13, fontweight="bold")

    x = np.arange(len(METHOD_ORDER))

    titles = ["VLM Score (/10)", "Physical Validity", "Intent Satisfied"]
    keys   = ["vlm_score",       "vlm_physical_valid", "vlm_intent_satisfied"]
    ylims  = [(0, 11),            (0, 1.15),             (0, 1.15)]

    for ax, title, key, (ymin, ymax) in zip(axes, titles, keys, ylims):
        vals = [vlm.get(m, {}).get(key, 0) for m in METHOD_ORDER]
        bars = ax.bar(x, vals, color=COLORS, alpha=0.88, edgecolor="white", linewidth=0.8)

        for bar, v in zip(bars, vals):
            label = f"{v:.1f}" if key == "vlm_score" else f"{v:.2f}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * (ymax - ymin),
                    label, ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_LABELS, fontsize=8)
        ax.set_ylim(ymin, ymax)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    path = OUT_DIR / "02_vlm_judge.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── 3. Clean vs Noise 하락폭 ─────────────────────────────────────────────────

def plot_noise_drop(clean: dict, noise: dict):
    metrics = ["CF", "IB", "Constraint Accuracy", "Success@1", "Success@5"]
    metric_labels = ["CF", "IB", "Const.\nAcc.", "S@1", "S@5"]

    fig, axes = plt.subplots(1, len(METHOD_ORDER), figsize=(16, 4))
    fig.suptitle("Sim-to-Real Robustness: Metric Drop (Clean → Noise)",
                 fontsize=12, fontweight="bold")

    x = np.arange(len(metrics))
    for ax, method, label, color in zip(axes, METHOD_ORDER, METHOD_LABELS, COLORS):
        drops = [
            clean.get(method, {}).get(m, 0) - noise.get(method, {}).get(m, 0)
            for m in metrics
        ]
        bars = ax.bar(x, drops, color=color, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, drops):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.003,
                    f"-{v:.3f}" if v >= 0 else f"+{-v:.3f}",
                    ha="center", va="bottom", fontsize=6.5)

        ax.set_title(label, fontsize=8, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=7)
        ax.set_ylim(-0.05, 0.18)
        ax.axhline(0, color="#333", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    path = OUT_DIR / "03_noise_robustness.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── 4. Radar chart: 종합 프로파일 ────────────────────────────────────────────

def plot_radar(clean: dict, vlm: dict):
    cats = ["CF", "IB", "Constraint\nAccuracy", "Success@5", "VLM\nScore"]
    N = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.suptitle("Method Profile Radar\n(normalised to [0,1])",
                 fontsize=12, fontweight="bold")

    # Normalisation ranges
    norm_max = [1.0, 1.0, 1.0, 1.0, 10.0]

    for method, label, color in zip(METHOD_ORDER, METHOD_LABELS, COLORS):
        m = clean.get(method, {})
        v = vlm.get(method, {})
        vals = [
            m.get("CF", 0) / norm_max[0],
            m.get("IB", 0) / norm_max[1],
            m.get("Constraint Accuracy", 0) / norm_max[2],
            m.get("Success@5", 0) / norm_max[3],
            v.get("vlm_score", 0) / norm_max[4],
        ]
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=2, label=label.replace("\n", " "))
        ax.fill(angles, vals, color=color, alpha=0.07)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

    plt.tight_layout()
    path = OUT_DIR / "04_radar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── 5. 종합 대시보드 ──────────────────────────────────────────────────────────

def plot_dashboard(clean: dict, noise: dict, vlm: dict):
    fig = plt.figure(figsize=(20, 11))
    fig.suptitle("SpaceFit Single-Target Benchmark — Full Dashboard",
                 fontsize=15, fontweight="bold", y=1.01)

    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35)

    x = np.arange(len(METHOD_ORDER))
    w = 0.38

    # Row 1: main metrics
    for col, (key, title) in enumerate([
        ("CF", "Collision-Free ↑"),
        ("Constraint Accuracy", "Constraint Accuracy ↑"),
        ("Success@1", "Success@1 ↑"),
        ("Success@5", "Success@5 ↑"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        cv = [clean.get(m, {}).get(key, 0) for m in METHOD_ORDER]
        nv = [noise.get(m, {}).get(key, 0) for m in METHOD_ORDER]
        ax.bar(x - w/2, cv, w, color=COLORS, alpha=0.9)
        ax.bar(x + w/2, nv, w, color=COLORS, alpha=0.4, edgecolor=COLORS, linewidth=1)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_LABELS, fontsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        for i, (c, n) in enumerate(zip(cv, nv)):
            ax.text(i - w/2, c + 0.01, f"{c:.2f}", ha="center", fontsize=5.5, fontweight="bold")
            ax.text(i + w/2, n + 0.01, f"{n:.2f}", ha="center", fontsize=5, color="#555")

    # Row 2 left-3: VLM metrics
    for col, (key, title) in enumerate([
        ("vlm_score",          "VLM Score (/10) ↑"),
        ("vlm_physical_valid", "VLM Physical Valid ↑"),
        ("vlm_intent_satisfied","VLM Intent Satisfied ↑"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        vals = [vlm.get(m, {}).get(key, 0) for m in METHOD_ORDER]
        bars = ax.bar(x, vals, color=COLORS, alpha=0.88, edgecolor="white")
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_LABELS, fontsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        for bar, v in zip(bars, vals):
            lbl = f"{v:.1f}" if "score" in key else f"{v:.2f}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    lbl, ha="center", fontsize=6.5, fontweight="bold")

    # Row 2 right: radar
    ax_radar = fig.add_subplot(gs[1, 3], polar=True)
    cats = ["CF", "IB", "CA", "S@5", "VLM"]
    N = len(cats)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    norm_max = [1.0, 1.0, 1.0, 1.0, 10.0]
    for method, label, color in zip(METHOD_ORDER, METHOD_LABELS, COLORS):
        m = clean.get(method, {}); v = vlm.get(method, {})
        vals = [m.get("CF",0)/norm_max[0], m.get("IB",0)/norm_max[1],
                m.get("Constraint Accuracy",0)/norm_max[2],
                m.get("Success@5",0)/norm_max[3], v.get("vlm_score",0)/norm_max[4]]
        vals += vals[:1]
        ax_radar.plot(angles, vals, color=color, lw=1.8,
                      label=label.replace("\n", " "))
        ax_radar.fill(angles, vals, color=color, alpha=0.06)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(cats, fontsize=8)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title("Radar", fontsize=9)

    # legend
    patches = [mpatches.Patch(color=c, label=l.replace("\n"," "))
               for c, l in zip(COLORS, METHOD_LABELS)]
    clean_p = mpatches.Patch(color="#888", alpha=0.9, label="■ Clean")
    noise_p = mpatches.Patch(color="#888", alpha=0.4, label="□ Noise")
    fig.legend(handles=patches + [clean_p, noise_p],
               loc="lower center", ncol=8, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.03))

    path = OUT_DIR / "00_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading results...")
    clean = load_results("spacefit_v2/results/experiment_v2/test_gpt_intent/results.json")
    noise = load_results("spacefit_v2/results/experiment_v2/test_roomplan_gpt_intent/results.json")
    vlm   = load_vlm("spacefit_v2/results/experiment_v2/test_gpt_intent/vlm_judgments.json")

    print("Generating charts...")
    plot_dashboard(clean, noise, vlm)
    plot_metric_bars(clean, noise, vlm)
    plot_vlm_scores(vlm)
    plot_noise_drop(clean, noise)
    plot_radar(clean, vlm)
    print(f"\nAll charts saved → {OUT_DIR}/")
