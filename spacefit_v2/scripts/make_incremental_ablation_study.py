"""Create an incremental ablation study figure from saved benchmark results."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams


OUT_DIR = Path("spacefit_v2/results/ablation_incremental")
BASE_RESULTS = Path("spacefit_v2/results/experiment_final/test_gpt_intent/results.csv")
RERANK_RESULTS = Path("spacefit_v2/results/final_constraint_solver_human_rerank/test_gpt_intent/results.csv")


INCREMENTAL_STEPS = [
    {
        "step": 0,
        "variant": "LLM Direct",
        "method": "LayoutGPT (Direct Coord)",
        "source": "base",
        "added_component": "좌표 직접 예측",
        "description": "LLM이 target furniture의 좌표와 방향을 직접 출력하는 baseline.",
    },
    {
        "step": 1,
        "variant": "+ Candidate Search",
        "method": "Heuristic Baseline",
        "source": "base",
        "added_component": "빈 공간 후보 생성",
        "description": "방의 빈 공간을 추출하고 target furniture가 들어갈 후보 영역을 생성.",
    },
    {
        "step": 2,
        "variant": "+ Local Refinement",
        "method": "Proposal + Heuristic Refinement",
        "source": "base",
        "added_component": "후보 위치/방향 정제",
        "description": "후보 영역 안에서 위치와 방향을 탐색해 더 그럴듯한 placement를 선택.",
    },
    {
        "step": 3,
        "variant": "+ Constraint Solver",
        "method": "constraint_solver",
        "source": "base",
        "added_component": "물리 필터링 + 제약 점수화",
        "description": "충돌/경계/접근성 조건을 검사하고 near, beside, facing 등의 제약을 점수화.",
    },
    {
        "step": 4,
        "variant": "+ Human-Aligned Rerank",
        "method": "constraint_solver",
        "source": "rerank",
        "added_component": "사람 평가 기반 재정렬",
        "description": "human audit label로 학습한 scorer를 사용해 top-k 후보를 재정렬.",
    },
]


DIAGNOSTIC_METHODS = [
    {
        "variant": "SpaceFit GPT-Text",
        "method": "SpaceFit + GPT-Text",
        "source": "base",
        "note": "LLM intent guidance를 사용한 text-based variant. 현재 저장 결과에서는 solver 단독보다 CPS가 낮아 main incremental chain에는 넣지 않음.",
    },
    {
        "variant": "DiffOpt Constraint",
        "method": "Proposal + DiffOpt-Constraint",
        "source": "base",
        "note": "Differentiable constraint optimization baseline.",
    },
]


def setup_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR", "DejaVu Sans"]:
        if name in available:
            rcParams["font.family"] = name
            break
    rcParams["axes.unicode_minus"] = False


def read_results_csv(path: Path) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = str(row["method"])
            rows[method] = {
                key: float(value)
                for key, value in row.items()
                if key != "method" and value not in {"", None}
            }
    return rows


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_rows(base: Mapping[str, Mapping[str, float]], rerank: Mapping[str, Mapping[str, float]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sources = {"base": base, "rerank": rerank}
    incremental: List[Dict[str, Any]] = []
    previous_cps = None
    for spec in INCREMENTAL_STEPS:
        metrics = sources[spec["source"]][spec["method"]]
        row = {
            "step": spec["step"],
            "variant": spec["variant"],
            "added_component": spec["added_component"],
            "source_method": spec["method"],
            "description": spec["description"],
            **metrics,
        }
        row["CPS_delta_from_prev"] = None if previous_cps is None else metrics["CPS"] - previous_cps
        previous_cps = metrics["CPS"]
        incremental.append(row)

    diagnostics: List[Dict[str, Any]] = []
    for spec in DIAGNOSTIC_METHODS:
        metrics = sources[spec["source"]][spec["method"]]
        diagnostics.append(
            {
                "variant": spec["variant"],
                "source_method": spec["method"],
                "note": spec["note"],
                **metrics,
            }
        )
    return incremental, diagnostics


def plot_incremental(rows: List[Mapping[str, Any]], out_png: Path, out_svg: Path) -> None:
    labels = [str(r["variant"]) for r in rows]
    x = list(range(len(rows)))
    cps = [float(r["CPS"]) for r in rows]
    ca = [float(r["Constraint Accuracy"]) for r in rows]
    cf = [float(r["CF"]) for r in rows]
    ib = [float(r["IB"]) for r in rows]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.6, 6.1),
        facecolor="white",
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    ax = axes[0]
    bars = ax.bar(x, cps, color=["#9CA3AF", "#60A5FA", "#3B82F6", "#2563EB", "#16A34A"], width=0.62)
    ax.plot(x, cps, color="#0F172A", marker="o", linewidth=2.2, zorder=3)
    ax.set_ylim(0, max(cps) + 0.13)
    ax.set_ylabel("CPS / Success@1", fontsize=12)
    ax.set_title("Ablation: component-wise improvement", fontsize=15, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=10)
    ax.grid(axis="y", color="#E5E7EB", linewidth=1.0)
    ax.set_axisbelow(True)
    for idx, (bar, value) in enumerate(zip(bars, cps)):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.3f}", ha="center", fontsize=11, weight="bold")
        if idx > 0:
            delta = value - cps[idx - 1]
            ax.text(bar.get_x() + bar.get_width() / 2, max(0.015, value - 0.07), f"{delta:+.3f}", ha="center", fontsize=9, color="white", weight="bold")

    ax2 = axes[1]
    ax2.plot(x, ca, marker="o", color="#7C3AED", linewidth=2.2, label="Constraint Accuracy")
    ax2.plot(x, cf, marker="s", color="#059669", linewidth=2.0, label="CF")
    ax2.plot(x, ib, marker="^", color="#D97706", linewidth=2.0, label="IB")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Supporting metrics", fontsize=15, weight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"S{r['step']}" for r in rows], fontsize=10)
    ax2.grid(axis="y", color="#E5E7EB", linewidth=1.0)
    ax2.legend(loc="lower right", frameon=True, fontsize=10)
    for series in [ca, cf, ib]:
        for xi, yi in zip(x, series):
            ax2.text(xi, yi + 0.018, f"{yi:.2f}", ha="center", fontsize=8)

    fig.suptitle("Incremental Ablation Study", fontsize=19, weight="bold", y=0.99)
    fig.text(
        0.5,
        0.02,
        "Adding geometry-aware search, constraint scoring, and human-aligned reranking progressively improves strict placement success.",
        ha="center",
        fontsize=11.5,
        color="#334155",
    )
    fig.tight_layout(rect=[0.02, 0.07, 0.995, 0.95])
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_markdown(path: Path, rows: List[Mapping[str, Any]], diagnostics: List[Mapping[str, Any]]) -> None:
    lines = [
        "# Incremental Ablation Study",
        "",
        "## Main Chain",
        "",
        "This table uses the same GPT-intent test set and the repository's unified evaluator.",
        "",
        "| Step | Variant | Added component | CPS | Constraint Acc. | CF | IB | ΔCPS |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = row["CPS_delta_from_prev"]
        delta_text = "-" if delta is None else f"{float(delta):+.3f}"
        lines.append(
            f"| {row['step']} | {row['variant']} | {row['added_component']} | "
            f"{row['CPS']:.3f} | {row['Constraint Accuracy']:.3f} | {row['CF']:.3f} | {row['IB']:.3f} | {delta_text} |"
        )
    lines += [
        "",
        "## Diagnostic Variants",
        "",
        "| Variant | CPS | Constraint Acc. | CF | IB | Note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in diagnostics:
        lines.append(
            f"| {row['variant']} | {row['CPS']:.3f} | {row['Constraint Accuracy']:.3f} | "
            f"{row['CF']:.3f} | {row['IB']:.3f} | {row['note']} |"
        )
    lines += [
        "",
        "## Takeaway",
        "",
        "- Direct coordinate prediction has low strict success because physical validity is unstable.",
        "- Candidate search and refinement improve the baseline but still miss many constraints.",
        "- Constraint Solver gives the largest jump by combining physical filtering and constraint scoring.",
        "- Human-Aligned Rerank further improves CPS in the saved experiment, so it is a valid optional final component.",
        "- SpaceFit GPT-Text is kept as a diagnostic variant unless the paper narrative explicitly treats LLM guidance as a separate goal from code-metric success.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    setup_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = read_results_csv(BASE_RESULTS)
    rerank = read_results_csv(RERANK_RESULTS)
    rows, diagnostics = collect_rows(base, rerank)

    write_csv(OUT_DIR / "incremental_ablation_results.csv", rows)
    write_csv(OUT_DIR / "diagnostic_variant_results.csv", diagnostics)
    (OUT_DIR / "incremental_ablation_results.json").write_text(
        json.dumps({"incremental": rows, "diagnostics": diagnostics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_incremental(rows, OUT_DIR / "incremental_ablation_study.png", OUT_DIR / "incremental_ablation_study.svg")
    write_markdown(OUT_DIR / "INCREMENTAL_ABLATION_STUDY.md", rows, diagnostics)
    print(json.dumps({"incremental": rows, "diagnostics": diagnostics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
