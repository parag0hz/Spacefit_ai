from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_spacefit_viz")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import gridspec
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from spacefit_v2.scripts.make_visualization_package import (
    RESULTS_DIR,
    _draw_scene,
    _evaluate_case_predictions,
    _load_cases,
    _load_predictions,
    _prediction_to_object,
)


OUT_DIR = RESULTS_DIR / "visualizations" / "motivation_slide"
EXP_A_PREDICTIONS = RESULTS_DIR / "exp_a" / "raw_predictions.json"

METHODS = [
    "heuristic_baseline",
    "proposal_diffopt_basic",
    "proposal_diffopt_constraint",
]

METHOD_TITLES = {
    "heuristic_baseline": "규칙 기반 방식",
    "proposal_diffopt_basic": "기본 최적화 방식",
    "proposal_diffopt_constraint": "제안 방식",
}

METHOD_CAPTIONS = {
    "heuristic_baseline": "안전하지만 조건 반영이 약함",
    "proposal_diffopt_basic": "조건은 반영하지만 물리적으로 불안정할 수 있음",
    "proposal_diffopt_constraint": "안전한 후보 생성 후 조건을 반영해 보정",
}

METHOD_COLORS = {
    "heuristic_baseline": "#4C78A8",
    "proposal_diffopt_basic": "#E45756",
    "proposal_diffopt_constraint": "#2CA02C",
}


def _configure_matplotlib_fonts() -> None:
    preferred_fonts = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ]
    for font_path in preferred_fonts:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK KR",
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


_configure_matplotlib_fonts()


def _save(fig: plt.Figure, stem: str) -> List[str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created: List[str] = []
    for ext in ("png", "pdf", "svg"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=220 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        created.append(str(path))
    plt.close(fig)
    return created


def _metric_text(metrics: Mapping[str, Any]) -> str:
    return (
        f"CF={metrics['cf']:.0f}, "
        f"IB={metrics['ib']:.0f}, "
        f"CA={metrics['constraint_accuracy']:.2f}, "
        f"CPS={int(metrics['cps'])}"
    )


def _panel_badge(method: str, metrics: Mapping[str, Any]) -> Tuple[str, str, str]:
    if method == "heuristic_baseline":
        if metrics["constraint_accuracy"] < 1.0:
            return ("조건 일부 미반영", "#FFF3CD", "#B58105")
        return ("안전한 배치", "#E3F2FD", "#2F6EA3")
    if method == "proposal_diffopt_basic":
        if metrics["cf"] < 1.0:
            return ("충돌 발생", "#FDE2E1", "#B8322B")
        if metrics["ib"] < 1.0:
            return ("경계 위반", "#FDE2E1", "#B8322B")
        return ("불안정 가능", "#FDE2E1", "#B8322B")
    if metrics["cps"] == 1:
        return ("균형 잡힌 성공", "#E3F6E8", "#2F7D32")
    return ("개선된 배치", "#E3F6E8", "#2F7D32")


def _add_panel_note(ax: plt.Axes, method: str, metrics: Mapping[str, Any]) -> None:
    badge, face, edge = _panel_badge(method, metrics)
    box = FancyBboxPatch(
        (0.66, 0.93),
        0.30,
        0.08,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.2,
        clip_on=False,
        zorder=30,
    )
    ax.add_patch(box)
    ax.text(0.81, 0.97, badge, transform=ax.transAxes, ha="center", va="center", fontsize=7.7, color=edge, weight="bold", zorder=31)


def _add_legend(ax: plt.Axes) -> None:
    ax.set_axis_off()
    items = [
        ("기존 가구", "#C9D7E3", "#6A7B8C"),
        ("타깃 가구", "#F28E6B", "#B54518"),
        ("문", "#D9534F", "white"),
        ("창문", "#4E79A7", "white"),
    ]
    x = 0.02
    for label, face, edge in items:
        if label in {"문", "창문"}:
            circ = Circle((x + 0.015, 0.5), 0.028, transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=0.9)
            ax.add_patch(circ)
        else:
            rect = Rectangle((x, 0.44), 0.03, 0.12, transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=1.0)
            ax.add_patch(rect)
        ax.text(x + 0.04, 0.50, label, transform=ax.transAxes, va="center", fontsize=9.5)
        x += 0.18
    ax.text(
        0.98,
        0.50,
        "메트릭 박스: CF / IB / CA / CPS",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=9.0,
        color="#555555",
    )


def _choose_best_case(
    cases: Mapping[str, Dict[str, Any]],
    eval_rows: Mapping[str, Mapping[str, Dict[str, Any]]],
) -> List[Tuple[str, float]]:
    ranked: List[Tuple[str, float]] = []
    for case_id, case in cases.items():
        heur = eval_rows["heuristic_baseline"][case_id]["top1"]
        basic = eval_rows["proposal_diffopt_basic"][case_id]["top1"]
        ours = eval_rows["proposal_diffopt_constraint"][case_id]["top1"]
        if not heur or not basic or not ours:
            continue
        if heur["cf"] != 1.0 or heur["ib"] != 1.0:
            continue
        if heur["constraint_accuracy"] >= ours["constraint_accuracy"]:
            continue
        if ours["cps"] != 1:
            continue
        if basic["cps"] == 1:
            continue

        score = 0.0
        score += 2.0 * (1.0 - heur["constraint_accuracy"])
        score += 2.0 * (ours["constraint_accuracy"] - basic["constraint_accuracy"])
        score += 1.6 * (1.0 - basic["ib"])
        score += 1.6 * (1.0 - basic["cf"])
        score += 0.4 if case["scene"]["room_type"] == "bedroom" else 0.0
        score += 0.2 if case["target_asset"]["category"] in {"bed", "sofa", "desk"} else 0.0
        score -= 0.015 * len(case["scene"]["objects"])
        ranked.append((case_id, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _make_case_figure(
    case: Mapping[str, Any],
    eval_rows: Mapping[str, Mapping[str, Dict[str, Any]]],
    *,
    stem: str,
) -> List[str]:
    fig = plt.figure(figsize=(15.6, 6.4))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.0, 0.11], hspace=0.12, wspace=0.12)

    for idx, method in enumerate(METHODS):
        ax = fig.add_subplot(gs[0, idx])
        prediction = eval_rows[method][case["id"]]["predictions"][0]
        metrics = eval_rows[method][case["id"]]["top1"]
        _draw_scene(
            ax,
            case,
            target=_prediction_to_object(case, prediction),
            title=METHOD_TITLES[method],
            subtitle=METHOD_CAPTIONS[method],
            metrics=metrics,
        )
        _add_panel_note(ax, method, metrics)

    legend_ax = fig.add_subplot(gs[1, :])
    _add_legend(legend_ax)

    fig.suptitle("왜 개선이 필요한가", fontsize=18, y=0.99, weight="bold")
    fig.text(
        0.5,
        0.93,
        "single-target placement에서는 안전성만으로도, 의도 최적화만으로도 충분하지 않다",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#444444",
    )
    return _save(fig, stem)


def _make_logic_figure() -> List[str]:
    fig, ax = plt.subplots(figsize=(12.8, 3.8))
    ax.set_axis_off()
    ax.set_title("제안 방향: 안전한 후보를 먼저 만들고, 그 다음 제약을 반영해 보정", fontsize=16, loc="left", pad=10)

    cards = [
        ("규칙 기반 방식", "충돌과 경계 위반은 잘 피하지만\n사용자 의도를 충분히 반영하지 못함", "#E9F1F8", "#4C78A8"),
        ("기본 최적화 방식", "의도는 맞추려 하지만\n충돌·경계 위반·불안정 배치가 생길 수 있음", "#FDE8E5", "#E45756"),
        ("제안 방식", "안전한 후보 위치를 먼저 찾고\nconstraint-aware refinement로 최종 보정", "#E6F4EA", "#2CA02C"),
    ]
    xs = [0.03, 0.355, 0.68]
    for i, (title, body, face, edge) in enumerate(cards):
        patch = FancyBboxPatch((xs[i], 0.20), 0.27, 0.56, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor=face, edgecolor=edge, linewidth=2.0, transform=ax.transAxes)
        ax.add_patch(patch)
        ax.text(xs[i] + 0.02, 0.66, title, transform=ax.transAxes, fontsize=12.5, weight="bold", color=edge)
        ax.text(xs[i] + 0.02, 0.48, body, transform=ax.transAxes, fontsize=10.4, va="top", color="#333333")
        if i < len(cards) - 1:
            ax.add_patch(FancyArrowPatch((xs[i] + 0.27, 0.48), (xs[i + 1] - 0.02, 0.48), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=16, linewidth=2.0, color="#777777"))
    ax.text(0.5, 0.09, "핵심: 안전성 확보와 의도 반영을 한 단계에서 동시에 해결하기보다, 두 단계를 분리해 균형을 맞춘다", transform=ax.transAxes, ha="center", fontsize=10.6, color="#444444")
    return _save(fig, "motivation_method_logic")


def _write_index(
    selected_case_ids: Sequence[str],
    cases: Mapping[str, Dict[str, Any]],
    eval_rows: Mapping[str, Mapping[str, Dict[str, Any]]],
    created_files: Sequence[str],
) -> str:
    best_case = cases[selected_case_ids[0]]
    lines = [
        "# Motivation Visualization Index",
        "",
        "## Best figure for PPT",
        "",
        "- `motivation_slide_best_figure.png`",
        "- Recommended for the slide titled `왜 개선이 필요한가`.",
        "",
        "## Selected cases",
        "",
    ]
    for case_id in selected_case_ids:
        case = cases[case_id]
        lines.append(f"- `{case_id}`")
        lines.append(f"  - room: `{case['scene']['room_type']}`")
        lines.append(f"  - target: `{case['target_asset']['category']}`")
        lines.append(f"  - intent: `{case['intent']['text']}`")
    lines.extend(
        [
            "",
            "## Methods visualized",
            "",
            "- `heuristic_baseline` -> 규칙 기반 방식",
            "- `proposal_diffopt_basic` -> 기본 최적화 방식",
            "- `proposal_diffopt_constraint` -> 제안 방식",
            "",
            "## Why the best case was selected",
            "",
        ]
    )
    heur = eval_rows["heuristic_baseline"][selected_case_ids[0]]["top1"]
    basic = eval_rows["proposal_diffopt_basic"][selected_case_ids[0]]["top1"]
    ours = eval_rows["proposal_diffopt_constraint"][selected_case_ids[0]]["top1"]
    lines.append(f"- Rule-based placement stays safe: `{_metric_text(heur)}`")
    lines.append(f"- Basic optimization shows the instability we want to highlight: `{_metric_text(basic)}`")
    lines.append(f"- Proposed method balances safety and intent satisfaction: `{_metric_text(ours)}`")
    lines.append("- The room is visually simple enough to read at slide scale, and the three panels tell a clear before/after story.")
    lines.extend(
        [
            "",
            "## Files created",
            "",
        ]
    )
    for path in created_files:
        lines.append(f"- `{Path(path).name}`")
    path = OUT_DIR / "motivation_visualization_index.md"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return str(path)


def main() -> None:
    cases = _load_cases()
    predictions = _load_predictions(EXP_A_PREDICTIONS)
    selected_predictions = {method: predictions[method] for method in METHODS}
    eval_rows = _evaluate_case_predictions(cases, selected_predictions)
    ranked = _choose_best_case(cases, eval_rows)
    if not ranked:
        raise RuntimeError("No suitable motivation case found from saved predictions.")

    best_case_id = ranked[0][0]
    case_ids = [best_case_id]
    if len(ranked) > 1 and ranked[1][1] > 0.95:
        case_ids.append(ranked[1][0])

    created: List[str] = []
    created.extend(_make_case_figure(cases[best_case_id], eval_rows, stem="motivation_compare_case_01"))
    created.extend(_make_case_figure(cases[best_case_id], eval_rows, stem="motivation_slide_best_figure"))
    created.extend(_make_logic_figure())
    index_path = _write_index(case_ids, cases, eval_rows, created)

    summary = {
        "selected_case_ids": case_ids,
        "best_case": best_case_id,
        "created_files": created + [index_path],
    }
    with open(OUT_DIR / "motivation_selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
