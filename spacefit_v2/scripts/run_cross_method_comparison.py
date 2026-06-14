from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_spacefit_cross_method")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.eval.unified_eval import (
    CandidateScene,
    NormalizedPlacement,
    NormalizedScene,
    _aggregate_method,
    _compute_candidate_metrics,
    load_holodeck_scenes,
    load_layoutgpt_scenes,
    load_layoutvlm_scenes,
)
from spacefit_v2.cross_method.layoutvlm_proxy import export_layoutvlm_proxy_tasks
from spacefit_v2.single_target.eval import aggregate_results, normalize_case_result


BENCHMARK_DIR = ROOT / "spacefit_v2" / "data" / "single_target_benchmark"
BENCHMARK_CASES = BENCHMARK_DIR / "cases.json"
OUR_FAIR_PREDICTIONS = ROOT / "spacefit_v2" / "results" / "exp_a" / "raw_predictions.json"
SELECTED_CASES = ROOT / "spacefit_v2" / "results" / "visualizations" / "selected_cases.json"

METHOD_KEYS = [
    "heuristic_baseline",
    "proposal_heuristic",
    "proposal_diffopt_basic",
    "proposal_diffopt_constraint",
]

METHOD_TITLES = {
    "heuristic_baseline": "Heuristic",
    "proposal_heuristic": "Proposal + Heuristic",
    "proposal_diffopt_basic": "Proposal + DiffOpt-Basic",
    "proposal_diffopt_constraint": "Proposal + DiffOpt-Constraint",
}

MAIN_TABLE_LABELS = {
    "Heuristic Baseline": "Heuristic baseline",
    "Proposal + Heuristic Refinement": "Proposal + Heuristic",
    "Proposal + DiffOpt-Basic": "Proposal + DiffOpt-Basic",
    "Proposal + DiffOpt-Constraint": "Proposal + DiffOpt-Constraint",
}

DIAGNOSTIC_METHODS = [
    (
        "LayoutVLM diagnostic",
        load_layoutvlm_scenes,
        "Different benchmark and Objaverse asset domain; evaluated only diagnostically.",
    ),
    (
        "LayoutGPT diagnostic",
        load_layoutgpt_scenes,
        "Saved full-scene bedroom generation outputs on a different 3D-FRONT split; not the current single-target benchmark.",
    ),
    (
        "Holodeck diagnostic",
        load_holodeck_scenes,
        "Single saved free-form living-room generation example with AI2-THOR assets; diagnostic only.",
    ),
]


def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _slug(text: str) -> str:
    return (
        str(text or "")
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("'", "_")
        .strip("_")
    )


def _corners(cx: float, cz: float, width: float, depth: float, yaw_deg: float) -> List[Tuple[float, float]]:
    import math

    yaw = math.radians(float(yaw_deg))
    c = math.cos(yaw)
    s = math.sin(yaw)
    hw = width / 2.0
    hd = depth / 2.0
    points = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
    return [(cx + x * c - z * s, cz + x * s + z * c) for x, z in points]


def _draw_floor(ax: plt.Axes, polygon: Sequence[Tuple[float, float]]) -> None:
    patch = MplPolygon(list(polygon), closed=True, facecolor="#FBF8F1", edgecolor="#3A3A3A", linewidth=1.8)
    ax.add_patch(patch)


def _draw_box(
    ax: plt.Axes,
    center_x: float,
    center_z: float,
    width: float,
    depth: float,
    yaw_deg: float,
    face: str,
    edge: str,
    alpha: float = 0.70,
    linestyle: str = "-",
    linewidth: float = 1.3,
) -> None:
    patch = MplPolygon(
        _corners(center_x, center_z, width, depth, yaw_deg),
        closed=True,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
    )
    ax.add_patch(patch)


def _object_box(item: Mapping[str, Any]) -> Tuple[float, float, float, float, float]:
    position = item["position"]
    if isinstance(position, dict):
        center_x = float(position["x"])
        center_z = float(position["z"])
    else:
        center_x = float(position[0])
        center_z = float(position[2])

    size = item["size"]
    if isinstance(size, dict):
        width = float(size["width"])
        depth = float(size["depth"])
    else:
        width = float(size[0])
        depth = float(size[2])

    yaw = float(item.get("yaw", item.get("rotation_y", 0.0)))
    return center_x, center_z, width, depth, yaw


def _prediction_box(item: Mapping[str, Any]) -> Tuple[float, float, float, float, float]:
    position = item["position"]
    size = item["size"]
    return (
        float(position["x"]),
        float(position["z"]),
        float(size["width"]),
        float(size["depth"]),
        float(item.get("rotation_y", 0.0)),
    )


def _set_bounds(ax: plt.Axes, polygon: Sequence[Tuple[float, float]]) -> None:
    xs = [p[0] for p in polygon]
    zs = [p[1] for p in polygon]
    margin = 0.35
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(zs) - margin, max(zs) + margin)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _metric_summary(metrics: Mapping[str, Any]) -> str:
    return (
        f"CF {metrics['cf']:.0f}  "
        f"IB {metrics['ib']:.0f}  "
        f"CA {metrics['constraint_accuracy']:.2f}  "
        f"CPS {int(metrics['cps'])}"
    )


def _text_panel(ax: plt.Axes, title: str, lines: Sequence[str]) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=11, loc="left")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F1E7", edgecolor="#C5B9A6"),
    )


def _load_test_cases() -> List[Dict[str, Any]]:
    return [case for case in _load_json(BENCHMARK_CASES) if case["split"] == "test"]


def _load_main_predictions() -> Dict[str, Dict[str, Sequence[Dict[str, Any]]]]:
    raw = _load_json(OUR_FAIR_PREDICTIONS)
    return {key: raw[key] for key in METHOD_KEYS if key in raw}


def _fair_main_rows(cases: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    case_ids = {case["id"] for case in cases}
    predictions = _load_main_predictions()
    filtered = {
        method: {case_id: rows for case_id, rows in by_case.items() if case_id in case_ids}
        for method, by_case in predictions.items()
    }
    rows = aggregate_results(cases, filtered)
    for row in rows:
        row["method"] = MAIN_TABLE_LABELS.get(row["method"], row["method"])
        row["comparison_type"] = "fair_direct"
    return rows


def _diagnostic_rows() -> List[Dict[str, Any]]:
    rows = []
    for method_name, loader, note in DIAGNOSTIC_METHODS:
        scenes, loader_note = loader()
        row = _aggregate_method(method_name, scenes, source_name=method_name, notes=f"{loader_note} {note}")
        row["comparison_type"] = "diagnostic_only"
        rows.append(row)
    return rows


def _row_for_csv(row: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = row["metrics"]
    return {
        "method": row["method"],
        "comparison_type": row.get("comparison_type", ""),
        "num_scenes": row.get("num_scenes", 0),
        "CF": metrics["CF"],
        "IB": metrics["IB"],
        "Constraint Accuracy": metrics["Constraint Accuracy"],
        "CPS": metrics["CPS"],
        "Success@1": metrics["Success@1"],
        "Success@5": metrics["Success@5"],
        "Reachability": metrics["Reachability"],
        "Walkability": metrics["Walkability"],
        "notes": row.get("notes", ""),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "method",
        "comparison_type",
        "num_scenes",
        "CF",
        "IB",
        "Constraint Accuracy",
        "CPS",
        "Success@1",
        "Success@5",
        "Reachability",
        "Walkability",
        "notes",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_for_csv(row))


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Method | CF | IB | Constraint Accuracy | CPS | Success@1 | Success@5 | Reachability | Walkability | n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| {row['method']} | {m['CF']:.3f} | {m['IB']:.3f} | {m['Constraint Accuracy']:.3f} | "
            f"{m['CPS']:.3f} | {m['Success@1']:.3f} | {m['Success@5']:.3f} | "
            f"{m['Reachability']:.3f} | {m['Walkability']:.3f} | {row['num_scenes']} |"
        )
    return "\n".join(lines)


def _case_method_metrics(case: Dict[str, Any], method: str, preds: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scene = normalize_case_result(case, method, preds)
    candidate_metrics = [_compute_candidate_metrics(scene, candidate) for candidate in scene.candidates]
    return {
        "scene": scene,
        "candidate_metrics": candidate_metrics,
        "top1": candidate_metrics[0] if candidate_metrics else None,
    }


def _pick_case_ids(cases: Sequence[Dict[str, Any]], predictions: Mapping[str, Mapping[str, Sequence[Dict[str, Any]]]]) -> Tuple[str, str]:
    selected = _load_json(SELECTED_CASES) if SELECTED_CASES.exists() else {}
    success_id = None
    failure_id = None

    for key in selected.get("success", []):
        if key in {case["id"] for case in cases}:
            success_id = key
            break
    failure_geom = selected.get("failure_geom")
    if isinstance(failure_geom, str) and failure_geom in {case["id"] for case in cases}:
        failure_id = failure_geom

    if success_id and failure_id:
        return success_id, failure_id

    best_success = None
    best_failure = None
    for case in cases:
        top = _case_method_metrics(case, "proposal_diffopt_constraint", predictions["proposal_diffopt_constraint"].get(case["id"], []))["top1"]
        heur = _case_method_metrics(case, "heuristic_baseline", predictions["heuristic_baseline"].get(case["id"], []))["top1"]
        if top is None:
            continue
        if top["cps"] == 1:
            score = (top["constraint_accuracy"], top["walkability"])
            if best_success is None or score > best_success[0]:
                best_success = (score, case["id"])
        else:
            score = (top["ib"] == 0 or top["cf"] == 0, 1.0 - top["constraint_accuracy"], 1.0 - top["ib"], 1.0 - top["cf"])
            if best_failure is None or score > best_failure[0]:
                best_failure = (score, case["id"])
        if heur and heur["cps"] == 0 and top["cps"] == 1 and best_success is None:
            best_success = ((2.0, top["constraint_accuracy"]), case["id"])

    success_id = success_id or (best_success[1] if best_success else cases[0]["id"])
    failure_id = failure_id or (best_failure[1] if best_failure else cases[-1]["id"])
    return success_id, failure_id


def _render_benchmark_panel(
    ax: plt.Axes,
    case: Dict[str, Any],
    prediction: Mapping[str, Any] | None,
    title: str,
    metrics: Mapping[str, Any] | None = None,
    show_reference_outline: bool = False,
) -> None:
    polygon = [(float(x), float(z)) for x, z in case["scene"]["floor"]["polygon"]]
    _draw_floor(ax, polygon)
    for obj in case["scene"].get("objects", []):
        cx, cz, width, depth, yaw = _object_box(obj)
        _draw_box(ax, cx, cz, width, depth, yaw, face="#D8D8D8", edge="#7A7A7A", alpha=0.92)

    if show_reference_outline:
        ref = case["reference_pose"]
        cx = float(ref["position"]["x"])
        cz = float(ref["position"]["z"])
        width = float(case["target_asset"]["size"]["width"])
        depth = float(case["target_asset"]["size"]["depth"])
        yaw = float(ref["rotation_y"])
        _draw_box(ax, cx, cz, width, depth, yaw, face="#000000", edge="#000000", alpha=0.08, linestyle="--", linewidth=1.2)

    if prediction is not None and prediction.get("status") == "placed":
        cx, cz, width, depth, yaw = _prediction_box(prediction)
        _draw_box(ax, cx, cz, width, depth, yaw, face="#7BBE6A", edge="#2F6B1F", alpha=0.80, linewidth=1.4)

    _set_bounds(ax, polygon)
    subtitle = _metric_summary(metrics) if metrics is not None else f"target: {case['target_asset']['category']}"
    ax.set_title(f"{title}\n{subtitle}", fontsize=10, loc="left")


def _render_scene_panel(
    ax: plt.Axes,
    scene: NormalizedScene,
    title: str,
    note: str,
    placement_face: str = "#E9A66B",
    placement_edge: str = "#934A09",
) -> None:
    _draw_floor(ax, scene.floor_polygon)
    for obj in scene.fixed_objects:
        poly = obj.footprint
        if poly is None and obj.position and obj.size:
            cx, _, cz = obj.position
            width, _, depth = obj.size
            yaw = float(obj.yaw_deg or 0.0)
            _draw_box(ax, float(cx), float(cz), float(width), float(depth), yaw, face="#D8D8D8", edge="#7A7A7A", alpha=0.92)
    if scene.candidates:
        for placement in scene.candidates[0].placements:
            if placement.status != "placed":
                continue
            if placement.footprint:
                patch = MplPolygon(list(placement.footprint), closed=True, facecolor=placement_face, edgecolor=placement_edge, linewidth=1.1, alpha=0.70)
                ax.add_patch(patch)
            elif placement.position and placement.size:
                cx, _, cz = placement.position
                width, _, depth = placement.size
                yaw = float(placement.yaw_deg or 0.0)
                _draw_box(ax, float(cx), float(cz), float(width), float(depth), yaw, face=placement_face, edge=placement_edge, alpha=0.72)
    _set_bounds(ax, scene.floor_polygon)
    ax.set_title(f"{title}\n{note}", fontsize=10, loc="left")


def _save_success_failure_figures(
    out_dir: Path,
    cases_by_id: Mapping[str, Dict[str, Any]],
    predictions: Mapping[str, Mapping[str, Sequence[Dict[str, Any]]]],
    success_id: str,
    failure_id: str,
) -> None:
    for index, case_id in enumerate([success_id, failure_id], start=1):
        case = cases_by_id[case_id]
        method_preds = {
            method: list(predictions[method].get(case_id, []))
            for method in METHOD_KEYS
        }
        method_metrics = {
            method: _case_method_metrics(case, method, method_preds[method])["top1"]
            for method in METHOD_KEYS
        }
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = list(axes.flat)
        _render_benchmark_panel(axes[0], case, None, "Input room (target removed)", None, show_reference_outline=True)
        _render_benchmark_panel(axes[1], case, method_preds["heuristic_baseline"][0] if method_preds["heuristic_baseline"] else None, METHOD_TITLES["heuristic_baseline"], method_metrics["heuristic_baseline"])
        _render_benchmark_panel(axes[2], case, method_preds["proposal_heuristic"][0] if method_preds["proposal_heuristic"] else None, METHOD_TITLES["proposal_heuristic"], method_metrics["proposal_heuristic"])
        _render_benchmark_panel(axes[3], case, method_preds["proposal_diffopt_basic"][0] if method_preds["proposal_diffopt_basic"] else None, METHOD_TITLES["proposal_diffopt_basic"], method_metrics["proposal_diffopt_basic"])
        _render_benchmark_panel(axes[4], case, method_preds["proposal_diffopt_constraint"][0] if method_preds["proposal_diffopt_constraint"] else None, METHOD_TITLES["proposal_diffopt_constraint"], method_metrics["proposal_diffopt_constraint"])
        _render_benchmark_panel(axes[5], case, {
            "status": "placed",
            "position": {
                "x": float(case["reference_pose"]["position"]["x"]),
                "y": float(case["reference_pose"]["position"]["y"]),
                "z": float(case["reference_pose"]["position"]["z"]),
            },
            "rotation_y": float(case["reference_pose"]["rotation_y"]),
            "size": case["target_asset"]["size"],
        }, "Reference pose", {"cf": 1.0, "ib": 1.0, "constraint_accuracy": 1.0, "cps": 1})
        headline = "Success comparison" if index == 1 else "Failure comparison"
        fig.suptitle(f"{headline}: {case_id}", fontsize=14)
        fig.tight_layout()
        fig.savefig(out_dir / f"qualitative_compare_case_0{index}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def _save_fairness_figure(
    out_dir: Path,
    fair_case: Dict[str, Any],
    fair_prediction: Mapping[str, Any] | None,
    fair_metrics: Mapping[str, Any] | None,
) -> None:
    layoutvlm_scene = load_layoutvlm_scenes()[0][0]
    layoutgpt_scene = load_layoutgpt_scenes()[0][0]
    holodeck_scene = load_holodeck_scenes()[0][0]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = list(axes.flat)

    _render_benchmark_panel(axes[0], fair_case, None, "Fair direct input", None, show_reference_outline=True)
    _render_benchmark_panel(axes[1], fair_case, fair_prediction, "Fair direct result", fair_metrics)
    _text_panel(
        axes[2],
        "Why fair",
        [
            "Same 36-case single-target benchmark",
            "Same removed-target protocol",
            "Same unified evaluator",
            "Only our methods currently have saved outputs on this exact setup",
        ],
    )

    _render_scene_panel(axes[3], layoutvlm_scene, "LayoutVLM diagnostic", "Own saved benchmark scene")
    _render_scene_panel(axes[4], layoutgpt_scene, "LayoutGPT diagnostic", "Full-scene saved bedroom output")
    _render_scene_panel(axes[5], holodeck_scene, "Holodeck diagnostic", "Single free-form living-room scene")

    fig.suptitle("Fair vs diagnostic comparison scope", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "qualitative_compare_case_03.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _comparability_audit() -> Dict[str, Any]:
    return {
        "LayoutVLM": {
            "classification": "partially_adaptable",
            "input_format": "Room boundary plus explicit 3D assets, optionally with existing assets rendered into the prompt context.",
            "output_format": "Per-asset 3D poses in layout.json plus auxiliary constraints/render artifacts.",
            "adapter_needed": (
                "Single-target benchmark cases must be converted into LayoutVLM tasks, existing furniture must be treated as fixed context, "
                "and 3D-FRONT furniture must be proxied into LayoutVLM's Objaverse asset catalog."
            ),
            "fair_comparison_possible": False,
            "why_not_fair_now": (
                "The local LayoutVLM asset domain is Objaverse rather than raw 3D-FRONT furniture, and no fresh single-target rerun was possible "
                "in this workspace because no OpenAI API key was configured."
            ),
            "runnable_now": "task export only",
        },
        "LayoutGPT": {
            "classification": "partially_adaptable",
            "input_format": "Room type and room size prompt for full-scene layout generation over LayoutGPT's preprocessed 3D-FRONT data.",
            "output_format": "Full object_list scene layouts with category, size, position, and orientation for all generated furniture.",
            "adapter_needed": (
                "Would require a new prompt/interface that conditions on existing furniture and predicts only the removed target or a deterministic extraction rule "
                "from a newly rerun scene-level output."
            ),
            "fair_comparison_possible": False,
            "why_not_fair_now": (
                "Saved outputs are full-scene bedroom generations on a different split and have zero overlap with the current 12 bedroom test cases in the single-target benchmark."
            ),
            "runnable_now": "diagnostic evaluation from saved outputs",
        },
        "Holodeck": {
            "classification": "diagnostic_only",
            "input_format": "Free-form natural-language room query with AI2-THOR/Objathor generation pipeline.",
            "output_format": "Generated room/scene JSON with room geometry, objects, walls, doors, and windows.",
            "adapter_needed": "A complete task reformulation would be required; the method does not natively operate on furnished-room single-target replacement.",
            "fair_comparison_possible": False,
            "why_not_fair_now": "Only one saved free-form living-room example is present locally, and the task definition is fundamentally different.",
            "runnable_now": "diagnostic evaluation from saved scene",
        },
    }


def _limitations_markdown(audit: Mapping[str, Any], proxy_manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Cross-Method Limitations",
        "",
        f"- Generated on {datetime.now(timezone.utc).isoformat()}",
        "- All quantitative metrics reuse the repository's unified evaluator and keep the benchmark/task distinction explicit.",
        "- No original-paper headline numbers are copied into these tables.",
        "",
        "## Core blockers",
        "",
        "- No `OPENAI_API_KEY` was configured in the local environment on 2026-04-23, so fresh `LayoutVLM` or `Holodeck` generations could not be rerun from this workspace.",
        "- `LayoutVLM` uses Objaverse assets locally, while the current benchmark is built from raw 3D-FRONT furniture. This requires a proxy asset layer and therefore remains adapted rather than fair-direct.",
        "- `LayoutGPT` saved outputs are full-scene generations on a different 3D-FRONT split, with zero overlap against the current single-target bedroom test subset.",
        "",
        "## Method-by-method status",
        "",
    ]
    for name, item in audit.items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Classification: `{item['classification']}`",
                f"- Runnable now: `{item['runnable_now']}`",
                f"- Input: {item['input_format']}",
                f"- Output: {item['output_format']}",
                f"- Adapter needed: {item['adapter_needed']}",
                f"- Fair direct comparison now: `{item['fair_comparison_possible']}`",
                f"- Why not fair now: {item['why_not_fair_now']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Exported LayoutVLM proxy tasks",
            "",
            f"- Proxy task count: `{proxy_manifest.get('num_tasks', 0)}`",
            f"- Manifest: `{proxy_manifest.get('note', '')}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _comparison_markdown(
    main_rows: Sequence[Mapping[str, Any]],
    diagnostic_rows: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    out_dir: Path,
) -> str:
    lines = [
        "# Cross-Method Comparison",
        "",
        "## What was actually runnable",
        "",
        "- Fair direct evaluation was available for the saved `spacefit_v2/results/exp_a/raw_predictions.json` single-target runs on the 36-case test split.",
        "- Diagnostic evaluation was available for the saved local artifacts from `LayoutVLM`, `LayoutGPT`, and `Holodeck` via the shared evaluator.",
        "- Fresh adapted `LayoutVLM` reruns were not executed because the workspace had no OpenAI API key and the asset domain still needs proxy mapping.",
        "",
        "## Comparability audit",
        "",
        f"- LayoutVLM: `{audit['LayoutVLM']['classification']}`",
        f"- LayoutGPT: `{audit['LayoutGPT']['classification']}`",
        f"- Holodeck: `{audit['Holodeck']['classification']}`",
        "",
        "## Table 1: Fair / main comparison",
        "",
        _markdown_table(main_rows),
        "",
        "Interpretation: this is the only apples-to-apples table in the current workspace because all rows share the same 36-case single-target benchmark and the same evaluator.",
        "",
        "## Table 2: Diagnostic comparison",
        "",
        _markdown_table(diagnostic_rows),
        "",
        "Interpretation: these rows reuse the same metrics but not the same task/split/input conditions, so they are diagnostic only and should not be read as head-to-head benchmark rankings.",
        "",
        "## Fair vs adapted vs diagnostic",
        "",
        "- Fair direct: the four `spacefit_v2` single-target methods in Table 1.",
        "- Adapted comparison: `LayoutVLM` is the closest candidate and proxy task JSONs were exported, but no fresh run was completed in this workspace.",
        "- Diagnostic only: `LayoutGPT`, `Holodeck`, and the saved `LayoutVLM` examples in Table 2.",
        "",
        "## Qualitative outputs",
        "",
        f"- Success case: `{out_dir / 'qualitative_compare_case_01.png'}`",
        f"- Failure case: `{out_dir / 'qualitative_compare_case_02.png'}`",
        f"- Fair-vs-diagnostic illustration: `{out_dir / 'qualitative_compare_case_03.png'}`",
        "",
        "## Exact rerun commands",
        "",
        "```bash",
        "python -m spacefit_v2.scripts.run_single_target_benchmark \\",
        "  --split test --max_cases 36 \\",
        "  --out_dir spacefit_v2/results/exp_a",
        "",
        "python -m spacefit_v2.scripts.run_cross_method_comparison \\",
        "  --out_dir spacefit_v2/results/cross_method_comparison",
        "```",
        "",
        "The second command reproduces the tables, proxy LayoutVLM task exports, markdown reports, and qualitative figures from the already-saved local artifacts.",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(ROOT / "spacefit_v2" / "results" / "cross_method_comparison"))
    return parser


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_test_cases()
    cases_by_id = {case["id"]: case for case in cases}
    predictions = _load_main_predictions()

    proxy_manifest = export_layoutvlm_proxy_tasks(cases, out_dir / "layoutvlm_proxy_tasks")
    audit = _comparability_audit()

    main_rows = _fair_main_rows(cases)
    diagnostic_rows = _diagnostic_rows()

    generated_at = datetime.now(timezone.utc).isoformat()
    main_payload = {
        "generated_at": generated_at,
        "benchmark": {
            "path": str(BENCHMARK_DIR),
            "split": "test",
            "num_cases": len(cases),
        },
        "comparison_type": "fair_direct",
        "methods": main_rows,
    }
    diagnostic_payload = {
        "generated_at": generated_at,
        "comparison_type": "diagnostic_only",
        "methods": diagnostic_rows,
    }

    _write_json(out_dir / "cross_method_results_main.json", main_payload)
    _write_json(out_dir / "cross_method_results_diagnostic.json", diagnostic_payload)
    _write_csv(out_dir / "cross_method_results_main.csv", main_rows)
    _write_csv(out_dir / "cross_method_results_diagnostic.csv", diagnostic_rows)

    success_id, failure_id = _pick_case_ids(cases, predictions)
    _save_success_failure_figures(out_dir, cases_by_id, predictions, success_id, failure_id)
    fair_case = cases_by_id[success_id]
    fair_prediction = predictions["proposal_diffopt_constraint"].get(success_id, [None])[0]
    fair_metrics = _case_method_metrics(fair_case, "proposal_diffopt_constraint", predictions["proposal_diffopt_constraint"].get(success_id, []))["top1"]
    _save_fairness_figure(out_dir, fair_case, fair_prediction, fair_metrics)

    limitations_md = _limitations_markdown(audit, proxy_manifest)
    comparison_md = _comparison_markdown(main_rows, diagnostic_rows, audit, out_dir)
    (out_dir / "CROSS_METHOD_LIMITATIONS.md").write_text(limitations_md)
    (out_dir / "CROSS_METHOD_COMPARISON.md").write_text(comparison_md)

    summary = {
        "out_dir": str(out_dir),
        "main_json": str(out_dir / "cross_method_results_main.json"),
        "diagnostic_json": str(out_dir / "cross_method_results_diagnostic.json"),
        "comparison_md": str(out_dir / "CROSS_METHOD_COMPARISON.md"),
        "limitations_md": str(out_dir / "CROSS_METHOD_LIMITATIONS.md"),
        "success_case_figure": str(out_dir / "qualitative_compare_case_01.png"),
        "failure_case_figure": str(out_dir / "qualitative_compare_case_02.png"),
        "fairness_figure": str(out_dir / "qualitative_compare_case_03.png"),
        "layoutvlm_proxy_manifest": str(out_dir / "layoutvlm_proxy_tasks" / "manifest.json"),
        "selected_success_case": success_id,
        "selected_failure_case": failure_id,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(build_parser().parse_args())
