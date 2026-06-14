from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_spacefit_layoutvlm_adapted")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spacefit_v2.cross_method.layoutvlm_adapted import load_benchmark_cases, run_layoutvlm_adapted_benchmark
from spacefit_v2.scripts.run_cross_method_comparison import (
    BENCHMARK_CASES,
    METHOD_TITLES,
    _case_method_metrics,
    _load_main_predictions,
    _render_benchmark_panel,
    _text_panel,
)
from spacefit_v2.single_target.eval import aggregate_results


OUT_DIR_DEFAULT = ROOT / "spacefit_v2" / "results" / "layoutvlm_adapted_benchmark"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


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
            metrics = row["metrics"]
            writer.writerow(
                {
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
            )


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Method | CF | IB | Constraint Accuracy | CPS | Success@1 | Success@5 | Reachability | Walkability | n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"| {row['method']} | {metrics['CF']:.3f} | {metrics['IB']:.3f} | {metrics['Constraint Accuracy']:.3f} | "
            f"{metrics['CPS']:.3f} | {metrics['Success@1']:.3f} | {metrics['Success@5']:.3f} | "
            f"{metrics['Reachability']:.3f} | {metrics['Walkability']:.3f} | {row['num_scenes']} |"
        )
    return "\n".join(lines)


def _comparison_rows(
    cases: Sequence[Dict[str, Any]],
    layoutvlm_predictions: Mapping[str, Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    raw = _load_main_predictions()
    keep_methods = {
        "heuristic_baseline": raw["heuristic_baseline"],
        "proposal_heuristic": raw["proposal_heuristic"],
        "proposal_diffopt_constraint": raw["proposal_diffopt_constraint"],
        "layoutvlm_adapted_backend": layoutvlm_predictions,
    }
    rows = aggregate_results(cases, keep_methods)
    rename = {
        "Heuristic Baseline": "Heuristic baseline",
        "Proposal + Heuristic Refinement": "Proposal + Heuristic",
        "Proposal + DiffOpt-Constraint": "Proposal + DiffOpt-Constraint",
        "layoutvlm_adapted_backend": "LayoutVLM adapted backend",
    }
    notes = {
        "LayoutVLM adapted backend": (
            "Reuses LayoutVLM's local constraint optimizer only. "
            "This is an adapted backend baseline on the same benchmark, not a fair direct rerun of the original LLM/VLM pipeline."
        ),
        "Heuristic baseline": "Same saved 36-case benchmark result from the repository.",
        "Proposal + Heuristic": "Same saved 36-case benchmark result from the repository.",
        "Proposal + DiffOpt-Constraint": "Same saved 36-case benchmark result from the repository.",
    }
    for row in rows:
        row["method"] = rename.get(row["method"], row["method"])
        row["comparison_type"] = "adapted_benchmark"
        row["notes"] = notes.get(row["method"], row.get("notes", ""))
    return rows


def _pick_cases(
    cases: Sequence[Dict[str, Any]],
    layoutvlm_predictions: Mapping[str, Sequence[Dict[str, Any]]],
) -> Tuple[str, str]:
    best_case = None
    fail_case = None
    for case in cases:
        metrics = _case_method_metrics(case, "layoutvlm_adapted_backend", layoutvlm_predictions.get(case["id"], []))["top1"]
        if metrics is None:
            continue
        if metrics["cps"] == 1:
            score = (metrics["constraint_accuracy"], metrics["walkability"], metrics["cf"], metrics["ib"])
            if best_case is None or score > best_case[0]:
                best_case = (score, case["id"])
        else:
            score = (1.0 - metrics["cps"], 1.0 - metrics["constraint_accuracy"], 1.0 - metrics["cf"], 1.0 - metrics["ib"])
            if fail_case is None or score > fail_case[0]:
                fail_case = (score, case["id"])
    return (
        best_case[1] if best_case is not None else cases[0]["id"],
        fail_case[1] if fail_case is not None else cases[-1]["id"],
    )


def _render_compare_figure(
    out_path: Path,
    case: Dict[str, Any],
    layoutvlm_predictions: Mapping[str, Sequence[Dict[str, Any]]],
    ours_predictions: Mapping[str, Sequence[Dict[str, Any]]],
    title: str,
    note_lines: Sequence[str],
) -> None:
    layout_preds = list(layoutvlm_predictions.get(case["id"], []))
    our_preds = list(ours_predictions.get(case["id"], []))
    layout_metrics = _case_method_metrics(case, "layoutvlm_adapted_backend", layout_preds)["top1"]
    our_metrics = _case_method_metrics(case, "proposal_diffopt_constraint", our_preds)["top1"]
    reference_pred = {
        "status": "placed",
        "position": {
            "x": float(case["reference_pose"]["position"]["x"]),
            "y": float(case["reference_pose"]["position"]["y"]),
            "z": float(case["reference_pose"]["position"]["z"]),
        },
        "rotation_y": float(case["reference_pose"]["rotation_y"]),
        "size": dict(case["target_asset"]["size"]),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))
    axes = list(axes.flat)
    _render_benchmark_panel(axes[0], case, None, "Input room (target removed)", None, show_reference_outline=True)
    _render_benchmark_panel(
        axes[1],
        case,
        our_preds[0] if our_preds else None,
        METHOD_TITLES["proposal_diffopt_constraint"],
        our_metrics,
    )
    _render_benchmark_panel(
        axes[2],
        case,
        layout_preds[0] if layout_preds else None,
        "LayoutVLM adapted backend",
        layout_metrics,
    )
    _text_panel(
        axes[3],
        "Reference / Notes",
        [
            f"case: {case['id']}",
            f"target: {case['target_asset']['category']}",
            f"intent: {case['intent']['text']}",
            "",
            *note_lines,
            "",
            f"reference yaw: {reference_pred['rotation_y']:.1f}",
        ],
    )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _render_overview_figure(
    out_path: Path,
    rows: Sequence[Mapping[str, Any]],
    limitations: Sequence[str],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis("off")
    lines = [
        "LayoutVLM Adapted Benchmark Summary",
        "",
        "Quantitative results",
        "",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"- {row['method']}: CF {metrics['CF']:.3f}, IB {metrics['IB']:.3f}, "
            f"CA {metrics['Constraint Accuracy']:.3f}, CPS {metrics['CPS']:.3f}"
        )
    lines.extend(
        [
            "",
            "Interpretation",
            "",
            *[f"- {line}" for line in limitations],
        ]
    )
    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F6F2E9", edgecolor="#C8BCA9"),
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _report_markdown(
    rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
    num_cases: int,
    num_restarts: int,
    iterations: int,
    learning_rate: float,
) -> str:
    lines = [
        "# LayoutVLM Adapted Benchmark Report",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Benchmark: `{BENCHMARK_CASES}`",
        f"- Test cases: `{num_cases}`",
        "",
        "## What was run",
        "",
        "- The run reuses LayoutVLM's local differentiable constraint optimizer.",
        "- It does not invoke the original OpenAI-based prompt-generation front-end.",
        "- Existing furniture is kept fixed from the benchmark scene, and only the removed target asset is optimized.",
        "- Supported adapted constraints: `against_wall`, `near`, `facing`.",
        "- Unsupported benchmark constraints are still judged by the unified evaluator, but they are not directly optimized inside the LayoutVLM backend.",
        "",
        "## Quantitative comparison",
        "",
        _markdown_table(rows),
        "",
        "Interpretation: `LayoutVLM adapted backend` is informative on the same benchmark, but it should still be reported as an adapted baseline rather than a fair direct rerun of the original LayoutVLM pipeline.",
        "",
        "## Saved files",
        "",
        f"- Raw predictions: `{out_dir / 'raw_predictions.json'}`",
        f"- Quantitative JSON: `{out_dir / 'results.json'}`",
        f"- Quantitative CSV: `{out_dir / 'results.csv'}`",
        f"- Success figure: `{out_dir / 'qualitative_layoutvlm_case_01.png'}`",
        f"- Failure figure: `{out_dir / 'qualitative_layoutvlm_case_02.png'}`",
        f"- Overview figure: `{out_dir / 'qualitative_layoutvlm_case_03.png'}`",
        "",
        "## Exact rerun command",
        "",
        "```bash",
        "conda run -n layoutvlm python -m spacefit_v2.scripts.run_layoutvlm_adapted_benchmark \\",
        f"  --out_dir {out_dir} \\",
        f"  --num_restarts {num_restarts} \\",
        f"  --iterations {iterations} \\",
        f"  --learning_rate {learning_rate}",
        "```",
        "",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_cases", default=str(BENCHMARK_CASES))
    parser.add_argument("--out_dir", default=str(OUT_DIR_DEFAULT))
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--num_restarts", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=220)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [case for case in load_benchmark_cases(args.benchmark_cases) if case["split"] == args.split]
    if args.max_cases:
        cases = cases[: args.max_cases]

    layoutvlm_predictions = run_layoutvlm_adapted_benchmark(
        cases,
        num_restarts=args.num_restarts,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    raw_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "layoutvlm_adapted_backend",
        "comparison_type": "adapted_benchmark",
        "predictions": layoutvlm_predictions,
    }
    _write_json(out_dir / "raw_predictions.json", raw_payload)

    rows = _comparison_rows(cases, layoutvlm_predictions)
    results_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparison_type": "adapted_benchmark",
        "methods": rows,
    }
    _write_json(out_dir / "results.json", results_payload)
    _write_csv(out_dir / "results.csv", rows)

    cases_by_id = {case["id"]: case for case in cases}
    our_preds = _load_main_predictions()["proposal_diffopt_constraint"]
    success_id, failure_id = _pick_cases(cases, layoutvlm_predictions)
    _render_compare_figure(
        out_dir / "qualitative_layoutvlm_case_01.png",
        case=cases_by_id[success_id],
        layoutvlm_predictions=layoutvlm_predictions,
        ours_predictions=our_preds,
        title="LayoutVLM adapted success case",
        note_lines=[
            "LayoutVLM backend only",
            "Existing furniture fixed",
            "Unified evaluator reused",
        ],
    )
    _render_compare_figure(
        out_dir / "qualitative_layoutvlm_case_02.png",
        case=cases_by_id[failure_id],
        layoutvlm_predictions=layoutvlm_predictions,
        ours_predictions=our_preds,
        title="LayoutVLM adapted failure case",
        note_lines=[
            "Typical failure mode: unsupported protocol constraints",
            "or poor wall / anchor initialization",
        ],
    )
    _render_overview_figure(
        out_dir / "qualitative_layoutvlm_case_03.png",
        rows=rows,
        limitations=[
            "This is adapted backend evaluation, not the original LayoutVLM full pipeline.",
            "The backend directly optimizes only against_wall / near / facing.",
            "keep_window_clear / access_zone / not_block_door remain evaluation-only constraints.",
        ],
    )

    report = _report_markdown(
        rows=rows,
        out_dir=out_dir,
        num_cases=len(cases),
        num_restarts=args.num_restarts,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
    )
    (out_dir / "LAYOUTVLM_ADAPTED_REPORT.md").write_text(report)

    summary = {
        "out_dir": str(out_dir),
        "num_cases": len(cases),
        "results_json": str(out_dir / "results.json"),
        "results_csv": str(out_dir / "results.csv"),
        "report_md": str(out_dir / "LAYOUTVLM_ADAPTED_REPORT.md"),
        "success_case": success_id,
        "failure_case": failure_id,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(build_parser().parse_args())
