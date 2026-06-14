from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_spacefit_layoutvlm_llm")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spacefit_v2.cross_method.layoutvlm_llm import get_openai_api_key, run_layoutvlm_llm_benchmark
from spacefit_v2.scripts.run_cross_method_comparison import (
    BENCHMARK_CASES,
    METHOD_TITLES,
    _case_method_metrics,
    _load_main_predictions,
    _render_benchmark_panel,
    _text_panel,
)
from spacefit_v2.single_target.eval import aggregate_results


OUT_DIR_DEFAULT = ROOT / "spacefit_v2" / "results" / "layoutvlm_llm_benchmark"


def _load_cases(path: str | Path, split: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return [case for case in json.load(f) if case["split"] == split]


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
        "layoutvlm_llm_single_target": layoutvlm_predictions,
    }
    rows = aggregate_results(cases, keep_methods)
    rename = {
        "Heuristic Baseline": "Heuristic baseline",
        "Proposal + Heuristic Refinement": "Proposal + Heuristic",
        "Proposal + DiffOpt-Constraint": "Proposal + DiffOpt-Constraint",
        "layoutvlm_llm_single_target": "LayoutVLM LLM adapted",
    }
    notes = {
        "LayoutVLM LLM adapted": (
            "Uses the actual LayoutVLM OpenAI-based prompt-generation path with fixed existing furniture and proxy Objaverse assets. "
            "This is an adapted benchmark run, not a fair direct rerun on raw 3D-FRONT assets."
        )
    }
    for row in rows:
        row["method"] = rename.get(row["method"], row["method"])
        row["comparison_type"] = "adapted_llm_benchmark"
        row["notes"] = notes.get(row["method"], row.get("notes", ""))
    return rows


def _pick_cases(
    cases: Sequence[Dict[str, Any]],
    layoutvlm_predictions: Mapping[str, Sequence[Dict[str, Any]]],
) -> Tuple[str, str]:
    best_case = None
    fail_case = None
    for case in cases:
        metrics = _case_method_metrics(case, "layoutvlm_llm_single_target", layoutvlm_predictions.get(case["id"], []))["top1"]
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
    layout_metrics = _case_method_metrics(case, "layoutvlm_llm_single_target", layout_preds)["top1"]
    our_metrics = _case_method_metrics(case, "proposal_diffopt_constraint", our_preds)["top1"]

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
        "LayoutVLM LLM adapted",
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
        ],
    )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _report_markdown(
    rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
    num_cases: int,
    model_name: str,
    max_attempts: int,
) -> str:
    lines = [
        "# LayoutVLM LLM Benchmark Report",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Benchmark: `{BENCHMARK_CASES}`",
        f"- Test cases: `{num_cases}`",
        f"- Model: `{model_name}`",
        f"- Max attempts per case: `{max_attempts}`",
        "",
        "## What was run",
        "",
        "- This run uses the actual LayoutVLM OpenAI-based prompt-generation path.",
        "- Existing furniture is injected as fixed context and only the removed target is solved.",
        "- Objaverse proxy assets are still used, so this remains an adapted comparison rather than a fair direct rerun on raw 3D-FRONT assets.",
        "",
        "## Quantitative comparison",
        "",
        _markdown_table(rows),
        "",
        "## Saved files",
        "",
        f"- Raw predictions: `{out_dir / 'raw_predictions.json'}`",
        f"- Quantitative JSON: `{out_dir / 'results.json'}`",
        f"- Quantitative CSV: `{out_dir / 'results.csv'}`",
        f"- Success figure: `{out_dir / 'qualitative_layoutvlm_llm_case_01.png'}`",
        f"- Failure figure: `{out_dir / 'qualitative_layoutvlm_llm_case_02.png'}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_cases", default=str(BENCHMARK_CASES))
    parser.add_argument("--out_dir", default=str(OUT_DIR_DEFAULT))
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--model_name", default="gpt-4o")
    parser.add_argument("--max_attempts", type=int, default=1)
    parser.add_argument("--api_key", default="")
    return parser


def main(args: argparse.Namespace) -> None:
    api_key = get_openai_api_key(args.api_key or None)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_cases(args.benchmark_cases, args.split)
    if args.max_cases:
        cases = cases[: args.max_cases]

    layoutvlm_predictions = run_layoutvlm_llm_benchmark(
        cases,
        out_dir=out_dir / "case_runs",
        api_key=api_key,
        model_name=args.model_name,
        max_attempts=args.max_attempts,
    )

    raw_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "layoutvlm_llm_single_target",
        "comparison_type": "adapted_llm_benchmark",
        "predictions": layoutvlm_predictions,
    }
    _write_json(out_dir / "raw_predictions.json", raw_payload)

    rows = _comparison_rows(cases, layoutvlm_predictions)
    results_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparison_type": "adapted_llm_benchmark",
        "methods": rows,
    }
    _write_json(out_dir / "results.json", results_payload)
    _write_csv(out_dir / "results.csv", rows)

    cases_by_id = {case["id"]: case for case in cases}
    our_preds = _load_main_predictions()["proposal_diffopt_constraint"]
    success_id, failure_id = _pick_cases(cases, layoutvlm_predictions)
    _render_compare_figure(
        out_dir / "qualitative_layoutvlm_llm_case_01.png",
        case=cases_by_id[success_id],
        layoutvlm_predictions=layoutvlm_predictions,
        ours_predictions=our_preds,
        title="LayoutVLM LLM success case",
        note_lines=[
            "Actual OpenAI-backed LayoutVLM prompt path",
            "Existing furniture fixed",
            "Objaverse proxy assets",
        ],
    )
    _render_compare_figure(
        out_dir / "qualitative_layoutvlm_llm_case_02.png",
        case=cases_by_id[failure_id],
        layoutvlm_predictions=layoutvlm_predictions,
        ours_predictions=our_preds,
        title="LayoutVLM LLM failure case",
        note_lines=[
            "Adapted run with proxy assets",
            "Failure can come from LLM constraint code or proxy mismatch",
        ],
    )

    report = _report_markdown(rows, out_dir=out_dir, num_cases=len(cases), model_name=args.model_name, max_attempts=args.max_attempts)
    (out_dir / "LAYOUTVLM_LLM_REPORT.md").write_text(report)

    summary = {
        "out_dir": str(out_dir),
        "num_cases": len(cases),
        "results_json": str(out_dir / "results.json"),
        "results_csv": str(out_dir / "results.csv"),
        "report_md": str(out_dir / "LAYOUTVLM_LLM_REPORT.md"),
        "success_case": success_id,
        "failure_case": failure_id,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(build_parser().parse_args())
