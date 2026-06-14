"""Apply the human-aligned seed scorer to real top-k benchmark candidates.

The training labels are the visual-audit top-1 labels. This script uses them as
a small pilot scorer, then reorders each method's existing top-k candidates and
re-evaluates the benchmark with the repository's unified metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from experiments.eval.unified_eval import _compute_candidate_metrics
from spacefit_v2.scripts.analyze_human_aligned_scorer import (
    _feature_row,
    _make_model,
    _overall,
    _read_jsonl,
    _rows_to_table,
)
from spacefit_v2.single_target.eval import aggregate_results, normalize_case_result, save_results


DEFAULT_METHODS = [
    "spacefit_gpt_text",
    "constraint_solver",
    "proposal_diffopt_constraint",
    "layoutgpt_direct",
]


def _load_cases(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(c["id"]): c for c in json.loads(path.read_text(encoding="utf-8"))}


def _target_category(case: Mapping[str, Any], prediction: Mapping[str, Any]) -> str:
    return str(
        prediction.get("category")
        or (case.get("target_asset") or {}).get("category")
        or ""
    )


def _candidate_metrics(case: Dict[str, Any], method: str, prediction: Dict[str, Any]) -> Dict[str, Any]:
    scene = normalize_case_result(case, method, [prediction])
    if not scene.candidates:
        return {}
    return _compute_candidate_metrics(scene, scene.candidates[0])


def _candidate_feature(
    case: Dict[str, Any],
    method: str,
    prediction: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    row = {
        "case_id": case["id"],
        "method": method,
        "target_category": _target_category(case, prediction),
        "room_type": (case.get("scene") or {}).get("room_type", ""),
        "prediction_status": prediction.get("status", ""),
        "metrics": {"top1": metrics},
    }
    return _feature_row(row, case=case, prediction=prediction)


def _train_model(
    labels_path: Path,
    cases: Mapping[str, Dict[str, Any]],
    predictions: Mapping[str, Any],
    model_kind: str,
) -> tuple[Any, Dict[str, Any]]:
    rows = _read_jsonl(labels_path)
    features, y_overall, _y_score, groups, _item_ids = _rows_to_table(rows, cases, predictions)
    if len(y_overall) == 0:
        raise RuntimeError("No usable human labels with overall_ok were found.")
    model = _make_model(model_kind)
    model.fit(pd.DataFrame(features), y_overall)
    summary = {
        "model_kind": model_kind,
        "n_train_items": int(len(y_overall)),
        "train_positive_rate": float(np.mean(y_overall)),
        "n_train_cases": int(len(set(groups.tolist()))),
    }
    return model, summary


def _score_and_rerank(
    cases: Mapping[str, Dict[str, Any]],
    predictions: Mapping[str, Any],
    methods: Sequence[str],
    model: Any,
) -> tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], List[Dict[str, Any]]]:
    reranked: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    score_rows: List[Dict[str, Any]] = []

    for method in methods:
        if method not in predictions:
            continue
        reranked[method] = {}
        for case_id, preds in predictions[method].items():
            case = cases.get(str(case_id))
            if not case:
                reranked[method][str(case_id)] = list(preds or [])
                continue
            scored: List[tuple[float, int, Dict[str, Any]]] = []
            for rank, pred in enumerate(preds or []):
                pred_copy = deepcopy(pred)
                metrics = _candidate_metrics(case, method, pred_copy)
                feature = _candidate_feature(case, method, pred_copy, metrics)
                prob = float(model.predict_proba(pd.DataFrame([feature]))[0, 1])
                pred_copy["human_aligned_score"] = prob
                pred_copy["human_aligned_original_rank"] = rank
                pred_copy["human_aligned_metrics"] = {
                    key: metrics.get(key)
                    for key in [
                        "cf",
                        "ib",
                        "constraint_accuracy",
                        "constraint_satisfied",
                        "reachability",
                        "walkability",
                        "cps",
                        "num_predicted",
                    ]
                }
                scored.append((prob, rank, pred_copy))
                score_rows.append(
                    {
                        "method": method,
                        "case_id": case_id,
                        "original_rank": rank,
                        "score": prob,
                        "status": pred_copy.get("status", ""),
                        "category": _target_category(case, pred_copy),
                        "cf": metrics.get("cf"),
                        "ib": metrics.get("ib"),
                        "constraint_accuracy": metrics.get("constraint_accuracy"),
                        "reachability": metrics.get("reachability"),
                        "walkability": metrics.get("walkability"),
                        "cps": metrics.get("cps"),
                    }
                )
            scored.sort(key=lambda item: (-item[0], item[1]))
            reranked[method][str(case_id)] = [item[2] for item in scored]
            for new_rank, pred in enumerate(reranked[method][str(case_id)]):
                pred["human_aligned_rank"] = new_rank
    return reranked, score_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _metric_delta(original_rows: Sequence[Mapping[str, Any]], rerank_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_name = {str(row["source_name"]): row for row in original_rows}
    out: List[Dict[str, Any]] = []
    for row in rerank_rows:
        source = str(row["source_name"])
        old = by_name.get(source)
        if not old:
            continue
        merged: Dict[str, Any] = {"method": source}
        for key, new_val in row["metrics"].items():
            old_val = old["metrics"].get(key)
            merged[f"{key}_original"] = old_val
            merged[f"{key}_reranked"] = new_val
            merged[f"{key}_delta"] = float(new_val) - float(old_val)
        out.append(merged)
    return out


def _write_report(out_dir: Path, train_summary: Mapping[str, Any], deltas: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Human-Aligned Top-k Rerank",
        "",
        "## Training",
        f"- Model: {train_summary['model_kind']}",
        f"- Human-labeled train items: {train_summary['n_train_items']}",
        f"- Train cases: {train_summary['n_train_cases']}",
        f"- Positive rate: {train_summary['train_positive_rate']:.3f}",
        "",
        "## Automatic Metric Delta",
        "",
        "| Method | CPS | Success@1 | Success@5 | Constraint Accuracy | CF | IB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in deltas:
        lines.append(
            f"| {row['method']} | "
            f"{row['CPS_delta']:+.3f} | {row['Success@1_delta']:+.3f} | {row['Success@5_delta']:+.3f} | "
            f"{row['Constraint Accuracy_delta']:+.3f} | {row['CF_delta']:+.3f} | {row['IB_delta']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Note",
            "- This is a pilot: labels cover top-1 outputs from 40 cases, not every top-k candidate.",
            "- A positive delta means the learned human-aligned score picked a better first candidate under code metrics.",
        ]
    )
    (out_dir / "HUMAN_ALIGNED_TOPK_RERANK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rerank benchmark top-k candidates with the human-aligned seed scorer.")
    p.add_argument("--labels", default="spacefit_v2/results/visual_audit_distribution/result/analysis/merged_visual_audit_labels.jsonl")
    p.add_argument("--cases", default="spacefit_v2/data/single_target_benchmark/gpt_intent_cases_test.json")
    p.add_argument("--manifest", default="spacefit_v2/data/single_target_benchmark/manifest.json")
    p.add_argument("--predictions", default="spacefit_v2/results/experiment_final/test_gpt_intent/raw_predictions.json")
    p.add_argument("--out_dir", default="spacefit_v2/results/visual_audit_distribution/result/analysis/human_aligned_topk_rerank")
    p.add_argument("--model", choices=["rf", "logreg"], default="rf")
    p.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    return p


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases_by_id = _load_cases(Path(args.cases))
    case_list = [cases_by_id[k] for k in cases_by_id]
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    methods = [m for m in args.methods if m in predictions]

    model, train_summary = _train_model(Path(args.labels), cases_by_id, predictions, args.model)
    reranked, score_rows = _score_and_rerank(cases_by_id, predictions, methods, model)

    selected_original = {method: predictions[method] for method in methods}
    original_aggregates = aggregate_results(case_list, selected_original)
    reranked_aggregates = aggregate_results(case_list, reranked)
    save_results(out_dir / "original_eval", manifest, original_aggregates)
    save_results(out_dir / "reranked_eval", manifest, reranked_aggregates)

    (out_dir / "raw_predictions_reranked.json").write_text(
        json.dumps(reranked, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(out_dir / "candidate_scores.csv", score_rows)
    deltas = _metric_delta(original_aggregates, reranked_aggregates)
    _write_csv(out_dir / "metric_deltas.csv", deltas)

    summary = {
        "inputs": {
            "labels": args.labels,
            "cases": args.cases,
            "predictions": args.predictions,
            "methods": methods,
        },
        "training": train_summary,
        "metric_deltas": deltas,
    }
    (out_dir / "rerank_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(out_dir, train_summary, deltas)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(build_parser().parse_args())
