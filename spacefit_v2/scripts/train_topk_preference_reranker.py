"""Train a candidate-level reranker from top-k human preference labels."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spacefit_v2.scripts import analyze_human_aligned_scorer as scorer
from spacefit_v2.scripts.analyze_human_aligned_scorer import _feature_row, _make_model
from spacefit_v2.scripts.analyze_topk_labels import _dedupe_rows, _load_jsonl
from spacefit_v2.single_target.eval import aggregate_results, save_results

for _extra_feature in ["current_rank", "current_score", "original_rank"]:
    if _extra_feature not in scorer.NUMERIC_FEATURES:
        scorer.NUMERIC_FEATURES.append(_extra_feature)


DEFAULT_LABELS_DIR = Path("spacefit_v2/results/topk")
DEFAULT_CASES = Path("spacefit_v2/data/single_target_benchmark/gpt_intent_cases_test.json")
DEFAULT_MANIFEST = Path("spacefit_v2/data/single_target_benchmark/manifest.json")
DEFAULT_PREDICTIONS = Path("spacefit_v2/results/final_constraint_solver_human_rerank/test_gpt_intent/raw_predictions_human_reranked.json")
DEFAULT_OUT_DIR = Path("spacefit_v2/results/topk/preference_reranker")


def _load_cases(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_label_rows(labels_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(labels_dir.glob("*.jsonl")):
        rows.extend(_load_jsonl(path))
    return _dedupe_rows(rows)


def _target_category(case: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("category") or (case.get("target_asset") or {}).get("category") or "")


def _candidate_feature(
    case: Mapping[str, Any],
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    pred = candidate.get("prediction") or {}
    feature_row = {
        "case_id": row.get("case_id"),
        "method": row.get("method", "constraint_solver"),
        "target_category": _target_category(case, candidate),
        "room_type": (case.get("scene") or {}).get("room_type", ""),
        "prediction_status": candidate.get("status", pred.get("status", "")),
        "metrics": {"top1": candidate.get("metrics") or {}},
    }
    feature = _feature_row(feature_row, case=case, prediction=pred)
    feature["current_rank"] = float(candidate.get("rank", 0))
    feature["current_score"] = float(candidate.get("human_aligned_score") or 0.0)
    feature["original_rank"] = float(candidate.get("original_rank", candidate.get("rank", 0)) or 0)
    return feature


def _build_dataset(
    label_rows: Sequence[Mapping[str, Any]],
    cases_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    features: List[Dict[str, Any]] = []
    labels: List[int] = []
    groups: List[str] = []
    meta: List[Dict[str, Any]] = []

    for row in label_rows:
        case_id = str(row.get("case_id"))
        case = cases_by_id.get(case_id)
        if not case:
            continue
        human_label = row.get("human_label") or {}
        best_id = human_label.get("best_candidate_id")
        if not best_id:
            continue
        acceptable = set(human_label.get("acceptable_candidate_ids") or [])
        for candidate in row.get("candidates") or []:
            cid = str(candidate.get("candidate_id"))
            features.append(_candidate_feature(case, row, candidate))
            labels.append(1 if cid == best_id else 0)
            groups.append(case_id)
            meta.append(
                {
                    "case_id": case_id,
                    "candidate_id": cid,
                    "rank": int(candidate.get("rank", 0)),
                    "original_rank": candidate.get("original_rank"),
                    "selected_best": int(cid == best_id),
                    "acceptable": int(cid in acceptable),
                    "human_aligned_score": candidate.get("human_aligned_score"),
                    "cps": (candidate.get("metrics") or {}).get("cps"),
                    "constraint_accuracy": (candidate.get("metrics") or {}).get("constraint_accuracy"),
                }
            )
    return pd.DataFrame(features), np.asarray(labels), np.asarray(groups), meta


def _choose_by_score(rows: Sequence[Mapping[str, Any]], score_key: str) -> Dict[str, Mapping[str, Any]]:
    best_by_case: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        case_id = str(row["case_id"])
        current = best_by_case.get(case_id)
        if current is None or (float(row[score_key]), -int(row["rank"])) > (
            float(current[score_key]),
            -int(current["rank"]),
        ):
            best_by_case[case_id] = row
    return best_by_case


def _evaluate_choice(chosen: Mapping[str, Mapping[str, Any]], completed_cases: Sequence[str]) -> Dict[str, Any]:
    rows = [chosen[cid] for cid in completed_cases if cid in chosen]
    if not rows:
        return {}
    return {
        "cases": len(rows),
        "best_agreement": float(np.mean([r["selected_best"] for r in rows])),
        "acceptable_rate": float(np.mean([r["acceptable"] for r in rows])),
        "avg_human_rank": float(np.mean([r["rank"] for r in rows])),
        "cps": float(np.mean([float(r["cps"] or 0.0) for r in rows])),
        "constraint_accuracy": float(np.mean([float(r["constraint_accuracy"] or 0.0) for r in rows])),
    }


def _cross_validate(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    meta: Sequence[Mapping[str, Any]],
    model_kind: str,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    logo = LeaveOneGroupOut()
    scored_rows = [dict(row) for row in meta]
    probs = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in logo.split(X, y, groups):
        model = _make_model(model_kind)
        model.fit(X.iloc[train_idx], y[train_idx])
        probs[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    for row, prob in zip(scored_rows, probs):
        row["cv_preference_score"] = float(prob)

    completed_cases = sorted(set(groups.tolist()))
    baseline = _evaluate_choice({cid: row for cid, row in _choose_by_score(scored_rows, "negative_rank").items()}, completed_cases)
    cv_choice = _evaluate_choice(_choose_by_score(scored_rows, "cv_preference_score"), completed_cases)
    try:
        auc = float(roc_auc_score(y, probs))
    except ValueError:
        auc = float("nan")
    summary = {
        "model_kind": model_kind,
        "n_candidates": int(len(y)),
        "n_cases": int(len(completed_cases)),
        "positive_rate": float(np.mean(y)),
        "candidate_auc": auc,
        "baseline_current_rank0": baseline,
        "cv_preference_reranker": cv_choice,
    }
    return summary, scored_rows


def _fit_full_model(X: pd.DataFrame, y: np.ndarray, model_kind: str) -> Any:
    model = _make_model(model_kind)
    model.fit(X, y)
    return model


def _candidate_feature_for_prediction(
    case: Mapping[str, Any],
    method: str,
    prediction: Mapping[str, Any],
    rank: int,
) -> Dict[str, Any]:
    metrics = prediction.get("human_aligned_metrics") or {}
    candidate = {
        "rank": rank,
        "original_rank": prediction.get("human_aligned_original_rank", rank),
        "human_aligned_score": prediction.get("human_aligned_score", 0.0),
        "status": prediction.get("status", ""),
        "category": prediction.get("category") or (case.get("target_asset") or {}).get("category"),
        "metrics": metrics,
        "prediction": prediction,
    }
    return _candidate_feature(case, {"case_id": case["id"], "method": method}, candidate)


def _rerank_predictions(
    cases: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    model: Any,
    method: str,
) -> tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], List[Dict[str, Any]]]:
    cases_by_id = {str(case["id"]): case for case in cases}
    output: Dict[str, Dict[str, List[Dict[str, Any]]]] = {m: {} for m in predictions}
    score_rows: List[Dict[str, Any]] = []
    for m, by_case in predictions.items():
        for case_id, preds in by_case.items():
            case = cases_by_id.get(str(case_id))
            if not case or m != method:
                output[m][str(case_id)] = [deepcopy(p) for p in preds]
                continue
            scored: List[tuple[float, int, Dict[str, Any]]] = []
            for rank, pred in enumerate(preds or []):
                pred_copy = deepcopy(pred)
                feature = _candidate_feature_for_prediction(case, m, pred_copy, rank)
                prob = float(model.predict_proba(pd.DataFrame([feature]))[0, 1])
                pred_copy["topk_preference_score"] = prob
                pred_copy["topk_preference_original_rank"] = rank
                scored.append((prob, rank, pred_copy))
                score_rows.append(
                    {
                        "method": m,
                        "case_id": case_id,
                        "original_rank": rank,
                        "topk_preference_score": prob,
                        "previous_human_aligned_score": pred_copy.get("human_aligned_score"),
                        "cps": (pred_copy.get("human_aligned_metrics") or {}).get("cps"),
                        "constraint_accuracy": (pred_copy.get("human_aligned_metrics") or {}).get("constraint_accuracy"),
                    }
                )
            scored.sort(key=lambda item: (-item[0], item[1]))
            output[m][str(case_id)] = [item[2] for item in scored]
            for new_rank, pred in enumerate(output[m][str(case_id)]):
                pred["topk_preference_rank"] = new_rank
    return output, score_rows


def _metric_delta(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    old_by_source = {str(row["source_name"]): row for row in before}
    out: List[Dict[str, Any]] = []
    for row in after:
        old = old_by_source.get(str(row["source_name"]))
        if not old:
            continue
        delta = {"method": row["source_name"]}
        for key, val in row["metrics"].items():
            old_val = old["metrics"].get(key)
            delta[f"{key}_before"] = old_val
            delta[f"{key}_after"] = val
            delta[f"{key}_delta"] = float(val) - float(old_val)
        out.append(delta)
    return out


def _write_report(path: Path, cv_summary: Mapping[str, Any], metric_deltas: Sequence[Mapping[str, Any]]) -> None:
    b = cv_summary["baseline_current_rank0"]
    c = cv_summary["cv_preference_reranker"]
    lines = [
        "# Top-k Preference Reranker",
        "",
        "## Cross-Validation on Human Top-k Labels",
        "",
        f"- Cases: {cv_summary['n_cases']}",
        f"- Candidates: {cv_summary['n_candidates']}",
        f"- Positive rate: {cv_summary['positive_rate']:.3f}",
        f"- Candidate ROC-AUC: {cv_summary['candidate_auc']:.3f}",
        "",
        "| Selector | Human best agreement | Acceptable rate | Avg selected rank | CPS | Constraint Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Current rank-1 | {b['best_agreement']:.3f} | {b['acceptable_rate']:.3f} | {b['avg_human_rank']:.2f} | {b['cps']:.3f} | {b['constraint_accuracy']:.3f} |",
        f"| CV preference model | {c['best_agreement']:.3f} | {c['acceptable_rate']:.3f} | {c['avg_human_rank']:.2f} | {c['cps']:.3f} | {c['constraint_accuracy']:.3f} |",
        "",
        "## Full-Model Automatic Metric Delta",
        "",
        "The full model is trained on all completed top-k labels and then applied to the current candidate list.",
        "",
        "| Method | CPS | Success@1 | Success@5 | Constraint Accuracy | CF | IB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_deltas:
        lines.append(
            f"| {row['method']} | {row['CPS_delta']:+.3f} | {row['Success@1_delta']:+.3f} | "
            f"{row['Success@5_delta']:+.3f} | {row['Constraint Accuracy_delta']:+.3f} | "
            f"{row['CF_delta']:+.3f} | {row['IB_delta']:+.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Use cross-validation numbers for honest human-preference fit.",
        "- Use automatic metric delta only as a sanity check; human preference and code CPS are not identical objectives.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels_dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--method", default="constraint_solver")
    parser.add_argument("--model", choices=["rf", "logreg"], default="rf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    label_rows = _load_label_rows(args.labels_dir)
    cases = _load_cases(args.cases)
    cases_by_id = {str(case["id"]): case for case in cases}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))

    X, y, groups, meta = _build_dataset(label_rows, cases_by_id)
    if len(y) == 0:
        raise RuntimeError("No completed top-k preference labels were found.")
    for row in meta:
        row["negative_rank"] = -float(row["rank"])

    cv_summary, cv_rows = _cross_validate(X, y, groups, meta, args.model)
    model = _fit_full_model(X, y, args.model)
    reranked, score_rows = _rerank_predictions(cases, predictions, model, args.method)

    before = aggregate_results(cases, {args.method: predictions[args.method]})
    after = aggregate_results(cases, {args.method: reranked[args.method]})
    metric_deltas = _metric_delta(before, after)

    save_results(args.out_dir / "before_eval", manifest, before)
    save_results(args.out_dir / "after_eval", manifest, after)
    (args.out_dir / "raw_predictions_topk_preference_reranked.json").write_text(
        json.dumps(reranked, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(args.out_dir / "cv_candidate_scores.csv", cv_rows)
    _write_csv(args.out_dir / "full_candidate_scores.csv", score_rows)
    _write_csv(args.out_dir / "metric_deltas.csv", metric_deltas)
    summary = {
        "inputs": {
            "labels_dir": str(args.labels_dir),
            "cases": str(args.cases),
            "predictions": str(args.predictions),
            "method": args.method,
        },
        "cv_summary": cv_summary,
        "metric_deltas": metric_deltas,
    }
    (args.out_dir / "topk_preference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(args.out_dir / "TOPK_PREFERENCE_RERANKER.md", cv_summary, metric_deltas)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
