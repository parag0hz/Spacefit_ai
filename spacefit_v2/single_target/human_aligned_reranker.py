"""Human-aligned top-k reranker for single-target placement predictions."""
from __future__ import annotations

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
    _read_jsonl,
    _rows_to_table,
)
from spacefit_v2.single_target.eval import normalize_case_result


def _case_map(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(c["id"]): dict(c) for c in cases}


def _target_category(case: Mapping[str, Any], prediction: Mapping[str, Any]) -> str:
    return str(prediction.get("category") or (case.get("target_asset") or {}).get("category") or "")


def candidate_metrics(case: Dict[str, Any], method: str, prediction: Dict[str, Any]) -> Dict[str, Any]:
    scene = normalize_case_result(case, method, [prediction])
    if not scene.candidates:
        return {}
    return _compute_candidate_metrics(scene, scene.candidates[0])


def candidate_feature(
    case: Dict[str, Any],
    method: str,
    prediction: Dict[str, Any],
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    row = {
        "case_id": case["id"],
        "method": method,
        "target_category": _target_category(case, prediction),
        "room_type": (case.get("scene") or {}).get("room_type", ""),
        "prediction_status": prediction.get("status", ""),
        "metrics": {"top1": dict(metrics)},
    }
    return _feature_row(row, case=case, prediction=prediction)


def train_human_aligned_model(
    *,
    labels_path: str | Path,
    cases: Sequence[Mapping[str, Any]] | Mapping[str, Dict[str, Any]],
    train_predictions: Mapping[str, Any] | str | Path,
    model_kind: str = "rf",
) -> tuple[Any, Dict[str, Any]]:
    if isinstance(cases, Mapping):
        cases_by_id = {str(k): v for k, v in cases.items()}
    else:
        cases_by_id = _case_map(cases)
    if isinstance(train_predictions, (str, Path)):
        predictions = json.loads(Path(train_predictions).read_text(encoding="utf-8"))
    else:
        predictions = train_predictions

    rows = _read_jsonl(Path(labels_path))
    features, y_overall, _y_score, groups, _item_ids = _rows_to_table(rows, cases_by_id, predictions)
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


def rerank_predictions(
    *,
    cases: Sequence[Mapping[str, Any]] | Mapping[str, Dict[str, Any]],
    predictions_by_method: Mapping[str, Mapping[str, Sequence[Dict[str, Any]]]],
    model: Any,
    methods: Sequence[str] | None = None,
) -> tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], List[Dict[str, Any]]]:
    cases_by_id = {str(k): v for k, v in cases.items()} if isinstance(cases, Mapping) else _case_map(cases)
    method_list = list(methods) if methods else list(predictions_by_method.keys())
    reranked: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    score_rows: List[Dict[str, Any]] = []

    for method, by_case in predictions_by_method.items():
        if method not in method_list:
            reranked[method] = {str(cid): [deepcopy(p) for p in preds] for cid, preds in by_case.items()}
            continue

        reranked[method] = {}
        for case_id, preds in by_case.items():
            case = cases_by_id.get(str(case_id))
            if not case:
                reranked[method][str(case_id)] = [deepcopy(p) for p in preds]
                continue

            scored: List[tuple[float, int, Dict[str, Any]]] = []
            for rank, pred in enumerate(preds or []):
                pred_copy = deepcopy(pred)
                metrics = candidate_metrics(case, method, pred_copy)
                feature = candidate_feature(case, method, pred_copy, metrics)
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
