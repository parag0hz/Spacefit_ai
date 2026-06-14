"""Analyze human labels and train a lightweight human-aligned reranker.

This script uses the visual-audit labels as a small seed set. It does not train
a production model yet; it answers:
  1. Which automatic features correlate with human acceptability?
  2. Can a simple scorer choose a better method per case than fixed methods?
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


QIDS = ["physical_ok", "relation_ok", "orientation_ok", "access_ok", "naturalness_ok", "overall_ok"]
NUMERIC_FEATURES = [
    "cf",
    "ib",
    "constraint_accuracy",
    "reachability",
    "walkability",
    "cps",
    "num_predicted",
    "constraint_satisfied",
    "complete_count_ok",
    "x",
    "z",
    "yaw_sin",
    "yaw_cos",
    "target_width",
    "target_depth",
    "target_area",
    "room_width",
    "room_depth",
    "room_area",
    "rel_x",
    "rel_z",
    "area_ratio",
    "nearest_wall_dist",
    "nearest_window_dist",
    "nearest_door_dist",
    "nearest_object_dist",
    "nearest_ref_dist",
    "num_objects",
    "num_windows",
    "num_doors",
]
CATEGORICAL_FEATURES = ["method", "target_category", "room_type", "prediction_status"]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _bool_float(value: Any) -> Optional[float]:
    if value is True:
        return 1.0
    if value is False:
        return 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _human_score(row: Mapping[str, Any]) -> Optional[float]:
    labels = row.get("human_labels") or {}
    vals = [_bool_float(labels.get(q)) for q in QIDS]
    vals = [v for v in vals if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _overall(row: Mapping[str, Any]) -> Optional[int]:
    val = (row.get("human_labels") or {}).get("overall_ok")
    if val is True:
        return 1
    if val is False:
        return 0
    return None


def _position_xz(item: Mapping[str, Any]) -> Tuple[float, float]:
    pos = item.get("position", {})
    if isinstance(pos, Mapping):
        return float(pos.get("x", 0.0)), float(pos.get("z", 0.0))
    if isinstance(pos, (list, tuple)):
        return float(pos[0]), float(pos[2] if len(pos) > 2 else pos[1])
    return float(item.get("x", 0.0)), float(item.get("z", 0.0))


def _size_wd(item: Mapping[str, Any]) -> Tuple[float, float]:
    size = item.get("size", {})
    if isinstance(size, Mapping):
        return float(size.get("width", 0.5)), float(size.get("depth", 0.5))
    if isinstance(size, (list, tuple)):
        return float(size[0]), float(size[2] if len(size) > 2 else size[-1])
    return float(item.get("width", 0.5)), float(item.get("depth", 0.5))


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _point_segment_dist(px: float, pz: float, ax: float, az: float, bx: float, bz: float) -> float:
    vx, vz = bx - ax, bz - az
    wx, wz = px - ax, pz - az
    denom = vx * vx + vz * vz
    t = 0.0 if denom <= 1e-9 else max(0.0, min(1.0, (wx * vx + wz * vz) / denom))
    cx, cz = ax + t * vx, az + t * vz
    return _dist((px, pz), (cx, cz))


def _nearest_wall_dist(point: Tuple[float, float], floor: List[List[float]]) -> float:
    pts = [(float(x), float(z)) for x, z in floor]
    if len(pts) < 2:
        return float("nan")
    best = float("inf")
    for i, (ax, az) in enumerate(pts):
        bx, bz = pts[(i + 1) % len(pts)]
        best = min(best, _point_segment_dist(point[0], point[1], ax, az, bx, bz))
    return float(best)


def _reference_categories(case: Optional[Mapping[str, Any]]) -> set[str]:
    cats: set[str] = set()
    if not case:
        return cats
    for c in case.get("intent", {}).get("constraints", []):
        if c.get("target_category"):
            cats.add(str(c["target_category"]))
    return cats


def _prediction_for(row: Mapping[str, Any], predictions: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    method = str(row.get("method", ""))
    cid = str(row.get("case_id", ""))
    preds = predictions.get(method, {}).get(cid) if predictions else None
    if preds:
        return preds[0]
    return None


def _feature_row(
    row: Mapping[str, Any],
    case: Optional[Mapping[str, Any]] = None,
    prediction: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    top1 = (row.get("metrics") or {}).get("top1") or {}
    out: Dict[str, Any] = {
        "method": row.get("method", ""),
        "target_category": row.get("target_category", ""),
        "room_type": row.get("room_type", ""),
        "prediction_status": row.get("prediction_status", ""),
    }
    for key in NUMERIC_FEATURES:
        out.setdefault(key, np.nan)
    for key in [
        "cf",
        "ib",
        "constraint_accuracy",
        "reachability",
        "walkability",
        "cps",
        "num_predicted",
        "constraint_satisfied",
        "complete_count_ok",
    ]:
        val = top1.get(key)
        if isinstance(val, bool):
            out[key] = float(val)
        elif isinstance(val, (int, float)):
            out[key] = float(val)

    if case and prediction and prediction.get("status") == "placed":
        x, z = _position_xz(prediction)
        w, d = _size_wd(prediction)
        yaw = math.radians(float(prediction.get("rotation_y", 0.0) or 0.0))
        scene = case.get("scene", {})
        floor = scene.get("floor", {}).get("polygon", [])
        floor_pts = [(float(px), float(pz)) for px, pz in floor]
        xs = [p[0] for p in floor_pts] or [0.0]
        zs = [p[1] for p in floor_pts] or [0.0]
        room_w = max(xs) - min(xs)
        room_d = max(zs) - min(zs)
        room_area = max(room_w * room_d, 1e-6)
        point = (x, z)
        objects = scene.get("objects", [])
        windows = scene.get("windows", [])
        doors = scene.get("doors", [])
        ref_cats = _reference_categories(case)
        obj_dists = [_dist(point, _position_xz(o)) for o in objects]
        ref_dists = [_dist(point, _position_xz(o)) for o in objects if str(o.get("category", "")) in ref_cats]
        win_dists = [_dist(point, _position_xz(o)) for o in windows]
        door_dists = [_dist(point, _position_xz(o)) for o in doors]
        out.update(
            {
                "x": x,
                "z": z,
                "yaw_sin": math.sin(yaw),
                "yaw_cos": math.cos(yaw),
                "target_width": w,
                "target_depth": d,
                "target_area": w * d,
                "room_width": room_w,
                "room_depth": room_d,
                "room_area": room_area,
                "rel_x": (x - min(xs)) / max(room_w, 1e-6),
                "rel_z": (z - min(zs)) / max(room_d, 1e-6),
                "area_ratio": (w * d) / room_area,
                "nearest_wall_dist": _nearest_wall_dist(point, floor),
                "nearest_window_dist": min(win_dists) if win_dists else np.nan,
                "nearest_door_dist": min(door_dists) if door_dists else np.nan,
                "nearest_object_dist": min(obj_dists) if obj_dists else np.nan,
                "nearest_ref_dist": min(ref_dists) if ref_dists else np.nan,
                "num_objects": float(len(objects)),
                "num_windows": float(len(windows)),
                "num_doors": float(len(doors)),
            }
        )
    return out


def _rows_to_table(
    rows: List[Mapping[str, Any]],
    cases: Mapping[str, Any],
    predictions: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, List[str]]:
    features: List[Dict[str, Any]] = []
    y_overall: List[int] = []
    y_score: List[float] = []
    groups: List[str] = []
    for row in rows:
        overall = _overall(row)
        score = _human_score(row)
        if overall is None or score is None:
            continue
        case = cases.get(str(row.get("case_id", "")))
        prediction = _prediction_for(row, predictions)
        features.append(_feature_row(row, case=case, prediction=prediction))
        y_overall.append(int(overall))
        y_score.append(float(score))
        groups.append(str(row["case_id"]))
    item_ids = [str(row["item_id"]) for row in rows if _overall(row) is not None and _human_score(row) is not None]
    return features, np.asarray(y_overall), np.asarray(y_score), np.asarray(groups), item_ids


def _make_preprocessor() -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    try:
        imputer = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
    except TypeError:
        imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", imputer), ("scaler", StandardScaler())]),
                NUMERIC_FEATURES,
            ),
            ("cat", encoder, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _make_model(kind: str) -> Pipeline:
    pre = _make_preprocessor()
    if kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
        )
    else:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])


def _splits(groups: np.ndarray) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    n_groups = len(set(groups.tolist()))
    if n_groups <= 10:
        return LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups)
    return GroupKFold(n_splits=5).split(np.zeros(len(groups)), groups=groups)


def _cv_predict(features: List[Dict[str, Any]], y: np.ndarray, groups: np.ndarray, kind: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    pred = np.full(len(y), np.nan)
    fold_rows: List[Dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(_splits(groups), start=1):
        model = _make_model(kind)
        x_train = pd.DataFrame([features[i] for i in train_idx])
        x_test = pd.DataFrame([features[i] for i in test_idx])
        model.fit(x_train, y[train_idx])
        prob = model.predict_proba(x_test)[:, 1]
        pred[test_idx] = prob
        yp = (prob >= 0.5).astype(int)
        yt = y[test_idx]
        fold_rows.append(
            {
                "fold": fold,
                "n": int(len(test_idx)),
                "accuracy": float(accuracy_score(yt, yp)),
                "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
                "auc": float(roc_auc_score(yt, prob)) if len(set(yt.tolist())) > 1 else None,
            }
        )
    return pred, fold_rows


def _metrics(y: np.ndarray, prob: np.ndarray) -> Dict[str, Any]:
    mask = ~np.isnan(prob)
    yt = y[mask]
    pp = prob[mask]
    yp = (pp >= 0.5).astype(int)
    return {
        "n": int(len(yt)),
        "positive_rate": float(np.mean(yt)) if len(yt) else None,
        "pred_positive_rate": float(np.mean(yp)) if len(yp) else None,
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "auc": float(roc_auc_score(yt, pp)) if len(set(yt.tolist())) > 1 else None,
    }


def _get_feature_names(model: Pipeline) -> List[str]:
    pre = model.named_steps["pre"]
    names: List[str] = []
    names.extend(NUMERIC_FEATURES)
    enc = pre.named_transformers_["cat"]
    cat_names = enc.get_feature_names_out(CATEGORICAL_FEATURES)
    names.extend([str(x) for x in cat_names])
    return names


def _full_fit_importance(features: List[Dict[str, Any]], y: np.ndarray, out_dir: Path) -> Dict[str, float]:
    model = _make_model("logreg")
    model.fit(pd.DataFrame(features), y)
    names = _get_feature_names(model)
    coefs = model.named_steps["clf"].coef_[0]
    pairs = sorted(zip(names, coefs), key=lambda x: abs(x[1]), reverse=True)
    with (out_dir / "feature_coefficients.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "coefficient", "abs_coefficient"])
        for name, coef in pairs:
            writer.writerow([name, float(coef), float(abs(coef))])
    top = pairs[:18]
    labels = [p[0] for p in top][::-1]
    vals = [p[1] for p in top][::-1]
    colors = ["#2ca25f" if v >= 0 else "#de2d26" for v in vals]
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=160)
    ax.barh(labels, vals, color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("Human overall scorer: strongest coefficients")
    ax.set_xlabel("Logistic regression coefficient")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_coefficients.png", bbox_inches="tight")
    plt.close(fig)
    return {name: float(coef) for name, coef in pairs}


def _rerank(rows: List[Mapping[str, Any]], scored_probs: Mapping[str, float], out_dir: Path) -> Dict[str, Any]:
    by_case: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("item_id") in scored_probs and _overall(row) is not None and _human_score(row) is not None:
            by_case[str(row["case_id"])].append(row)

    selection_rows: List[Dict[str, Any]] = []
    selected_overall: List[int] = []
    selected_scores: List[float] = []
    oracle_overall: List[int] = []
    oracle_scores: List[float] = []
    method_overalls: Dict[str, List[int]] = defaultdict(list)
    method_scores: Dict[str, List[float]] = defaultdict(list)

    for cid, items in sorted(by_case.items()):
        if not items:
            continue
        chosen = max(items, key=lambda r: scored_probs[str(r["item_id"])])
        oracle = max(items, key=lambda r: (_human_score(r) or 0.0, _overall(r) or 0))
        co = int(_overall(chosen) or 0)
        cs = float(_human_score(chosen) or 0.0)
        oo = int(_overall(oracle) or 0)
        os = float(_human_score(oracle) or 0.0)
        selected_overall.append(co)
        selected_scores.append(cs)
        oracle_overall.append(oo)
        oracle_scores.append(os)
        for item in items:
            method = str(item["method"])
            method_overalls[method].append(int(_overall(item) or 0))
            method_scores[method].append(float(_human_score(item) or 0.0))
        selection_rows.append(
            {
                "case_id": cid,
                "selected_method": chosen["method"],
                "selected_prob": float(scored_probs[str(chosen["item_id"])]),
                "selected_overall": co,
                "selected_human_score": cs,
                "oracle_method": oracle["method"],
                "oracle_overall": oo,
                "oracle_human_score": os,
                "instruction": chosen.get("instruction", ""),
            }
        )

    with (out_dir / "reranked_selection.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = list(selection_rows[0].keys()) if selection_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selection_rows)

    summary: Dict[str, Any] = {
        "cases": len(selection_rows),
        "reranker_overall_yes_rate": float(np.mean(selected_overall)) if selected_overall else None,
        "reranker_avg_human_score": float(np.mean(selected_scores)) if selected_scores else None,
        "oracle_overall_yes_rate": float(np.mean(oracle_overall)) if oracle_overall else None,
        "oracle_avg_human_score": float(np.mean(oracle_scores)) if oracle_scores else None,
        "selected_method_counts": dict(Counter(r["selected_method"] for r in selection_rows)),
        "method_baselines": {},
    }
    for method in sorted(method_scores):
        summary["method_baselines"][method] = {
            "overall_yes_rate": float(np.mean(method_overalls[method])),
            "avg_human_score": float(np.mean(method_scores[method])),
            "n": len(method_scores[method]),
        }
    return summary


def _plot_rerank(summary: Mapping[str, Any], out_dir: Path) -> None:
    labels = ["reranker", "oracle"] + sorted(summary["method_baselines"].keys())
    yes = [summary["reranker_overall_yes_rate"], summary["oracle_overall_yes_rate"]]
    score = [summary["reranker_avg_human_score"], summary["oracle_avg_human_score"]]
    for method in sorted(summary["method_baselines"].keys()):
        yes.append(summary["method_baselines"][method]["overall_yes_rate"])
        score.append(summary["method_baselines"][method]["avg_human_score"])
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=160)
    ax.bar(x - w / 2, yes, w, label="overall yes rate", color="#31a354")
    ax.bar(x + w / 2, score, w, label="avg human score", color="#3182bd")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_title("Case-level method selection by human-aligned scorer")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for i, v in enumerate(yes):
        ax.text(i - w / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(score):
        ax.text(i + w / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "reranker_vs_methods.png", bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze human labels and train a lightweight reranker.")
    p.add_argument("--labels", default="spacefit_v2/results/visual_audit_distribution/result/analysis/merged_visual_audit_labels.jsonl")
    p.add_argument("--cases", default="spacefit_v2/data/single_target_benchmark/gpt_intent_cases_test.json")
    p.add_argument("--predictions", default="spacefit_v2/results/experiment_final/test_gpt_intent/raw_predictions.json")
    p.add_argument("--out_dir", default="spacefit_v2/results/visual_audit_distribution/result/analysis/human_aligned_scorer")
    return p


def main(args: argparse.Namespace) -> None:
    rows = _read_jsonl(Path(args.labels))
    cases = {str(c["id"]): c for c in json.loads(Path(args.cases).read_text(encoding="utf-8"))}
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features, y_overall, y_score, groups, item_ids = _rows_to_table(rows, cases, predictions)

    summaries: Dict[str, Any] = {
        "input": args.labels,
        "n_labeled_overall": int(len(y_overall)),
        "positive_rate": float(np.mean(y_overall)) if len(y_overall) else None,
        "unique_cases": int(len(set(groups.tolist()))),
        "models": {},
    }
    best_name = "logreg"
    best_prob: Optional[np.ndarray] = None
    for kind in ["logreg", "rf"]:
        prob, fold_rows = _cv_predict(features, y_overall, groups, kind)
        summaries["models"][kind] = {
            "overall_cv": _metrics(y_overall, prob),
            "folds": fold_rows,
        }
        with (out_dir / f"{kind}_cv_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["item_id", "case_id", "method", "target", "prob", "human_score"])
            j = 0
            for row in rows:
                if _overall(row) is None or _human_score(row) is None:
                    continue
                writer.writerow([row["item_id"], row["case_id"], row["method"], int(y_overall[j]), float(prob[j]), float(y_score[j])])
                j += 1
        if best_prob is None or (summaries["models"][kind]["overall_cv"]["balanced_accuracy"] or 0) > (
            summaries["models"][best_name]["overall_cv"]["balanced_accuracy"] or 0
        ):
            best_name = kind
            best_prob = prob

    coeffs = _full_fit_importance(features, y_overall, out_dir)
    scored_probs = {item_id: float(prob) for item_id, prob in zip(item_ids, best_prob.tolist()) if not math.isnan(float(prob))}
    rerank_summary = _rerank(rows, scored_probs, out_dir)
    summaries["best_cv_model"] = best_name
    summaries["reranking"] = rerank_summary
    summaries["top_coefficients"] = dict(list(coeffs.items())[:25])
    (out_dir / "human_aligned_scorer_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_rerank(rerank_summary, out_dir)

    report = [
        "# Human-Aligned Scorer Analysis",
        "",
        f"- Labeled items with overall answer: {summaries['n_labeled_overall']}",
        f"- Overall positive rate: {summaries['positive_rate']:.3f}",
        f"- Best CV model: {best_name}",
        "",
        "## CV Performance",
    ]
    for kind, data in summaries["models"].items():
        m = data["overall_cv"]
        report.append(
            f"- {kind}: acc={m['accuracy']:.3f}, bal_acc={m['balanced_accuracy']:.3f}, auc={m['auc']:.3f}, pred_pos={m['pred_positive_rate']:.3f}"
        )
    report.extend(
        [
            "",
            "## Reranking",
            f"- Reranker overall yes rate: {rerank_summary['reranker_overall_yes_rate']:.3f}",
            f"- Reranker avg human score: {rerank_summary['reranker_avg_human_score']:.3f}",
            f"- Oracle overall yes rate: {rerank_summary['oracle_overall_yes_rate']:.3f}",
            f"- Oracle avg human score: {rerank_summary['oracle_avg_human_score']:.3f}",
            f"- Selected method counts: {rerank_summary['selected_method_counts']}",
            "",
            "## Interpretation",
            "- This is a seed reranker analysis, not a production scorer.",
            "- If reranker selection beats fixed methods under case-level CV, the next step is to add richer geometric features and apply it to candidate selection.",
        ]
    )
    (out_dir / "HUMAN_ALIGNED_SCORER_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(build_parser().parse_args())
