"""Analyze top-k human preference labels for SpaceFit candidate reranking."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spacefit_v2.single_target.eval import aggregate_results


DEFAULT_LABELS_DIR = Path("spacefit_v2/results/topk")
DEFAULT_CASES = Path("spacefit_v2/data/single_target_benchmark/gpt_intent_cases_test.json")
DEFAULT_OUT_DIR = DEFAULT_LABELS_DIR / "analysis"


def _setup_font() -> None:
    candidates = [
        "Malgun Gothic",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "DejaVu Sans",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rcParams["font.family"] = name
            break
    rcParams["axes.unicode_minus"] = False


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
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


def _label_completeness(row: Mapping[str, Any]) -> int:
    label = row.get("human_label") or {}
    score = 0
    if label.get("best_candidate_id"):
        score += 1000
    score += len(label.get("acceptable_candidate_ids") or [])
    score += sum(len(v or []) for v in (label.get("candidate_issue_tags") or {}).values())
    if label.get("case_notes"):
        score += 1
    return score


def _dedupe_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_item: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("item_id") or row.get("case_id"))
        previous = by_item.get(key)
        if previous is None or _label_completeness(row) > _label_completeness(previous):
            by_item[key] = dict(row)
    return [by_item[key] for key in sorted(by_item)]


def _metric(prefix: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        f"{prefix}_cf": metrics.get("cf"),
        f"{prefix}_ib": metrics.get("ib"),
        f"{prefix}_constraint_accuracy": metrics.get("constraint_accuracy"),
        f"{prefix}_cps": metrics.get("cps"),
        f"{prefix}_reachability": metrics.get("reachability"),
        f"{prefix}_walkability": metrics.get("walkability"),
    }


def _mean(values: Iterable[Any]) -> float | None:
    nums = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(nums) / len(nums) if nums else None


def _load_labels(labels_dir: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    files = sorted(labels_dir.glob("*.jsonl"))
    all_rows: List[Dict[str, Any]] = []
    file_summary = []
    for path in files:
        rows = _load_jsonl(path)
        all_rows.extend(rows)
        file_summary.append(
            {
                "file": path.name,
                "rows": len(rows),
                "completed_best": sum(1 for row in rows if (row.get("human_label") or {}).get("best_candidate_id")),
                "acceptable_marks": sum(
                    len((row.get("human_label") or {}).get("acceptable_candidate_ids") or []) for row in rows
                ),
                "case_notes": sum(1 for row in rows if (row.get("human_label") or {}).get("case_notes")),
            }
        )
    return _dedupe_rows(all_rows), {"files": file_summary, "raw_rows": len(all_rows), "unique_rows": 0}


def _build_tables(rows: Sequence[Mapping[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    candidate_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []
    best_rank_counter: Counter[int] = Counter()
    acceptable_by_rank: Dict[int, List[int]] = defaultdict(list)
    issue_counter: Counter[str] = Counter()
    issue_by_rank: Counter[tuple[int, str]] = Counter()
    top1_issue_counter: Counter[str] = Counter()

    for row in rows:
        label = row.get("human_label") or {}
        best_id = label.get("best_candidate_id")
        acceptable = set(label.get("acceptable_candidate_ids") or [])
        issue_tags = label.get("candidate_issue_tags") or {}
        candidates = list(row.get("candidates") or [])
        best_candidate = next((c for c in candidates if c.get("candidate_id") == best_id), None)
        top1_candidate = next((c for c in candidates if int(c.get("rank", 999)) == 0), candidates[0] if candidates else None)
        best_rank = best_candidate.get("rank") if best_candidate else None
        top1_id = top1_candidate.get("candidate_id") if top1_candidate else None

        if best_rank is not None:
            best_rank_counter[int(best_rank)] += 1

        for cand in candidates:
            cid = str(cand.get("candidate_id"))
            rank = int(cand.get("rank", 0))
            tags = list(issue_tags.get(cid) or [])
            issue_counter.update(tags)
            for tag in tags:
                issue_by_rank[(rank, tag)] += 1
            if rank == 0:
                top1_issue_counter.update(tags)
            acceptable_by_rank[rank].append(1 if cid in acceptable else 0)
            metrics = cand.get("metrics") or {}
            candidate_rows.append(
                {
                    "case_id": row.get("case_id"),
                    "item_id": row.get("item_id"),
                    "method": row.get("method"),
                    "rank": rank,
                    "original_rank": cand.get("original_rank"),
                    "candidate_id": cid,
                    "selected_best": int(cid == best_id),
                    "acceptable": int(cid in acceptable),
                    "issue_tags": "|".join(tags),
                    "human_aligned_score": cand.get("human_aligned_score"),
                    "category": cand.get("category"),
                    "rotation_y": cand.get("rotation_y"),
                    **_metric("", metrics),
                }
            )

        best_metrics = (best_candidate or {}).get("metrics") or {}
        top1_metrics = (top1_candidate or {}).get("metrics") or {}
        case_rows.append(
            {
                "case_id": row.get("case_id"),
                "item_id": row.get("item_id"),
                "method": row.get("method"),
                "target_category": row.get("target_category"),
                "room_type": row.get("room_type"),
                "num_candidates": len(candidates),
                "completed": int(best_candidate is not None),
                "best_rank": best_rank,
                "best_original_rank": (best_candidate or {}).get("original_rank"),
                "top1_is_best": int(best_candidate is not None and best_id == top1_id),
                "top1_acceptable": int(top1_id in acceptable) if top1_id is not None else 0,
                "num_acceptable": len(acceptable),
                "case_notes": label.get("case_notes") or "",
                **_metric("top1", top1_metrics),
                **_metric("human_best", best_metrics),
            }
        )

    completed = [row for row in case_rows if row["completed"]]
    summary = {
        "total_cases": len(case_rows),
        "completed_cases": len(completed),
        "incomplete_cases": len(case_rows) - len(completed),
        "total_candidates": len(candidate_rows),
        "avg_candidates_per_case": _mean(row["num_candidates"] for row in case_rows),
        "human_top1_agreement": _mean(row["top1_is_best"] for row in completed),
        "top1_acceptable_rate": _mean(row["top1_acceptable"] for row in completed),
        "avg_best_rank": _mean(row["best_rank"] for row in completed),
        "best_rank_counts": {str(k): v for k, v in sorted(best_rank_counter.items())},
        "acceptable_rate_by_rank": {
            str(k): _mean(v) for k, v in sorted(acceptable_by_rank.items())
        },
        "issue_tag_counts": dict(issue_counter.most_common()),
        "top1_issue_tag_counts": dict(top1_issue_counter.most_common()),
    }
    return candidate_rows, case_rows, summary


def _oracle_eval(
    rows: Sequence[Mapping[str, Any]],
    cases_path: Path,
    method: str,
) -> List[Dict[str, Any]]:
    cases_all = json.load(cases_path.open(encoding="utf-8"))
    case_by_id = {str(case["id"]): case for case in cases_all}
    selected_cases = []
    current_predictions: Dict[str, List[Dict[str, Any]]] = {}
    human_oracle_predictions: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        case_id = str(row.get("case_id"))
        if case_id not in case_by_id:
            continue
        label = row.get("human_label") or {}
        best_id = label.get("best_candidate_id")
        candidates = list(row.get("candidates") or [])
        best_candidate = next((c for c in candidates if c.get("candidate_id") == best_id), None)
        if not best_candidate:
            continue
        current = [dict(c.get("prediction") or {}) for c in sorted(candidates, key=lambda c: int(c.get("rank", 0)))]
        best_prediction = dict(best_candidate.get("prediction") or {})
        remainder = [p for p in current if p.get("human_aligned_rank") != best_prediction.get("human_aligned_rank")]
        selected_cases.append(case_by_id[case_id])
        current_predictions[case_id] = current
        human_oracle_predictions[case_id] = [best_prediction] + remainder

    if not selected_cases:
        return []
    aggregates = aggregate_results(
        selected_cases,
        {
            f"{method}_current_top1": current_predictions,
            f"{method}_human_oracle_top1": human_oracle_predictions,
        },
    )
    return [
        {
            "method": row.get("method"),
            "source_name": row.get("source_name"),
            "notes": row.get("notes"),
            "num_scenes": row.get("num_scenes"),
            "metrics": row.get("metrics"),
        }
        for row in aggregates
    ]


def _plot_best_rank(summary: Mapping[str, Any], out_path: Path) -> None:
    counts = {int(k): v for k, v in (summary.get("best_rank_counts") or {}).items()}
    xs = list(range(5))
    ys = [counts.get(x, 0) for x in xs]
    plt.figure(figsize=(7.5, 4.2), facecolor="white")
    bars = plt.bar(xs, ys, color="#4c78a8")
    plt.title("Human-selected best candidate rank", fontsize=15, weight="bold")
    plt.xlabel("Current reranker rank")
    plt.ylabel("Cases")
    plt.xticks(xs, [f"Rank {x + 1}" for x in xs])
    for bar, val in zip(bars, ys):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.6, str(val), ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_acceptable(summary: Mapping[str, Any], out_path: Path) -> None:
    rates = {int(k): float(v) for k, v in (summary.get("acceptable_rate_by_rank") or {}).items() if v is not None}
    xs = list(range(5))
    ys = [rates.get(x, 0.0) for x in xs]
    plt.figure(figsize=(7.5, 4.2), facecolor="white")
    bars = plt.bar(xs, ys, color="#59a14f")
    plt.ylim(0, 1.0)
    plt.title("Acceptable rate by current rank", fontsize=15, weight="bold")
    plt.xlabel("Current reranker rank")
    plt.ylabel("Acceptable rate")
    plt.xticks(xs, [f"Rank {x + 1}" for x in xs])
    for bar, val in zip(bars, ys):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.025, f"{val:.1%}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_issue_tags(summary: Mapping[str, Any], out_path: Path) -> None:
    counts = list((summary.get("issue_tag_counts") or {}).items())[:10]
    if not counts:
        return
    labels = [k for k, _ in counts][::-1]
    values = [v for _, v in counts][::-1]
    plt.figure(figsize=(8.2, 4.8), facecolor="white")
    bars = plt.barh(labels, values, color="#e15759")
    plt.title("Issue tags marked by annotators", fontsize=15, weight="bold")
    plt.xlabel("Count")
    for bar, val in zip(bars, values):
        plt.text(val + 2, bar.get_y() + bar.get_height() / 2, str(val), va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_oracle(aggregates: Sequence[Mapping[str, Any]], out_path: Path) -> None:
    if not aggregates:
        return
    metrics = ["CPS", "Constraint Accuracy", "Success@1", "Success@5"]
    names = [row["method"] for row in aggregates]
    x = range(len(metrics))
    width = 0.34
    plt.figure(figsize=(8.5, 4.6), facecolor="white")
    for i, row in enumerate(aggregates):
        vals = [row["metrics"].get(m, 0.0) for m in metrics]
        offsets = [j + (i - 0.5) * width for j in x]
        plt.bar(offsets, vals, width=width, label=names[i], color=["#4c78a8", "#59a14f"][i % 2])
    plt.ylim(0, 1.05)
    plt.xticks(list(x), metrics, rotation=15, ha="right")
    plt.ylabel("Metric")
    plt.title("Current top-1 vs human-preferred top-1", fontsize=15, weight="bold")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def _write_markdown(
    path: Path,
    summary: Mapping[str, Any],
    oracle: Sequence[Mapping[str, Any]],
    files: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Top-k Human Label Analysis",
        "",
        "## Label Completion",
        "",
        f"- Total unique cases: {summary['total_cases']}",
        f"- Completed best selections: {summary['completed_cases']}",
        f"- Incomplete cases: {summary['incomplete_cases']}",
        f"- Total candidates: {summary['total_candidates']}",
        f"- Average candidates per case: {summary['avg_candidates_per_case']:.2f}",
        "",
        "| File | Rows | Best selected | Acceptable marks | Case notes |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in files:
        lines.append(
            f"| {item['file']} | {item['rows']} | {item['completed_best']} | "
            f"{item['acceptable_marks']} | {item['case_notes']} |"
        )
    lines += [
        "",
        "## Human Preference vs Current Reranker",
        "",
        f"- Human top-1 agreement: {summary['human_top1_agreement']:.3f}",
        f"- Current top-1 acceptable rate: {summary['top1_acceptable_rate']:.3f}",
        f"- Average human-selected rank: {summary['avg_best_rank']:.2f} (0 means current top-1)",
        "",
        "Best-rank distribution:",
        "",
        "| Current rank | Cases |",
        "|---:|---:|",
    ]
    for rank, count in (summary.get("best_rank_counts") or {}).items():
        lines.append(f"| {int(rank) + 1} | {count} |")
    lines += [
        "",
        "## Main Issue Tags",
        "",
        "| Issue tag | Count |",
        "|---|---:|",
    ]
    for tag, count in list((summary.get("issue_tag_counts") or {}).items())[:12]:
        lines.append(f"| {tag} | {count} |")
    if oracle:
        lines += [
            "",
            "## Metric Oracle Check",
            "",
            "This compares current top-1 ordering against an oracle that moves the human-selected best candidate to rank 1 on completed cases only.",
            "",
            "| Method | CF | IB | Constraint Accuracy | CPS | Success@1 | Success@5 | Reachability | Walkability |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in oracle:
            m = row["metrics"]
            lines.append(
                f"| {row['method']} | {m['CF']:.3f} | {m['IB']:.3f} | "
                f"{m['Constraint Accuracy']:.3f} | {m['CPS']:.3f} | {m['Success@1']:.3f} | "
                f"{m['Success@5']:.3f} | {m['Reachability']:.3f} | {m['Walkability']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels_dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--method", default="constraint_solver")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_font()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, load_summary = _load_labels(args.labels_dir)
    load_summary["unique_rows"] = len(rows)
    candidate_rows, case_rows, summary = _build_tables(rows)
    summary["load_summary"] = load_summary
    oracle = _oracle_eval(rows, args.cases, args.method)
    summary["oracle_eval"] = oracle

    _write_jsonl(args.out_dir / "merged_topk_labels.jsonl", rows)
    _write_csv(args.out_dir / "candidate_labels.csv", candidate_rows)
    _write_csv(args.out_dir / "case_summary.csv", case_rows)
    _dump_json(args.out_dir / "topk_label_summary.json", summary)
    _dump_json(args.out_dir / "oracle_eval_results.json", {"methods": oracle})

    _plot_best_rank(summary, args.out_dir / "best_rank_distribution.png")
    _plot_acceptable(summary, args.out_dir / "acceptable_by_rank.png")
    _plot_issue_tags(summary, args.out_dir / "issue_tag_counts.png")
    _plot_oracle(oracle, args.out_dir / "human_vs_current_metrics.png")
    _write_markdown(
        args.out_dir / "TOPK_LABEL_ANALYSIS.md",
        summary,
        oracle,
        load_summary["files"],
    )

    console_summary = {
        key: value
        for key, value in summary.items()
        if key in {
            "total_cases",
            "completed_cases",
            "incomplete_cases",
            "total_candidates",
            "human_top1_agreement",
            "top1_acceptable_rate",
            "avg_best_rank",
            "best_rank_counts",
            "issue_tag_counts",
            "oracle_eval",
        }
    }
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
