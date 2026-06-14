"""Merge API backbone benchmark results into the open LLM summary CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "spacefit_v2" / "results" / "open_llm_backbone_comparison"
SUMMARY = OUT_DIR / "open_llm_full_summary.csv"


BACKBONES = [
    {
        "backbone": "Qwen3.5-9B",
        "model_id": "Qwen/Qwen3.5-9B",
        "results": ROOT / "spacefit_v2" / "results" / "open_llm_qwen35_9b_full" / "test_gpt_intent" / "results.csv",
        "predictions": ROOT / "spacefit_v2" / "results" / "open_llm_qwen35_9b_full" / "test_gpt_intent" / "raw_predictions.json",
        "cases": "181",
        "placed": "173",
    },
    {
        "backbone": "DeepSeek V4 Flash",
        "model_id": "deepseek-v4-flash",
        "results": ROOT / "spacefit_v2" / "results" / "deepseek_v4_flash_full" / "test_gpt_intent" / "results.csv",
        "predictions": ROOT / "spacefit_v2" / "results" / "deepseek_v4_flash_full" / "test_gpt_intent" / "raw_predictions.json",
    },
    {
        "backbone": "Gemini 3.5 Flash",
        "model_id": "gemini-3.5-flash",
        "results": ROOT / "spacefit_v2" / "results" / "gemini_3_5_flash_full" / "test_gpt_intent" / "results.csv",
        "predictions": ROOT / "spacefit_v2" / "results" / "gemini_3_5_flash_full" / "test_gpt_intent" / "raw_predictions.json",
    },
]


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backbone",
        "model_id",
        "cases",
        "placed",
        "CF",
        "IB",
        "Constraint Accuracy",
        "CPS",
        "Success@1",
        "Success@5",
        "Reachability",
        "Walkability",
        "source_results",
        "source_predictions",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(spec: Dict[str, object]) -> Dict[str, str] | None:
    results_path = Path(spec["results"])
    rows = read_rows(results_path)
    if not rows:
        return None
    # The aggregate row is usually last; prefer an explicit aggregate-like row if present.
    aggregate = rows[-1]
    for row in rows:
        method = (row.get("method") or row.get("Method") or "").lower()
        if method in {"aggregate", "summary", "spacefit_gpt_text"}:
            aggregate = row
    def get(*names: str) -> str:
        for name in names:
            if name in aggregate and aggregate[name] != "":
                return aggregate[name]
        return ""
    return {
        "backbone": str(spec["backbone"]),
        "model_id": str(spec["model_id"]),
        "cases": str(spec.get("cases") or get("cases", "n", "num_cases")),
        "placed": str(spec.get("placed") or get("placed")),
        "CF": get("CF", "cf"),
        "IB": get("IB", "ib"),
        "Constraint Accuracy": get("Constraint Accuracy", "constraint_accuracy"),
        "CPS": get("CPS", "cps"),
        "Success@1": get("Success@1", "success@1"),
        "Success@5": get("Success@5", "success@5"),
        "Reachability": get("Reachability", "reachability"),
        "Walkability": get("Walkability", "walkability"),
        "source_results": str(results_path.relative_to(ROOT)).replace("\\", "/"),
        "source_predictions": str(Path(spec["predictions"]).relative_to(ROOT)).replace("\\", "/"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args()

    summary_path = args.summary if args.summary.is_absolute() else ROOT / args.summary
    existing = read_rows(summary_path)
    by_backbone = {row.get("backbone", ""): row for row in existing}

    added = []
    for spec in BACKBONES:
        row = metric_row(spec)
        if row is None:
            print(f"missing: {spec['backbone']} ({spec['results']})")
            continue
        by_backbone[row["backbone"]] = row
        added.append(row["backbone"])

    ordered = []
    preferred = ["GPT-4o", "Qwen3-8B", "Qwen3.5-9B", "Gemma 4 E4B", "DeepSeek V4 Flash", "Gemini 3.5 Flash"]
    for name in preferred:
        if name in by_backbone:
            ordered.append(by_backbone.pop(name))
    ordered.extend(by_backbone.values())
    write_rows(summary_path, ordered)
    print(f"updated={summary_path}")
    print("added_or_refreshed=" + ", ".join(added) if added else "added_or_refreshed=none")


if __name__ == "__main__":
    main()
