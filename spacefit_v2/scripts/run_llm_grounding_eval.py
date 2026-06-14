"""Experiment C: evaluate LLM/VLM open-vocabulary intent grounding.

For each test case:
  1. Generate a free-form NL instruction from the structured constraints.
  2. Ground that instruction via the configured provider (openai or deterministic fallback).
  3. Compare the grounded constraints against the ground-truth structured constraints.
  4. Report per-case qualitative outputs + aggregate precision/recall/F1.

Usage:
    # Deterministic fallback only (no API key needed):
    python -m spacefit_v2.scripts.run_llm_grounding_eval \
        --n_cases 15 --provider deterministic \
        --out_dir spacefit_v2/results/exp_c

    # With OpenAI (requires OPENAI_API_KEY env var):
    python -m spacefit_v2.scripts.run_llm_grounding_eval \
        --n_cases 15 --provider openai \
        --out_dir spacefit_v2/results/exp_c
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set

from spacefit_v2.single_target.intent_grounder import IntentGrounder


# ── constraint comparison helpers ────────────────────────────────────────────

def _type_set(constraints: List[Dict[str, Any]]) -> Set[str]:
    return {str(c.get("constraint_type", "")) for c in constraints if c.get("constraint_type")}


def _constraint_match_score(pred: List[Dict[str, Any]], gt: List[Dict[str, Any]]) -> Dict[str, float]:
    """Type-level precision, recall, F1 between predicted and ground-truth constraints."""
    pred_types = _type_set(pred)
    gt_types = _type_set(gt)
    if not gt_types:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    tp = len(pred_types & gt_types)
    fp = len(pred_types - gt_types)
    fn = len(gt_types - pred_types)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _near_category_match(pred: List[Dict[str, Any]], gt: List[Dict[str, Any]]) -> bool:
    """Check whether 'near' constraint target_category is correctly identified."""
    pred_near = next((c for c in pred if c.get("constraint_type") == "near"), None)
    gt_near = next((c for c in gt if c.get("constraint_type") == "near"), None)
    if gt_near is None:
        return True  # N/A
    if pred_near is None:
        return False
    return str(pred_near.get("target_category", "")).lower() == str(gt_near.get("target_category", "")).lower()


# ── main evaluation ──────────────────────────────────────────────────────────

def run_grounding_eval(
    cases: List[Dict[str, Any]],
    provider: str,
    model_name: str | None,
) -> List[Dict[str, Any]]:
    grounder = IntentGrounder(provider=provider, model_name=model_name)
    results: List[Dict[str, Any]] = []

    for case in cases:
        target_asset = case["target_asset"]
        gt_constraints: List[Dict[str, Any]] = list(case["intent"].get("constraints", []))
        category = str(target_asset.get("category", "furniture"))

        # 1. Generate NL instruction from ground-truth structured constraints
        nl_text = grounder._to_natural_language(category, gt_constraints)

        # 2. Ground the NL text (bypass structured passthrough — force text path)
        intent_text_only = {"text": nl_text, "constraints": []}
        grounded = grounder.ground(target_asset, intent=intent_text_only)
        grounded_constraints: List[Dict[str, Any]] = grounded.get("constraints", [])

        # 3. Evaluate
        scores = _constraint_match_score(grounded_constraints, gt_constraints)
        near_ok = _near_category_match(grounded_constraints, gt_constraints)

        results.append({
            "case_id": case["id"],
            "category": category,
            "room_type": case["scene"].get("room_type"),
            "nl_instruction": nl_text,
            "gt_constraints": [c.get("constraint_type") for c in gt_constraints],
            "grounded_constraints": [c.get("constraint_type") for c in grounded_constraints],
            "grounded_detail": grounded_constraints,
            "provider": grounded.get("provider"),
            "mode": grounded.get("mode"),
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"],
            "near_category_match": near_ok,
            "tp": scores["tp"],
            "fp": scores["fp"],
            "fn": scores["fn"],
        })

    return results


def aggregate_grounding(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}
    avg_p = sum(r["precision"] for r in results) / n
    avg_r = sum(r["recall"] for r in results) / n
    avg_f1 = sum(r["f1"] for r in results) / n
    near_cases = [r for r in results if r["gt_constraints"] and "near" in r["gt_constraints"]]
    near_acc = sum(1 for r in near_cases if r["near_category_match"]) / len(near_cases) if near_cases else None
    return {
        "n_cases": n,
        "avg_precision": round(avg_p, 4),
        "avg_recall": round(avg_r, 4),
        "avg_f1": round(avg_f1, 4),
        "near_category_accuracy": round(near_acc, 4) if near_acc is not None else "N/A",
        "n_near_cases": len(near_cases),
    }


def _build_markdown(results: List[Dict[str, Any]], agg: Dict[str, Any], provider: str) -> str:
    lines = [
        "# Experiment C: LLM/VLM Intent Grounding Evaluation",
        "",
        f"**Provider:** `{provider}`  |  **Cases:** {agg.get('n_cases', 0)}",
        "",
        "## Aggregate Metrics (constraint-type level)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Precision | {agg.get('avg_precision', 0):.3f} |",
        f"| Recall | {agg.get('avg_recall', 0):.3f} |",
        f"| F1 | {agg.get('avg_f1', 0):.3f} |",
        f"| Near-category accuracy | {agg.get('near_category_accuracy', 'N/A')} |",
        "",
        "## Per-case Qualitative Results",
        "",
    ]
    for i, r in enumerate(results, 1):
        gt_str = ", ".join(r["gt_constraints"]) or "(none)"
        pred_str = ", ".join(r["grounded_constraints"]) or "(none)"
        match_icon = "✓" if r["f1"] >= 1.0 else ("~" if r["f1"] >= 0.5 else "✗")
        lines.append(f"### Case {i}: `{r['case_id']}`")
        lines.append(f"- **Category:** {r['category']}  **Room:** {r['room_type']}")
        lines.append(f"- **NL instruction:** _{r['nl_instruction']}_")
        lines.append(f"- **GT constraints:** `{gt_str}`")
        lines.append(f"- **Grounded constraints:** `{pred_str}`")
        lines.append(f"- **F1:** {r['f1']:.2f}  {match_icon}")
        lines.append("")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_dir", default="spacefit_v2/data/single_target_benchmark")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n_cases", type=int, default=15)
    parser.add_argument("--provider", default="deterministic", choices=["deterministic", "openai"])
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--out_dir", default="spacefit_v2/results/exp_c")
    return parser


def main(args: argparse.Namespace) -> None:
    benchmark_dir = Path(args.benchmark_dir)
    with open(benchmark_dir / "cases.json") as f:
        all_cases = json.load(f)
    cases = [c for c in all_cases if c["split"] == args.split][: args.n_cases]
    print(f"Running Exp C grounding eval: {len(cases)} cases, provider={args.provider}")

    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set — will use deterministic fallback for all cases.")

    results = run_grounding_eval(cases, provider=args.provider, model_name=args.model_name)
    agg = aggregate_grounding(results)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "grounding_results.json", "w") as f:
        json.dump({"aggregate": agg, "cases": results}, f, indent=2)

    md = _build_markdown(results, agg, args.provider)
    with open(out / "GROUNDING_EVAL.md", "w") as f:
        f.write(md)

    print(json.dumps(agg, indent=2))
    print(f"\nOutputs saved to {out}/")


if __name__ == "__main__":
    main(build_parser().parse_args())
