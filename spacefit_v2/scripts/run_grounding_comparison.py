"""Experiment 2: Rule vs Extended-Rule vs LLM vs Oracle grounding comparison.

Evaluates grounding quality and (optionally) downstream placement quality
across grounding methods, paraphrase styles, and constraint types.

Usage:
    # Grounding-only (fast, no placement)
    python -m spacefit_v2.scripts.run_grounding_comparison

    # With downstream placement evaluation (slow)
    python -m spacefit_v2.scripts.run_grounding_comparison --downstream --max_cases 20

    # Include LLM grounding (requires OPENAI_API_KEY)
    python -m spacefit_v2.scripts.run_grounding_comparison --include_llm
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from spacefit_v2.grounding.extended_rule_grounder import ExtendedRuleGrounder
from spacefit_v2.grounding.metrics import (
    aggregate_by_method,
    aggregate_by_style,
    aggregate_grounding_metrics,
    anchor_match,
    constraint_type_scores,
    make_grounding_table,
    per_type_recall,
)
from spacefit_v2.single_target.intent_grounder import IntentGrounder


# ── Grounding methods ─────────────────────────────────────────────────────────

_GROUNDER_SIMPLE = IntentGrounder(provider="deterministic")
_GROUNDER_EXTENDED = ExtendedRuleGrounder()


def _ground_simple(text: str, target_asset: Dict[str, Any]) -> List[Dict[str, Any]]:
    subject_id = str(target_asset.get("id", "target"))
    result = _GROUNDER_SIMPLE.ground(target_asset=target_asset, text=text)
    return result.get("constraints", [])


def _ground_extended(text: str, target_asset: Dict[str, Any]) -> List[Dict[str, Any]]:
    subject_id = str(target_asset.get("id", "target"))
    result = _GROUNDER_EXTENDED.ground(target_asset=target_asset, text=text)
    return result.get("constraints", [])


def _ground_llm(text: str, target_asset: Dict[str, Any], model: str = "gpt-4o-mini") -> List[Dict[str, Any]]:
    grounder = IntentGrounder(provider="openai", model_name=model)
    result = grounder.ground(target_asset=target_asset, text=text)
    return result.get("constraints", [])


def _ground_oracle(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(c) for c in item.get("gt_constraints", [])]


# ── Per-item grounding evaluation ─────────────────────────────────────────────

_ALL_CTYPES = ("against_wall", "near", "facing", "not_block_door", "keep_window_clear", "access_zone")


def _eval_grounding_item(
    item: Dict[str, Any],
    method_name: str,
    pred_constraints: List[Dict[str, Any]],
) -> Dict[str, Any]:
    gt = item.get("gt_constraints", [])
    scores = constraint_type_scores(pred_constraints, gt)
    result: Dict[str, Any] = {
        "item_id": item["id"],
        "case_id": item["case_id"],
        "style": item["style"],
        "method": method_name,
        "text": item.get("text", ""),
        "target_category": item.get("target_category"),
        "room_type": item.get("room_type"),
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1": scores["f1"],
        "tp": scores["tp"],
        "fp": scores["fp"],
        "fn": scores["fn"],
        "pred_constraint_types": [c.get("constraint_type") for c in pred_constraints],
        "gt_constraint_types": item.get("gt_constraint_types", []),
    }
    # Per constraint-type recall
    for ctype in _ALL_CTYPES:
        v = per_type_recall(pred_constraints, gt, ctype)
        result[f"recall_{ctype}"] = v
    # Anchor accuracy
    result["near_anchor_match"] = anchor_match(pred_constraints, gt, "near")
    result["facing_anchor_match"] = anchor_match(pred_constraints, gt, "facing")
    return result


# ── Main grounding eval loop ──────────────────────────────────────────────────

def run_grounding_eval(
    paraphrase_cases: List[Dict[str, Any]],
    original_cases_by_id: Dict[str, Dict[str, Any]],
    include_llm: bool = False,
    llm_model: str = "gpt-4o-mini",
) -> List[Dict[str, Any]]:
    """Evaluate grounding quality for simple-rule / extended-rule / oracle (+ optional LLM)."""
    all_results: List[Dict[str, Any]] = []

    for i, item in enumerate(paraphrase_cases):
        case_id = item["case_id"]
        orig_case = original_cases_by_id.get(case_id)
        if orig_case is None:
            continue
        target_asset = orig_case["target_asset"]
        text = item.get("text", "")

        # Simple rule
        pred_simple = _ground_simple(text, target_asset)
        all_results.append(_eval_grounding_item(item, "simple_rule", pred_simple))

        # Extended rule
        pred_extended = _ground_extended(text, target_asset)
        all_results.append(_eval_grounding_item(item, "extended_rule", pred_extended))

        # Oracle (GT pass-through)
        pred_oracle = _ground_oracle(item)
        all_results.append(_eval_grounding_item(item, "oracle", pred_oracle))

        # LLM (optional, expensive)
        if include_llm:
            try:
                pred_llm = _ground_llm(text, target_asset, model=llm_model)
                all_results.append(_eval_grounding_item(item, f"llm_{llm_model}", pred_llm))
            except Exception as exc:
                print(f"  [warn] LLM grounding failed for {item['id']}: {exc}")

        if (i + 1) % 25 == 0:
            print(f"  Grounding eval: {i + 1}/{len(paraphrase_cases)} items done")

    return all_results


# ── Downstream placement (optional) ──────────────────────────────────────────

def run_downstream_placement(
    original_cases: List[Dict[str, Any]],
    original_cases_by_id: Dict[str, Dict[str, Any]],
    paraphrase_style: str = "casual",
    paraphrase_cases_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    topk: int = 5,
    device: str = "cpu",
    diffopt_iters: int = 80,
    include_llm: bool = False,
    llm_model: str = "gpt-4o-mini",
) -> List[Dict[str, Any]]:
    """Run diffopt placement with constraints grounded from paraphrase text.

    Uses the 'casual' style (hardest for rule parser) by default to maximise
    the gap between rule-based and oracle grounding.
    """
    from spacefit_v2.single_target.eval import aggregate_results
    from spacefit_v2.single_target.methods import run_diffopt

    downstream_results: List[Dict[str, Any]] = []

    grounding_methods: Dict[str, Any] = {
        "simple_rule": lambda text, asset: _GROUNDER_SIMPLE.ground(target_asset=asset, text=text),
        "extended_rule": lambda text, asset: _GROUNDER_EXTENDED.ground(target_asset=asset, text=text),
        "oracle": None,  # handled separately
    }
    if include_llm:
        grounding_methods[f"llm_{llm_model}"] = lambda text, asset: IntentGrounder(
            provider="openai", model_name=llm_model
        ).ground(target_asset=asset, text=text)

    predictions_by_method: Dict[str, Dict[str, List[Dict]]] = {m: {} for m in grounding_methods}

    for case in original_cases:
        case_id = case["id"]
        # Look up the paraphrase item for this case + style
        para_item = (paraphrase_cases_by_id or {}).get(f"{case_id}__{paraphrase_style}")
        text = para_item["text"] if para_item else None
        gt_constraints = list(case["intent"].get("constraints", []))

        for method_name, grounder_fn in grounding_methods.items():
            if method_name == "oracle":
                grounded = {
                    "provider": "oracle",
                    "mode": "oracle",
                    "constraints": [dict(c) for c in gt_constraints],
                    "priorities": {"physics": 1.0, "constraints": 1.0 if gt_constraints else 0.0},
                }
            elif text:
                grounded = grounder_fn(text, case["target_asset"])
            else:
                # Fallback to empty constraints if no paraphrase found
                grounded = {"provider": method_name, "mode": "fallback", "constraints": [], "priorities": {"physics": 1.0, "constraints": 0.0}}

            try:
                preds = run_diffopt(
                    case=case,
                    grounded=grounded,
                    topk=topk,
                    device=device,
                    n_iters=diffopt_iters,
                    use_constraints=bool(grounded.get("constraints")),
                    use_heuristic_seed=True,
                )
            except Exception as exc:
                print(f"  [warn] diffopt failed for {case_id}/{method_name}: {exc}")
                preds = []

            predictions_by_method[method_name][case_id] = preds

    # Aggregate placement metrics per grounding method
    for method_name in grounding_methods:
        agg_list = aggregate_results(original_cases, {method_name: predictions_by_method[method_name]})
        if agg_list:
            entry = dict(agg_list[0])
            entry["grounding_method"] = method_name
            entry["paraphrase_style"] = paraphrase_style
            downstream_results.append(entry)

    return downstream_results


# ── Saving outputs ────────────────────────────────────────────────────────────

def _save_grounding_results(
    out_dir: Path,
    all_results: List[Dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "grounding_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # CSV
    fieldnames = [
        "item_id", "case_id", "style", "method", "precision", "recall", "f1",
        "tp", "fp", "fn",
        "recall_against_wall", "recall_near", "recall_facing",
        "recall_not_block_door", "recall_keep_window_clear", "recall_access_zone",
        "near_anchor_match", "facing_anchor_match",
        "target_category", "room_type",
    ]
    with open(out_dir / "grounding_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    # Aggregate tables
    by_method = aggregate_by_method(all_results)
    by_style = aggregate_by_style(all_results)

    # Method × style F1 breakdown
    by_style_by_method: Dict[str, Dict[str, Any]] = {}
    for method in by_method:
        method_rows = [r for r in all_results if r["method"] == method]
        by_style_by_method[method] = aggregate_by_style(method_rows)

    table_str = make_grounding_table(by_method, by_style_by_method)
    with open(out_dir / "GROUNDING_COMPARISON_TABLE.md", "w") as f:
        f.write(table_str)

    # Summary JSON
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_items": len(all_results),
        "methods": list(by_method.keys()),
        "by_method": by_method,
        "by_style": by_style,
    }
    with open(out_dir / "grounding_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Grounding results: {out_dir}/grounding_results.json")
    print(f"  Comparison table:  {out_dir}/GROUNDING_COMPARISON_TABLE.md")
    _print_summary(by_method)


def _save_downstream_results(out_dir: Path, downstream: List[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "downstream_from_grounding_results.json", "w") as f:
        json.dump(downstream, f, indent=2)

    # Markdown table
    lines = [
        "## Downstream Placement by Grounding Method",
        f"*(paraphrase style: {downstream[0].get('paraphrase_style', '?') if downstream else '?'})*",
        "",
        "| Grounding | CF ↑ | IB ↑ | Constraint Acc ↑ | CPS ↑ | Success@1 ↑ | Success@5 ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in downstream:
        m = row.get("metrics", {})
        lines.append(
            f"| {row['grounding_method']} | {m.get('CF', 0):.3f} | {m.get('IB', 0):.3f} | "
            f"{m.get('Constraint Accuracy', 0):.3f} | {m.get('CPS', 0):.3f} | "
            f"{m.get('Success@1', 0):.3f} | {m.get('Success@5', 0):.3f} |"
        )
    with open(out_dir / "DOWNSTREAM_COMPARISON_TABLE.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Downstream results: {out_dir}/downstream_from_grounding_results.json")
    print(f"  Downstream table:   {out_dir}/DOWNSTREAM_COMPARISON_TABLE.md")


def _write_protocol(out_dir: Path) -> None:
    lines = [
        "# Grounding Comparison Evaluation Protocol",
        "",
        "## Experiment Setup",
        "",
        "- **Source**: grounding benchmark (`data/grounding_benchmark/cases.json`)",
        "  - 5 paraphrase styles per original case: formal / casual / descriptive / indirect / richer",
        "  - Each style uses systematically different vocabulary to expose rule-parser vocabulary gaps",
        "",
        "## Grounding Methods",
        "",
        "| Method | Description |",
        "|---|---|",
        "| `simple_rule` | Deterministic regex parser (`IntentGrounder(provider='deterministic')`). Catches only canonical vocabulary (e.g. 'against the wall', 'near the X', 'access'). |",
        "| `extended_rule` | Broader regex patterns (`ExtendedRuleGrounder`). Catches casual/indirect synonyms (e.g. 'along the wall', 'next to', 'easy to reach'). |",
        "| `oracle` | GT constraints passed through directly. Upper-bound performance. |",
        "| `llm_gpt-4o-mini` | OpenAI LLM parser. Activated with --include_llm (requires OPENAI_API_KEY). |",
        "",
        "## Grounding Metrics",
        "",
        "- **Precision / Recall / F1**: constraint-type level (set-based)",
        "- **Per-type recall**: recall broken down per constraint type",
        "- **Near/Facing anchor accuracy**: whether the anchor category/kind was correctly recovered",
        "",
        "## Downstream Placement Metrics (--downstream)",
        "",
        "- Uses `run_diffopt()` (Proposal + DiffOpt-Constraint variant) with grounded constraints",
        "- Evaluated on `casual` paraphrase style by default (hardest for simple-rule parser)",
        "- **CF**: collision-free rate, **IB**: in-boundary rate",
        "- **Constraint Accuracy**: fraction of GT constraints satisfied by placement",
        "- **CPS**: complete placement success (CF=IB=all constraints=reachability=1)",
        "- **Success@1/5**: CPS achieved by top-1 / any-of-top-5 predictions",
        "",
        "## Key Findings",
        "",
        "- The simple rule parser systematically misses `access_zone`, `facing`, `not_block_door`,",
        "  `near`, and `keep_window_clear` for casual/descriptive/indirect/richer styles.",
        "- The extended rule parser recovers most constraint types from all 5 styles.",
        "- Oracle sets the upper bound — any gap between oracle and rule parsers is recoverable",
        "  with better grounding (LLM or extended rules).",
        "- Downstream placement with rule-grounded constraints degrades for non-formal styles",
        "  because uncaptured constraints are treated as unconstrained → optimizer picks suboptimal poses.",
    ]
    with open(out_dir / "GROUNDING_PROTOCOL.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Protocol: {out_dir}/GROUNDING_PROTOCOL.md")


def _print_summary(by_method: Dict[str, Any]) -> None:
    print("\n── Grounding Summary ─────────────────────────────")
    print(f"  {'Method':<30} {'P':>6} {'R':>6} {'F1':>6}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6}")
    for method, m in sorted(by_method.items()):
        p = m.get("precision", 0)
        r = m.get("recall", 0)
        f1 = m.get("f1", 0)
        print(f"  {method:<30} {p:>6.3f} {r:>6.3f} {f1:>6.3f}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--grounding_benchmark_dir", default="spacefit_v2/data/grounding_benchmark")
    p.add_argument("--single_target_benchmark_dir", default="spacefit_v2/data/single_target_benchmark")
    p.add_argument("--out_dir", default="spacefit_v2/results/grounding_comparison")
    p.add_argument("--splits", nargs="+", default=["test"])
    p.add_argument("--max_cases", type=int, default=None, help="Limit original benchmark cases")
    p.add_argument("--downstream", action="store_true", help="Run downstream placement eval (slow)")
    p.add_argument("--downstream_style", default="casual", help="Paraphrase style to use for downstream eval")
    p.add_argument("--downstream_max_cases", type=int, default=20)
    p.add_argument("--include_llm", action="store_true", help="Include LLM grounding (requires OPENAI_API_KEY)")
    p.add_argument("--llm_model", default="gpt-4o-mini")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--diffopt_iters", type=int, default=80)
    return p


def main(args: argparse.Namespace) -> None:
    grounding_dir = Path(args.grounding_benchmark_dir)
    single_target_dir = Path(args.single_target_benchmark_dir)
    out_dir = Path(args.out_dir)

    # Load grounding benchmark
    with open(grounding_dir / "cases.json") as f:
        paraphrase_cases: List[Dict[str, Any]] = json.load(f)
    with open(grounding_dir / "manifest.json") as f:
        grounding_manifest = json.load(f)

    # Load original single-target benchmark
    with open(single_target_dir / "cases.json") as f:
        original_cases_all: List[Dict[str, Any]] = json.load(f)

    selected_original = [c for c in original_cases_all if c["split"] in args.splits]
    if args.max_cases:
        selected_original = selected_original[: args.max_cases]

    original_ids = {c["id"] for c in selected_original}
    original_by_id = {c["id"]: c for c in selected_original}

    # Filter paraphrase cases to selected original cases
    selected_para = [item for item in paraphrase_cases if item["case_id"] in original_ids]
    print(f"Grounding benchmark: {len(selected_para)} paraphrase items ({len(original_ids)} cases × 5 styles)")

    # ── Part 1: Grounding quality eval ───────────────────────────────────────
    print("Running grounding quality evaluation...")
    grounding_results = run_grounding_eval(
        paraphrase_cases=selected_para,
        original_cases_by_id=original_by_id,
        include_llm=args.include_llm,
        llm_model=args.llm_model,
    )
    _save_grounding_results(out_dir, grounding_results)
    _write_protocol(out_dir)

    # ── Part 2: Downstream placement eval (optional) ─────────────────────────
    if args.downstream:
        print(f"\nRunning downstream placement eval (style='{args.downstream_style}', max_cases={args.downstream_max_cases})...")
        downstream_cases = selected_original[: args.downstream_max_cases]
        para_by_composed_id = {item["id"]: item for item in selected_para}

        downstream_results = run_downstream_placement(
            original_cases=downstream_cases,
            original_cases_by_id=original_by_id,
            paraphrase_style=args.downstream_style,
            paraphrase_cases_by_id=para_by_composed_id,
            topk=args.topk,
            device=args.device,
            diffopt_iters=args.diffopt_iters,
            include_llm=args.include_llm,
            llm_model=args.llm_model,
        )
        _save_downstream_results(out_dir, downstream_results)

    print("\nDone.")


if __name__ == "__main__":
    main(build_parser().parse_args())
