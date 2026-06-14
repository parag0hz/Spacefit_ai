"""Experiment 3: VLM grounding subset evaluation.

Tests grounding quality for visual-language instructions that require
understanding the room layout from a top-down image.

Grounding methods compared:
  - simple_rule: deterministic text parser on vlm instruction text
  - extended_rule: broader regex patterns on vlm instruction text
  - oracle: GT constraints from original benchmark case

VLM inference (--vlm_provider openai) passes the top-down render image + instruction
to GPT-4V and extracts structured constraints from the response.

Usage:
    # Rule-only eval (fast, no API needed)
    python -m spacefit_v2.scripts.run_vlm_grounding

    # Include VLM inference (requires OPENAI_API_KEY, costs ~$0.01/image)
    python -m spacefit_v2.scripts.run_vlm_grounding --vlm_provider openai --vlm_model gpt-4o
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from spacefit_v2.grounding.extended_rule_grounder import ExtendedRuleGrounder
from spacefit_v2.grounding.metrics import (
    aggregate_by_method,
    aggregate_grounding_metrics,
    anchor_match,
    constraint_type_scores,
    make_grounding_table,
    per_type_recall,
)
from spacefit_v2.grounding.render import render_topdown
from spacefit_v2.single_target.intent_grounder import IntentGrounder


_GROUNDER_SIMPLE = IntentGrounder(provider="deterministic")
_GROUNDER_EXTENDED = ExtendedRuleGrounder()

_ALL_CTYPES = ("against_wall", "near", "facing", "not_block_door", "keep_window_clear", "access_zone")

# ── VLM inference ─────────────────────────────────────────────────────────────

_VLM_SYSTEM = (
    "You are a spatial reasoning assistant. You will be shown a top-down floor plan of a room "
    "and given a furniture placement instruction. "
    "Extract structured placement constraints as a JSON array. "
    "Each object must have a 'constraint_type'. Allowed types: "
    "against_wall, near, facing, not_block_door, keep_window_clear, access_zone. "
    "For 'near': add 'target_category' (string). "
    "For 'facing': add 'target_kind' (window/door/center) or 'target_category'. "
    "Return ONLY a JSON array, no explanation."
)


def _encode_image_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("utf-8")


def _call_vlm_openai(
    png_path: Optional[str],
    instruction_text: str,
    subject_id: str,
    model: str = "gpt-4o",
) -> List[Dict[str, Any]]:
    """Call GPT-4V with floor plan image + instruction, return parsed constraints."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    import openai
    import re

    client = openai.OpenAI(api_key=api_key)

    # Build message content
    content: List[Any] = [{"type": "text", "text": f"Instruction: {instruction_text}"}]

    if png_path and Path(png_path).exists():
        with open(png_path, "rb") as fh:
            b64 = _encode_image_b64(fh.read())
        content.insert(0, {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
        })

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _VLM_SYSTEM},
            {"role": "user", "content": content},
        ],
        temperature=0.0,
        max_tokens=400,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    parsed = json.loads(raw)
    result = []
    for item in parsed:
        item = dict(item)
        item.setdefault("subject_id", subject_id)
        result.append(item)
    return result


# ── Per-item eval ──────────────────────────────────────────────────────────────

def _eval_item(
    vlm_item: Dict[str, Any],
    method_name: str,
    pred_constraints: List[Dict[str, Any]],
) -> Dict[str, Any]:
    gt = vlm_item.get("gt_constraints", [])
    scores = constraint_type_scores(pred_constraints, gt)
    result: Dict[str, Any] = {
        "item_id": vlm_item["id"],
        "case_id": vlm_item["case_id"],
        "style": vlm_item.get("style", "vlm"),
        "visual_cue": vlm_item.get("visual_cue"),
        "method": method_name,
        "text": vlm_item.get("text", ""),
        "target_category": vlm_item.get("target_category"),
        "room_type": vlm_item.get("room_type"),
        "n_windows": vlm_item.get("n_windows", 0),
        "n_objects": vlm_item.get("n_objects", 0),
        "render_path": vlm_item.get("render_path"),
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1": scores["f1"],
        "tp": scores["tp"],
        "fp": scores["fp"],
        "fn": scores["fn"],
        "pred_constraint_types": [c.get("constraint_type") for c in pred_constraints],
        "gt_constraint_types": vlm_item.get("gt_constraint_types", []),
    }
    for ctype in _ALL_CTYPES:
        result[f"recall_{ctype}"] = per_type_recall(pred_constraints, gt, ctype)
    result["near_anchor_match"] = anchor_match(pred_constraints, gt, "near")
    result["facing_anchor_match"] = anchor_match(pred_constraints, gt, "facing")
    return result


# ── Main eval loop ────────────────────────────────────────────────────────────

def run_vlm_grounding_eval(
    vlm_cases: List[Dict[str, Any]],
    original_cases_by_id: Dict[str, Dict[str, Any]],
    vlm_provider: Optional[str] = None,
    vlm_model: str = "gpt-4o",
    render_missing: bool = True,
) -> List[Dict[str, Any]]:
    all_results: List[Dict[str, Any]] = []

    for i, item in enumerate(vlm_cases):
        case_id = item["case_id"]
        orig_case = original_cases_by_id.get(case_id)
        if orig_case is None:
            continue

        target_asset = orig_case["target_asset"]
        subject_id = str(target_asset.get("id", "target"))
        text = item.get("text", "")

        # Ensure render exists if needed
        render_path = item.get("render_path")
        if render_missing and (not render_path or not Path(render_path).exists()):
            render_path = _ensure_render(orig_case, item)
            item = dict(item)
            item["render_path"] = render_path

        # Simple rule on vlm instruction text
        pred_simple = _GROUNDER_SIMPLE.ground(target_asset=target_asset, text=text).get("constraints", [])
        all_results.append(_eval_item(item, "simple_rule", pred_simple))

        # Extended rule
        pred_extended = _GROUNDER_EXTENDED.ground(target_asset=target_asset, text=text).get("constraints", [])
        all_results.append(_eval_item(item, "extended_rule", pred_extended))

        # Oracle
        gt_constraints = [dict(c) for c in item.get("gt_constraints", [])]
        all_results.append(_eval_item(item, "oracle", gt_constraints))

        # VLM (image + text)
        if vlm_provider == "openai":
            try:
                pred_vlm = _call_vlm_openai(
                    png_path=render_path,
                    instruction_text=text,
                    subject_id=subject_id,
                    model=vlm_model,
                )
                all_results.append(_eval_item(item, f"vlm_{vlm_model}", pred_vlm))
            except Exception as exc:
                print(f"  [warn] VLM call failed for {item['id']}: {exc}")

        if (i + 1) % 10 == 0:
            print(f"  VLM grounding eval: {i + 1}/{len(vlm_cases)} items done")

    return all_results


def _ensure_render(orig_case: Dict[str, Any], vlm_item: Dict[str, Any]) -> Optional[str]:
    """Render top-down PNG if missing, save to vlm_subset/renders/."""
    try:
        renders_dir = Path("spacefit_v2/data/vlm_subset/renders")
        renders_dir.mkdir(parents=True, exist_ok=True)
        png_path = renders_dir / f"{orig_case['id']}.png"
        if not png_path.exists():
            render_topdown(
                scene=orig_case["scene"],
                target_asset=orig_case["target_asset"],
                save_path=str(png_path),
            )
        return str(png_path) if png_path.exists() else None
    except Exception:
        return None


# ── Save outputs ──────────────────────────────────────────────────────────────

def _save_results(out_dir: Path, all_results: List[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "vlm_grounding_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    fieldnames = [
        "item_id", "case_id", "style", "visual_cue", "method",
        "precision", "recall", "f1", "tp", "fp", "fn",
        "recall_against_wall", "recall_near", "recall_facing",
        "recall_not_block_door", "recall_keep_window_clear", "recall_access_zone",
        "near_anchor_match", "facing_anchor_match",
        "target_category", "room_type", "n_windows", "n_objects",
    ]
    with open(out_dir / "vlm_grounding_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    by_method = aggregate_by_method(all_results)

    # Per visual_cue breakdown
    by_cue: Dict[str, List[Dict]] = {}
    for r in all_results:
        cue = r.get("visual_cue") or "unknown"
        by_cue.setdefault(cue, []).append(r)
    by_cue_agg = {cue: aggregate_grounding_metrics(rows) for cue, rows in by_cue.items()}

    table_str = make_grounding_table(by_method)
    with open(out_dir / "VLM_GROUNDING_TABLE.md", "w") as f:
        cue_lines = ["\n## F1 by Method × Visual Cue\n"]
        all_cues = sorted(by_cue_agg.keys())
        cue_lines.append("| Method | " + " | ".join(all_cues) + " |")
        cue_lines.append("|---|" + "---:|" * len(all_cues))
        for method_name in by_method:
            method_rows = [r for r in all_results if r["method"] == method_name]
            by_cue_method = {}
            for r in method_rows:
                cue = r.get("visual_cue") or "unknown"
                by_cue_method.setdefault(cue, []).append(r)
            row = f"| {method_name} |"
            for cue in all_cues:
                rows_here = by_cue_method.get(cue, [])
                agg = aggregate_grounding_metrics(rows_here) if rows_here else {}
                f1 = agg.get("f1")
                row += f" {f1:.3f} |" if f1 is not None else " N/A |"
            cue_lines.append(row)
        f.write(table_str + "\n".join(cue_lines) + "\n")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_items": len(all_results),
        "methods": list(by_method.keys()),
        "by_method": by_method,
        "by_visual_cue": by_cue_agg,
    }
    with open(out_dir / "vlm_grounding_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  VLM results: {out_dir}/vlm_grounding_results.json")
    print(f"  VLM table:   {out_dir}/VLM_GROUNDING_TABLE.md")

    print("\n── VLM Grounding Summary ──────────────────────────")
    print(f"  {'Method':<30} {'P':>6} {'R':>6} {'F1':>6}")
    for method_name, m in sorted(by_method.items()):
        print(f"  {method_name:<30} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--vlm_subset_dir", default="spacefit_v2/data/vlm_subset")
    p.add_argument("--single_target_benchmark_dir", default="spacefit_v2/data/single_target_benchmark")
    p.add_argument("--out_dir", default="spacefit_v2/results/vlm_grounding")
    p.add_argument("--vlm_provider", default=None, choices=[None, "openai"],
                   help="VLM API provider. None = rule-only eval.")
    p.add_argument("--vlm_model", default="gpt-4o")
    p.add_argument("--render_missing", action="store_true", default=True,
                   help="Re-render top-down PNGs for cases where the file is missing")
    return p


def main(args: argparse.Namespace) -> None:
    vlm_dir = Path(args.vlm_subset_dir)
    single_target_dir = Path(args.single_target_benchmark_dir)
    out_dir = Path(args.out_dir)

    with open(vlm_dir / "cases.json") as f:
        vlm_cases: List[Dict[str, Any]] = json.load(f)
    with open(single_target_dir / "cases.json") as f:
        original_cases: List[Dict[str, Any]] = json.load(f)

    original_by_id = {c["id"]: c for c in original_cases}
    n_cases = len({item["case_id"] for item in vlm_cases})
    print(f"VLM grounding eval: {len(vlm_cases)} instructions across {n_cases} cases")
    if args.vlm_provider:
        print(f"  Using VLM provider: {args.vlm_provider} / {args.vlm_model}")
    else:
        print("  Rule-only eval (pass --vlm_provider openai to use VLM)")

    results = run_vlm_grounding_eval(
        vlm_cases=vlm_cases,
        original_cases_by_id=original_by_id,
        vlm_provider=args.vlm_provider,
        vlm_model=args.vlm_model,
        render_missing=args.render_missing,
    )

    _save_results(out_dir, results)
    print("Done.")


if __name__ == "__main__":
    main(build_parser().parse_args())
