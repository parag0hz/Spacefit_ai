"""Grounding evaluation metrics.

All metrics operate at the constraint-type level and at the anchor level.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple


# ── per-instance metrics ──────────────────────────────────────────────────────

def _type_set(constraints: List[Dict[str, Any]]) -> Set[str]:
    return {str(c.get("constraint_type", "")) for c in constraints if c.get("constraint_type")}


def constraint_type_scores(
    pred: List[Dict[str, Any]],
    gt: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Precision, recall, F1 at the constraint-type level."""
    pred_types = _type_set(pred)
    gt_types = _type_set(gt)
    if not gt_types and not pred_types:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    tp = len(pred_types & gt_types)
    fp = len(pred_types - gt_types)
    fn = len(gt_types - pred_types)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def per_type_recall(
    pred: List[Dict[str, Any]],
    gt: List[Dict[str, Any]],
    constraint_type: str,
) -> Optional[bool]:
    """Whether a specific constraint type was correctly recalled. None if not in GT."""
    gt_types = _type_set(gt)
    if constraint_type not in gt_types:
        return None
    pred_types = _type_set(pred)
    return constraint_type in pred_types


def anchor_match(
    pred: List[Dict[str, Any]],
    gt: List[Dict[str, Any]],
    constraint_type: str = "near",
) -> Optional[bool]:
    """Whether the anchor (target_category / target_kind) matches for a given constraint type.

    Returns None if no GT constraint of that type exists.
    """
    gt_c = next((c for c in gt if c.get("constraint_type") == constraint_type), None)
    if gt_c is None:
        return None
    pred_c = next((c for c in pred if c.get("constraint_type") == constraint_type), None)
    if pred_c is None:
        return False
    # Compare target_kind first, then target_category
    for field in ("target_kind", "target_category"):
        gt_val = str(gt_c.get(field) or "").lower().strip()
        pred_val = str(pred_c.get(field) or "").lower().strip()
        if gt_val:
            return gt_val == pred_val
    return True  # type matched with no anchor to check


# ── aggregate over many instances ─────────────────────────────────────────────

def aggregate_grounding_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-instance grounding metrics."""
    n = len(results)
    if n == 0:
        return {}

    avg_p = sum(r["precision"] for r in results) / n
    avg_r = sum(r["recall"] for r in results) / n
    avg_f1 = sum(r["f1"] for r in results) / n

    # Per constraint type recall (among cases where GT has that type)
    type_recalls: Dict[str, List[bool]] = {}
    for ctype in ("against_wall", "near", "facing", "not_block_door", "keep_window_clear", "access_zone"):
        vals = [r.get(f"recall_{ctype}") for r in results if r.get(f"recall_{ctype}") is not None]
        if vals:
            type_recalls[ctype] = float(sum(vals) / len(vals))

    # Near/facing anchor accuracy
    near_anchor = [r["near_anchor_match"] for r in results if r.get("near_anchor_match") is not None]
    facing_anchor = [r["facing_anchor_match"] for r in results if r.get("facing_anchor_match") is not None]

    return {
        "n": n,
        "precision": round(avg_p, 4),
        "recall": round(avg_r, 4),
        "f1": round(avg_f1, 4),
        "per_type_recall": {k: round(v, 4) for k, v in type_recalls.items()},
        "near_anchor_accuracy": round(sum(near_anchor) / len(near_anchor), 4) if near_anchor else None,
        "facing_anchor_accuracy": round(sum(facing_anchor) / len(facing_anchor), 4) if facing_anchor else None,
    }


def aggregate_by_style(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate metrics grouped by paraphrase style."""
    by_style: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        style = r.get("style", "unknown")
        by_style.setdefault(style, []).append(r)
    return {style: aggregate_grounding_metrics(rows) for style, rows in by_style.items()}


def aggregate_by_method(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate metrics grouped by grounding method."""
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        method = r.get("method", "unknown")
        by_method.setdefault(method, []).append(r)
    return {method: aggregate_grounding_metrics(rows) for method, rows in by_method.items()}


def make_grounding_table(
    by_method: Dict[str, Dict[str, Any]],
    by_style_by_method: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
) -> str:
    """Render a markdown grounding comparison table."""
    lines = [
        "## Grounding Metrics by Method",
        "",
        "| Method | Precision ↑ | Recall ↑ | F1 ↑ | Near Anchor Acc ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, m in by_method.items():
        na = m.get("near_anchor_accuracy")
        na_str = f"{na:.3f}" if na is not None else "N/A"
        lines.append(
            f"| {method} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {na_str} |"
        )

    if by_style_by_method:
        lines += ["", "## F1 by Method × Style", ""]
        all_styles = sorted({s for d in by_style_by_method.values() for s in d})
        header = "| Method | " + " | ".join(all_styles) + " |"
        lines.append(header)
        lines.append("|---|" + "---:|" * len(all_styles))
        for method, by_style in by_style_by_method.items():
            row = f"| {method} |"
            for style in all_styles:
                f1 = by_style.get(style, {}).get("f1", None)
                row += f" {f1:.3f} |" if f1 is not None else " N/A |"
            lines.append(row)

    return "\n".join(lines) + "\n"
