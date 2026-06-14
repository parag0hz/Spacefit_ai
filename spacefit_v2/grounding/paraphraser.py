"""Template-based paraphrase generator for placement intent grounding benchmark.

Generates 5 linguistically distinct paraphrase styles per benchmark case.
Each style uses systematically different vocabulary to expose vocabulary-level
gaps in rule-based parsers vs LLM-based grounding.

Style vocabulary challenge matrix (what the simple rule parser catches):
                    against_wall  near  facing  not_block_door  keep_window_clear  access_zone
  formal            ✓             ✓     ✓*      ✓               ✓                  ✓
  casual            ✓             ✗     ✗       ✗               ✗                  ✗
  descriptive       ✗             ✗     ✗       ✗               ✗                  ✗
  indirect          ✓             ✗     ✗       ✗               ✓                  ✗
  richer            ✓             ✗     ✗       ✗               ✗                  ✗

* facing only caught if target is "window" (exact match "facing the window")
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional

# ── per-constraint phrase banks per style ─────────────────────────────────────

_PHRASES: Dict[str, Dict[str, str]] = {
    "against_wall": {
        # rule parser: needs "against" AND "wall"
        "formal":      "against the wall",
        "casual":      "up against the wall",           # rule: ✓ (both words present)
        "descriptive": "along the wall",                # rule: ✗ (no "against")
        "indirect":    "pushed against the wall",       # rule: ✓
        "richer":      "flush against the nearest wall", # rule: ✓
    },
    "near": {
        # rule parser: needs exact "near the <category>"
        "formal":      "near the {anchor}",             # rule: ✓
        "casual":      "right next to the {anchor}",    # rule: ✗
        "descriptive": "positioned close to the {anchor}", # rule: ✗
        "indirect":    "set beside the {anchor}",       # rule: ✗
        "richer":      "within comfortable reach of the {anchor}", # rule: ✗
    },
    "facing": {
        # rule parser: only "facing the window" (exact)
        "formal":      "facing the {target}",           # rule: ✓ only if target=="window"
        "casual":      "turned toward the {target}",    # rule: ✗
        "descriptive": "oriented so it faces the {target}", # rule: ✗
        "indirect":    "angled toward the {target}",    # rule: ✗
        "richer":      "with its front directed at the {target}", # rule: ✗
    },
    "not_block_door": {
        # rule parser: needs "door" AND ("not block" OR "without blocking")
        "formal":      "without blocking the door",     # rule: ✓
        "casual":      "so the door stays clear",       # rule: ✗
        "descriptive": "clear of the doorway",          # rule: ✗
        "indirect":    "keeping the door passable",     # rule: ✗
        "richer":      "so the entrance remains open and accessible", # rule: ✗ (has "accessible"→access_zone cross-fire; careful: "accessible" contains "access" → rule catches access_zone!)
    },
    "keep_window_clear": {
        # rule parser: needs "window" AND ("clear" OR "keep")
        "formal":      "keeping the window area clear", # rule: ✓
        "casual":      "not blocking the window",       # rule: ✗
        "descriptive": "with the window left unobstructed", # rule: ✗
        "indirect":    "leaving the window unblocked",  # rule: ✗
        "richer":      "so natural light can flow through unhindered", # rule: ✗
    },
    "access_zone": {
        # rule parser: needs "access" anywhere in text
        "formal":      "preserving access around it",   # rule: ✓ ("access" present)
        "casual":      "easy to reach",                 # rule: ✗
        "descriptive": "with adequate clearance for movement", # rule: ✗
        "indirect":    "leaving room to move around it", # rule: ✗
        "richer":      "giving enough space for comfortable use", # rule: ✗
    },
}

# ── sentence openers per style ────────────────────────────────────────────────

_OPENERS: Dict[str, str] = {
    "formal":      "Place the {cat}",
    "casual":      "I'd put the {cat}",
    "descriptive": "The {cat} should go",
    "indirect":    "Set the {cat}",
    "richer":      "Position the {cat}",
}

# ── assembly templates per style ─────────────────────────────────────────────
# Each style assembles a full sentence differently so the sentence structure varies.

def _assemble(style: str, cat: str, constraint_phrases: List[str]) -> str:
    opener = _OPENERS[style].format(cat=cat)
    if not constraint_phrases:
        return opener + "."

    if style == "formal":
        return opener + " " + ", ".join(constraint_phrases) + "."

    if style == "casual":
        # Split into placement part and clarification part
        placement = [p for p in constraint_phrases if not _is_clarification(p)]
        clarifications = [p for p in constraint_phrases if _is_clarification(p)]
        parts = [opener]
        if placement:
            parts.append(" ".join(placement))
        base = " ".join(parts)
        if clarifications:
            base += ". " + _capitalize(", ".join(clarifications)) + "."
        else:
            base += "."
        return base

    if style == "descriptive":
        return opener + " " + ", ".join(constraint_phrases) + "."

    if style == "indirect":
        main = [p for p in constraint_phrases if not _is_clarification(p)]
        clarifications = [p for p in constraint_phrases if _is_clarification(p)]
        base = opener
        if main:
            base += " " + " and ".join(main)
        base += "."
        if clarifications:
            base += " " + _capitalize(", ".join(clarifications)) + "."
        return base

    if style == "richer":
        # Use two sentences
        half = max(1, len(constraint_phrases) // 2)
        first_half = constraint_phrases[:half]
        second_half = constraint_phrases[half:]
        first_sentence = opener + " " + ", ".join(first_half) + "."
        if second_half:
            second_sentence = _capitalize(", ".join(second_half)) + "."
            return first_sentence + " " + second_sentence
        return first_sentence

    return opener + " " + ", ".join(constraint_phrases) + "."


def _is_clarification(phrase: str) -> bool:
    """Phrases that work better as follow-up sentences."""
    starters = ("easy", "so the", "clear of", "keeping the", "not block", "without block",
                "giving", "with adequate", "leaving room", "so natural")
    return any(phrase.lower().startswith(s) for s in starters)


def _capitalize(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


# ── public API ────────────────────────────────────────────────────────────────

STYLES = ("formal", "casual", "descriptive", "indirect", "richer")


def generate_paraphrase(
    case: Dict[str, Any],
    style: str,
) -> Dict[str, Any]:
    """Generate one paraphrase for a benchmark case at a given style level.

    Returns a dict with:
      id, case_id, style, text, gt_constraint_types, gt_constraints
    """
    assert style in STYLES, f"unknown style {style!r}"

    cat = str(case["target_asset"]["category"]).replace("_", " ")
    gt_constraints = list(case["intent"].get("constraints", []))

    # Build constraint phrases in a fixed order
    phrase_list: List[str] = []
    for c in gt_constraints:
        ctype = c.get("constraint_type")
        bank = _PHRASES.get(ctype)
        if bank is None:
            continue
        tmpl = bank[style]
        # Fill in anchor/target
        anchor = _resolve_anchor(c, case)
        phrase = tmpl.format(anchor=anchor, target=anchor, cat=cat)
        phrase_list.append(phrase)

    text = _assemble(style, cat, phrase_list)
    uid = hashlib.md5(f"{case['id']}__{style}".encode()).hexdigest()[:8]
    return {
        "id": f"{case['id']}__{style}",
        "uid": uid,
        "case_id": case["id"],
        "split": case.get("split", "test"),
        "style": style,
        "text": text,
        "target_category": case["target_asset"]["category"],
        "room_type": case["scene"].get("room_type"),
        "gt_constraints": [dict(c) for c in gt_constraints],
        "gt_constraint_types": [c["constraint_type"] for c in gt_constraints],
    }


def generate_all_paraphrases(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all 5 style paraphrases for a case."""
    return [generate_paraphrase(case, style) for style in STYLES]


def _resolve_anchor(constraint: Dict[str, Any], case: Dict[str, Any]) -> str:
    """Resolve near/facing target to a human-readable name."""
    target_kind = constraint.get("target_kind")
    if target_kind:
        return str(target_kind).replace("_", " ")
    target_cat = constraint.get("target_category")
    if target_cat:
        return str(target_cat).replace("_", " ")
    target_id = constraint.get("target_id")
    if target_id:
        for obj in case.get("scene", {}).get("objects", []):
            if str(obj.get("id")) == str(target_id):
                cat = obj.get("category", "furniture")
                return str(cat).replace("_", " ")
    return "furniture"
