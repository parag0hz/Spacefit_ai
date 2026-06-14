"""Extended rule-based grounder with broader vocabulary coverage.

Catches the vocabulary gaps that the simple deterministic parser misses:
- 'along/flush/toward the wall' for against_wall
- 'next to/beside/close to/adjacent to' for near
- 'oriented/turned/angled toward' for facing
- 'door stays clear/passable/entrance' for not_block_door
- 'window unobstructed/light/don't block window' for keep_window_clear
- 'reach/clearance/movement/comfortable/space' for access_zone

This parser approximates what an LLM would infer from context.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class ExtendedRuleGrounder:
    """Rule-based grounder with broader vocabulary coverage than the simple parser."""

    def __init__(self) -> None:
        pass

    def ground(
        self,
        target_asset: Dict[str, Any],
        intent: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        subject_id = str((target_asset or {}).get("id", "target"))
        nl = str(text or (intent or {}).get("text", "")).lower()
        constraints = self._parse(nl, subject_id, target_asset)
        return {
            "provider": "extended_rule",
            "mode": "extended_rule",
            "constraints": constraints,
            "priorities": {"physics": 1.0, "constraints": 1.0 if constraints else 0.0},
        }

    def _parse(self, text: str, subject_id: str, target_asset: Dict[str, Any]) -> List[Dict[str, Any]]:
        constraints: List[Dict[str, Any]] = []

        # against_wall ──────────────────────────────────────────────────────────
        _WALL_PATTERNS = [
            r"\bagainst\b.*\bwall\b",
            r"\balong\b.*\bwall\b",
            r"\bflush\b.*\bwall\b",
            r"\bpushed?\b.*\bwall\b",
            r"\bwall\b.{0,12}\bhugging\b",
            r"\btucked\b.*\bwall\b",
            r"\bto\s+the\s+wall\b",
            r"\bwall[- ]adjacent\b",
        ]
        if any(re.search(p, text) for p in _WALL_PATTERNS):
            constraints.append({"constraint_type": "against_wall", "subject_id": subject_id})

        # near ──────────────────────────────────────────────────────────────────
        _NEAR_PATTERNS = [
            r"\bnear\s+the\s+([a-z][a-z_ ]{1,30})",
            r"\bnext\s+to\s+the\s+([a-z][a-z_ ]{1,30})",
            r"\bbeside\s+the\s+([a-z][a-z_ ]{1,30})",
            r"\bclose\s+to\s+the\s+([a-z][a-z_ ]{1,30})",
            r"\badjacent\s+to\s+the\s+([a-z][a-z_ ]{1,30})",
            r"\bby\s+the\s+([a-z][a-z_ ]{1,30})",
            r"\bwithin\b.{0,20}\bof\s+the\s+([a-z][a-z_ ]{1,30})",
            r"\balongside\s+the\s+([a-z][a-z_ ]{1,30})",
        ]
        near_match = None
        for p in _NEAR_PATTERNS:
            m = re.search(p, text)
            if m:
                anchor = m.group(1).strip().rstrip("., ").replace(" ", "_")
                if anchor not in {"door", "window", "wall", "entrance"}:
                    near_match = anchor
                    break
        if near_match:
            constraints.append({
                "constraint_type": "near",
                "subject_id": subject_id,
                "target_category": near_match,
                "max_distance": 1.4,
            })

        # facing ────────────────────────────────────────────────────────────────
        _FACING_PATTERNS = [
            r"\bfacing\s+the\s+([a-z][a-z_ ]{1,20})",
            r"\bturned?\s+toward\s+the\s+([a-z][a-z_ ]{1,20})",
            r"\boriented\b.{0,25}\btoward\s+the\s+([a-z][a-z_ ]{1,20})",
            r"\bangled\s+toward\s+the\s+([a-z][a-z_ ]{1,20})",
            r"\bdirected\s+at\s+the\s+([a-z][a-z_ ]{1,20})",
            r"\bfront\b.{0,20}\bfacing\s+the\s+([a-z][a-z_ ]{1,20})",
            r"\bfront\b.{0,20}\btoward\s+the\s+([a-z][a-z_ ]{1,20})",
            r"\bloking\s+at\s+the\s+([a-z][a-z_ ]{1,20})",
        ]
        for p in _FACING_PATTERNS:
            m = re.search(p, text)
            if m:
                target = m.group(1).strip().rstrip("., ").replace(" ", "_")
                c: Dict[str, Any] = {"constraint_type": "facing", "subject_id": subject_id}
                if target in {"window", "door", "center"}:
                    c["target_kind"] = target
                else:
                    c["target_category"] = target
                constraints.append(c)
                break

        # not_block_door ────────────────────────────────────────────────────────
        _DOOR_BLOCKED_PATTERNS = [
            r"\bdoor\b.{0,30}\b(not block|without blocking|stays? clear|passable|clear|open|accessible|free|unobstructed)\b",
            r"\b(not block|without blocking|keep clear|don'?t block|clear of)\b.{0,30}\bdoor\b",
            r"\bdoorway\b.{0,25}\b(clear|free|passable|accessible|open)\b",
            r"\b(entrance|doorway)\b.{0,20}\b(remains?|stays?|kept?)\b.{0,20}\b(open|clear|passable)\b",
            r"\bso\s+the\s+(door|entrance)\b.{0,30}\b(open|clear|accessible)\b",
        ]
        if any(re.search(p, text) for p in _DOOR_BLOCKED_PATTERNS):
            constraints.append({"constraint_type": "not_block_door", "subject_id": subject_id})

        # keep_window_clear ─────────────────────────────────────────────────────
        _WINDOW_CLEAR_PATTERNS = [
            r"\bwindow\b.{0,30}\b(clear|keep|unobstructed|open|free|accessible)\b",
            r"\b(keep|keeping|don'?t|not)\b.{0,20}\bwindow\b.{0,20}\b(block|obstruct)\b",
            r"\bwindow\b.{0,20}\b(stay|left|remain)\b.{0,15}\b(clear|open|unblocked|unobstructed)\b",
            r"\blight\b.{0,40}\b(unhindered|flow|through|come through)\b",
            r"\bnatural\s+light\b",
        ]
        if any(re.search(p, text) for p in _WINDOW_CLEAR_PATTERNS):
            constraints.append({"constraint_type": "keep_window_clear", "subject_id": subject_id})

        # access_zone ───────────────────────────────────────────────────────────
        _ACCESS_PATTERNS = [
            r"\baccess\b",
            r"\breach\b",
            r"\beasy\s+to\s+(get|reach|use)\b",
            r"\b(get|reach)\s+to\s+(it|the)\b",
            r"\bclearance\b",
            r"\bwalk\s+around\b",
            r"\bmove\s+around\b",
            r"\bmovement\b",
            r"\bcomforta\w+\s+(use|access|reach)\b",
            r"\bcomforta\w+\b.{0,20}\bspace\b",
            r"\benough\s+space\b",
            r"\broom\s+(to|around|for)\b",
            r"\bspace\s+around\b",
            r"\bcirculation\b",
        ]
        if any(re.search(p, text) for p in _ACCESS_PATTERNS):
            # Avoid double-counting from not_block_door "accessible" in door context
            # (check that access cue is NOT only inside a door-access phrase)
            if not self._only_door_accessible(text):
                constraints.append({"constraint_type": "access_zone", "subject_id": subject_id})

        return constraints

    @staticmethod
    def _only_door_accessible(text: str) -> bool:
        """Return True if 'accessible/access' only appears in a door-access context."""
        # Count standalone access cues outside door phrases
        door_ctx = re.sub(r"\b(door|entrance|doorway)\b.{0,50}", "", text)
        standalone_access = any(
            re.search(p, door_ctx)
            for p in [r"\baccess\b", r"\breach\b", r"\bclearance\b",
                      r"\broom\s+(to|around)\b", r"\bspace\s+around\b"]
        )
        return not standalone_access
