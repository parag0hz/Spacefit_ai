"""Intent grounder: deterministic fallback + optional OpenAI LLM provider."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence


class IntentGrounder:
    def __init__(
        self,
        provider: str = "deterministic",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.provider = str(provider)
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    def _from_text(self, text: str, subject_id: str) -> List[Dict[str, Any]]:
        text = str(text or "").lower()
        constraints: List[Dict[str, Any]] = []
        if "against" in text and "wall" in text:
            constraints.append({"constraint_type": "against_wall", "subject_id": subject_id})
        if "door" in text and ("not block" in text or "without blocking" in text):
            constraints.append({"constraint_type": "not_block_door", "subject_id": subject_id})
        if "window" in text and ("clear" in text or "keep" in text):
            constraints.append({"constraint_type": "keep_window_clear", "subject_id": subject_id})
        if "access" in text:
            constraints.append({"constraint_type": "access_zone", "subject_id": subject_id})
        if "facing the window" in text:
            constraints.append({"constraint_type": "facing", "subject_id": subject_id, "target_kind": "window"})
        near_match = re.search(r"near the ([a-z_ ]+)", text)
        if near_match:
            target = near_match.group(1).strip().replace(" ", "_")
            if target not in {"door", "window"}:
                constraints.append({"constraint_type": "near", "subject_id": subject_id, "target_category": target, "max_distance": 1.4})
        return constraints

    # ── helpers ───────────────────────────────────────────────────────────────

    def _to_natural_language(self, target_category: str, constraints: List[Dict[str, Any]]) -> str:
        """Convert structured constraints to a free-form placement instruction."""
        cat = target_category.replace("_", " ")
        has_wall = any(c.get("constraint_type") == "against_wall" for c in constraints)
        near_c = next((c for c in constraints if c.get("constraint_type") == "near"), None)
        facing_c = next((c for c in constraints if c.get("constraint_type") == "facing"), None)
        has_door = any(c.get("constraint_type") == "not_block_door" for c in constraints)
        has_win = any(c.get("constraint_type") == "keep_window_clear" for c in constraints)
        has_access = any(c.get("constraint_type") == "access_zone" for c in constraints)

        desc = f"Place the {cat}"
        if has_wall:
            desc += " against the wall"
        if near_c:
            anchor = str(near_c.get("target_category") or "furniture").replace("_", " ")
            desc += f", near the {anchor}"
        if facing_c:
            kind = facing_c.get("target_kind") or str(facing_c.get("target_category") or "").replace("_", " ")
            if kind:
                desc += f", facing the {kind}"
        clarifications: List[str] = []
        if has_door:
            clarifications.append("without blocking the door")
        if has_win:
            clarifications.append("keeping the window area clear")
        if has_access:
            clarifications.append("with enough room to reach it")
        if clarifications:
            desc += " " + ", ".join(clarifications)
        return desc + "."

    def _ground_with_openai(self, text: str, subject_id: str) -> List[Dict[str, Any]]:
        """Call OpenAI to parse free-form placement text into structured constraints."""
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        base_url = self.base_url or os.environ.get("OPENAI_BASE_URL")
        if not api_key:
            return self._from_text(text, subject_id)
        try:
            import openai
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            client = openai.OpenAI(**kwargs)
            system = (
                "You parse indoor furniture placement instructions into a structured JSON list of constraints. "
                "Each object has a 'constraint_type' key. Allowed types: "
                "against_wall, near, facing, not_block_door, keep_window_clear, access_zone. "
                "For 'near': add target_category (string) and max_distance (float, default 1.4). "
                "For 'facing': add target_kind ('window', 'door', or 'center') or target_category. "
                "Return ONLY a JSON array, no explanation."
            )
            user = f'Instruction: "{text}"'
            response = client.chat.completions.create(
                model=self.model_name or "gpt-4o-mini",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=400,
            )
            content = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if content.startswith("```"):
                content = re.sub(r"^```[a-z]*\n?", "", content)
                content = re.sub(r"\n?```$", "", content)
            parsed = json.loads(content)
            result = []
            for item in parsed:
                item = dict(item)
                item.setdefault("subject_id", subject_id)
                result.append(item)
            return result
        except Exception:
            return self._from_text(text, subject_id)

    # ── public interface ──────────────────────────────────────────────────────

    def ground(
        self,
        target_asset: Dict[str, Any],
        intent: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        subject_id = str((target_asset or {}).get("id", "target"))
        source_constraints = list((intent or {}).get("constraints") or [])

        if self.provider == "openai":
            target_category = str((target_asset or {}).get("category", "furniture"))
            if source_constraints:
                nl_text = self._to_natural_language(target_category, source_constraints)
            else:
                nl_text = text or str((intent or {}).get("text", ""))
            grounded_constraints = self._ground_with_openai(nl_text, subject_id)
            return {
                "provider": self.provider,
                "model_name": self.model_name or "gpt-4o-mini",
                "mode": "llm_grounded",
                "nl_text": nl_text,
                "constraints": grounded_constraints,
                "priorities": {"physics": 1.0, "constraints": 1.0 if grounded_constraints else 0.0},
            }

        # deterministic (default) ─────────────────────────────────────────────
        if source_constraints:
            constraints = [dict(item) for item in source_constraints]
        else:
            constraints = self._from_text(text or (intent or {}).get("text", ""), subject_id=subject_id)

        priorities = {
            "physics": 1.0,
            "constraints": 1.0 if constraints else 0.0,
        }
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "mode": "fallback_structured" if self.provider == "deterministic" else "grounded",
            "constraints": constraints,
            "priorities": priorities,
        }
