"""GPT-based natural language intent generation for single-target benchmark."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


_SYSTEM_PROMPT = """\
You are a user who wants to place a piece of furniture in their room.
Given a room description (type, existing furniture, doors, windows), write a realistic,
specific placement instruction for the target item — the way a real person would ask.

Your instruction must:
- Reference only furniture categories listed under "Existing furniture"
- Be natural and specific (include spatial relationships like near, facing, against wall)
- Optionally include functional context (for watching TV, for working, etc.)

After writing the instruction, also output a structured constraint list.

Allowed constraint types:
- "against_wall": placed against or close to a wall
- "near": near another item {target_category: str, max_distance: float (default 1.4), min_distance: float (optional)}
- "beside": directly adjacent to another item, very close {target_category: str}
- "facing": faces something {target_kind: "window"|"door"|"center", OR target_category: str}
- "in_front_of": placed in front of a reference item (from that item's perspective) {target_category: str}
- "left_of": placed to the left of a reference item {target_category: str}
- "right_of": placed to the right of a reference item {target_category: str}
- "not_block_door": must not block a door/doorway
- "keep_window_clear": must not block a window
- "access_zone": should have clear surrounding space for access

Rules for constraints:
- For relational constraints (near, beside, facing, in_front_of, left_of, right_of): ONLY use categories that appear in "Existing furniture"
- subject_id is always "target"
- Include 2–5 constraints that best capture the instruction. Do not invent constraints not implied by the instruction.
- Prefer specific constraints (beside, in_front_of) over generic ones (near) when the instruction is specific.

Return ONLY valid JSON (no markdown):
{
  "user_instruction": "<natural language instruction>",
  "constraints": [
    {"constraint_type": "...", "subject_id": "target", ...}
  ]
}
"""


def _describe_room(case: Dict[str, Any]) -> str:
    scene = case["scene"]
    target_cat = str(case["target_asset"]["category"]).replace("_", " ")
    room_type = str(scene.get("room_type", "room")).replace("_", " ")

    # Existing furniture: list unique categories with counts
    cats: Dict[str, int] = {}
    for obj in scene.get("objects", []):
        cat = str(obj.get("category", "")).replace("_", " ").strip()
        if cat:
            cats[cat] = cats.get(cat, 0) + 1
    furniture_lines = [f"  - {cat} ×{cnt}" for cat, cnt in sorted(cats.items())]

    n_doors = len(scene.get("doors", []))
    n_windows = len(scene.get("windows", []))

    lines = [
        f"Room type: {room_type}",
        f"Existing furniture:",
        *furniture_lines,
        f"Doors: {n_doors}",
        f"Windows: {n_windows}",
        f"Item to place: {target_cat}",
        "",
        f"Generate a realistic user instruction for placing the {target_cat} in this room.",
    ]
    return "\n".join(lines)


def generate_gpt_intent(
    case: Dict[str, Any],
    openai_client: Any,
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """Generate a GPT natural language intent + structured constraints for one case.

    Returns a dict with keys: text, constraints, model, raw_response.
    Falls back to original template intent on failure.
    """
    user_message = _describe_room(case)

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,  # some variety in instructions
            max_tokens=500,
        )
        content = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        content = re.sub(r"^```[a-z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)

        parsed = json.loads(content)
        instruction = str(parsed.get("user_instruction", "")).strip()
        constraints = list(parsed.get("constraints", []))

        # Always use the real asset ID as subject_id so it matches predicted placement asset_id
        subject_id = str(case["target_asset"]["id"])
        for c in constraints:
            c["subject_id"] = subject_id

        # Verify referenced categories exist in scene
        existing_cats = {
            str(obj.get("category", "")).lower().replace(" ", "_")
            for obj in case["scene"].get("objects", [])
        }
        valid_constraints = []
        for c in constraints:
            tc = c.get("target_category")
            if tc is not None:
                tc_norm = str(tc).lower().replace(" ", "_")
                if tc_norm not in existing_cats:
                    continue  # drop constraint referencing non-existent furniture
            valid_constraints.append(c)

        return {
            "text": instruction,
            "constraints": valid_constraints,
            "model": model,
            "raw_response": content,
        }

    except Exception as exc:
        # Fallback to original template intent
        original = case.get("intent", {})
        return {
            "text": original.get("text", ""),
            "constraints": list(original.get("constraints", [])),
            "model": "fallback",
            "error": str(exc),
        }


def generate_gpt_intents_for_cases(
    cases: List[Dict[str, Any]],
    openai_client: Any,
    model: str = "gpt-4o",
    split: Optional[str] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Generate GPT intents for a list of cases, returning augmented cases.

    Each returned case is a copy of the original with `gpt_intent` added.
    The original `intent` field is preserved as `template_intent`.
    """
    import copy

    augmented = []
    targets = [c for c in cases if split is None or c.get("split") == split]

    for idx, case in enumerate(targets, start=1):
        result = copy.deepcopy(case)
        result["template_intent"] = copy.deepcopy(case.get("intent", {}))

        gpt_intent = generate_gpt_intent(case, openai_client, model)
        result["gpt_intent"] = gpt_intent
        result["intent"] = {
            "text": gpt_intent["text"],
            "constraints": gpt_intent["constraints"],
        }

        augmented.append(result)

        if verbose:
            status = "ok" if gpt_intent.get("model") != "fallback" else "fallback"
            print(f"[gpt_intent] {idx}/{len(targets)} [{status}] {case['id'][:60]}")
            if status == "ok":
                print(f"             → {gpt_intent['text'][:80]}")

    return augmented
