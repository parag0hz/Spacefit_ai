"""Stage 4: LLM-guided candidate selection for each new furniture item.

Input:
  scene, candidates, zones, new_furniture list, instruction.
Output:
  list of selections — each maps a new-furniture id to a chosen candidate region id,
  plus semantic hints for the solver (face_toward, preferred_wall, reason).

When a new_furniture item has `_catalog_candidates` (populated from 3D-FUTURE),
the LLM also picks a `selected_model_id` from the candidates. The caller resolves
that to real (width, depth, height) before running the solver.

The LLM is NOT asked for coordinates — only which pre-computed candidate region
each new item should go in, and in what general orientation.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .client import LLMClient
from .zone_recognition import _summarize_candidates, _summarize_scene

_SYSTEM = """You are a spatial-reasoning assistant that assigns new furniture to
pre-computed candidate regions in a room. You will receive:
 - the room summary and existing furniture,
 - a list of candidate regions (id, size, position, adjacencies),
 - detected functional zones,
 - a list of new furniture to place, each possibly with a set of real
   3D-FUTURE catalog candidates (with real width/depth/height), and
 - a natural-language instruction from the user.

You MUST NOT generate numeric coordinates. You must only pick:
 (a) ONE 3D-FUTURE model_id per item (when `catalog_candidates` is provided), and
 (b) ONE candidate region id per item,
plus semantic hints for the downstream physics solver.

Rules:
 1. Every new furniture item must be assigned to exactly one candidate region id.
 2. If the item has catalog_candidates, pick the model_id that best fits the
    instruction AND physically fits the chosen region (width/depth < region
    rect dimensions after accounting for the item's orientation).
 3. Each candidate region may receive at most one new item unless its max_rect
    area comfortably exceeds the combined footprints.
 4. If no region/model combo plausibly fits, return "selected_region": null and
    "selected_model_id": null and briefly explain in "reason".
 5. Provide "face_toward" as one of {"center","wall","door","window",
    "existing:<target>"} where <target> can be:
      - prefix of an existing scene object's UUID
      - a category name ("table","sofa","window","door","chair","bed","desk")
      - a newly-placed item's furniture_id (e.g. "table-new-0")
    The solver will snap the item's rotation to face this target.
 6. Provide "preferred_wall" as "yes"/"no".

Semantic guidance (IMPORTANT — the solver has fine rotation control, so use it):
 - Seating around a table: if the instruction places chairs with a table,
   the TABLE should be placed FIRST (listed first in your output), then each
   chair's face_toward MUST be "existing:<table_furniture_id>" so it actually
   faces the table. preferred_wall: "no".
 - Sofa + coffee_table: sofa face_toward the room center or a window;
   coffee_table face_toward "existing:<sofa_id>".
 - Bookshelf / bench / wardrobe / cabinet: preferred_wall: "yes" so it sits
   flush against the nearest wall with its back to it.
 - Armchair / reading_chair: face_toward "window" for a reading nook, or
   face_toward "existing:<sofa_id>" for a conversation pairing.
 - Desk + chair: desk preferred_wall: "yes", chair face_toward
   "existing:<desk_id>".
 - When multiple identical items are placed (e.g. 4 chairs around a table),
   assign them to the same region as the table — the solver will position
   each one around the table's perimeter.

Output order matters: list items in dependency order. If B faces A, place A
before B in the list.

Return strict JSON:
{
  "placements": [
    {
      "furniture_id": "id_from_input",
      "selected_region": <int or null>,
      "selected_model_id": "<model_id or null>",  // null if no catalog candidates
      "reason": "one short sentence",
      "face_toward": "center" | "wall" | "door" | "window" | "existing:<prefix>",
      "preferred_wall": "yes" | "no",
      "backup_region": <int or null>
    }
  ]
}
"""


def _summarize_new_furniture(new_furniture: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for f in new_furniture:
        entry = {
            "id": f["id"],
            "category": f.get("category") or f.get("category_query", ""),
        }
        if "size" in f and f["size"]:
            entry["size"] = {"width": round(f["size"]["width"], 2),
                              "depth": round(f["size"]["depth"], 2),
                              "height": round(f["size"]["height"], 2)}
        if f.get("_catalog_candidates"):
            entry["catalog_candidates"] = [
                {"model_id": c["model_id"],
                 "category": c["category"],
                 "style": c.get("style"),
                 "width": round(c["width"], 2),
                 "depth": round(c["depth"], 2),
                 "height": round(c["height"], 2)}
                for c in f["_catalog_candidates"]
            ]
        out.append(entry)
    return out


def _summarize_zones(zones_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{
        "name": z.get("name"),
        "furniture_ids": z.get("furniture_ids", []),
        "anchor_region_ids": z.get("anchor_region_ids", []),
        "complementary_categories": z.get("complementary_categories", []),
        "incompatible_categories": z.get("incompatible_categories", []),
    } for z in zones_dict.get("zones", [])]


def select_placements(scene: Dict[str, Any], candidates: List, zones: Dict[str, Any],
                      new_furniture: List[Dict[str, Any]], instruction: str,
                      llm: Optional[LLMClient] = None) -> Dict[str, Any]:
    llm = llm or LLMClient()
    user_payload = {
        "instruction": instruction,
        "scene": _summarize_scene(scene),
        "candidates": _summarize_candidates(candidates),
        "zones": _summarize_zones(zones),
        "new_furniture": _summarize_new_furniture(new_furniture),
    }
    result = llm.chat_json(
        system=_SYSTEM,
        user=json.dumps(user_payload, ensure_ascii=False),
        tag="placement_select",
    )

    raw = result.get("placements", [])
    valid_ids = {c.id for c in candidates}
    # map furniture_id → set of valid catalog model_ids
    valid_models: Dict[str, set] = {}
    for f in new_furniture:
        if f.get("_catalog_candidates"):
            valid_models[f["id"]] = {c["model_id"] for c in f["_catalog_candidates"]}
    cleaned = []
    for entry in raw:
        fid = entry.get("furniture_id")
        sel = entry.get("selected_region")
        backup = entry.get("backup_region")
        model_id = entry.get("selected_model_id")
        # coerce + validate
        sel_int = int(sel) if isinstance(sel, (int, float)) else None
        if sel_int is not None and sel_int not in valid_ids:
            sel_int = None
        backup_int = int(backup) if isinstance(backup, (int, float)) else None
        if backup_int is not None and backup_int not in valid_ids:
            backup_int = None
        if fid in valid_models and (not model_id or model_id not in valid_models[fid]):
            # invalid/missing — pick the smallest candidate as a safe fallback
            cand_pool = next(f for f in new_furniture if f["id"] == fid)["_catalog_candidates"]
            model_id = min(cand_pool,
                           key=lambda c: c["width"] * c["depth"])["model_id"]
        cleaned.append({
            "furniture_id": fid,
            "selected_region": sel_int,
            "backup_region": backup_int,
            "selected_model_id": model_id,
            "reason": entry.get("reason", ""),
            "face_toward": entry.get("face_toward", "center"),
            "preferred_wall": str(entry.get("preferred_wall", "no")).lower() == "yes",
        })

    # Ensure every new furniture has an entry (fallback to largest available region)
    seen = {e["furniture_id"] for e in cleaned}
    sorted_cands = sorted(candidates, key=lambda c: -c.area)
    fallback_idx = 0
    for f in new_furniture:
        if f["id"] not in seen:
            sel = sorted_cands[fallback_idx % len(sorted_cands)].id if sorted_cands else None
            fallback_idx += 1
            cleaned.append({
                "furniture_id": f["id"],
                "selected_region": sel,
                "backup_region": None,
                "reason": "fallback — LLM did not return this item",
                "face_toward": "center",
                "preferred_wall": False,
            })
    return {"placements": cleaned, "_llm_usage": llm.usage.__dict__}
