"""Stage 3: Functional-zone recognition via GPT-4o.

Input : parsed scene + candidate regions.
Output: list of zones — {name, furniture_ids, complementary_categories, incompatible_categories,
        bbox (optional), description}.

The LLM is ONLY asked to cluster existing furniture into zones and comment on
which new-furniture categories would fit or clash. It is never asked for
coordinates.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .client import LLMClient

_SYSTEM = """You are a spatial-reasoning assistant for a room-layout pipeline.
You will be given a room scan and its candidate free-space regions.
Your job is to group existing furniture into functional zones
(e.g. dining zone, work zone, lounge zone, sleeping zone, storage zone,
 circulation path). You DO NOT generate coordinates.

Return strict JSON with this schema:
{
  "zones": [
    {
      "name": "short lowercase identifier, e.g. 'dining'",
      "description": "one short sentence",
      "furniture_ids": ["existing_obj_id1", ...],
      "anchor_region_ids": [candidate_region_id, ...],  // up to 3 region ids that
                                                         // overlap/neighbor this zone
      "complementary_categories": ["category_name", ...], // new furniture that fits well
      "incompatible_categories": ["category_name", ...]   // categories to avoid here
    }
  ],
  "global_notes": "one short paragraph of any whole-room observations"
}
"""


def _summarize_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    f = scene["floor"]
    xmin, zmin, xmax, zmax = f["bounds"]
    objs = [{
        "id": o["id"][:8],
        "category": o["category"],
        "subtype": o["subtype"],
        "position": {"x": round(o["position"][0], 2), "z": round(o["position"][2], 2)},
        "size": {"width": round(o["size"][0], 2),
                 "height": round(o["size"][1], 2),
                 "depth": round(o["size"][2], 2)},
        "yaw": round(o["yaw"], 0),
    } for o in scene["objects"]]
    doors = [{"x": round(d["position"][0], 2), "z": round(d["position"][2], 2),
              "width": round(d["width"], 2)} for d in scene.get("doors", [])]
    windows = [{"x": round(w["position"][0], 2), "z": round(w["position"][2], 2),
                "width": round(w["width"], 2), "height": round(w["height"], 2)}
               for w in scene.get("windows", [])]
    return {
        "room_bounds": {"xmin": round(xmin, 2), "xmax": round(xmax, 2),
                        "zmin": round(zmin, 2), "zmax": round(zmax, 2),
                        "area_m2": round(f["area_m2"], 1)},
        "room_type_guess": scene.get("room_type", "unknown"),
        "doors": doors,
        "windows": windows,
        "objects": objs,
    }


def _summarize_candidates(candidates) -> List[Dict[str, Any]]:
    out = []
    for c in candidates:
        cx, cz, w, d, _ = c.max_rect
        out.append({
            "id": c.id,
            "area_m2": round(c.area, 2),
            "centroid": {"x": round(cx, 2), "z": round(cz, 2)},
            "rect": {"width": round(w, 2), "depth": round(d, 2)},
            "near_wall": c.near_wall,
            "near_door": c.near_door,
            "near_window": c.near_window,
        })
    return out


def recognize_zones(scene: Dict[str, Any], candidates: List,
                    llm: Optional[LLMClient] = None) -> Dict[str, Any]:
    llm = llm or LLMClient()
    scene_summary = _summarize_scene(scene)
    cand_summary = _summarize_candidates(candidates)

    # Short-circuit: empty room or no existing furniture → single zone over whole room.
    if not scene["objects"]:
        return {
            "zones": [{
                "name": "empty_room",
                "description": "Room has no existing furniture; treat entire floor as one zone.",
                "furniture_ids": [],
                "anchor_region_ids": [c.id for c in candidates[:3]],
                "complementary_categories": [],
                "incompatible_categories": [],
            }],
            "global_notes": "Empty room.",
            "_llm_usage": llm.usage.__dict__,
        }

    user = json.dumps({
        "scene": scene_summary,
        "candidates": cand_summary,
    }, ensure_ascii=False)
    result = llm.chat_json(system=_SYSTEM, user=user, tag="zone_recognition")

    zones = result.get("zones", [])
    if not isinstance(zones, list) or not zones:
        zones = [{
            "name": "whole_room",
            "description": "fallback — treat entire floor as one zone",
            "furniture_ids": [o["id"] for o in scene["objects"]],
            "anchor_region_ids": [c.id for c in candidates[:5]],
            "complementary_categories": [],
            "incompatible_categories": [],
        }]
    # normalize anchor_region_ids to ints within range
    max_id = max((c.id for c in candidates), default=-1)
    for z in zones:
        ids = z.get("anchor_region_ids", []) or []
        z["anchor_region_ids"] = [int(i) for i in ids if isinstance(i, (int, float)) and 0 <= int(i) <= max_id]
    return {
        "zones": zones,
        "global_notes": result.get("global_notes", ""),
        "_llm_usage": llm.usage.__dict__,
    }
