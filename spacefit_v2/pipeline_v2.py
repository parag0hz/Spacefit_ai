"""SpaceFit v2 pipeline: reuse Stage 1-4, replace Stage 5 with diff-opt."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from spacefit_v2.legacy.core.free_space import extract_candidates
from spacefit_v2.legacy.core.occupancy import create_occupancy_grid
from spacefit_v2.legacy.core.scene_parser import parse_room
from spacefit_v2.legacy.llm.placement_select import select_placements
from spacefit_v2.legacy.llm.zone_recognition import recognize_zones
from spacefit_v2.optim.diff_refine import DifferentiableRefiner, _normalize_category, _size_dict


def heuristic_select(candidates: Sequence[Any], new_furniture: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        return {"placements": []}
    def _area(item: Any) -> float:
        if hasattr(item, "area"):
            return float(item.area)
        return float(item.get("area", 0.0))

    def _region_id(item: Any) -> int:
        if hasattr(item, "id"):
            return int(item.id)
        return int(item.get("id", 0))

    ordered = sorted(candidates, key=_area, reverse=True)
    out = []
    for idx, furn in enumerate(new_furniture):
        region = ordered[idx % len(ordered)]
        out.append({
            "furniture_id": furn["id"],
            "selected_region": _region_id(region),
        })
    return {"placements": out}


class SpaceFitV2Pipeline:
    def __init__(self, scorer_path: str | None = None, use_llm: bool = False, device: str = "cpu") -> None:
        self.refiner = DifferentiableRefiner(scorer_path=scorer_path, device=device)
        self.use_llm = bool(use_llm)

    def place(self, room_json_path: str, new_furniture: List[Dict[str, Any]], instruction: str = "") -> List[Dict[str, Any]]:
        scene = parse_room(room_json_path)
        occ = create_occupancy_grid(scene)
        candidates = extract_candidates(occ, scene)

        if self.use_llm:
            zones = recognize_zones(scene, candidates)
            selections = select_placements(scene, candidates, zones, new_furniture, instruction)
        else:
            selections = heuristic_select(candidates, new_furniture)

        return self.refiner.refine(
            scene=scene,
            new_furniture=new_furniture,
            candidate_regions=candidates,
            selected_regions=selections,
        )
