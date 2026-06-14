"""Stage-5 replacement: differentiable placement refinement."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.optim as optim
from shapely.geometry import Polygon

from spacefit_v2 import config
from spacefit_v2.device import resolve_torch_device
from spacefit_v2.model.features import (
    CATEGORY_TO_IDX,
    Furniture,
    PlacementContext,
    category_idx,
    extract_placement_features,
)
from spacefit_v2.model.losses import loss_physics, loss_semantic, loss_style
from spacefit_v2.model.scorer import load_scorer_checkpoint
from spacefit_v2.optim.initializer import build_region_box, initialize_pose


PLACEMENT_PRIORITY = {
    "double_bed": 100,
    "single_bed": 95,
    "kids_bed": 95,
    "sofa": 90,
    "dining_table": 85,
    "table": 80,
    "wardrobe": 75,
    "bookshelf": 70,
    "desk": 70,
    "cabinet": 65,
    "tv_stand": 60,
    "nightstand": 45,
    "coffee_table": 45,
    "chair": 40,
    "dining_chair": 40,
    "floor_lamp": 20,
    "ceiling_lamp": 10,
}


def _normalize_category(name: str) -> str:
    return str(name or "").strip().lower().replace(" ", "_").replace("-", "_")


def _size_dict(item: Dict[str, Any]) -> Dict[str, float]:
    size = item.get("size")
    if isinstance(size, dict):
        return {
            "width": float(size.get("width", item.get("width", 0.0))),
            "depth": float(size.get("depth", item.get("depth", 0.0))),
            "height": float(size.get("height", item.get("height", 0.0))),
        }
    if isinstance(size, (list, tuple)):
        if len(size) >= 3:
            return {"width": float(size[0]), "depth": float(size[2]), "height": float(size[1])}
        return {"width": float(size[0]), "depth": float(size[1]), "height": 0.0}
    return {
        "width": float(item.get("width", 0.0)),
        "depth": float(item.get("depth", 0.0)),
        "height": float(item.get("height", 0.0)),
    }


def _item_position(item: Dict[str, Any]) -> tuple[float, float]:
    pos = item.get("position")
    if isinstance(pos, dict):
        return float(pos["x"]), float(pos["z"])
    if isinstance(pos, (list, tuple)):
        if len(pos) >= 3:
            return float(pos[0]), float(pos[2])
        return float(pos[0]), float(pos[1])
    return float(item.get("x", 0.0)), float(item.get("z", 0.0))


def _scene_floor_polygon(scene: Dict[str, Any]) -> Sequence[Sequence[float]]:
    if "floor_polygon" in scene:
        return scene["floor_polygon"]
    if "floor_plan_vertices" in scene:
        return scene["floor_plan_vertices"]
    return scene["floor"]["polygon"]


def _scene_walls(scene: Dict[str, Any]) -> Sequence[Any]:
    if "walls" in scene and scene["walls"]:
        walls = scene["walls"]
        if isinstance(walls[0], dict):
            return walls
        return walls
    return []


def _scene_doors(scene: Dict[str, Any]) -> Sequence[Any]:
    return scene.get("doors", [])


def _existing_items(scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "existing_furniture" in scene:
        return [dict(item) for item in scene.get("existing_furniture", [])]
    if "objects" in scene:
        out = []
        for obj in scene["objects"]:
            out.append({
                "id": obj.get("id"),
                "category": _normalize_category(obj.get("category", "")),
                "size": {
                    "width": float(obj["size"][0]),
                    "depth": float(obj["size"][2]),
                    "height": float(obj["size"][1]),
                },
                "position": {"x": float(obj["position"][0]), "z": float(obj["position"][2])},
                "rotation_y": float(obj.get("yaw", 0.0)),
            })
        return out
    if "furniture" in scene:
        return [dict(item) for item in scene["furniture"]]
    return []


def _to_furniture(item: Dict[str, Any]) -> Furniture:
    size = _size_dict(item)
    x, z = _item_position(item)
    yaw_deg = float(item.get("rotation_y", item.get("yaw", 0.0)))
    yaw = math.radians(yaw_deg) if abs(yaw_deg) > 2.0 * math.pi + 1e-3 else yaw_deg
    return Furniture(
        x=x,
        z=z,
        yaw=yaw,
        width=size["width"],
        depth=size["depth"],
        height=size["height"],
        category=_normalize_category(item.get("category", "unknown")),
    )


def _selection_map(selected_regions: Any) -> Dict[str, int]:
    if selected_regions is None:
        return {}
    if isinstance(selected_regions, dict) and "placements" in selected_regions:
        return {
            str(entry["furniture_id"]): int(entry["selected_region"])
            for entry in selected_regions["placements"]
            if entry.get("selected_region") is not None
        }
    if isinstance(selected_regions, list):
        return {
            str(entry["furniture_id"]): int(entry["selected_region"])
            for entry in selected_regions
            if entry.get("selected_region") is not None
        }
    return dict(selected_regions)


def _priority(item: Dict[str, Any]) -> tuple[int, float]:
    cat = _normalize_category(item.get("category", ""))
    size = _size_dict(item)
    area = size["width"] * size["depth"]
    return (PLACEMENT_PRIORITY.get(cat, 30), area)


def _sorted_furniture(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda item: _priority(item), reverse=True)


def _placement_polygon(x: float, z: float, width: float, depth: float, yaw_deg: float) -> Polygon:
    yaw = math.radians(yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    hw = width * 0.5
    hd = depth * 0.5
    pts = []
    for dx, dz in ((hw, hd), (-hw, hd), (-hw, -hd), (hw, -hd)):
        pts.append((x + dx * c - dz * s, z + dx * s + dz * c))
    return Polygon(pts)


def _is_feasible(
    x: float,
    z: float,
    yaw_deg: float,
    size: Dict[str, float],
    floor_polygon: Sequence[Sequence[float]],
    placed_so_far: Sequence[Dict[str, Any]],
    outside_tol_area: float = 1e-3,
    overlap_eps: float = 1e-4,
) -> bool:
    poly = _placement_polygon(x, z, size["width"], size["depth"], yaw_deg)
    floor = Polygon([(float(px), float(pz)) for px, pz in floor_polygon])
    try:
        if poly.difference(floor).area > outside_tol_area:
            return False
    except Exception:
        return False

    for item in placed_so_far:
        if "position" not in item:
            continue
        ox, oz = _item_position(item)
        osize = _size_dict(item)
        oyaw = float(item.get("rotation_y", item.get("yaw", 0.0)))
        other = _placement_polygon(ox, oz, osize["width"], osize["depth"], oyaw)
        try:
            if poly.intersection(other).area > overlap_eps:
                return False
        except Exception:
            return False
    return True


class DifferentiableRefiner:
    def __init__(self, scorer_path: Optional[str] = None, device: str = "auto") -> None:
        self.device = resolve_torch_device(device)
        self.scorer = None
        self.category_mapping = dict(CATEGORY_TO_IDX)

        if scorer_path:
            self.scorer, metadata = load_scorer_checkpoint(scorer_path, device=self.device)
            if metadata.get("category_to_idx"):
                self.category_mapping = dict(metadata["category_to_idx"])

        self.lambda_physics = config.LAMBDA_PHYSICS
        self.lambda_semantic = config.LAMBDA_SEMANTIC
        self.lambda_style = config.LAMBDA_STYLE
        self.lambda_scorer = config.LAMBDA_SCORER
        self.lambda_region = config.LAMBDA_REGION

    def refine(
        self,
        scene: Dict[str, Any],
        new_furniture: Sequence[Dict[str, Any]],
        candidate_regions: Sequence[Any],
        selected_regions: Any | None = None,
        n_iters: int = config.DIFFOPT_ITERS,
        lr: float = config.DIFFOPT_LR,
        n_restarts: int = 5,
    ) -> List[Dict[str, Any]]:
        floor_polygon = _scene_floor_polygon(scene)
        walls = _scene_walls(scene)
        doors = _scene_doors(scene)
        existing_items = _existing_items(scene)
        region_by_id = {
            build_region_box(region, floor_polygon)["id"]: build_region_box(region, floor_polygon)
            for region in candidate_regions
        }
        if not region_by_id:
            full_region = build_region_box(None, floor_polygon)
            region_by_id = {full_region["id"]: full_region}
        selection_map = _selection_map(selected_regions)

        placements: List[Dict[str, Any]] = []
        placed_so_far: List[Dict[str, Any]] = list(existing_items)

        for furn in _sorted_furniture(list(new_furniture)):
            size = _size_dict(furn)
            fid = str(furn.get("id", f"{furn.get('category', 'item')}-{len(placements)}"))
            cat = _normalize_category(furn.get("category", "unknown"))
            region = region_by_id.get(selection_map.get(fid, 0), next(iter(region_by_id.values())))
            region_usage = sum(1 for p in placements if p.get("region_id") == region["id"])
            best_feasible_state: Optional[tuple[float, float, float]] = None
            best_feasible_loss = float("inf")
            best_state: Optional[tuple[float, float, float]] = None
            best_loss = float("inf")

            for restart_idx in range(max(1, int(n_restarts))):
                x, z, yaw = initialize_pose(
                    region,
                    walls=walls,
                    floor_polygon=floor_polygon,
                    seed_index=region_usage + restart_idx,
                    device=self.device,
                )
                optimizer = optim.Adam([x, z, yaw], lr=lr)

                for _step in range(int(n_iters)):
                    optimizer.zero_grad()

                    pos = torch.stack([x, z])
                    l_phys = loss_physics(
                        positions=[pos],
                        rotations=[yaw],
                        sizes=[size],
                        floor_polygon=floor_polygon,
                        existing_obstacles=placed_so_far,
                        doors=doors,
                    )
                    l_sem = loss_semantic([pos], [cat], placed_so_far)
                    l_style = loss_style([pos], [yaw], [cat], walls=walls, floor_polygon=floor_polygon)

                    cx, cz = region["centroid"]
                    region_pull = ((x - cx) ** 2 + (z - cz) ** 2)

                    total = (
                        self.lambda_physics * l_phys
                        + self.lambda_semantic * l_sem
                        + self.lambda_style * l_style
                        + self.lambda_region * region_pull
                    )

                    if self.scorer is not None:
                        context = PlacementContext(
                            floor_polygon=floor_polygon,
                            walls=walls if walls and not isinstance(walls[0], dict) else [],
                            doors=[_item_position(d) if isinstance(d, dict) else d for d in doors],
                            windows=[],
                            existing_furniture=[_to_furniture(item) for item in placed_so_far],
                        )
                        feat = extract_placement_features(
                            x=x,
                            z=z,
                            yaw=yaw,
                            width=size["width"],
                            depth=size["depth"],
                            category=cat,
                            context=context,
                        ).unsqueeze(0)
                        cat_tensor = torch.tensor(
                            [category_idx(cat, self.category_mapping)],
                            dtype=torch.long,
                            device=self.device,
                        )
                        scorer_logit = self.scorer(feat, category_idx=cat_tensor).squeeze()
                        total = total - self.lambda_scorer * torch.sigmoid(scorer_logit)

                    total.backward()
                    optimizer.step()

                    with torch.no_grad():
                        x.clamp_(region["xmin"], region["xmax"])
                        z.clamp_(region["zmin"], region["zmax"])
                        yaw.copy_(torch.atan2(torch.sin(yaw), torch.cos(yaw)))

                    total_value = float(total.detach().cpu())
                    state = (float(x.detach().cpu()), float(z.detach().cpu()), float(yaw.detach().cpu()))
                    if total_value < best_loss:
                        best_loss = total_value
                        best_state = state

                    feasible = _is_feasible(
                        x=state[0],
                        z=state[1],
                        yaw_deg=math.degrees(state[2]) % 360.0,
                        size=size,
                        floor_polygon=floor_polygon,
                        placed_so_far=placed_so_far,
                    )
                    if feasible and total_value < best_feasible_loss:
                        best_feasible_loss = total_value
                        best_feasible_state = state

            chosen_state = best_feasible_state if best_feasible_state is not None else None
            chosen_loss = best_feasible_loss if best_feasible_state is not None else None

            if chosen_state is None and best_state is not None:
                fallback_feasible = _is_feasible(
                    x=best_state[0],
                    z=best_state[1],
                    yaw_deg=math.degrees(best_state[2]) % 360.0,
                    size=size,
                    floor_polygon=floor_polygon,
                    placed_so_far=placed_so_far,
                )
                if fallback_feasible:
                    chosen_state = best_state
                    chosen_loss = best_loss

            if chosen_state is None:
                placements.append({
                    "furniture_id": fid,
                    "category": cat,
                    "status": "unplaced",
                    "reason": "no feasible placement found",
                })
                continue

            bx, bz, byaw = chosen_state
            placement = {
                "furniture_id": fid,
                "category": cat,
                "position": {"x": bx, "z": bz},
                "rotation_y": (math.degrees(byaw) % 360.0),
                "size": size,
                "region_id": region["id"],
                "score": -float(chosen_loss if chosen_loss is not None else best_loss),
                "status": "placed",
            }
            placements.append(placement)
            placed_so_far.append(placement)

        return placements
