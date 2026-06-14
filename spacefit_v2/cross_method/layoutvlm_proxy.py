"""Prepare adapted LayoutVLM task JSONs from the single-target benchmark.

The current benchmark uses raw 3D-FRONT furniture, while the local LayoutVLM
snapshot expects Objaverse-style asset ids. This helper therefore builds
explicit proxy-task JSONs and records the proxy mapping in the exported task.

These exports are intentionally labeled adapted/proxy rather than fair-direct.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[2]
LAYOUTVLM_ASSET_DIR = ROOT / "LayoutVLM" / "objaverse_processed"


_CATEGORY_ALIASES = {
    "armchair": ["armchair", "chair", "accent chair", "lounge chair"],
    "bed": ["bed", "double bed", "single bed"],
    "bookshelf": ["bookshelf", "bookcase", "shelf"],
    "coffee_table": ["coffee table", "table"],
    "desk": ["desk", "table"],
    "dining_chair": ["dining chair", "chair"],
    "nightstand": ["nightstand", "side table", "table"],
    "sofa": ["sofa", "couch"],
    "tv_stand": ["tv stand", "television", "media console", "cabinet"],
    "wardrobe": ["wardrobe", "cabinet", "closet"],
}


def _slug(text: str) -> str:
    return (
        str(text or "")
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("'", "_")
        .strip("_")
    )


@dataclass
class ProxyAsset:
    uid: str
    category: str
    width: float
    height: float
    depth: float

    @property
    def size(self) -> tuple[float, float, float]:
        return (self.width, self.height, self.depth)


def build_asset_catalog(asset_dir: str | Path = LAYOUTVLM_ASSET_DIR) -> List[ProxyAsset]:
    asset_dir = Path(asset_dir)
    catalog: List[ProxyAsset] = []
    for asset_path in sorted(asset_dir.iterdir()):
        data_path = asset_path / "data.json"
        if not data_path.exists():
            continue
        try:
            data = json.loads(data_path.read_text())
            bbox = data["assetMetadata"]["boundingBox"]
            catalog.append(
                ProxyAsset(
                    uid=asset_path.name,
                    category=_slug(data["annotations"]["category"]),
                    width=float(bbox["y"]),
                    height=float(bbox["z"]),
                    depth=float(bbox["x"]),
                )
            )
        except Exception:
            continue
    return catalog


def _size_distance(target_size: tuple[float, float, float], proxy_size: tuple[float, float, float]) -> float:
    # Compare in log-space to avoid large objects dominating the match score.
    score = 0.0
    for a, b in zip(target_size, proxy_size):
        aa = max(float(a), 1e-4)
        bb = max(float(b), 1e-4)
        score += abs(math.log(aa) - math.log(bb))
    return score


def choose_proxy_asset(
    category: str,
    size: tuple[float, float, float],
    catalog: Iterable[ProxyAsset],
) -> Optional[ProxyAsset]:
    category_slug = _slug(category)
    aliases = [_slug(name) for name in _CATEGORY_ALIASES.get(category_slug, [category_slug])]

    exact = [asset for asset in catalog if asset.category in aliases]
    if exact:
        exact.sort(key=lambda asset: (_size_distance(size, asset.size), asset.uid))
        return exact[0]

    fuzzy = []
    compact_aliases = [alias.replace("_", "") for alias in aliases]
    for asset in catalog:
        compact = asset.category.replace("_", "")
        if any(alias in compact or compact in alias for alias in compact_aliases):
            fuzzy.append(asset)
    if fuzzy:
        fuzzy.sort(key=lambda asset: (_size_distance(size, asset.size), asset.uid))
        return fuzzy[0]

    all_assets = list(catalog)
    if not all_assets:
        return None
    all_assets.sort(key=lambda asset: (_size_distance(size, asset.size), asset.uid))
    return all_assets[0]


def benchmark_case_to_layoutvlm_task(
    case: Dict[str, Any],
    catalog: Optional[Iterable[ProxyAsset]] = None,
) -> Dict[str, Any]:
    """Export one adapted LayoutVLM task from one single-target benchmark case.

    The exported JSON keeps LayoutVLM's expected top-level keys and adds a
    `spacefit_context` payload that records fixed existing poses plus the proxy
    mapping used to approximate the 3D-FRONT furniture with Objaverse assets.
    """
    if catalog is None:
        catalog = build_asset_catalog()
    catalog = list(catalog)

    floor_vertices = [[float(x), float(z), 0.0] for x, z in case["scene"]["floor"]["polygon"]]
    task_assets: Dict[str, Dict[str, Any]] = {}
    existing_assets: List[Dict[str, Any]] = []

    for idx, obj in enumerate(case["scene"].get("objects", [])):
        size = (float(obj["size"][0]), float(obj["size"][1]), float(obj["size"][2]))
        proxy = choose_proxy_asset(str(obj["category"]), size, catalog)
        proxy_uid = proxy.uid if proxy is not None else f"missing_proxy_existing_{idx}"
        asset_key = f"{proxy_uid}-{idx}"
        task_assets[asset_key] = {}
        existing_assets.append(
            {
                "asset_key": asset_key,
                "source_id": str(obj["id"]),
                "source_category": str(obj["category"]),
                "proxy_uid": proxy_uid,
                "proxy_category": proxy.category if proxy is not None else None,
                "position": [float(obj["position"][0]), float(obj["position"][2]), float(obj["position"][1])],
                "rotation": [0.0, 0.0, float(obj.get("yaw", 0.0))],
                "source_size": {
                    "width": float(obj["size"][0]),
                    "height": float(obj["size"][1]),
                    "depth": float(obj["size"][2]),
                },
            }
        )

    target_size = (
        float(case["target_asset"]["size"]["width"]),
        float(case["target_asset"]["size"]["height"]),
        float(case["target_asset"]["size"]["depth"]),
    )
    target_proxy = choose_proxy_asset(str(case["target_asset"]["category"]), target_size, catalog)
    target_proxy_uid = target_proxy.uid if target_proxy is not None else "missing_proxy_target"
    target_key = f"{target_proxy_uid}-target"
    task_assets[target_key] = {}

    constraints = []
    for item in case["intent"].get("constraints", []):
        record = {
            "constraint_type": str(item.get("constraint_type")),
            "subject_id": str(item.get("subject_id", case["target_asset"]["id"])),
        }
        if item.get("target_id") is not None:
            record["target_id"] = str(item["target_id"])
        if item.get("target_category") is not None:
            record["target_category"] = str(item["target_category"])
        if item.get("target_kind") is not None:
            record["target_kind"] = str(item["target_kind"])
        if item.get("min_distance") is not None:
            record["min_distance"] = float(item["min_distance"])
        if item.get("max_distance") is not None:
            record["max_distance"] = float(item["max_distance"])
        constraints.append(record)

    return {
        "task_description": str(case["intent"]["text"]),
        "layout_criteria": (
            "Place only the target asset while keeping all existing furniture fixed. "
            "Use the provided fixed scene context as immutable support geometry."
        ),
        "boundary": {
            "floor_vertices": floor_vertices,
            "wall_height": float(case["scene"].get("floor", {}).get("height", 2.7) or 2.7),
        },
        "assets": task_assets,
        "spacefit_context": {
            "case_id": str(case["id"]),
            "comparison_status": "adapted_proxy_only",
            "source_room_type": str(case["scene"]["room_type"]),
            "existing_assets": existing_assets,
            "target_asset": {
                "asset_key": target_key,
                "source_id": str(case["target_asset"]["id"]),
                "source_category": str(case["target_asset"]["category"]),
                "proxy_uid": target_proxy_uid,
                "proxy_category": target_proxy.category if target_proxy is not None else None,
                "source_size": {
                    "width": float(case["target_asset"]["size"]["width"]),
                    "height": float(case["target_asset"]["size"]["height"]),
                    "depth": float(case["target_asset"]["size"]["depth"]),
                },
                "reference_pose": {
                    "position": [
                        float(case["reference_pose"]["position"]["x"]),
                        float(case["reference_pose"]["position"]["z"]),
                        float(case["reference_pose"]["position"]["y"]),
                    ],
                    "rotation": [0.0, 0.0, float(case["reference_pose"]["rotation_y"])],
                },
            },
            "benchmark_constraints": constraints,
        },
    }


def export_layoutvlm_proxy_tasks(
    cases: Iterable[Dict[str, Any]],
    out_dir: str | Path,
    asset_dir: str | Path = LAYOUTVLM_ASSET_DIR,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = build_asset_catalog(asset_dir)
    exported = []
    for case in cases:
        payload = benchmark_case_to_layoutvlm_task(case, catalog=catalog)
        out_path = out_dir / f"{case['id']}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        exported.append(
            {
                "case_id": str(case["id"]),
                "task_json": str(out_path),
                "target_proxy_uid": payload["spacefit_context"]["target_asset"]["proxy_uid"],
                "num_existing_assets": len(payload["spacefit_context"]["existing_assets"]),
            }
        )

    manifest = {
        "exported_tasks": exported,
        "asset_dir": str(asset_dir),
        "num_tasks": len(exported),
        "note": (
            "These tasks are adapted proxy exports for LayoutVLM. They preserve the "
            "single-target benchmark geometry and intent structure, but 3D-FRONT "
            "furniture is approximated with Objaverse proxy assets from the local "
            "LayoutVLM asset directory."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
