"""Generate scorer training samples from 3D-FRONT ground truth."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from experiments.adapters.threedfront_adapter import load_3dfront_scenes, resolve_layoutgpt_dataset_dir
from spacefit_v2.data.negative_sampler import (
    sample_collision_negative,
    sample_misaligned_negative,
    sample_random_negative,
)
from spacefit_v2.model.features import Furniture, PlacementContext, category_idx, extract_placement_features


def load_category_mapping(data_dir: str | Path, room_type: str) -> Dict[str, int]:
    dataset_dir = resolve_layoutgpt_dataset_dir(data_dir, room_type)
    with open(dataset_dir / "dataset_stats.txt", "r") as f:
        stats = json.load(f)
    categories = list(stats.get("object_types", []))
    return {str(name): idx for idx, name in enumerate(categories)}


def _feature_from_item(
    item: Dict,
    scene: Dict,
    context: PlacementContext,
    category_to_idx: Dict[str, int],
) -> Dict:
    pos = item["position"]
    yaw = torch.tensor(float(item["rotation_y"]) * torch.pi / 180.0, dtype=torch.float32)
    feat = extract_placement_features(
        x=torch.tensor(float(pos["x"]), dtype=torch.float32),
        z=torch.tensor(float(pos["z"]), dtype=torch.float32),
        yaw=yaw,
        width=float(item["size"]["width"]),
        depth=float(item["size"]["depth"]),
        category=str(item["category"]),
        context=context,
    ).detach()
    cat = str(item["category"])
    return {
        "features": feat,
        "category_idx": int(category_idx(cat, category_to_idx)),
        "label": 1.0,
        "category": cat,
        "scene_id": scene["id"],
    }


def generate_training_data(
    data_dir: str | Path,
    room_type: str = "bedroom",
    split: str = "train",
    limit: int | None = None,
    seed: int = 0,
) -> List[Dict]:
    rng = random.Random(seed)
    category_to_idx = load_category_mapping(data_dir, room_type)
    scenes = load_3dfront_scenes(data_dir, room_type=room_type, split=split, limit=limit)

    samples: List[Dict] = []
    for scene in scenes:
        for furn in scene["furniture"]:
            others = [item for item in scene["furniture"] if item["id"] != furn["id"]]
            context = PlacementContext(
                floor_polygon=scene["floor_plan_vertices"],
                existing_furniture=[Furniture.from_dict(item) for item in others],
            )
            positive = _feature_from_item(furn, scene, context, category_to_idx)
            samples.append(positive)

            for sampler in (sample_random_negative, sample_collision_negative, sample_misaligned_negative):
                if sampler is sample_collision_negative:
                    neg_item = sampler(furn, others, scene["floor_plan_vertices"], rng)
                else:
                    neg_item = sampler(furn, scene["floor_plan_vertices"], rng)
                neg = _feature_from_item(neg_item, scene, context, category_to_idx)
                neg["label"] = 0.0
                samples.append(neg)
    return samples
