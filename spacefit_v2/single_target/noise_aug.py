"""Sim-to-real noise augmentation for single-target benchmark cases.

Simulates the measurement errors that arise from Apple RoomPlan scans:
  - wall_sigma: Gaussian noise on floor polygon vertices (default ±2 cm)
  - furniture_sigma: Gaussian noise on existing furniture positions (default ±5 cm)
  - size_sigma: Gaussian noise on existing furniture dimensions (default ±3 cm)
  - drop_prob: probability of dropping an existing furniture item (sensor miss)
"""
from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List, Optional


ROOMPLAN_PRESET = {
    "wall_sigma": 0.02,       # ±2 cm wall vertex jitter
    "furniture_sigma": 0.05,  # ±5 cm position noise on existing furniture
    "size_sigma": 0.03,       # ±3 cm size noise on existing furniture
    "drop_prob": 0.05,        # 5% chance a furniture item is undetected
}

NONE_PRESET: Dict[str, float] = {
    "wall_sigma": 0.0,
    "furniture_sigma": 0.0,
    "size_sigma": 0.0,
    "drop_prob": 0.0,
}


def _gauss(sigma: float, rng: random.Random) -> float:
    return rng.gauss(0.0, sigma) if sigma > 0 else 0.0


def _jitter_polygon(polygon: List[List[float]], sigma: float, rng: random.Random) -> List[List[float]]:
    return [
        [x + _gauss(sigma, rng), z + _gauss(sigma, rng)]
        for x, z in polygon
    ]


def _jitter_position(pos: List[float], sigma: float, rng: random.Random) -> List[float]:
    return [
        pos[0] + _gauss(sigma, rng),
        pos[1],
        pos[2] + _gauss(sigma, rng),
    ]


def _jitter_size(size: List[float], sigma: float, rng: random.Random) -> List[float]:
    return [max(0.1, size[0] + _gauss(sigma, rng)),
            size[1],
            max(0.1, size[2] + _gauss(sigma, rng))]


def augment_case(
    case: Dict[str, Any],
    wall_sigma: float = ROOMPLAN_PRESET["wall_sigma"],
    furniture_sigma: float = ROOMPLAN_PRESET["furniture_sigma"],
    size_sigma: float = ROOMPLAN_PRESET["size_sigma"],
    drop_prob: float = ROOMPLAN_PRESET["drop_prob"],
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a deep-copied case with RoomPlan-style noise injected.

    The target_asset and reference_pose are untouched — only the scene
    context (floor polygon, existing furniture) is perturbed.
    """
    rng = random.Random(seed)
    noisy = copy.deepcopy(case)
    scene = noisy["scene"]

    # Jitter floor polygon vertices
    if wall_sigma > 0 and scene.get("floor", {}).get("polygon"):
        scene["floor"]["polygon"] = _jitter_polygon(scene["floor"]["polygon"], wall_sigma, rng)

    # Jitter opening positions (doors/windows) with same wall sigma
    for opening_key in ("doors", "windows", "openings"):
        for item in scene.get(opening_key, []):
            if isinstance(item.get("position"), list) and wall_sigma > 0:
                item["position"] = _jitter_position(item["position"], wall_sigma, rng)

    # Jitter existing furniture
    surviving = []
    for obj in scene.get("objects", []):
        if drop_prob > 0 and rng.random() < drop_prob:
            continue  # sensor miss
        noisy_obj = dict(obj)
        if furniture_sigma > 0 and isinstance(noisy_obj.get("position"), list):
            noisy_obj["position"] = _jitter_position(noisy_obj["position"], furniture_sigma, rng)
        if size_sigma > 0 and isinstance(noisy_obj.get("size"), list):
            noisy_obj["size"] = _jitter_size(noisy_obj["size"], size_sigma, rng)
        surviving.append(noisy_obj)
    scene["objects"] = surviving

    noisy["_noise_config"] = {
        "wall_sigma": wall_sigma,
        "furniture_sigma": furniture_sigma,
        "size_sigma": size_sigma,
        "drop_prob": drop_prob,
        "seed": seed,
    }
    return noisy


def augment_cases(
    cases: List[Dict[str, Any]],
    preset: str = "roomplan",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Augment a list of cases using a named noise preset."""
    if preset == "roomplan":
        params = ROOMPLAN_PRESET
    elif preset == "none":
        params = NONE_PRESET
    else:
        raise ValueError(f"unknown noise preset '{preset}'. Use 'roomplan' or 'none'.")
    return [
        augment_case(case, seed=seed + i, **params)
        for i, case in enumerate(cases)
    ]
