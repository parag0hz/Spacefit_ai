"""Run the actual LayoutVLM LLM/VLM pipeline on the single-target benchmark.

This adapter keeps the original LayoutVLM prompt-generation path and OpenAI API
usage, but constrains the task to our benchmark by:
1. mapping 3D-FRONT assets to local Objaverse proxies,
2. fixing existing furniture poses in the sandbox, and
3. asking LayoutVLM to place only the removed target asset.

Because the asset domain is still proxied, the result should be reported as an
adapted comparison rather than a fair direct rerun.
"""
from __future__ import annotations

import contextlib
import collections
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from dd0nw.LayoutVLM.prompts.layoutvlm import base_prompt
from dd0nw.LayoutVLM.src.layoutvlm.layoutvlm import LayoutVLM
from dd0nw.LayoutVLM.src.layoutvlm.sandbox import SandBoxEnv
from dd0nw.LayoutVLM.src.layoutvlm.scene import AssetInstance

from spacefit_v2.cross_method.layoutvlm_proxy import LAYOUTVLM_ASSET_DIR, ProxyAsset, build_asset_catalog, choose_proxy_asset


load_dotenv(ROOT / "LayoutVLM" / ".env")


def _asset_path(uid: str, asset_dir: str | Path = LAYOUTVLM_ASSET_DIR) -> Path:
    return Path(asset_dir) / uid / "data.json"


def _normalize_category(name: str) -> str:
    return (
        str(name or "")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("'", "_")
        .replace("/", "_")
        .replace(",", "_")
        .lower()
    )


def _proxy_selection(
    category: str,
    size: Tuple[float, float, float],
    catalog: Iterable[ProxyAsset],
) -> ProxyAsset:
    proxy = choose_proxy_asset(category, size, catalog)
    if proxy is None:
        raise ValueError(f"Could not find proxy asset for {category}")
    return proxy


def _solver_position_from_scene(position_xyz: Sequence[float], height: float) -> List[float]:
    return [float(position_xyz[0]), float(position_xyz[2]), float(height) * 0.5]


def _deg_rotation(yaw_deg: float) -> List[float]:
    return [0.0, 0.0, float(yaw_deg)]


def _load_proxy_data(uid: str, asset_dir: str | Path = LAYOUTVLM_ASSET_DIR) -> Dict[str, Any]:
    with open(_asset_path(uid, asset_dir), "r") as f:
        return json.load(f)


def _build_prepared_assets(
    selections: Sequence[Dict[str, Any]],
    asset_dir: str | Path = LAYOUTVLM_ASSET_DIR,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    all_data: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for item in selections:
        data = _load_proxy_data(item["proxy_uid"], asset_dir=asset_dir)
        data["path"] = str(Path(asset_dir) / item["proxy_uid"] / f"{item['proxy_uid']}.glb")
        data["_source_meta"] = item
        all_data[item["proxy_uid"]].append(data)

    category_count: Dict[str, int] = collections.defaultdict(int)
    for uid, duplicated_assets in all_data.items():
        base_category = _normalize_category(duplicated_assets[0]["annotations"]["category"])
        category_count[base_category] += 1

    assets: Dict[str, Dict[str, Any]] = {}
    asset_key_by_source_id: Dict[str, str] = {}
    category_idx: Dict[str, int] = collections.defaultdict(int)
    for uid, duplicated_assets in all_data.items():
        base_category = _normalize_category(duplicated_assets[0]["annotations"]["category"])
        category_idx[base_category] += 1
        asset_var_name = base_category
        if category_count[base_category] > 1:
            asset_var_name = f"{base_category}_{chr(ord('A') + category_idx[base_category] - 1)}"
        for instance_idx, data in enumerate(duplicated_assets):
            entry = {
                "uid": uid,
                "count": len(duplicated_assets),
                "instance_var_name": f"{asset_var_name}_{instance_idx}" if len(duplicated_assets) > 1 else asset_var_name,
                "asset_var_name": asset_var_name,
                "instance_idx": instance_idx,
                "annotations": data["annotations"],
                "category": data["annotations"]["category"],
                "description": data["annotations"]["description"],
                "path": data["path"],
                "onCeiling": data["annotations"]["onCeiling"],
                "onFloor": data["annotations"]["onFloor"],
                "onWall": data["annotations"]["onWall"],
                "onObject": data["annotations"]["onObject"],
                "frontView": data["annotations"]["frontView"],
                "assetMetadata": {
                    "boundingBox": {
                        "x": float(data["assetMetadata"]["boundingBox"]["y"]),
                        "y": float(data["assetMetadata"]["boundingBox"]["x"]),
                        "z": float(data["assetMetadata"]["boundingBox"]["z"]),
                    }
                },
                "_source_meta": data["_source_meta"],
            }
            asset_key = f"{asset_var_name}-{instance_idx}"
            assets[asset_key] = entry
            asset_key_by_source_id[str(data["_source_meta"]["source_id"])] = asset_key
    return assets, asset_key_by_source_id


def build_single_target_task(
    case: Mapping[str, Any],
    catalog: Optional[Iterable[ProxyAsset]] = None,
    asset_dir: str | Path = LAYOUTVLM_ASSET_DIR,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], str]:
    if catalog is None:
        catalog = build_asset_catalog(asset_dir)
    catalog = list(catalog)

    selections: List[Dict[str, Any]] = []
    existing_fixed: Dict[str, Dict[str, Any]] = {}

    for obj in case["scene"].get("objects", []):
        size = (float(obj["size"][0]), float(obj["size"][1]), float(obj["size"][2]))
        proxy = _proxy_selection(str(obj["category"]), size, catalog)
        selections.append(
            {
                "proxy_uid": proxy.uid,
                "source_id": str(obj["id"]),
                "source_category": str(obj["category"]),
                "size": size,
                "is_target": False,
                "position_solver": _solver_position_from_scene(obj["position"], height=float(obj["size"][1])),
                "rotation_deg": _deg_rotation(float(obj.get("yaw", 0.0))),
            }
        )

    target_size = (
        float(case["target_asset"]["size"]["width"]),
        float(case["target_asset"]["size"]["height"]),
        float(case["target_asset"]["size"]["depth"]),
    )
    target_proxy = _proxy_selection(str(case["target_asset"]["category"]), target_size, catalog)
    selections.append(
        {
            "proxy_uid": target_proxy.uid,
            "source_id": str(case["target_asset"]["id"]),
            "source_category": str(case["target_asset"]["category"]),
            "size": target_size,
            "is_target": True,
        }
    )

    assets, asset_key_by_source_id = _build_prepared_assets(selections, asset_dir=asset_dir)
    target_key = asset_key_by_source_id[str(case["target_asset"]["id"])]

    for obj in case["scene"].get("objects", []):
        asset_key = asset_key_by_source_id[str(obj["id"])]
        existing_fixed[asset_key] = {
            **assets[asset_key],
            "position": _solver_position_from_scene(obj["position"], height=float(obj["size"][1])),
            "rotation": _deg_rotation(float(obj.get("yaw", 0.0))),
        }

    floor_vertices = [[float(x), float(z), 0.0] for x, z in case["scene"]["floor"]["polygon"]]
    task = {
        "task_description": str(case["intent"]["text"]),
        "layout_criteria": (
            "Place only the target asset while keeping all existing furniture fixed in their provided poses. "
            f"Instruction: {case['intent']['text']}"
        ),
        "boundary": {
            "floor_vertices": floor_vertices,
            "wall_height": float(case["scene"].get("floor", {}).get("height", 2.7) or 2.7),
        },
        "assets": assets,
        "spacefit_context": {
            "case_id": str(case["id"]),
            "comparison_type": "adapted_proxy_llm",
            "target_asset_key": target_key,
        },
    }
    return task, existing_fixed, target_key


class SingleTargetSandBoxEnv(SandBoxEnv):
    def setup_optimization_param(self, placed_instance_ids, new_instance_ids, new_constraints):
        all_instance_ids = placed_instance_ids + new_instance_ids
        for constraint, instance_ids in self.all_constraints + new_constraints:
            all_instance_ids.extend(instance_ids)
        solver_assets = self.setup_initial_assets()
        for instance_id in all_instance_ids:
            if instance_id.startswith("walls_") or instance_id == "room_0":
                continue
            asset_var_name = "_".join(instance_id.split("_")[:-1])
            instance_idx = int(instance_id.split("_")[-1])
            if len(self.local_vars[asset_var_name].placements[instance_idx].rotation) == 1:
                self.local_vars[asset_var_name].placements[instance_idx].rotation = [0, 0, self.local_vars[asset_var_name].placements[instance_idx].rotation[0]]
            is_new = instance_id in new_instance_ids
            optimize = 1 if (is_new and not instance_id.startswith("fixed_point")) else 0
            solver_assets[instance_id] = AssetInstance(
                id=instance_id,
                position=self.local_vars[asset_var_name].placements[instance_idx].position,
                rotation=self.local_vars[asset_var_name].placements[instance_idx].rotation,
                size=self.local_vars[asset_var_name].size,
                onCeiling=self.local_vars[asset_var_name].onCeiling,
                optimize=optimize,
            )
        return solver_assets


def _sync_fixed_assets_in_sandbox(layout_solver: LayoutVLM, fixed_assets: Mapping[str, Mapping[str, Any]]) -> None:
    lines: List[str] = []
    for asset_key, asset in fixed_assets.items():
        var_name = asset["asset_var_name"]
        instance_idx = int(asset_key.split("-")[-1])
        lines.append(f"{var_name}[{instance_idx}].position = {list(asset['position'])}")
        lines.append(f"{var_name}[{instance_idx}].rotation = {list(asset['rotation'])}")
        lines.append(f"{var_name}[{instance_idx}].optimize = 0")
    if lines:
        layout_solver.sandbox.execute_code("\n".join(lines) + "\n")


def _prediction_from_layout(case: Mapping[str, Any], layout: Mapping[str, Any], target_key: str) -> List[Dict[str, Any]]:
    if target_key not in layout:
        return [
            {
                "furniture_id": str(case["target_asset"]["id"]),
                "category": str(case["target_asset"]["category"]),
                "status": "unplaced",
                "reason": "layoutvlm_llm_no_target_output",
                "size": dict(case["target_asset"]["size"]),
            }
        ]
    result = layout[target_key]
    return [
        {
            "furniture_id": str(case["target_asset"]["id"]),
            "category": str(case["target_asset"]["category"]),
            "position": {
                "x": float(result["position"][0]),
                "y": 0.0,
                "z": float(result["position"][1]),
            },
            "rotation_y": float(result["rotation"][-1]),
            "size": dict(case["target_asset"]["size"]),
            "region_id": 0,
            "score": 0.0,
            "status": "placed",
            "metadata": {
                "runner": "layoutvlm_llm_single_target",
                "asset_key": target_key,
            },
        }
    ]


def get_openai_api_key(explicit_key: str | None = None) -> str:
    key = explicit_key or os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key
    raise ValueError("OPENAI_API_KEY is not available in the current session.")


def run_layoutvlm_llm_case(
    case: Mapping[str, Any],
    save_dir: str | Path,
    api_key: str | None = None,
    model_name: str = "gpt-4o",
    max_attempts: int = 1,
    asset_dir: str | Path = LAYOUTVLM_ASSET_DIR,
    catalog: Optional[Iterable[ProxyAsset]] = None,
) -> List[Dict[str, Any]]:
    key = get_openai_api_key(api_key)
    os.environ["OPENAI_API_KEY"] = key
    task, fixed_assets, target_key = build_single_target_task(case, catalog=catalog, asset_dir=asset_dir)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "task.json", "w") as f:
        json.dump(task, f, indent=2)

    layout_solver = LayoutVLM(
        mode="one_shot",
        save_dir=str(save_dir),
        asset_source="objaverse",
        gpt_4o_model_name=model_name,
    )
    layout_solver.sandbox = SingleTargetSandBoxEnv(task, mode=layout_solver.mode, save_dir=str(save_dir))
    task_program = layout_solver.get_task_program(list(task["assets"].keys()), task)
    layout_solver.sandbox.execute_code(base_prompt.CODE_FOR_SANDBOX + "\n" + task_program)
    layout_solver.sandbox.assign_instance_ids()
    layout_solver.sandbox.initialize_variables()
    _sync_fixed_assets_in_sandbox(layout_solver, fixed_assets)

    placed_assets = dict(fixed_assets)
    group_assets = {target_key}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result_assets = layout_solver._solve_single_group(
            task,
            task["layout_criteria"],
            placed_assets,
            group_assets,
            str(save_dir / "group_0"),
            include_image=True,
            MAX_ATTEMPTS=max_attempts,
        )
    with open(save_dir / "layout.json", "w") as f:
        json.dump(result_assets, f, indent=2)
    return _prediction_from_layout(case, result_assets, target_key)


def run_layoutvlm_llm_benchmark(
    cases: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    api_key: str | None = None,
    model_name: str = "gpt-4o",
    max_attempts: int = 1,
    asset_dir: str | Path = LAYOUTVLM_ASSET_DIR,
) -> Dict[str, List[Dict[str, Any]]]:
    catalog = build_asset_catalog(asset_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions: Dict[str, List[Dict[str, Any]]] = {}
    for idx, case in enumerate(cases):
        case_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(case["id"]))
        case_save_dir = out_dir / f"{idx:02d}_{case_slug}"
        try:
            predictions[str(case["id"])] = run_layoutvlm_llm_case(
                case,
                save_dir=case_save_dir,
                api_key=api_key,
                model_name=model_name,
                max_attempts=max_attempts,
                asset_dir=asset_dir,
                catalog=catalog,
            )
        except Exception as exc:
            predictions[str(case["id"])] = [
                {
                    "furniture_id": str(case["target_asset"]["id"]),
                    "category": str(case["target_asset"]["category"]),
                    "status": "unplaced",
                    "reason": f"layoutvlm_llm_failed: {type(exc).__name__}",
                    "error": str(exc),
                    "size": dict(case["target_asset"]["size"]),
                }
            ]
    return predictions
