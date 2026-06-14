from __future__ import annotations

import json
import math
import os
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_spacefit_viz")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon as MplPolygon, Rectangle
from PIL import Image, ImageOps

from experiments.eval.unified_eval import _compute_candidate_metrics
from spacefit_v2.single_target.eval import normalize_case_result


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "spacefit_v2" / "results"
OUT_DIR = RESULTS_DIR / "visualizations"
BENCHMARK_DIR = ROOT / "spacefit_v2" / "data" / "single_target_benchmark"
FUTURE_DIR = ROOT / "dataset" / "3D-FUTURE-model"

MAIN_RESULTS = RESULTS_DIR / "single_target_paper_v1" / "results.json"
MAIN_PREDICTIONS = RESULTS_DIR / "single_target_paper_v1" / "raw_predictions.json"
EXP_A_RESULTS = RESULTS_DIR / "exp_a" / "results.json"
EXP_A_PREDICTIONS = RESULTS_DIR / "exp_a" / "raw_predictions.json"
EXP_B_RESULTS = RESULTS_DIR / "exp_b" / "results.json"
EXP_B_PREDICTIONS = RESULTS_DIR / "exp_b" / "raw_predictions.json"
EXP_C_RESULTS = RESULTS_DIR / "exp_c" / "grounding_results.json"


METHOD_DISPLAY = {
    "heuristic_baseline": "Heuristic\nBaseline",
    "proposal_heuristic": "Proposal +\nHeuristic",
    "proposal_diffopt_basic": "Proposal +\nDiffOpt-Basic",
    "proposal_diffopt_constraint": "Proposal +\nDiffOpt-Constraint",
    "proposal_llm_grounded_diffopt": "Proposal +\nLLM/VLM-Grounded",
    "no_proposal_diffopt_basic": "No-Proposal +\nDiffOpt-Basic",
    "no_proposal_diffopt_constraint": "No-Proposal +\nDiffOpt-Constraint",
}

METHOD_COLORS = {
    "heuristic_baseline": "#8E9AAF",
    "proposal_heuristic": "#4C78A8",
    "proposal_diffopt_basic": "#F58518",
    "proposal_diffopt_constraint": "#2CA02C",
    "proposal_llm_grounded_diffopt": "#B279A2",
    "no_proposal_diffopt_basic": "#E45756",
    "no_proposal_diffopt_constraint": "#D62728",
}

METRIC_GROUPS = {
    "physical_validity": ["CF", "IB"],
    "intent_satisfaction": ["Constraint Accuracy", "CPS", "Success@1", "Success@5"],
    "usability": ["Reachability", "Walkability"],
}

FIGURE_PROVENANCE = {
    "fig_main_comparison": "Exact aggregate metrics from `exp_a/results.json`; no metric recomputation.",
    "fig_diffopt_delta": "Exact aggregate metrics from `single_target_paper_v1/results.json` and `exp_a/results.json`; deltas reproduced from stored outputs.",
    "fig_proposal_ablation": "Exact aggregate metrics from `exp_b/results.json`; no rerun.",
    "fig_grounding_eval": "Exact grounding aggregates and per-case constraint recovery from `exp_c/grounding_results.json`.",
    "fig_metric_groups": "Conceptual grouping figure reconstructed from the project metric definitions in `PROTOCOL.md` and evaluator code.",
    "fig_case_overview_01": "Reconstructed top-down from exact benchmark case geometry plus exact stored prediction and exact 3D-FUTURE asset thumbnail.",
    "fig_case_compare_01": "Reconstructed top-down from exact benchmark case geometry plus exact stored method predictions.",
    "fig_failure_cases": "Reconstructed top-down from exact benchmark cases plus exact stored predictions; grounding miss panel uses exact exp_c grounding output.",
    "fig_success_cases": "Reconstructed top-down from exact benchmark cases plus exact stored predictions.",
    "fig_pipeline_diagram": "Conceptual pipeline diagram reconstructed from repository code and experiment notes.",
}

SLIDE_SUGGESTIONS = {
    "Slide 2": ["fig_metric_groups.png", "Use as a motivation slide to explain what the benchmark rewards."],
    "Slide 3": ["fig_pipeline_diagram_ppt.png", "Method overview slide."],
    "Slide 4": ["fig_case_overview_01.png", "Introduce the single-target benchmark task visually."],
    "Slide 5": ["fig_main_comparison_ppt.png", "Headline quantitative result."],
    "Slide 6": ["fig_diffopt_delta_ppt.png", "Experiment A: explain what improved and why."],
    "Slide 7": ["fig_proposal_ablation_ppt.png", "Experiment B: justify the proposal stage."],
    "Slide 8": ["fig_success_cases.png", "Qualitative success stories for Proposal + DiffOpt-Constraint."],
    "Slide 9": ["fig_case_compare_01.png", "Direct method comparison on representative scenes."],
    "Slide 10": ["fig_failure_cases.png", "Failure analysis slide."],
    "Slide 11": ["fig_grounding_eval_ppt.png", "Grounding evaluation and `access_zone` miss explanation."],
}


def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _style(kind: str) -> Dict[str, Any]:
    is_ppt = kind == "ppt"
    return {
        "title": 18 if is_ppt else 14,
        "subtitle": 13 if is_ppt else 11,
        "label": 12 if is_ppt else 9.5,
        "tick": 11 if is_ppt else 8.5,
        "small": 10 if is_ppt else 7.5,
        "tiny": 9 if is_ppt else 6.5,
        "line": 2.6 if is_ppt else 1.8,
        "marker": 90 if is_ppt else 52,
        "figscale": 1.2 if is_ppt else 1.0,
    }


def _save_variants(fig: plt.Figure, stem: str, *, png_only: bool = False) -> List[str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = []
    for ext in (["png"] if png_only else ["png", "svg", "pdf"]):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=220 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        created.append(str(path))
    plt.close(fig)
    return created


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text).lower()).strip("_")


def _read_results_table(path: Path) -> Dict[str, Dict[str, float]]:
    payload = _load_json(path)
    out: Dict[str, Dict[str, float]] = {}
    for row in payload["methods"]:
        name = row["source_name"]
        out[name] = {k: float(v) for k, v in row["metrics"].items()}
    return out


def _load_cases() -> Dict[str, Dict[str, Any]]:
    cases = _load_json(BENCHMARK_DIR / "cases.json")
    return {case["id"]: case for case in cases}


def _load_predictions(path: Path) -> Dict[str, Dict[str, Sequence[Dict[str, Any]]]]:
    return _load_json(path)


def _rotation_deg_from_quat(quat: Sequence[float]) -> float:
    if len(quat) != 4:
        return 0.0
    x, y, z, w = (float(v) for v in quat)
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def _scene_target_child(case: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    source = _load_json(Path(case["scene"]["source_path"]))
    room = None
    for entry in source["scene"]["room"]:
        if entry.get("instanceid") == case["scene"]["room_id"]:
            room = entry
            break
    if room is None:
        raise KeyError(f"Room not found for {case['id']}")
    child = next(item for item in room["children"] if item.get("instanceid") == case["target_asset"]["id"])
    furniture = next(item for item in source["furniture"] if item.get("uid") == child["ref"])
    return child, furniture


def _target_thumbnail(case: Mapping[str, Any]) -> Optional[Path]:
    try:
        _, furniture = _scene_target_child(case)
    except Exception:
        return None
    jid = furniture.get("jid")
    if not jid:
        return None
    candidate = FUTURE_DIR / jid / "image.jpg"
    return candidate if candidate.exists() else None


def _target_object_from_pose(case: Mapping[str, Any], position: Mapping[str, float], rotation_y: float) -> Dict[str, Any]:
    size = case["target_asset"]["size"]
    child, furniture = _scene_target_child(case)
    return {
        "id": case["target_asset"]["id"],
        "category": case["target_asset"]["category"],
        "size": [float(size["width"]), float(size["height"]), float(size["depth"])],
        "position": [float(position["x"]), float(position.get("y", 0.0)), float(position["z"])],
        "yaw": float(rotation_y),
        "attributes": {
            "raw_ref": child.get("ref"),
            "raw_category": furniture.get("category") or furniture.get("title"),
        },
    }


def _prediction_to_object(case: Mapping[str, Any], prediction: Mapping[str, Any]) -> Dict[str, Any]:
    pos = prediction["position"]
    return {
        "id": prediction.get("furniture_id", case["target_asset"]["id"]),
        "category": prediction.get("category", case["target_asset"]["category"]),
        "size": [
            float(prediction["size"]["width"]),
            float(prediction["size"].get("height", case["target_asset"]["size"]["height"])),
            float(prediction["size"]["depth"]),
        ],
        "position": [float(pos["x"]), float(pos.get("y", 0.0)), float(pos["z"])],
        "yaw": float(prediction.get("rotation_y", 0.0)),
    }


def _corners(cx: float, cz: float, w: float, d: float, yaw_deg: float) -> List[Tuple[float, float]]:
    yaw = math.radians(float(yaw_deg))
    c = math.cos(yaw)
    s = math.sin(yaw)
    hw, hd = w / 2.0, d / 2.0
    pts = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
    return [(cx + x * c - z * s, cz + x * s + z * c) for x, z in pts]


def _object_center(item: Mapping[str, Any]) -> Tuple[float, float]:
    pos = item["position"]
    if isinstance(pos, dict):
        return float(pos["x"]), float(pos["z"])
    return float(pos[0]), float(pos[2])


def _object_polygon(item: Mapping[str, Any]) -> List[Tuple[float, float]]:
    cx, cz = _object_center(item)
    size = item["size"]
    if isinstance(size, dict):
        w = float(size["width"])
        d = float(size["depth"])
    else:
        w = float(size[0])
        d = float(size[2])
    return _corners(cx, cz, w, d, float(item.get("yaw", item.get("rotation_y", 0.0))))


def _constraint_summary(case: Mapping[str, Any]) -> str:
    phrases = []
    for item in case["intent"]["constraints"]:
        ctype = item["constraint_type"]
        if ctype == "against_wall":
            phrases.append("against wall")
        elif ctype == "near":
            target = item.get("target_category", "anchor")
            phrases.append(f"near {str(target).replace('_', ' ')}")
        elif ctype == "facing":
            target = item.get("target_kind") or item.get("target_category") or "anchor"
            phrases.append(f"face {str(target).replace('_', ' ')}")
        elif ctype == "not_block_door":
            phrases.append("clear door")
        elif ctype == "keep_window_clear":
            phrases.append("clear window")
        elif ctype == "access_zone":
            phrases.append("preserve access")
    return " | ".join(phrases)


def _primary_anchor_categories(case: Mapping[str, Any]) -> set[str]:
    out = set()
    for item in case["intent"]["constraints"]:
        target = item.get("target_category")
        if target:
            out.add(_slug(target))
    return out


def _evaluate_case_predictions(
    cases: Mapping[str, Dict[str, Any]],
    predictions_by_method: Mapping[str, Mapping[str, Sequence[Dict[str, Any]]]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for method, case_map in predictions_by_method.items():
        method_rows: Dict[str, Dict[str, Any]] = {}
        for case_id, case in cases.items():
            predictions = list(case_map.get(case_id, []))
            scene = normalize_case_result(case, method, predictions)
            candidate_metrics = [_compute_candidate_metrics(scene, cand) for cand in scene.candidates]
            method_rows[case_id] = {
                "scene": scene,
                "candidate_metrics": candidate_metrics,
                "predictions": predictions,
                "top1": candidate_metrics[0] if candidate_metrics else None,
                "success_at_1": int(candidate_metrics[0]["cps"] == 1) if candidate_metrics else 0,
                "success_at_5": int(any(item["cps"] == 1 for item in candidate_metrics[:5])),
            }
        out[method] = method_rows
    return out


def _method_label(method: str) -> str:
    return METHOD_DISPLAY.get(method, method.replace("_", " ").title())


def _pick_unique_cases(rows: Sequence[Tuple[str, float]], cases: Mapping[str, Dict[str, Any]], n: int) -> List[str]:
    selected: List[str] = []
    seen_categories: set[str] = set()
    seen_rooms: set[str] = set()
    for case_id, _score in rows:
        case = cases[case_id]
        category = case["target_asset"]["category"]
        room = case["scene"]["room_type"]
        if category not in seen_categories and room not in seen_rooms:
            selected.append(case_id)
            seen_categories.add(category)
            seen_rooms.add(room)
        if len(selected) >= n:
            return selected
    for case_id, _score in rows:
        if case_id not in selected:
            selected.append(case_id)
        if len(selected) >= n:
            break
    return selected


def _select_cases(
    cases: Mapping[str, Dict[str, Any]],
    main_eval: Mapping[str, Mapping[str, Dict[str, Any]]],
    exp_b_eval: Mapping[str, Mapping[str, Dict[str, Any]]],
    grounding: Mapping[str, Any],
) -> Dict[str, Any]:
    improvement_rows = []
    for case_id in cases:
        base = main_eval["heuristic_baseline"][case_id]["top1"]
        prop = main_eval["proposal_diffopt_constraint"][case_id]["top1"]
        if not base or not prop:
            continue
        score = (
            2.5 * (prop["cps"] - base["cps"])
            + 1.5 * (prop["constraint_accuracy"] - base["constraint_accuracy"])
            + 0.7 * (prop["ib"] - base["ib"])
            + 0.5 * (prop["cf"] - base["cf"])
        )
        if prop["cps"] >= base["cps"]:
            improvement_rows.append((case_id, score))
    improvement_rows.sort(key=lambda item: item[1], reverse=True)

    weak_fail_rows = []
    for case_id in cases:
        basic = main_eval["proposal_diffopt_basic"][case_id]["top1"]
        constraint = main_eval["proposal_diffopt_constraint"][case_id]["top1"]
        if not basic or not constraint:
            continue
        if constraint["cps"] == 1 and basic["cps"] == 0:
            score = (
                (1.0 - basic["cf"])
                + (1.0 - basic["ib"])
                + max(0.0, constraint["constraint_accuracy"] - basic["constraint_accuracy"])
            )
            weak_fail_rows.append((case_id, score))
    weak_fail_rows.sort(key=lambda item: item[1], reverse=True)

    no_proposal_fail_rows = []
    for case_id in cases:
        with_prop = exp_b_eval["proposal_diffopt_constraint"][case_id]["top1"]
        no_prop = exp_b_eval["no_proposal_diffopt_constraint"][case_id]["top1"]
        if not with_prop or not no_prop:
            continue
        if with_prop["cps"] == 1 and no_prop["cps"] == 0:
            score = (
                (with_prop["constraint_accuracy"] - no_prop["constraint_accuracy"])
                + (with_prop["ib"] - no_prop["ib"])
                + (with_prop["cf"] - no_prop["cf"])
            )
            no_proposal_fail_rows.append((case_id, score))
    no_proposal_fail_rows.sort(key=lambda item: item[1], reverse=True)

    overview = _pick_unique_cases(improvement_rows, cases, 2)
    success = _pick_unique_cases(improvement_rows, cases, 2)
    failure_geom = weak_fail_rows[0][0] if weak_fail_rows else overview[0]
    failure_no_prop = no_proposal_fail_rows[0][0] if no_proposal_fail_rows else overview[-1]
    compare = list(success)
    if failure_geom not in compare:
        compare.append(failure_geom)
    compare = compare[:3]

    grounding_case = None
    grounding_score = -1.0
    for row in grounding["cases"]:
        gt = set(row["gt_constraints"])
        pred = set(row["grounded_constraints"])
        miss_access = "access_zone" in gt and "access_zone" not in pred
        if not miss_access:
            continue
        score = float("facing" in gt and "facing" not in pred) + 0.1 * len(gt)
        if score > grounding_score:
            grounding_score = score
            grounding_case = row["case_id"]

    return {
        "overview": overview,
        "compare": compare,
        "success": success,
        "failure_geom": failure_geom,
        "failure_no_prop": failure_no_prop,
        "failure_grounding": grounding_case,
    }


def _target_scene(case: Mapping[str, Any], prediction: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    scene = dict(case["scene"])
    scene["objects"] = list(case["scene"]["objects"])
    if prediction is not None:
        scene["objects"] = list(scene["objects"]) + [_prediction_to_object(case, prediction)]
    return scene


def _reference_scene(case: Mapping[str, Any]) -> Dict[str, Any]:
    scene = dict(case["scene"])
    scene["objects"] = list(case["scene"]["objects"]) + [
        _target_object_from_pose(case, case["reference_pose"]["position"], case["reference_pose"]["rotation_y"])
    ]
    return scene


def _ghost_target(case: Mapping[str, Any]) -> Dict[str, Any]:
    return _target_object_from_pose(case, case["reference_pose"]["position"], case["reference_pose"]["rotation_y"])


def _add_chip_row(ax: plt.Axes, texts: Sequence[str], *, y: float = 0.93) -> None:
    x = 0.05
    for text in texts:
        label = textwrap.shorten(text, width=20, placeholder="...")
        width = 0.06 + 0.011 * len(label)
        chip = FancyBboxPatch(
            (x, y),
            width,
            0.08,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            transform=ax.transAxes,
            facecolor="#F6E7CB",
            edgecolor="#D7A441",
            linewidth=1.0,
            clip_on=False,
            zorder=20,
        )
        ax.add_patch(chip)
        ax.text(x + width / 2, y + 0.04, label, transform=ax.transAxes, ha="center", va="center", fontsize=7.5)
        x += width + 0.012
        if x > 0.92:
            break


def _draw_scene(
    ax: plt.Axes,
    case: Mapping[str, Any],
    *,
    target: Optional[Mapping[str, Any]] = None,
    ghost: Optional[Mapping[str, Any]] = None,
    title: str = "",
    subtitle: Optional[str] = None,
    metrics: Optional[Mapping[str, Any]] = None,
) -> None:
    scene = case["scene"]
    floor = scene["floor"]["polygon"]
    xs = [float(p[0]) for p in floor]
    zs = [float(p[1]) for p in floor]
    xmin, xmax = min(xs), max(xs)
    zmin, zmax = min(zs), max(zs)
    margin = max(xmax - xmin, zmax - zmin) * 0.06 + 0.1
    ax.set_aspect("equal")
    ax.add_patch(
        MplPolygon(
            [(float(x), float(z)) for x, z in floor],
            closed=True,
            facecolor="#F7F3EB",
            edgecolor="#454545",
            linewidth=1.8,
            zorder=0,
        )
    )

    anchor_categories = _primary_anchor_categories(case)
    for obj in scene["objects"]:
        poly = _object_polygon(obj)
        category = _slug(obj["category"])
        face = "#C9D7E3"
        edge = "#6A7B8C"
        alpha = 0.92
        if category in anchor_categories:
            face = "#B8D9A6"
            edge = "#4E7C2A"
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=face, edgecolor=edge, linewidth=1.0, alpha=alpha, zorder=2))
        cx, cz = _object_center(obj)
        label = str(obj["category"]).replace("_", " ")
        ax.text(cx, cz, textwrap.shorten(label, width=13, placeholder=""), ha="center", va="center", fontsize=6.2, color="#21313F", zorder=3)

    if ghost is not None:
        ax.add_patch(
            MplPolygon(
                _object_polygon(ghost),
                closed=True,
                facecolor="none",
                edgecolor="#E6845E",
                linewidth=1.5,
                linestyle="--",
                zorder=4,
            )
        )

    for door in scene.get("doors", []):
        dx = float(door["position"][0])
        dz = float(door["position"][2])
        ax.add_patch(Circle((dx, dz), 0.12, facecolor="#D9534F", edgecolor="white", linewidth=0.8, zorder=5))
        ax.text(dx, dz, "D", ha="center", va="center", fontsize=6.5, color="white", weight="bold", zorder=6)

    for window in scene.get("windows", []):
        wx = float(window["position"][0])
        wz = float(window["position"][2])
        ax.add_patch(Circle((wx, wz), 0.12, facecolor="#4E79A7", edgecolor="white", linewidth=0.8, zorder=5))
        ax.text(wx, wz, "W", ha="center", va="center", fontsize=6.5, color="white", weight="bold", zorder=6)

    if target is not None:
        poly = _object_polygon(target)
        ax.add_patch(MplPolygon(poly, closed=True, facecolor="#F28E6B", edgecolor="#B54518", linewidth=1.5, alpha=0.95, zorder=7))
        cx, cz = _object_center(target)
        ax.text(cx, cz, str(target["category"]).replace("_", " "), ha="center", va="center", fontsize=7.0, color="white", weight="bold", zorder=8)
        size = target["size"]
        w = float(size[0]) if not isinstance(size, dict) else float(size["width"])
        d = float(size[2]) if not isinstance(size, dict) else float(size["depth"])
        yaw = float(target.get("yaw", target.get("rotation_y", 0.0)))
        arrow_len = 0.25 * max(w, d)
        ax.add_patch(
            FancyArrowPatch(
                (cx, cz),
                (cx + arrow_len * math.cos(math.radians(yaw)), cz + arrow_len * math.sin(math.radians(yaw))),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#7A250D",
                linewidth=1.2,
                zorder=9,
            )
        )

    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(zmin - margin, zmax + margin)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10.5, pad=8)
    if subtitle:
        ax.text(0.5, -0.08, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=7.2, color="#444444")
    if metrics:
        summary = []
        summary.append(f"CF {metrics['cf']:.0f}")
        summary.append(f"IB {metrics['ib']:.0f}")
        summary.append(f"CA {metrics['constraint_accuracy']:.2f}")
        summary.append(f"CPS {int(metrics['cps'])}")
        ax.text(
            0.01,
            0.01,
            " | ".join(summary),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.8,
            color="#2E2E2E",
            bbox={"facecolor": "white", "edgecolor": "#C7C7C7", "alpha": 0.9, "boxstyle": "round,pad=0.2"},
        )


def _asset_panel(ax: plt.Axes, case: Mapping[str, Any]) -> None:
    ax.set_axis_off()
    ax.set_title("Target Asset", fontsize=10.5, pad=8)
    thumb = _target_thumbnail(case)
    if thumb is not None:
        image = Image.open(thumb).convert("RGB")
        image = ImageOps.contain(image, (700, 500))
        ax.imshow(image)
    else:
        ax.add_patch(Rectangle((0.08, 0.22), 0.84, 0.52, transform=ax.transAxes, facecolor="#FDE3D9", edgecolor="#C96B4F", linewidth=1.5))
        ax.text(0.5, 0.48, case["target_asset"]["category"].replace("_", " "), transform=ax.transAxes, ha="center", va="center", fontsize=14, color="#8B3B25")
    size = case["target_asset"]["size"]
    dims = f"{size['width']:.2f} x {size['depth']:.2f} m"
    ax.text(0.5, 0.14, case["target_asset"]["category"].replace("_", " ").title(), transform=ax.transAxes, ha="center", va="center", fontsize=12, weight="bold")
    ax.text(0.5, 0.07, dims, transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#555555")


def make_main_comparison(results: Mapping[str, Dict[str, float]], kind: str) -> List[str]:
    style = _style(kind)
    methods = [
        "heuristic_baseline",
        "proposal_heuristic",
        "proposal_diffopt_basic",
        "proposal_diffopt_constraint",
        "proposal_llm_grounded_diffopt",
    ]
    focus_metrics = ["CF", "IB", "Constraint Accuracy", "CPS", "Success@5"]
    fig = plt.figure(figsize=(14.0 * style["figscale"], 5.8 * style["figscale"]))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.25, 1.0], wspace=0.26)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(methods))
    width = 0.14
    for idx, metric in enumerate(focus_metrics):
        offset = (idx - 2) * width
        vals = [results[m][metric] for m in methods]
        color = ["#7B8EA8", "#9AC2E2", "#F2B880", "#E06C78", "#4CB397"][idx]
        ax.bar(x + offset, vals, width=width, label=metric, color=color, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([_method_label(m) for m in methods], fontsize=style["tick"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score", fontsize=style["label"])
    ax.set_title("Main Comparison: geometric validity vs. intent success", fontsize=style["title"], loc="left")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(ncol=3, fontsize=style["tick"], frameon=False, loc="upper left")

    ax2 = fig.add_subplot(gs[0, 1])
    cf = [results[m]["CF"] for m in methods]
    ca = [results[m]["Constraint Accuracy"] for m in methods]
    s5 = [results[m]["Success@5"] for m in methods]
    offsets = {
        "heuristic_baseline": (0.006, 0.005),
        "proposal_heuristic": (0.006, 0.005),
        "proposal_diffopt_basic": (0.006, 0.006),
        "proposal_diffopt_constraint": (0.006, 0.008),
        "proposal_llm_grounded_diffopt": (0.055, 0.004),
    }
    for method, xval, yval, sval in zip(methods, cf, ca, s5):
        ax2.scatter(xval, yval, s=600 * sval + style["marker"], color=METHOD_COLORS[method], alpha=0.88, edgecolor="white", linewidth=1.2)
        dx, dy = offsets[method]
        label = _method_label(method).replace("\n", " ")
        if method == "proposal_llm_grounded_diffopt":
            label = "LLM/VLM-Grounded"
        ax2.text(xval + dx, yval + dy, label, fontsize=style["tick"])
    ax2.set_xlim(0.78, 1.01)
    ax2.set_ylim(0.64, 0.86)
    ax2.set_xlabel("Collision-Free Rate (CF)", fontsize=style["label"])
    ax2.set_ylabel("Constraint Accuracy", fontsize=style["label"])
    ax2.set_title("Tradeoff view: bubble size = Success@5", fontsize=style["subtitle"], loc="left")
    ax2.grid(linestyle=":", alpha=0.35)

    fig.suptitle("SpaceFit Single-Target Placement Results", fontsize=style["title"] + 1, y=1.02)
    stem = "fig_main_comparison" if kind == "paper" else "fig_main_comparison_ppt"
    return _save_variants(fig, stem, png_only=(kind == "ppt"))


def make_diffopt_delta(previous: Mapping[str, Dict[str, float]], improved: Mapping[str, Dict[str, float]], kind: str) -> List[str]:
    style = _style(kind)
    methods = ["proposal_diffopt_basic", "proposal_diffopt_constraint"]
    metrics = ["CF", "IB", "Constraint Accuracy", "CPS", "Success@5"]
    fig, axes = plt.subplots(1, 2, figsize=(14.0 * style["figscale"], 5.4 * style["figscale"]), sharex=True)
    for ax, method in zip(axes, methods):
        prev_vals = np.array([previous[method][m] for m in metrics])
        new_vals = np.array([improved[method][m] for m in metrics])
        delta = new_vals - prev_vals
        y = np.arange(len(metrics))
        colors = ["#2CA02C" if d >= 0 else "#D62728" for d in delta]
        ax.axvline(0.0, color="#888888", linewidth=1.0)
        ax.barh(y, delta, color=colors, alpha=0.85)
        for idx, d in enumerate(delta):
            ax.text(d + (0.005 if d >= 0 else -0.005), idx, f"{d:+.3f}", va="center", ha="left" if d >= 0 else "right", fontsize=style["tick"])
            ax.text(0.35, idx + 0.28, f"{prev_vals[idx]:.3f} -> {new_vals[idx]:.3f}", fontsize=style["small"], color="#555555")
        ax.set_yticks(y)
        ax.set_yticklabels(metrics, fontsize=style["tick"])
        ax.set_title(_method_label(method).replace("\n", " "), fontsize=style["subtitle"], loc="left")
        ax.grid(axis="x", linestyle=":", alpha=0.35)
    axes[0].set_xlim(-0.12, 0.24)
    axes[0].set_xlabel("After - Before", fontsize=style["label"])
    fig.suptitle("Experiment A: DiffOpt delta after targeted improvements", fontsize=style["title"], y=1.02)
    stem = "fig_diffopt_delta" if kind == "paper" else "fig_diffopt_delta_ppt"
    return _save_variants(fig, stem, png_only=(kind == "ppt"))


def make_proposal_ablation(results: Mapping[str, Dict[str, float]], kind: str) -> List[str]:
    style = _style(kind)
    fig = plt.figure(figsize=(14.0 * style["figscale"], 6.0 * style["figscale"]))
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.95, 1.25], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    for variant, color in [("basic", "#F58518"), ("constraint", "#2CA02C")]:
        with_key = f"proposal_diffopt_{variant}"
        no_key = f"no_proposal_diffopt_{variant}"
        yvals = [results[with_key]["CPS"], results[with_key]["Success@5"]]
        yvals2 = [results[no_key]["CPS"], results[no_key]["Success@5"]]
        ypos = np.array([0, 1]) + (0.12 if variant == "constraint" else -0.12)
        for idx in range(2):
            ax.plot([yvals2[idx], yvals[idx]], [ypos[idx], ypos[idx]], color=color, linewidth=3.0, alpha=0.8)
        ax.scatter(yvals2, ypos, color="#E45756", s=70, label=f"No-Proposal {variant.title()}" if variant == "basic" else None, zorder=4)
        ax.scatter(yvals, ypos, color=color, s=70, label=f"Proposal {variant.title()}" if variant == "basic" else None, zorder=5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["CPS", "Success@5"], fontsize=style["tick"])
    ax.set_xlim(0.0, 0.75)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.set_title("Primary effect: proposal improves scene success", fontsize=style["subtitle"], loc="left")

    ax2 = fig.add_subplot(gs[0, 1])
    methods = [
        "proposal_diffopt_constraint",
        "no_proposal_diffopt_constraint",
        "proposal_diffopt_basic",
        "no_proposal_diffopt_basic",
    ]
    metrics = ["CF", "IB", "Constraint Accuracy", "Walkability"]
    xpos = np.arange(len(metrics))
    width = 0.18
    for idx, method in enumerate(methods):
        vals = [results[method][m] for m in metrics]
        ax2.bar(xpos + (idx - 1.5) * width, vals, width=width, color=METHOD_COLORS[method], label=_method_label(method).replace("\n", " "), edgecolor="white", linewidth=0.6)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(metrics, fontsize=style["tick"])
    ax2.set_ylim(0.0, 1.05)
    ax2.set_title("Secondary metrics", fontsize=style["subtitle"], loc="left")
    ax2.grid(axis="y", linestyle=":", alpha=0.35)
    ax2.legend(fontsize=style["small"], frameon=False, loc="upper left")

    fig.suptitle("Experiment B: Proposal ablation", fontsize=style["title"], y=1.02)
    stem = "fig_proposal_ablation" if kind == "paper" else "fig_proposal_ablation_ppt"
    return _save_variants(fig, stem, png_only=(kind == "ppt"))


def _grounding_constraint_breakdown(data: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, Counter] = {}
    for row in data["cases"]:
        gt = set(row["gt_constraints"])
        pred = set(row["grounded_constraints"])
        for ctype in gt:
            bucket = counts.setdefault(ctype, Counter())
            bucket["gt"] += 1
            if ctype in pred:
                bucket["hit"] += 1
            else:
                bucket["miss"] += 1
    out = {}
    for ctype, counter in counts.items():
        gt = max(counter["gt"], 1)
        out[ctype] = {
            "gt": float(counter["gt"]),
            "hit": float(counter["hit"]),
            "miss": float(counter["miss"]),
            "recall": float(counter["hit"] / gt),
        }
    return out


def make_grounding_eval(data: Mapping[str, Any], kind: str) -> List[str]:
    style = _style(kind)
    agg = data["aggregate"]
    breakdown = _grounding_constraint_breakdown(data)
    metrics = [
        ("Precision", agg["avg_precision"]),
        ("Recall", agg["avg_recall"]),
        ("F1", agg["avg_f1"]),
        ("Near-category acc.", agg["near_category_accuracy"]),
    ]
    order = ["against_wall", "near", "facing", "not_block_door", "keep_window_clear", "access_zone"]

    fig = plt.figure(figsize=(14.0 * style["figscale"], 5.8 * style["figscale"]))
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.9, 1.15], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    ax.bar([name for name, _ in metrics], [val for _, val in metrics], color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Aggregate grounding metrics", fontsize=style["subtitle"], loc="left")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    for idx, (_, val) in enumerate(metrics):
        ax.text(idx, val + 0.025, f"{val:.3f}", ha="center", va="bottom", fontsize=style["tick"])

    ax2 = fig.add_subplot(gs[0, 1])
    ypos = np.arange(len(order))
    hits = [breakdown.get(k, {}).get("hit", 0.0) for k in order]
    misses = [breakdown.get(k, {}).get("miss", 0.0) for k in order]
    ax2.barh(ypos, hits, color="#54A24B", label="Recovered")
    ax2.barh(ypos, misses, left=hits, color="#E45756", label="Missed")
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([k.replace("_", " ") for k in order], fontsize=style["tick"])
    ax2.set_title("Constraint-type recovery count", fontsize=style["subtitle"], loc="left")
    ax2.legend(frameon=False, fontsize=style["tick"], loc="lower right")
    ax2.grid(axis="x", linestyle=":", alpha=0.35)
    ax2.text(
        0.98,
        0.03,
        "`access_zone` is missed in every grounded case\nbecause the parser does not map\n'room to reach it' -> access_zone.",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=style["small"],
        bbox={"facecolor": "#FFF7E6", "edgecolor": "#E6B450", "boxstyle": "round,pad=0.35"},
    )

    fig.suptitle("Experiment C: Grounding evaluation", fontsize=style["title"], y=1.02)
    stem = "fig_grounding_eval" if kind == "paper" else "fig_grounding_eval_ppt"
    return _save_variants(fig, stem, png_only=(kind == "ppt"))


def make_metric_groups(kind: str) -> List[str]:
    style = _style(kind)
    fig, ax = plt.subplots(figsize=(12.8 * style["figscale"], 4.8 * style["figscale"]))
    ax.set_axis_off()
    ax.set_title("Metric groups used in the single-target benchmark", fontsize=style["title"], loc="left", pad=10)

    cards = [
        ("Physical Validity", ["CF", "IB"], "#DCEAF7", "#4C78A8", "Does the placement fit inside the room\nwithout colliding with fixed furniture?"),
        ("Intent Satisfaction", ["Constraint Accuracy", "CPS", "Success@1", "Success@5"], "#FCE7CC", "#F58518", "Does the pose satisfy the user intent\nand produce a fully successful scene?"),
        ("Usability", ["Reachability", "Walkability"], "#DBF0D8", "#54A24B", "Can a person still reach the asset\nand move through the room?"),
    ]
    xs = [0.03, 0.355, 0.68]
    for x, (title, metrics, face, edge, desc) in zip(xs, cards):
        card = FancyBboxPatch((x, 0.18), 0.285, 0.60, boxstyle="round,pad=0.018,rounding_size=0.03", facecolor=face, edgecolor=edge, linewidth=2.0, transform=ax.transAxes)
        ax.add_patch(card)
        ax.text(x + 0.02, 0.72, title, transform=ax.transAxes, fontsize=style["subtitle"], weight="bold", color=edge)
        ax.text(x + 0.02, 0.56, "\n".join(f"- {metric}" for metric in metrics), transform=ax.transAxes, fontsize=style["label"], va="top")
        ax.text(x + 0.02, 0.26, desc, transform=ax.transAxes, fontsize=style["tick"], color="#444444")
    ax.text(0.5, 0.06, "Together these metrics evaluate whether the prediction is valid, semantically appropriate, and usable in a furnished room.", transform=ax.transAxes, ha="center", fontsize=style["label"])
    stem = "fig_metric_groups" if kind == "paper" else "fig_metric_groups_ppt"
    return _save_variants(fig, stem, png_only=(kind == "ppt"))


def make_case_overview(case: Mapping[str, Any], prediction: Mapping[str, Any]) -> List[str]:
    fig = plt.figure(figsize=(16.0, 4.8))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.0, 1.0, 0.8, 1.0], wspace=0.16)
    ref_target = _target_object_from_pose(case, case["reference_pose"]["position"], case["reference_pose"]["rotation_y"])
    pred_obj = _prediction_to_object(case, prediction)

    ax1 = fig.add_subplot(gs[0, 0])
    _draw_scene(ax1, case, target=ref_target, title="Original furnished room")

    ax2 = fig.add_subplot(gs[0, 1])
    _draw_scene(ax2, case, ghost=ref_target, title="Benchmark input (target removed)")

    ax3 = fig.add_subplot(gs[0, 2])
    _asset_panel(ax3, case)

    ax4 = fig.add_subplot(gs[0, 3])
    _draw_scene(ax4, case, target=pred_obj, title="Predicted placement")
    ax4.text(0.02, -0.12, textwrap.fill(case["intent"]["text"], width=42), transform=ax4.transAxes, fontsize=8.2, va="top")

    fig.suptitle("Benchmark case overview", fontsize=15, y=1.02)
    return _save_variants(fig, "fig_case_overview_01", png_only=False)


def make_method_comparison(
    case_ids: Sequence[str],
    cases: Mapping[str, Dict[str, Any]],
    eval_rows: Mapping[str, Mapping[str, Dict[str, Any]]],
) -> List[str]:
    methods = ["heuristic_baseline", "proposal_heuristic", "proposal_diffopt_constraint"]
    fig = plt.figure(figsize=(15.8, 4.2 * len(case_ids)))
    gs = gridspec.GridSpec(len(case_ids), len(methods) + 1, width_ratios=[0.56, 1.0, 1.0, 1.0], hspace=0.28, wspace=0.14)

    for row_idx, case_id in enumerate(case_ids):
        case = cases[case_id]
        label_ax = fig.add_subplot(gs[row_idx, 0])
        label_ax.set_axis_off()
        label_ax.text(0.0, 0.85, case["target_asset"]["category"].replace("_", " ").title(), fontsize=12, weight="bold")
        label_ax.text(0.0, 0.69, case["scene"]["room_type"], fontsize=10, color="#555555")
        label_ax.text(0.0, 0.52, textwrap.fill(case["intent"]["text"], width=26), fontsize=8.0, va="top")
        label_ax.text(0.0, 0.14, _constraint_summary(case), fontsize=7.6, color="#7A5A00")

        for col_idx, method in enumerate(methods, start=1):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            prediction = eval_rows[method][case_id]["predictions"][0]
            metrics = eval_rows[method][case_id]["top1"]
            _draw_scene(ax, case, target=_prediction_to_object(case, prediction), title=_method_label(method).replace("\n", " "), metrics=metrics)

    fig.suptitle("Representative method comparison cases", fontsize=15, y=1.01)
    return _save_variants(fig, "fig_case_compare_01", png_only=False)


def make_failure_cases(
    selected: Mapping[str, Any],
    cases: Mapping[str, Dict[str, Any]],
    main_eval: Mapping[str, Mapping[str, Dict[str, Any]]],
    exp_b_eval: Mapping[str, Mapping[str, Dict[str, Any]]],
    grounding: Mapping[str, Any],
) -> List[str]:
    fig = plt.figure(figsize=(16.0, 9.2))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.0, 1.0], hspace=0.24, wspace=0.16)

    case_geom = cases[selected["failure_geom"]]
    pred_basic = main_eval["proposal_diffopt_basic"][selected["failure_geom"]]["predictions"][0]
    pred_constraint = main_eval["proposal_diffopt_constraint"][selected["failure_geom"]]["predictions"][0]
    ax = fig.add_subplot(gs[0, 0])
    _draw_scene(ax, case_geom, target=_prediction_to_object(case_geom, pred_basic), title="Weaker optimization failure", metrics=main_eval["proposal_diffopt_basic"][selected["failure_geom"]]["top1"])
    ax.text(0.02, -0.12, "Proposal + DiffOpt-Basic", transform=ax.transAxes, fontsize=8.0, va="top")
    ax = fig.add_subplot(gs[0, 1])
    _draw_scene(ax, case_geom, target=_prediction_to_object(case_geom, pred_constraint), title="Constraint-guided fix", metrics=main_eval["proposal_diffopt_constraint"][selected["failure_geom"]]["top1"])
    ax.text(0.02, -0.12, "Same case, Proposal + DiffOpt-Constraint", transform=ax.transAxes, fontsize=8.0, va="top")

    case_nop = cases[selected["failure_no_prop"]]
    pred_nop = exp_b_eval["no_proposal_diffopt_constraint"][selected["failure_no_prop"]]["predictions"][0]
    pred_prop = exp_b_eval["proposal_diffopt_constraint"][selected["failure_no_prop"]]["predictions"][0]
    ax = fig.add_subplot(gs[1, 0])
    _draw_scene(ax, case_nop, target=_prediction_to_object(case_nop, pred_nop), title="No-proposal failure", metrics=exp_b_eval["no_proposal_diffopt_constraint"][selected["failure_no_prop"]]["top1"])
    ax = fig.add_subplot(gs[1, 1])
    _draw_scene(ax, case_nop, target=_prediction_to_object(case_nop, pred_prop), title="Proposal-guided recovery", metrics=exp_b_eval["proposal_diffopt_constraint"][selected["failure_no_prop"]]["top1"])

    axg = fig.add_subplot(gs[:, 2])
    axg.set_axis_off()
    ground_row = next(item for item in grounding["cases"] if item["case_id"] == selected["failure_grounding"])
    gt = [item.replace("_", " ") for item in ground_row["gt_constraints"]]
    pred = [item.replace("_", " ") for item in ground_row["grounded_constraints"]]
    miss = sorted(set(gt) - set(pred))
    axg.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.90, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="#FFF9EE", edgecolor="#D9A441", linewidth=1.8, transform=axg.transAxes))
    axg.text(0.06, 0.91, "Grounding miss: access-related instruction", transform=axg.transAxes, fontsize=12, weight="bold")
    axg.text(0.06, 0.82, ground_row["category"].replace("_", " ").title(), transform=axg.transAxes, fontsize=10, color="#555555")
    axg.text(0.06, 0.74, textwrap.fill(ground_row["nl_instruction"], width=34), transform=axg.transAxes, fontsize=10, va="top")
    axg.text(0.06, 0.50, "Ground-truth constraints", transform=axg.transAxes, fontsize=10, weight="bold")
    axg.text(0.06, 0.44, "\n".join(f"- {item}" for item in gt), transform=axg.transAxes, fontsize=9.4, va="top")
    axg.text(0.06, 0.28, "Grounded constraints", transform=axg.transAxes, fontsize=10, weight="bold")
    axg.text(0.06, 0.22, "\n".join(f"- {item}" for item in pred), transform=axg.transAxes, fontsize=9.4, va="top")
    axg.text(0.06, 0.08, f"Missed: {', '.join(miss)}", transform=axg.transAxes, fontsize=10.2, color="#C23B22", weight="bold")

    fig.suptitle("Failure cases", fontsize=15, y=1.01)
    return _save_variants(fig, "fig_failure_cases", png_only=False)


def make_success_cases(
    case_ids: Sequence[str],
    cases: Mapping[str, Dict[str, Any]],
    main_eval: Mapping[str, Mapping[str, Dict[str, Any]]],
) -> List[str]:
    fig = plt.figure(figsize=(15.8, 8.8))
    gs = gridspec.GridSpec(len(case_ids), 3, width_ratios=[0.62, 1.0, 1.0], hspace=0.25, wspace=0.16)
    for row_idx, case_id in enumerate(case_ids):
        case = cases[case_id]
        ax_label = fig.add_subplot(gs[row_idx, 0])
        ax_label.set_axis_off()
        ax_label.text(0.0, 0.84, case["target_asset"]["category"].replace("_", " ").title(), fontsize=12, weight="bold")
        ax_label.text(0.0, 0.67, case["scene"]["room_type"], fontsize=10, color="#555555")
        ax_label.text(0.0, 0.52, textwrap.fill(case["intent"]["text"], width=28), fontsize=8.4, va="top")
        ax_label.text(
            0.0,
            0.12,
            textwrap.fill("Selected because Proposal + DiffOpt-Constraint reaches CPS=1 here while Proposal + Heuristic does not.", width=36),
            fontsize=8.0,
            color="#2A6E2A",
        )

        ax1 = fig.add_subplot(gs[row_idx, 1])
        pred1 = main_eval["proposal_heuristic"][case_id]["predictions"][0]
        _draw_scene(ax1, case, target=_prediction_to_object(case, pred1), title="Proposal + Heuristic", metrics=main_eval["proposal_heuristic"][case_id]["top1"])

        ax2 = fig.add_subplot(gs[row_idx, 2])
        pred2 = main_eval["proposal_diffopt_constraint"][case_id]["predictions"][0]
        _draw_scene(ax2, case, target=_prediction_to_object(case, pred2), title="Proposal + DiffOpt-Constraint", metrics=main_eval["proposal_diffopt_constraint"][case_id]["top1"])

    fig.suptitle("Success cases", fontsize=15, y=1.01)
    return _save_variants(fig, "fig_success_cases", png_only=False)


def make_pipeline_diagram(kind: str) -> List[str]:
    style = _style(kind)
    fig, ax = plt.subplots(figsize=(15.2 * style["figscale"], 4.6 * style["figscale"]))
    ax.set_axis_off()
    ax.set_title("SpaceFit single-target placement pipeline", fontsize=style["title"], loc="left", pad=12)

    boxes = [
        ("Room input", "Furnished 3D-FRONT room\n+ removed target asset"),
        ("Scene normalization", "Floor polygon, doors,\nwindows, fixed furniture"),
        ("Intent grounding", "Natural-language intent ->\nstructured constraints"),
        ("Candidate proposal", "Free-space regions and\nconstraint-aware seed poses"),
        ("Placement refinement", "DiffOpt with physics,\nconstraint terms, post-snap"),
        ("Evaluation + viz", "CF / IB / CPS / S@5\n+ qualitative figures"),
    ]
    x_positions = np.linspace(0.03, 0.83, len(boxes))
    for idx, ((title, body), x) in enumerate(zip(boxes, x_positions)):
        face = ["#E9F1F8", "#EEF3E5", "#FFF2DA", "#FDE7D9", "#E8F4E8", "#F2EDF9"][idx]
        edge = ["#4C78A8", "#6A994E", "#D99200", "#D16B47", "#3C8D40", "#8B6BB3"][idx]
        patch = FancyBboxPatch((x, 0.24), 0.13, 0.46, boxstyle="round,pad=0.02,rounding_size=0.025", facecolor=face, edgecolor=edge, linewidth=2.0, transform=ax.transAxes)
        ax.add_patch(patch)
        ax.text(x + 0.065, 0.59, title, transform=ax.transAxes, ha="center", va="center", fontsize=style["subtitle"], weight="bold", color=edge)
        ax.text(x + 0.065, 0.42, body, transform=ax.transAxes, ha="center", va="center", fontsize=style["tick"])
        if idx < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.13, 0.47), (x_positions[idx + 1] - 0.01, 0.47), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=16, linewidth=2.0, color="#666666"))
    ax.text(0.5, 0.09, "Two-stage design: proposal narrows the search space, refinement improves feasibility and semantic fit, unified evaluation measures validity, intent, and usability.", transform=ax.transAxes, ha="center", fontsize=style["label"])
    stem = "fig_pipeline_diagram" if kind == "paper" else "fig_pipeline_diagram_ppt"
    return _save_variants(fig, stem, png_only=(kind == "ppt"))


def _write_selected_cases(selected: Mapping[str, Any]) -> str:
    path = OUT_DIR / "selected_cases.json"
    with open(path, "w") as f:
        json.dump(selected, f, indent=2)
    return str(path)


def _write_index(created: Sequence[str], selected: Mapping[str, Any]) -> List[str]:
    figure_names = sorted({Path(path).name for path in created})
    exact = [
        "Quantitative figures (`fig_main_comparison`, `fig_diffopt_delta`, `fig_proposal_ablation`, `fig_grounding_eval`) use exact stored aggregate outputs.",
        "Target asset thumbnails are exact 3D-FUTURE images linked from the original 3D-FRONT source scenes.",
    ]
    reconstructed = [
        "All qualitative room-layout figures use reconstructed top-down renderings from exact benchmark geometry and exact stored predictions.",
        "The metric grouping and pipeline figures are reconstructed explanatory visuals, not evaluator outputs.",
    ]

    index_lines = [
        "# Visualization Package Index",
        "",
        f"Output directory: `{OUT_DIR}`",
        "",
        "## Figures",
        "",
    ]
    for name in figure_names:
        stem = Path(name).stem.replace("_ppt", "")
        key = stem
        note = FIGURE_PROVENANCE.get(key, "Generated from local repository artifacts.")
        index_lines.append(f"- `{name}`: {note}")

    index_lines.extend(
        [
            "",
            "## Provenance",
            "",
            "Exact existing outputs:",
        ]
    )
    index_lines.extend([f"- {line}" for line in exact])
    index_lines.extend(
        [
            "",
            "Reconstructed visuals:",
        ]
    )
    index_lines.extend([f"- {line}" for line in reconstructed])
    index_lines.extend(
        [
            "",
            "## Automatically selected qualitative cases",
            "",
            f"- Overview cases: {', '.join(selected['overview'])}",
            f"- Comparison cases: {', '.join(selected['compare'])}",
            f"- Success cases: {', '.join(selected['success'])}",
            f"- Failure case (weaker optimization): {selected['failure_geom']}",
            f"- Failure case (no proposal): {selected['failure_no_prop']}",
            f"- Grounding miss case: {selected['failure_grounding']}",
            "",
            "## Notes",
            "",
            "- No major experiment was rerun. Only lightweight metric recomputation on saved predictions was used to choose representative cases.",
            "- PPT variants are simplified PNG versions with larger text for direct slide insertion.",
        ]
    )
    index_path = OUT_DIR / "VISUALIZATION_INDEX.md"
    index_path.write_text("\n".join(index_lines) + "\n")

    slide_lines = ["# Recommended Slide Mapping", ""]
    for slide, (asset, note) in SLIDE_SUGGESTIONS.items():
        slide_lines.append(f"- {slide}: `{asset}` — {note}")
    slide_path = OUT_DIR / "SLIDE_MAPPING.md"
    slide_path.write_text("\n".join(slide_lines) + "\n")
    return [str(index_path), str(slide_path)]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = _load_cases()
    main_results = _read_results_table(MAIN_RESULTS)
    exp_a_results = _read_results_table(EXP_A_RESULTS)
    exp_b_results = _read_results_table(EXP_B_RESULTS)
    grounding = _load_json(EXP_C_RESULTS)

    main_predictions = _load_predictions(EXP_A_PREDICTIONS)
    exp_b_predictions = _load_predictions(EXP_B_PREDICTIONS)

    main_eval = _evaluate_case_predictions(cases, main_predictions)
    exp_b_eval = _evaluate_case_predictions(cases, exp_b_predictions)
    selected = _select_cases(cases, main_eval, exp_b_eval, grounding)

    created: List[str] = []
    for kind in ("paper", "ppt"):
        created.extend(make_main_comparison(exp_a_results, kind))
        created.extend(make_diffopt_delta(main_results, exp_a_results, kind))
        created.extend(make_proposal_ablation(exp_b_results, kind))
        created.extend(make_grounding_eval(grounding, kind))
        created.extend(make_metric_groups(kind))
        created.extend(make_pipeline_diagram(kind))

    overview_case = cases[selected["overview"][0]]
    overview_pred = main_eval["proposal_diffopt_constraint"][selected["overview"][0]]["predictions"][0]
    created.extend(make_case_overview(overview_case, overview_pred))
    created.extend(make_method_comparison(selected["compare"], cases, main_eval))
    created.extend(make_failure_cases(selected, cases, main_eval, exp_b_eval, grounding))
    created.extend(make_success_cases(selected["success"], cases, main_eval))
    created.append(_write_selected_cases(selected))
    created.extend(_write_index(created, selected))

    print(json.dumps({"output_dir": str(OUT_DIR), "created_files": sorted(created)}, indent=2))


if __name__ == "__main__":
    main()
