from __future__ import annotations

import json
import math
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_spacefit_exp_a")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon as MplPolygon, Rectangle
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.eval.unified_eval import (
    _compute_candidate_metrics,
    _connected_components,
    _constraint_satisfied,
    _nearest_free_cell,
    _object_access_polygon,
    _opening_keepout_polygon,
    _placement_polygon,
    _raster_scene,
)
from spacefit_v2.single_target.eval import normalize_case_result


RESULTS_DIR = ROOT / "spacefit_v2" / "results"
BENCHMARK_CASES = ROOT / "spacefit_v2" / "data" / "single_target_benchmark" / "cases.json"
BEFORE_PREDICTIONS = RESULTS_DIR / "single_target_paper_v1" / "raw_predictions.json"
AFTER_PREDICTIONS = RESULTS_DIR / "exp_a" / "raw_predictions.json"
OUT_DIR = RESULTS_DIR / "visualizations" / "experiment_a_qualitative"


def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text).lower()).strip("_")


def _style_target(kind: str) -> Dict[str, str]:
    if kind == "before":
        return {"face": "#F4A261", "edge": "#C2410C", "arrow": "#8A2E0A"}
    if kind == "after":
        return {"face": "#5BA7D1", "edge": "#0B5E8E", "arrow": "#083B63"}
    if kind == "heur":
        return {"face": "#90BE6D", "edge": "#4C7A2B", "arrow": "#36541E"}
    return {"face": "#D9D9D9", "edge": "#666666", "arrow": "#444444"}


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
    yaw = float(item.get("yaw", item.get("rotation_y", 0.0)))
    return _corners(cx, cz, w, d, yaw)


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


def _primary_anchor_categories(case: Mapping[str, Any]) -> set[str]:
    out = set()
    for item in case["intent"]["constraints"]:
        target = item.get("target_category")
        if target:
            out.add(_slug(target))
    return out


def _target_label(case: Mapping[str, Any]) -> str:
    return str(case["target_asset"]["category"]).replace("_", " ")


def _room_label(case: Mapping[str, Any]) -> str:
    return str(case["scene"]["room_type"]).replace("_", " ")


@dataclass
class SceneDiagnostics:
    outside_area: float
    fixed_hits: List[str]
    door_hits: int
    window_hits: int
    failed_constraints: List[str]
    theme: str
    caption: str


@dataclass
class CaseRow:
    case_id: str
    case: Dict[str, Any]
    before_prediction: Dict[str, Any]
    after_prediction: Dict[str, Any]
    heur_prediction: Dict[str, Any]
    before_metrics: Dict[str, Any]
    after_metrics: Dict[str, Any]
    heur_metrics: Dict[str, Any]
    before_diag: SceneDiagnostics
    after_diag: SceneDiagnostics
    score: float
    move_distance: float
    yaw_delta: float


def _angle_delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _center_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    pa = a["position"]
    pb = b["position"]
    return math.hypot(float(pa["x"]) - float(pb["x"]), float(pa["z"]) - float(pb["z"]))


def _reachability_map(scene: Any, candidate: Any) -> Dict[str, bool]:
    floor_poly = Polygon(scene.floor_polygon)
    predicted = [p for p in candidate.placements if p.status == "placed"]
    fixed_polys = [(obj, _placement_polygon(obj)) for obj in scene.fixed_objects]
    keepouts = [_opening_keepout_polygon(door, depth=0.45) for door in scene.doors]
    obstacle_polys = [poly for _, poly in fixed_polys]
    obstacle_polys.extend([_placement_polygon(p) for p in predicted])
    access_polys = [_object_access_polygon(p) for p in predicted]
    raster = _raster_scene(floor_poly=floor_poly, obstacle_polys=obstacle_polys + keepouts, access_polys=access_polys)
    labels = _connected_components(raster["free"])
    entrance = scene.entrance_point
    if entrance is None:
        centroid = floor_poly.centroid
        entrance = (float(centroid.x), float(centroid.y))
    start_cell = _nearest_free_cell(raster["free"], raster["xs"], raster["zs"], entrance)
    start_label = labels[start_cell] if start_cell is not None else 0
    reachability: Dict[str, bool] = {}
    for placement, access_mask in zip(predicted, raster["access_masks"]):
        reachable = False
        if start_label != 0 and np.any(access_mask):
            access_labels = labels[access_mask]
            reachable = bool(np.any(access_labels == start_label))
        reachability[placement.asset_id] = reachable
    return reachability


def _scene_diagnostics(case: Dict[str, Any], prediction: Dict[str, Any]) -> SceneDiagnostics:
    scene = normalize_case_result(case, "diag", [prediction])
    candidate = scene.candidates[0]
    placement = candidate.placements[0]
    poly = _placement_polygon(placement)
    floor = Polygon(scene.floor_polygon)
    outside_area = float(poly.difference(floor).area)

    fixed_hits: List[str] = []
    for obj in scene.fixed_objects:
        if poly.intersection(_placement_polygon(obj)).area > 1e-4:
            fixed_hits.append(str(obj.category).replace("_", " "))

    door_hits = 0
    for door in scene.doors:
        if poly.intersection(_opening_keepout_polygon(door, depth=0.45)).area > 1e-4:
            door_hits += 1

    window_hits = 0
    for window in scene.windows:
        if poly.intersection(_opening_keepout_polygon(window, depth=0.30)).area > 1e-4:
            window_hits += 1

    reachability_map = _reachability_map(scene, candidate)
    failed_constraints: List[str] = []
    for constraint in scene.constraints:
        ok = _constraint_satisfied(scene, candidate, constraint, reachability_map=reachability_map)
        if not ok:
            failed_constraints.append(str(constraint.constraint_type))

    if fixed_hits and outside_area > 1e-4:
        theme = "collision_boundary"
        caption = "collision and boundary repaired"
    elif fixed_hits:
        theme = "collision"
        caption = f"collision with {fixed_hits[0]} removed"
    elif outside_area > 1e-4 and window_hits > 0:
        theme = "window_boundary"
        caption = "moved inside and cleared window zone"
    elif outside_area > 1e-4:
        theme = "boundary"
        caption = "moved fully inside room boundary"
    elif door_hits > 0:
        theme = "door_keepout"
        caption = "door keep-out clearance repaired"
    elif window_hits > 0:
        theme = "window_keepout"
        caption = "window clearance repaired"
    elif "against_wall" in failed_constraints:
        theme = "wall_alignment"
        caption = "better wall alignment"
    elif failed_constraints:
        theme = "semantic"
        caption = "better semantic constraint fit"
    else:
        theme = "valid"
        caption = "valid placement"

    return SceneDiagnostics(
        outside_area=outside_area,
        fixed_hits=fixed_hits,
        door_hits=door_hits,
        window_hits=window_hits,
        failed_constraints=failed_constraints,
        theme=theme,
        caption=caption,
    )


def _pick_representative_cases(rows: Sequence[CaseRow], n: int = 4) -> List[CaseRow]:
    selected: List[CaseRow] = []
    seen_themes: set[str] = set()
    seen_rooms: set[str] = set()
    seen_categories: set[str] = set()

    for row in rows:
        if row.before_metrics["cps"] != 0 or row.after_metrics["cps"] != 1:
            continue
        room = _room_label(row.case)
        category = _target_label(row.case)
        theme = row.before_diag.theme
        if theme not in seen_themes and room not in seen_rooms and category not in seen_categories:
            selected.append(row)
            seen_themes.add(theme)
            seen_rooms.add(room)
            seen_categories.add(category)
        if len(selected) >= n:
            return selected

    for row in rows:
        if row.before_metrics["cps"] != 0 or row.after_metrics["cps"] != 1:
            continue
        if row.case_id not in {item.case_id for item in selected}:
            selected.append(row)
        if len(selected) >= n:
            break
    return selected


def _evaluate_cases() -> Tuple[List[CaseRow], List[CaseRow]]:
    cases = {case["id"]: case for case in _load_json(BENCHMARK_CASES)}
    before = _load_json(BEFORE_PREDICTIONS)["proposal_diffopt_constraint"]
    after = _load_json(AFTER_PREDICTIONS)["proposal_diffopt_constraint"]
    heur = _load_json(AFTER_PREDICTIONS)["proposal_heuristic"]

    rows: List[CaseRow] = []
    for case_id, case in cases.items():
        if case_id not in before or case_id not in after or case_id not in heur:
            continue

        before_scene = normalize_case_result(case, "before", before[case_id])
        after_scene = normalize_case_result(case, "after", after[case_id])
        heur_scene = normalize_case_result(case, "heur", heur[case_id])
        if not before_scene.candidates or not after_scene.candidates or not heur_scene.candidates:
            continue

        before_metrics = _compute_candidate_metrics(before_scene, before_scene.candidates[0])
        after_metrics = _compute_candidate_metrics(after_scene, after_scene.candidates[0])
        heur_metrics = _compute_candidate_metrics(heur_scene, heur_scene.candidates[0])
        before_prediction = before[case_id][0]
        after_prediction = after[case_id][0]
        heur_prediction = heur[case_id][0]
        before_diag = _scene_diagnostics(case, before_prediction)
        after_diag = _scene_diagnostics(case, after_prediction)
        move_distance = _center_distance(before_prediction, after_prediction)
        yaw_delta = _angle_delta(float(before_prediction.get("rotation_y", 0.0)), float(after_prediction.get("rotation_y", 0.0)))
        score = (
            4.0 * (after_metrics["cps"] - before_metrics["cps"])
            + 2.0 * (after_metrics["cf"] - before_metrics["cf"])
            + 1.5 * (after_metrics["ib"] - before_metrics["ib"])
            + 1.2 * (after_metrics["constraint_accuracy"] - before_metrics["constraint_accuracy"])
            + 0.25 * min(move_distance, 3.0)
            + 0.003 * yaw_delta
        )
        rows.append(
            CaseRow(
                case_id=case_id,
                case=case,
                before_prediction=before_prediction,
                after_prediction=after_prediction,
                heur_prediction=heur_prediction,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                heur_metrics=heur_metrics,
                before_diag=before_diag,
                after_diag=after_diag,
                score=score,
                move_distance=move_distance,
                yaw_delta=yaw_delta,
            )
        )

    rows.sort(key=lambda item: item.score, reverse=True)
    selected = _pick_representative_cases(rows, n=4)
    return rows, selected


def _keep_window_clear(case: Mapping[str, Any]) -> bool:
    return any(item.get("constraint_type") == "keep_window_clear" for item in case["intent"]["constraints"])


def _draw_target(ax: plt.Axes, item: Mapping[str, Any], kind: str, *, alpha: float = 0.95, linestyle: str = "-", zorder: int = 8) -> None:
    colors = _style_target(kind)
    poly = _object_polygon(item)
    ax.add_patch(
        MplPolygon(
            poly,
            closed=True,
            facecolor=colors["face"],
            edgecolor=colors["edge"],
            linewidth=1.7,
            alpha=alpha,
            linestyle=linestyle,
            zorder=zorder,
        )
    )
    cx, cz = _object_center(item)
    size = item["size"]
    w = float(size[0]) if not isinstance(size, dict) else float(size["width"])
    d = float(size[2]) if not isinstance(size, dict) else float(size["depth"])
    yaw = float(item.get("yaw", item.get("rotation_y", 0.0)))
    arrow_len = max(0.28, 0.28 * max(w, d))
    ax.add_patch(
        FancyArrowPatch(
            (cx, cz),
            (cx + arrow_len * math.cos(math.radians(yaw)), cz + arrow_len * math.sin(math.radians(yaw))),
            arrowstyle="-|>",
            mutation_scale=12,
            color=colors["arrow"],
            linewidth=1.2,
            zorder=zorder + 1,
        )
    )


def _draw_scene(
    ax: plt.Axes,
    case: Mapping[str, Any],
    *,
    target: Optional[Mapping[str, Any]] = None,
    target_kind: str = "after",
    overlay_targets: Optional[Sequence[Tuple[Mapping[str, Any], str, float, str]]] = None,
    title: str = "",
    metrics: Optional[Mapping[str, Any]] = None,
    show_keepouts: bool = True,
    subtitle: Optional[str] = None,
) -> None:
    scene = case["scene"]
    floor = scene["floor"]["polygon"]
    xs = [float(p[0]) for p in floor]
    zs = [float(p[1]) for p in floor]
    xmin, xmax = min(xs), max(xs)
    zmin, zmax = min(zs), max(zs)
    margin = max(xmax - xmin, zmax - zmin) * 0.06 + 0.08
    ax.set_aspect("equal")

    ax.add_patch(
        MplPolygon(
            [(float(x), float(z)) for x, z in floor],
            closed=True,
            facecolor="#FCFBF8",
            edgecolor="#2F2F2F",
            linewidth=1.9,
            zorder=0,
        )
    )

    if show_keepouts:
        for door in scene.get("doors", []):
            keepout = _opening_keepout_polygon(
                {"segment": door.get("segment"), "position": door["position"], "width": door.get("width", 0.9)},
                depth=0.45,
            )
            coords = list(keepout.exterior.coords)
            ax.add_patch(MplPolygon(coords, closed=True, facecolor="#FDE2DE", edgecolor="#D96C66", linewidth=0.8, alpha=0.65, zorder=1))

        if _keep_window_clear(case):
            for window in scene.get("windows", []):
                keepout = _opening_keepout_polygon(
                    {"segment": window.get("segment"), "position": window["position"], "width": window.get("width", 1.0)},
                    depth=0.30,
                )
                coords = list(keepout.exterior.coords)
                ax.add_patch(MplPolygon(coords, closed=True, facecolor="#DFF3FB", edgecolor="#56A9C7", linewidth=0.8, alpha=0.7, zorder=1))

    anchor_categories = _primary_anchor_categories(case)
    for obj in scene["objects"]:
        poly = _object_polygon(obj)
        category = _slug(obj["category"])
        face = "#CCD6DF"
        edge = "#6F7D87"
        if category in anchor_categories:
            face = "#BDD7A7"
            edge = "#5A7F3A"
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=face, edgecolor=edge, linewidth=1.0, alpha=0.94, zorder=2))
        cx, cz = _object_center(obj)
        ax.text(cx, cz, textwrap.shorten(str(obj["category"]).replace("_", " "), width=11, placeholder=""), ha="center", va="center", fontsize=6.3, color="#253240", zorder=3)

    for door in scene.get("doors", []):
        dx = float(door["position"][0])
        dz = float(door["position"][2])
        ax.scatter([dx], [dz], s=28, c="#C83D3D", marker="s", zorder=4)

    for window in scene.get("windows", []):
        wx = float(window["position"][0])
        wz = float(window["position"][2])
        ax.scatter([wx], [wz], s=28, c="#2B91B5", marker="s", zorder=4)

    if overlay_targets:
        for item, kind, alpha, linestyle in overlay_targets:
            _draw_target(ax, item, kind, alpha=alpha, linestyle=linestyle, zorder=7)

    if target is not None:
        _draw_target(ax, target, target_kind, alpha=0.95, linestyle="-", zorder=8)

    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(zmin - margin, zmax + margin)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10.8, pad=8)
    if subtitle:
        ax.text(0.5, -0.08, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=7.4, color="#555555")
    if metrics:
        summary = [
            f"CF {metrics['cf']:.0f}",
            f"IB {metrics['ib']:.0f}",
            f"CA {metrics['constraint_accuracy']:.2f}",
            f"CPS {int(metrics['cps'])}",
        ]
        ax.text(
            0.02,
            0.02,
            " | ".join(summary),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.9,
            color="#2E2E2E",
            bbox={"facecolor": "white", "edgecolor": "#C7C7C7", "alpha": 0.9, "boxstyle": "round,pad=0.22"},
        )


def _save_variants(fig: plt.Figure, stem: str) -> List[str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created: List[str] = []
    for ext in ["png", "svg", "pdf"]:
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=220 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        created.append(str(path))
    plt.close(fig)
    return created


def _metrics_delta_text(before_metrics: Mapping[str, Any], after_metrics: Mapping[str, Any]) -> List[str]:
    parts = []
    for key, label in [("cf", "CF"), ("ib", "IB"), ("constraint_accuracy", "CA"), ("cps", "CPS")]:
        delta = float(after_metrics[key]) - float(before_metrics[key])
        parts.append(f"{label} {delta:+.2f}")
    return parts


def _before_after_caption(row: CaseRow) -> str:
    before = row.before_diag
    after = row.after_diag
    labels: List[str] = []
    if before.fixed_hits and not after.fixed_hits:
        labels.append("collision removed")
    if before.outside_area > 1e-4 and after.outside_area <= 1e-4:
        labels.append("inside boundary")
    if before.window_hits > 0 and after.window_hits == 0:
        labels.append("window clear")
    if before.door_hits > 0 and after.door_hits == 0:
        labels.append("door clear")
    if row.after_metrics["constraint_accuracy"] > row.before_metrics["constraint_accuracy"]:
        labels.append("better semantic fit")
    if not labels:
        labels.append(after.caption)
    return " | ".join(labels[:3])


def make_before_after_cases(rows: Sequence[CaseRow]) -> List[str]:
    created: List[str] = []
    for idx, row in enumerate(rows, start=1):
        fig = plt.figure(figsize=(17.4, 4.9))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.72], wspace=0.16)
        ax0 = fig.add_subplot(gs[0, 0])
        _draw_scene(ax0, row.case, title="Input room", subtitle="target removed")

        ax1 = fig.add_subplot(gs[0, 1])
        _draw_scene(
            ax1,
            row.case,
            target=_prediction_to_object(row.case, row.before_prediction),
            target_kind="before",
            title="Before improvement",
            metrics=row.before_metrics,
            subtitle=row.before_diag.caption,
        )

        ax2 = fig.add_subplot(gs[0, 2])
        _draw_scene(
            ax2,
            row.case,
            target=_prediction_to_object(row.case, row.after_prediction),
            target_kind="after",
            title="After improvement",
            metrics=row.after_metrics,
            subtitle=_before_after_caption(row),
        )

        ax3 = fig.add_subplot(gs[0, 3])
        ax3.set_axis_off()
        ax3.add_patch(
            FancyBboxPatch(
                (0.03, 0.05),
                0.94,
                0.90,
                boxstyle="round,pad=0.02,rounding_size=0.03",
                facecolor="#FBF6ED",
                edgecolor="#D5B27C",
                linewidth=1.5,
                transform=ax3.transAxes,
            )
        )
        ax3.text(0.08, 0.90, f"Case {idx:02d}", transform=ax3.transAxes, fontsize=12, weight="bold")
        ax3.text(0.08, 0.83, _target_label(row.case).title(), transform=ax3.transAxes, fontsize=10.5)
        ax3.text(0.08, 0.77, _room_label(row.case).title(), transform=ax3.transAxes, fontsize=9.3, color="#555555")
        ax3.text(0.08, 0.66, textwrap.fill(row.case["intent"]["text"], width=27), transform=ax3.transAxes, fontsize=8.5, va="top")
        ax3.text(0.08, 0.38, "What improved", transform=ax3.transAxes, fontsize=9.0, weight="bold")
        ax3.text(0.08, 0.32, textwrap.fill(_before_after_caption(row), width=27), transform=ax3.transAxes, fontsize=8.6, color="#0B5E8E")
        ax3.text(0.08, 0.22, "Metric delta", transform=ax3.transAxes, fontsize=9.0, weight="bold")
        ax3.text(
            0.08,
            0.16,
            "\n".join(_metrics_delta_text(row.before_metrics, row.after_metrics)),
            transform=ax3.transAxes,
            fontsize=7.9,
            color="#333333",
            va="top",
            linespacing=1.0,
        )
        fig.suptitle(f"Experiment A Before vs After | Case {idx:02d}", fontsize=14.2, y=0.99)
        created.extend(_save_variants(fig, f"expA_before_after_case_{idx:02d}"))
    return created


def make_failure_to_success(rows: Sequence[CaseRow]) -> List[str]:
    fig = plt.figure(figsize=(14.8, 10.0))
    gs = gridspec.GridSpec(2, 2, hspace=0.22, wspace=0.14)
    for ax, row in zip([fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)], rows):
        before_obj = _prediction_to_object(row.case, row.before_prediction)
        after_obj = _prediction_to_object(row.case, row.after_prediction)
        _draw_scene(
            ax,
            row.case,
            target=after_obj,
            target_kind="after",
            overlay_targets=[(before_obj, "before", 0.30, "--")],
            title=f"{_target_label(row.case).title()} | {_room_label(row.case)}",
            subtitle=row.before_diag.caption,
        )
        ax.text(
            0.02,
            0.98,
            "before: orange dashed   after: blue",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            color="#444444",
            bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.92, "boxstyle": "round,pad=0.18"},
        )
    fig.suptitle("Experiment A Failure-to-Success Cases", fontsize=15.0, y=0.98)
    return _save_variants(fig, "expA_failure_to_success")


def _draw_breakdown_card(ax: plt.Axes, title: str, body: str, accent: str, step: int) -> None:
    ax.set_axis_off()
    ax.add_patch(
        FancyBboxPatch(
            (0.03, 0.05),
            0.94,
            0.90,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor="#FBFBFB",
            edgecolor=accent,
            linewidth=2.0,
            transform=ax.transAxes,
        )
    )
    ax.text(0.08, 0.88, f"{step}. {title}", transform=ax.transAxes, fontsize=11.5, weight="bold", color=accent)
    ax.text(0.08, 0.71, textwrap.fill(body, width=23), transform=ax.transAxes, fontsize=8.9, va="top")

    room = Rectangle((0.12, 0.16), 0.72, 0.30, transform=ax.transAxes, facecolor="#F7F3EB", edgecolor="#333333", linewidth=1.3)
    ax.add_patch(room)
    ax.add_patch(Rectangle((0.19, 0.25), 0.18, 0.10, transform=ax.transAxes, facecolor="#CAD6DF", edgecolor="#6A7B8C", linewidth=1.0))
    ax.add_patch(Rectangle((0.62, 0.18), 0.12, 0.10, transform=ax.transAxes, facecolor="#FDE2DE", edgecolor="#D96C66", linewidth=1.0, alpha=0.85))

    if step == 1:
        ax.add_patch(Rectangle((0.28, 0.29), 0.18, 0.08, transform=ax.transAxes, facecolor="#F4A261", edgecolor="#C2410C", linewidth=1.4, alpha=0.65))
        ax.add_patch(Rectangle((0.49, 0.29), 0.18, 0.08, transform=ax.transAxes, facecolor="#5BA7D1", edgecolor="#0B5E8E", linewidth=1.6))
        ax.annotate("", xy=(0.50, 0.52), xytext=(0.28, 0.52), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "color": accent, "lw": 1.4})
    elif step == 2:
        ax.add_patch(Rectangle((0.43, 0.22), 0.22, 0.10, transform=ax.transAxes, facecolor="#F4A261", edgecolor="#C2410C", linewidth=1.4))
        ax.annotate("20", xy=(0.47, 0.36), xytext=(0.34, 0.58), xycoords=ax.transAxes, textcoords=ax.transAxes, fontsize=8.8, color="#A84B09", arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#A84B09"})
        ax.annotate("40", xy=(0.62, 0.33), xytext=(0.70, 0.58), xycoords=ax.transAxes, textcoords=ax.transAxes, fontsize=8.8, color="#0B5E8E", arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#0B5E8E"})
    elif step == 3:
        ax.add_patch(Rectangle((0.25, 0.24), 0.18, 0.08, transform=ax.transAxes, facecolor="#F4A261", edgecolor="#C2410C", linewidth=1.4))
        ax.add_patch(Rectangle((0.40, 0.30), 0.18, 0.08, transform=ax.transAxes, facecolor="#5BA7D1", edgecolor="#0B5E8E", linewidth=1.4))
        ax.annotate("", xy=(0.42, 0.35), xytext=(0.34, 0.29), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "color": accent, "lw": 1.4})
    else:
        ax.add_patch(Rectangle((0.60, 0.18), 0.12, 0.10, transform=ax.transAxes, facecolor="#FDE2DE", edgecolor="#D96C66", linewidth=1.0, alpha=0.85))
        ax.add_patch(Rectangle((0.58, 0.20), 0.18, 0.08, transform=ax.transAxes, facecolor="#F4A261", edgecolor="#C2410C", linewidth=1.4, alpha=0.7))
        ax.add_patch(Rectangle((0.39, 0.25), 0.18, 0.08, transform=ax.transAxes, facecolor="#5BA7D1", edgecolor="#0B5E8E", linewidth=1.5))
        ax.annotate("", xy=(0.49, 0.40), xytext=(0.67, 0.30), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "color": accent, "lw": 1.4})


def make_improvement_breakdown() -> List[str]:
    fig = plt.figure(figsize=(17.2, 4.4))
    gs = gridspec.GridSpec(1, 4, wspace=0.12)
    cards = [
        ("Heuristic init", "Restart 0 begins from a heuristic pose near the likely semantic anchor instead of wasting one restart on a poor seed.", "#B16A2B"),
        ("Physics 20 -> 40", "A stronger physics term penalizes collisions and out-of-room placements more aggressively during refinement.", "#C04C40"),
        ("Local regularizer", "A local anchor term discourages the optimizer from drifting away from a good nearby placement region.", "#4D8F6B"),
        ("Post-snap repair", "After snapping to a clean orientation or wall alignment, a lightweight repair step restores feasibility if the snap introduced clipping.", "#3D6FA3"),
    ]
    for idx, (title, body, accent) in enumerate(cards, start=1):
        ax = fig.add_subplot(gs[0, idx - 1])
        _draw_breakdown_card(ax, title, body, accent, idx)
    fig.suptitle("Experiment A: What changed in the optimization pipeline", fontsize=15.0, y=1.02)
    return _save_variants(fig, "expA_improvement_breakdown")


def make_best_success_cases(rows: Sequence[CaseRow]) -> List[str]:
    fig = plt.figure(figsize=(16.8, 8.6))
    gs = gridspec.GridSpec(len(rows), 4, width_ratios=[0.62, 1.0, 1.0, 1.0], hspace=0.24, wspace=0.16)
    for row_idx, row in enumerate(rows):
        ax_label = fig.add_subplot(gs[row_idx, 0])
        ax_label.set_axis_off()
        ax_label.text(0.0, 0.86, _target_label(row.case).title(), fontsize=12.0, weight="bold")
        ax_label.text(0.0, 0.70, _room_label(row.case).title(), fontsize=10.0, color="#555555")
        ax_label.text(0.0, 0.54, textwrap.fill(row.case["intent"]["text"], width=28), fontsize=8.4, va="top")
        ax_label.text(
            0.0,
            0.10,
            textwrap.fill("Selected because Proposal + Heuristic still misses a constraint, while the improved DiffOpt result is valid and visually cleaner.", width=30),
            fontsize=8.1,
            color="#0B5E8E",
        )

        ax0 = fig.add_subplot(gs[row_idx, 1])
        _draw_scene(ax0, row.case, title="Input room", subtitle="target removed")

        ax1 = fig.add_subplot(gs[row_idx, 2])
        _draw_scene(
            ax1,
            row.case,
            target=_prediction_to_object(row.case, row.heur_prediction),
            target_kind="heur",
            title="Proposal + Heuristic",
            metrics=row.heur_metrics,
            subtitle="still misses a constraint",
        )

        ax2 = fig.add_subplot(gs[row_idx, 3])
        _draw_scene(
            ax2,
            row.case,
            target=_prediction_to_object(row.case, row.after_prediction),
            target_kind="after",
            title="Improved DiffOpt-Constraint",
            metrics=row.after_metrics,
            subtitle=row.after_diag.caption,
        )

    fig.suptitle("Best Qualitative Success Cases", fontsize=15.0, y=0.99)
    return _save_variants(fig, "expA_best_success_cases")


def _write_manifest(selected: Sequence[CaseRow], best: Sequence[CaseRow]) -> str:
    payload = {
        "before_source": str(BEFORE_PREDICTIONS),
        "after_source": str(AFTER_PREDICTIONS),
        "before_meaning": "Pre-improvement Proposal + DiffOpt-Constraint snapshot from single_target_paper_v1.",
        "after_meaning": "Improved Proposal + DiffOpt-Constraint snapshot from exp_a.",
        "before_after_cases": [
            {
                "figure": f"expA_before_after_case_{idx:02d}",
                "case_id": row.case_id,
                "target_category": _target_label(row.case),
                "room_type": _room_label(row.case),
                "why_selected": _before_after_caption(row),
                "score": row.score,
            }
            for idx, row in enumerate(selected, start=1)
        ],
        "best_success_cases": [
            {
                "case_id": row.case_id,
                "target_category": _target_label(row.case),
                "room_type": _room_label(row.case),
                "why_selected": row.after_diag.caption,
            }
            for row in best
        ],
    }
    path = OUT_DIR / "expA_case_manifest.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return str(path)


def _write_index(created_files: Sequence[str], selected: Sequence[CaseRow], best: Sequence[CaseRow], manifest_path: str) -> str:
    lines = [
        "# Experiment A Qualitative Visualization Index",
        "",
        f"Output directory: `{OUT_DIR}`",
        "",
        "## Provenance",
        "",
        "- Exact stored outputs were reused for all method poses.",
        f"- `Before improvement` = `{BEFORE_PREDICTIONS}` (`single_target_paper_v1`, Proposal + DiffOpt-Constraint).",
        f"- `After improvement` = `{AFTER_PREDICTIONS}` (`exp_a`, Proposal + DiffOpt-Constraint).",
        "- `Proposal + Heuristic Refinement` comparisons also come from the exact saved `exp_a/raw_predictions.json` bundle.",
        "- No benchmark-wide rerun was performed.",
        "- The visual layouts are reconstructed top-down renderings from exact benchmark geometry plus exact stored predictions.",
        "",
        "## Figures",
        "",
    ]

    figure_notes = {
        "expA_before_after_case_01": "Three-panel input / before / after comparison for representative case 01.",
        "expA_before_after_case_02": "Three-panel input / before / after comparison for representative case 02.",
        "expA_before_after_case_03": "Three-panel input / before / after comparison for representative case 03.",
        "expA_before_after_case_04": "Three-panel input / before / after comparison for representative case 04.",
        "expA_failure_to_success": "Compact overlay summary showing before failure (orange dashed) versus after success (blue).",
        "expA_improvement_breakdown": "Conceptual mini-panel explaining the four optimization modifications in Experiment A.",
        "expA_best_success_cases": "Two best highlight cases comparing Proposal + Heuristic against the improved DiffOpt result.",
        "expA_case_manifest": "JSON manifest recording selected case IDs and rationale.",
    }

    for path in sorted(created_files):
        name = Path(path).name
        stem = Path(path).stem
        if stem in figure_notes:
            lines.append(f"- `{name}`: {figure_notes[stem]}")
    lines.append(f"- `{Path(manifest_path).name}`: {figure_notes['expA_case_manifest']}")

    lines.extend(
        [
            "",
            "## Selected Before/After Cases",
            "",
        ]
    )
    for idx, row in enumerate(selected, start=1):
        lines.append(
            f"- `Case {idx:02d}`: `{row.case_id}` | {_target_label(row.case)} | {_room_label(row.case)} | selected for `{_before_after_caption(row)}`."
        )

    lines.extend(
        [
            "",
            "## Best Highlight Cases",
            "",
        ]
    )
    for row in best:
        lines.append(
            f"- `{row.case_id}` | {_target_label(row.case)} | {_room_label(row.case)} | improved DiffOpt succeeds while Proposal + Heuristic still misses at least one constraint."
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- For the PPT slide `Performance optimization process / systematic improvement (Experiment A)`, use `expA_failure_to_success.png` as the main visual and pair it with the stored metric delta numbers.",
            "- Use `expA_improvement_breakdown.png` as the supporting explanatory panel if the slide has enough width or if you split the story across two slides.",
        ]
    )

    index_path = OUT_DIR / "expA_visualization_index.md"
    with open(index_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return str(index_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, selected = _evaluate_cases()
    best = [row for row in rows if row.after_metrics["cps"] == 1 and row.heur_metrics["cps"] == 0][:2]

    created: List[str] = []
    created.extend(make_before_after_cases(selected))
    created.extend(make_failure_to_success(selected))
    created.extend(make_improvement_breakdown())
    created.extend(make_best_success_cases(best))
    manifest_path = _write_manifest(selected, best)
    index_path = _write_index(created, selected, best, manifest_path)

    print(json.dumps({"out_dir": str(OUT_DIR), "files": created + [manifest_path, index_path]}, indent=2))


if __name__ == "__main__":
    main()
