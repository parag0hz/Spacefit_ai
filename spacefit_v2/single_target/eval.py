"""Evaluation wrapper for the single-target benchmark using unified metrics."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from experiments.eval.unified_eval import (
    CandidateScene,
    NormalizedConstraint,
    NormalizedPlacement,
    NormalizedScene,
    _aggregate_method,
)


METHOD_LABELS = {
    "heuristic_baseline": "Heuristic Baseline",
    "proposal_heuristic": "Proposal + Heuristic Refinement",
    "proposal_diffopt_basic": "Proposal + DiffOpt-Basic",
    "proposal_diffopt_constraint": "Proposal + DiffOpt-Constraint",
    "proposal_llm_grounded_diffopt": "Proposal + LLM/VLM-Grounded DiffOpt",
    # GPT API methods
    "layoutgpt_direct": "LayoutGPT (Direct Coord)",
    "spacefit_gpt_text": "SpaceFit + GPT-Text",
    # Experiment B: no-proposal ablation
    "no_proposal_diffopt_basic": "No-Proposal + DiffOpt-Basic",
    "no_proposal_diffopt_constraint": "No-Proposal + DiffOpt-Constraint",
}


def _constraint_from_dict(item: Dict[str, Any]) -> NormalizedConstraint:
    return NormalizedConstraint(
        constraint_type=str(item.get("constraint_type")),
        subject_id=item.get("subject_id"),
        target_id=item.get("target_id"),
        target_category=item.get("target_category"),
        target_kind=item.get("target_kind"),
        min_distance=item.get("min_distance"),
        max_distance=item.get("max_distance"),
        angle_deg=item.get("angle_deg"),
        source="benchmark",
        description=str(item.get("description", "")),
    )


def _fixed_object(item: Dict[str, Any]) -> NormalizedPlacement:
    return NormalizedPlacement(
        asset_id=str(item["id"]),
        category=str(item["category"]),
        position=(float(item["position"][0]), float(item["position"][1]), float(item["position"][2])),
        yaw_deg=float(item.get("yaw", 0.0)),
        size=(float(item["size"][0]), float(item["size"][1]), float(item["size"][2])),
    )


def _predicted_object(item: Dict[str, Any]) -> NormalizedPlacement:
    pos = item.get("position") or {"x": 0.0, "y": 0.0, "z": 0.0}
    size = item.get("size") or {"width": 0.0, "height": 0.0, "depth": 0.0}
    return NormalizedPlacement(
        asset_id=str(item.get("furniture_id", "target")),
        category=str(item.get("category", "unknown")),
        position=(float(pos["x"]), float(pos.get("y", 0.0)), float(pos["z"])),
        yaw_deg=float(item.get("rotation_y", 0.0)),
        size=(float(size["width"]), float(size.get("height", 0.0)), float(size["depth"])),
        status=str(item.get("status", "placed")),
        metadata={"region_id": item.get("region_id"), "score": item.get("score")},
    )


def normalize_case_result(case: Dict[str, Any], method: str, predictions: Sequence[Dict[str, Any]]) -> NormalizedScene:
    scene = case["scene"]
    doors = []
    for entry in scene.get("doors", []):
        doors.append(
            {
                "center": (float(entry["position"][0]), float(entry["position"][2])),
                "width": float(entry.get("width", 0.9)),
                "segment": entry.get("segment"),
                "position": entry["position"],
            }
        )
    windows = []
    for entry in scene.get("windows", []):
        windows.append(
            {
                "center": (float(entry["position"][0]), float(entry["position"][2])),
                "width": float(entry.get("width", 1.0)),
                "segment": entry.get("segment"),
                "position": entry["position"],
            }
        )

    candidates = []
    for rank, prediction in enumerate(predictions[:5]):
        placements = []
        if prediction.get("status") == "placed":
            placements.append(_predicted_object(prediction))
        candidates.append(CandidateScene(rank=rank, placements=placements, metadata={"raw_prediction": prediction}))

    entrance = scene.get("entrance_point")
    return NormalizedScene(
        scene_id=str(case["id"]),
        query_id=str(case["id"]),
        method=method,
        prompt=str(case["intent"].get("text", "")),
        room_type=str(scene.get("room_type")),
        floor_polygon=[(float(x), float(z)) for x, z in scene["floor"]["polygon"]],
        fixed_objects=[_fixed_object(item) for item in scene.get("objects", [])],
        doors=doors,
        windows=windows,
        constraints=[_constraint_from_dict(item) for item in case["intent"].get("constraints", [])],
        candidates=candidates,
        expected_asset_count=1,
        entrance_point=tuple(entrance) if entrance is not None else None,
        metadata={"case_id": case["id"], "method": method},
    )


def aggregate_results(
    benchmark_cases: Sequence[Dict[str, Any]],
    predictions_by_method: Dict[str, Dict[str, Sequence[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    aggregates = []
    for method, by_case in predictions_by_method.items():
        scenes = [normalize_case_result(case, method, by_case.get(case["id"], [])) for case in benchmark_cases]
        aggregates.append(
            _aggregate_method(
                method=METHOD_LABELS.get(method, method),
                scenes=scenes,
                source_name=method,
                notes="Single-target benchmark evaluated with the repo's unified code-based metrics.",
            )
        )
    return aggregates


def save_results(out_dir: str | Path, benchmark_manifest: Dict[str, Any], aggregates: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_manifest": benchmark_manifest,
        "methods": list(aggregates),
    }
    json_path = out_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    csv_path = out_path / "results.csv"
    fieldnames = ["method", "CF", "IB", "Constraint Accuracy", "CPS", "Success@1", "Success@5", "Reachability", "Walkability"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregates:
            metrics = row["metrics"]
            writer.writerow({"method": row["method"], **metrics})

    table_path = out_path / "COMPARISON_TABLE.md"
    lines = [
        "| Method | CF ↑ | IB ↑ | Constraint Accuracy ↑ | CPS ↑ | Success@1 ↑ | Success@5 ↑ | Reachability ↑ | Walkability ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        m = row["metrics"]
        lines.append(
            f"| {row['method']} | {m['CF']:.3f} | {m['IB']:.3f} | {m['Constraint Accuracy']:.3f} | "
            f"{m['CPS']:.3f} | {m['Success@1']:.3f} | {m['Success@5']:.3f} | {m['Reachability']:.3f} | {m['Walkability']:.3f} |"
        )
    with open(table_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    protocol_path = out_path / "PROTOCOL.md"
    protocol_lines = [
        "# Single-Target Evaluation Protocol",
        "",
        "- Main metrics reuse the repository's unified code-based evaluation logic from `experiments/eval/unified_eval.py`.",
        "- `CF`: collision-free rate against fixed furniture and door keep-out zones.",
        "- `IB`: in-boundary rate inside the room polygon.",
        "- `Constraint Accuracy`: fraction of normalized structured constraints satisfied.",
        "- `CPS`: strict scene success requiring `CF=1`, `IB=1`, all constraints satisfied, and reachability=1.",
        "- `Success@1`: whether the top-ranked prediction achieves `CPS=1`.",
        "- `Success@5`: whether any of the top 5 ranked predictions achieves `CPS=1`.",
        "- `Reachability` and `Walkability` come from the same rasterized connectivity map used in the unified evaluator.",
        "",
        "## Benchmark Notes",
        "",
        "- Each task removes exactly one eligible furniture item from a furnished 3D-FRONT room.",
        "- The remaining furniture stays fixed.",
        "- The removed object becomes the target asset to place.",
        "- Reference pose is kept only for analysis and constraint templating, not as the headline metric.",
    ]
    with open(protocol_path, "w") as f:
        f.write("\n".join(protocol_lines) + "\n")

    return {
        "results_json": str(json_path),
        "results_csv": str(csv_path),
        "comparison_table": str(table_path),
        "protocol": str(protocol_path),
    }

