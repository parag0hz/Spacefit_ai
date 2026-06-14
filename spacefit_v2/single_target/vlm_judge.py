"""VLM-as-judge: render a top-down placement image and ask GPT-4V to evaluate it.

Score: 0-10 (how well the placement satisfies the user's intent).
"""
from __future__ import annotations

import base64
import io
import json
import math
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ── rendering ─────────────────────────────────────────────────────────────────

def _rotated_corners(cx, cz, w, d, yaw_deg):
    hw, hd = max(w, 0.05) / 2, max(d, 0.05) / 2
    rad = math.radians(yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    return [(cx + dx * c - dz * s, cz + dx * s + dz * c)
            for dx, dz in [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]]


def _render_placement_png(case: Dict[str, Any], prediction: Dict[str, Any]) -> bytes:
    """Render a top-down room view with the predicted placement as PNG bytes."""
    fig, ax = plt.subplots(figsize=(5, 5))
    scene = case["scene"]

    # Floor
    floor_pts = [(float(x), float(z)) for x, z in scene["floor"]["polygon"]]
    ax.add_patch(mpatches.Polygon(floor_pts, closed=True,
                                  facecolor="#F5F5EE", edgecolor="#444", lw=1.5, zorder=0))

    # Fixed furniture
    for obj in scene.get("objects", []):
        px, _, pz = obj["position"]
        w, _, d = obj["size"]
        pts = _rotated_corners(float(px), float(pz), float(w), float(d), float(obj.get("yaw", 0)))
        ax.add_patch(mpatches.Polygon(pts, closed=True, facecolor="#AAAAAA",
                                      edgecolor="#777", lw=0.8, alpha=0.85, zorder=1))
        cat = str(obj.get("category", "")).replace("_", " ")[:8]
        ax.text(float(px), float(pz), cat, fontsize=4, ha="center", va="center", color="#333", zorder=3)

    # Doors (brown squares)
    for door in scene.get("doors", []):
        p = door.get("position", [0, 0, 0])
        ax.plot(float(p[0]), float(p[2]), "s", color="#8B4513", ms=7, zorder=2)

    # Windows (blue diamonds)
    for win in scene.get("windows", []):
        p = win.get("position", [0, 0, 0])
        ax.plot(float(p[0]), float(p[2]), "D", color="#5DADE2", ms=5, zorder=2)

    # Target placement
    pos = prediction.get("position")
    size = prediction.get("size")
    if pos and size and prediction.get("status") == "placed":
        px, pz = float(pos["x"]), float(pos["z"])
        pw, pd = float(size["width"]), float(size["depth"])
        yaw = float(prediction.get("rotation_y", 0))
        pts = _rotated_corners(px, pz, pw, pd, yaw)
        ax.add_patch(mpatches.Polygon(pts, closed=True, facecolor="#2ECC71",
                                      edgecolor="white", lw=2, alpha=0.95, zorder=4))
        # Direction arrow
        rad = math.radians(yaw)
        length = min(pw, pd) * 0.4
        ax.annotate("", xy=(px + math.sin(rad) * length, pz + math.cos(rad) * length),
                    xytext=(px, pz),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.5), zorder=5)
        cat = str(case["target_asset"]["category"]).replace("_", " ")
        ax.text(px, pz, cat[:10], fontsize=5, ha="center", va="center",
                color="white", fontweight="bold", zorder=6)
    else:
        ax.text(0.5, 0.5, "NOT PLACED", transform=ax.transAxes,
                ha="center", va="center", color="red", fontsize=12)

    xs, zs = zip(*floor_pts)
    margin = 0.5
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(zs) - margin, max(zs) + margin)
    ax.set_aspect("equal")
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ── VLM judge ─────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are an expert evaluator of indoor furniture placement.
You will see a top-down floor plan with:
- Gray rectangles: existing furniture (fixed)
- Green rectangle with arrow: the newly placed item (arrow = facing direction)
- Brown squares: doors, Blue diamonds: windows

Evaluate how well the placement satisfies the user's intent.

Respond with ONLY valid JSON (no markdown):
{
  "score": <integer 0-10>,
  "physical_valid": <true|false>,
  "intent_satisfied": <true|false>,
  "reasoning": "<1-2 sentence explanation>"
}

Scoring guide:
- 9-10: Perfect placement, clearly satisfies the intent
- 7-8: Good placement, mostly satisfies the intent
- 5-6: Acceptable but missing some aspects of the intent
- 3-4: Poor placement, partially wrong
- 0-2: Completely wrong (collision, out-of-bounds, or ignores intent)
"""


def judge_placement(
    case: Dict[str, Any],
    prediction: Dict[str, Any],
    openai_client: Any,
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """Ask GPT-4V to score a placement against the user's intent.

    Returns dict with: score (0-10), physical_valid, intent_satisfied, reasoning, error.
    """
    intent_text = case.get("intent", {}).get("text", "")
    target_cat = str(case["target_asset"]["category"]).replace("_", " ")

    if prediction.get("status") != "placed":
        return {
            "score": 0,
            "physical_valid": False,
            "intent_satisfied": False,
            "reasoning": "Item was not placed.",
            "error": None,
        }

    try:
        # Pre-compute physical validity for the prompt
        from spacefit_v2.scripts.visualize_results import _check_placement
        cf, ib = _check_placement(prediction, case)
        physics_note = ""
        if not cf:
            physics_note = "\n⚠️ NOTE: This placement has a COLLISION with existing furniture."
        elif not ib:
            physics_note = "\n⚠️ NOTE: This placement is PARTIALLY OUT OF THE ROOM BOUNDARY."

        png_bytes = _render_placement_png(case, prediction)
        b64 = base64.b64encode(png_bytes).decode("utf-8")

        user_content = [
            {
                "type": "text",
                "text": (
                    f"Target item: {target_cat}\n"
                    f"User's intent: \"{intent_text}\"{physics_note}\n\n"
                    "Please evaluate the placement shown in the floor plan image."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
            },
        ]

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        import re
        content = re.sub(r"^```[a-z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
        result = json.loads(content)
        result["error"] = None
        return result

    except Exception as exc:
        return {
            "score": -1,
            "physical_valid": None,
            "intent_satisfied": None,
            "reasoning": "",
            "error": str(exc),
        }


def judge_all_methods(
    cases: List[Dict[str, Any]],
    predictions_by_method: Dict[str, Dict[str, List[Dict[str, Any]]]],
    openai_client: Any,
    model: str = "gpt-4o",
    methods: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run VLM judge for all methods × cases. Returns {method: {case_id: judgment}}."""
    if methods is None:
        methods = list(predictions_by_method.keys())

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for method in methods:
        results[method] = {}
        for idx, case in enumerate(cases, 1):
            cid = case["id"]
            preds = predictions_by_method.get(method, {}).get(cid, [])
            top_pred = preds[0] if preds else {"status": "error"}
            judgment = judge_placement(case, top_pred, openai_client, model)
            results[method][cid] = judgment
            if verbose:
                score = judgment.get("score", -1)
                print(f"[vlm_judge|{method[:20]}] {idx}/{len(cases)} score={score}  {cid[:50]}")

    return results


def aggregate_vlm_scores(
    judgments: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, float]]:
    """Compute per-method mean VLM score, physical_valid rate, intent_satisfied rate."""
    summary = {}
    for method, by_case in judgments.items():
        scores = [v["score"] for v in by_case.values() if isinstance(v.get("score"), (int, float)) and v["score"] >= 0]
        phys = [v["physical_valid"] for v in by_case.values() if v.get("physical_valid") is not None]
        intent = [v["intent_satisfied"] for v in by_case.values() if v.get("intent_satisfied") is not None]
        summary[method] = {
            "vlm_score": sum(scores) / len(scores) if scores else 0.0,
            "vlm_physical_valid": sum(phys) / len(phys) if phys else 0.0,
            "vlm_intent_satisfied": sum(intent) / len(intent) if intent else 0.0,
            "n": len(scores),
        }
    return summary
