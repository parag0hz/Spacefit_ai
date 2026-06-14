"""Create qualitative comparison figures for x/z grid-step ablation.

The figure compares the same single-target placement cases under different
pose-search grid steps. Existing furniture and room geometry are fixed; only
the selected target placement candidates differ.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


GRID_SPECS = [
    ("0.10 m", "010", "#2563eb"),
    ("0.18 m", "018", "#16a34a"),
    ("0.25 m", "025", "#f97316"),
]


def setup_korean_font() -> None:
    candidates = [
        "Malgun Gothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "NanumGothic",
        "AppleGothic",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for family in candidates:
        if family in available:
            matplotlib.rcParams["font.family"] = family
            break
    else:
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rotated_corners(cx: float, cz: float, w: float, d: float, yaw_deg: float) -> List[Tuple[float, float]]:
    hw, hd = max(w, 0.04) / 2.0, max(d, 0.04) / 2.0
    rad = math.radians(yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    local = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
    return [(cx + dx * c - dz * s, cz + dx * s + dz * c) for dx, dz in local]


def draw_rotated_box(
    ax,
    cx: float,
    cz: float,
    w: float,
    d: float,
    yaw_deg: float,
    face: str,
    edge: str,
    alpha: float = 0.9,
    lw: float = 1.2,
    linestyle: str = "-",
    zorder: int = 3,
    label: str | None = None,
) -> None:
    poly = patches.Polygon(
        rotated_corners(cx, cz, w, d, yaw_deg),
        closed=True,
        facecolor=face,
        edgecolor=edge,
        alpha=alpha,
        linewidth=lw,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(poly)
    if label:
        ax.text(cx, cz, label, ha="center", va="center", fontsize=6.5, color="#334155", zorder=zorder + 1)


def draw_yaw_arrow(ax, cx: float, cz: float, yaw_deg: float, length: float, color: str, zorder: int = 8) -> None:
    rad = math.radians(yaw_deg)
    dx, dz = math.cos(rad) * length, math.sin(rad) * length
    ax.annotate(
        "",
        xy=(cx + dx, cz + dz),
        xytext=(cx, cz),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, shrinkA=0, shrinkB=0),
        zorder=zorder,
    )


def draw_room(ax, case: Dict[str, Any]) -> None:
    scene = case["scene"]
    floor_pts = [(float(x), float(z)) for x, z in scene["floor"]["polygon"]]
    ax.add_patch(
        patches.Polygon(
            floor_pts,
            closed=True,
            facecolor="#f8fafc",
            edgecolor="#334155",
            linewidth=1.7,
            zorder=0,
        )
    )

    for obj in scene.get("objects", []):
        x, _, z = obj["position"]
        w, _, d = obj["size"]
        yaw = float(obj.get("yaw", 0.0))
        cat = str(obj.get("category", "obj")).replace("_", " ")
        draw_rotated_box(
            ax,
            float(x),
            float(z),
            float(w),
            float(d),
            yaw,
            face="#dbeafe",
            edge="#64748b",
            alpha=0.72,
            lw=1.0,
            label=cat[:10],
            zorder=2,
        )

    for door in scene.get("doors", []):
        pts = door.get("polygon") or []
        if len(pts) >= 2:
            xs = [float(p[0]) for p in pts]
            zs = [float(p[1]) for p in pts]
            ax.plot(xs, zs, color="#92400e", linewidth=3.0, solid_capstyle="round", zorder=4)
        else:
            p = door.get("position", [0, 0, 0])
            ax.scatter([float(p[0])], [float(p[2])], marker="s", s=30, color="#92400e", zorder=4)
        ax.text(0.02, 0.04, "Door", transform=ax.transAxes, color="#92400e", fontsize=7, weight="bold")

    for win in scene.get("windows", []):
        pts = win.get("polygon") or []
        if len(pts) >= 2:
            xs = [float(p[0]) for p in pts]
            zs = [float(p[1]) for p in pts]
            ax.plot(xs, zs, color="#0284c7", linewidth=3.0, solid_capstyle="round", zorder=4)
        else:
            p = win.get("position", [0, 0, 0])
            ax.scatter([float(p[0])], [float(p[2])], marker="D", s=24, color="#0284c7", zorder=4)

    ref = case.get("reference_pose")
    if ref:
        size = case["target_asset"]["size"]
        pos = ref["position"]
        draw_rotated_box(
            ax,
            float(pos["x"]),
            float(pos["z"]),
            float(size["width"]),
            float(size["depth"]),
            float(ref.get("rotation_y", 0.0)),
            face="none",
            edge="#eab308",
            alpha=1.0,
            lw=1.8,
            linestyle="--",
            zorder=5,
        )

    xs = [p[0] for p in floor_pts]
    zs = [p[1] for p in floor_pts]
    ax.set_xlim(min(xs) - 0.45, max(xs) + 0.45)
    ax.set_ylim(min(zs) - 0.45, max(zs) + 0.45)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_candidate(ax, pred: Dict[str, Any], color: str, rank: int, primary: bool) -> None:
    if pred.get("status") != "placed":
        return
    pos = pred.get("position") or {}
    size = pred.get("size") or {}
    x, z = float(pos["x"]), float(pos["z"])
    w, d = float(size["width"]), float(size["depth"])
    yaw = float(pred.get("rotation_y", 0.0))
    if primary:
        draw_rotated_box(ax, x, z, w, d, yaw, face=color, edge="#0f172a", alpha=0.86, lw=1.8, zorder=7)
        draw_yaw_arrow(ax, x, z, yaw, max(0.18, min(w, d) * 0.35), color="#ffffff", zorder=8)
        ax.text(x, z, "1", ha="center", va="center", fontsize=8, color="#ffffff", weight="bold", zorder=9)
    else:
        draw_rotated_box(
            ax,
            x,
            z,
            w,
            d,
            yaw,
            face=color,
            edge=color,
            alpha=0.16,
            lw=1.1,
            linestyle=":",
            zorder=6,
        )
        ax.text(x, z, str(rank), ha="center", va="center", fontsize=6, color=color, weight="bold", zorder=7)


def short_intent(text: str, max_chars: int = 96) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def load_grid_data(results_root: Path) -> Tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], Dict[str, Dict[str, Any]]]:
    preds: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    metrics: Dict[str, Dict[str, Any]] = {}
    for _, suffix, _ in GRID_SPECS:
        grid_dir = results_root / f"grid_{suffix}" / "test_gpt_intent"
        raw = load_json(grid_dir / "raw_predictions.json")
        preds[suffix] = raw["constraint_solver"]
        result = load_json(grid_dir / "results.json")
        scenes = result["methods"][0]["scenes"]
        metrics[suffix] = {scene["scene_id"]: scene for scene in scenes}
    return preds, metrics


def top1_position(preds: List[Dict[str, Any]]) -> Tuple[float, float]:
    if not preds or preds[0].get("status") != "placed":
        return (0.0, 0.0)
    p = preds[0]["position"]
    return (float(p["x"]), float(p["z"]))


def select_cases(
    cases: Iterable[Dict[str, Any]],
    preds: Dict[str, Dict[str, List[Dict[str, Any]]]],
    metrics: Dict[str, Dict[str, Any]],
    n: int,
) -> List[Dict[str, Any]]:
    scored = []
    suffixes = [s for _, s, _ in GRID_SPECS]
    for case in cases:
        cid = case["id"]
        if not all(cid in preds[s] and cid in metrics[s] for s in suffixes):
            continue
        s1 = [int(metrics[s][cid].get("success_at_1", 0)) for s in suffixes]
        s5 = [int(metrics[s][cid].get("success_at_5", 0)) for s in suffixes]
        positions = [top1_position(preds[s][cid]) for s in suffixes]
        spread = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                spread = max(spread, math.dist(positions[i], positions[j]))
        ca = [float((metrics[s][cid]["candidate_metrics"][0] or {}).get("constraint_accuracy", 0.0)) for s in suffixes]
        ca_spread = max(ca) - min(ca)
        disagreement = len(set(s1)) * 5 + len(set(s5)) * 2
        score = disagreement + spread + ca_spread
        scored.append((score, case))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [case for _, case in scored[:n]]


def draw_metrics_badge(ax, scene_metrics: Dict[str, Any], color: str) -> None:
    cm = scene_metrics["candidate_metrics"][0]
    lines = [
        f"Top-1 CPS={int(cm.get('cps', 0))}",
        f"CA={float(cm.get('constraint_accuracy', 0.0)):.3f}",
        f"S@5={int(scene_metrics.get('success_at_5', 0))}",
    ]
    text = "\n".join(lines)
    ax.text(
        0.98,
        0.03,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor=color, linewidth=1.4, alpha=0.92),
        zorder=20,
    )


def build_figure(
    cases: List[Dict[str, Any]],
    preds: Dict[str, Dict[str, List[Dict[str, Any]]]],
    metrics: Dict[str, Dict[str, Any]],
    out_dir: Path,
) -> None:
    rows = len(cases)
    cols = len(GRID_SPECS)
    fig, axes = plt.subplots(rows, cols, figsize=(5.7 * cols, 4.7 * rows), squeeze=False)
    fig.patch.set_facecolor("white")

    for r, case in enumerate(cases):
        cid = case["id"]
        target = str(case["target_asset"]["category"]).replace("_", " ")
        intent = short_intent(case.get("intent", {}).get("text", ""))
        for c, (label, suffix, color) in enumerate(GRID_SPECS):
            ax = axes[r][c]
            draw_room(ax, case)
            candidates = preds[suffix][cid]
            for rank, pred in enumerate(candidates[:5], 1):
                draw_candidate(ax, pred, color=color, rank=rank, primary=(rank == 1))
            draw_metrics_badge(ax, metrics[suffix][cid], color=color)
            if r == 0:
                ax.set_title(f"Grid step {label}", fontsize=15, weight="bold", color=color, pad=8)
            if c == 0:
                room = str(case["scene"].get("room_type", "")).replace("_", " ")
                ax.text(
                    -0.04,
                    0.5,
                    f"Case {r + 1}\n{room}\nTarget: {target}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=10,
                    weight="bold",
                    color="#0f172a",
                )
            ax.text(
                0.02,
                0.98,
                "진한 박스: Top-1 / 점선: Top-2~5",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                color="#475569",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.85),
            )
        axes[r][1].text(
            0.5,
            -0.08,
            f"Intent: {intent}",
            transform=axes[r][1].transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="#334155",
        )

    handles = [
        patches.Patch(facecolor="#dbeafe", edgecolor="#64748b", label="Existing furniture"),
        patches.Patch(facecolor="none", edgecolor="#eab308", linestyle="--", label="GT reference"),
        patches.Patch(facecolor="#16a34a", edgecolor="#0f172a", label="Selected top-1 target"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle("Grid Step Qualitative Comparison", fontsize=21, weight="bold", y=0.985)
    fig.text(
        0.5,
        0.045,
        "Grid를 촘촘하게 만들면 후보 수는 늘지만, top-1 선택 품질과 실행 시간은 별도의 trade-off를 가진다.",
        ha="center",
        va="center",
        fontsize=12,
        color="#0f172a",
    )
    fig.tight_layout(rect=[0.045, 0.075, 1.0, 0.95], h_pad=2.0, w_pad=0.7)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "grid_step_qualitative_comparison.png"
    svg = out_dir / "grid_step_qualitative_comparison.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png)
    print(svg)
    print("cases:")
    for case in cases:
        print(f"- {case['id']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="spacefit_v2/data/single_target_benchmark/gpt_intent_cases_test.json")
    parser.add_argument("--results_root", default="spacefit_v2/results/grid_step_ablation")
    parser.add_argument("--out_dir", default="spacefit_v2/results/grid_step_ablation/qualitative")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--case_id", action="append", default=[], help="Specific case id to include; can be repeated.")
    return parser.parse_args()


def main() -> None:
    setup_korean_font()
    args = parse_args()
    cases = load_json(ROOT / args.cases)
    preds, metrics = load_grid_data(ROOT / args.results_root)
    if args.case_id:
        selected = [case for case in cases if case["id"] in set(args.case_id)]
    else:
        selected = select_cases(cases, preds, metrics, n=args.n)
    if not selected:
        raise SystemExit("No matching cases found.")
    build_figure(selected, preds, metrics, ROOT / args.out_dir)


if __name__ == "__main__":
    main()
