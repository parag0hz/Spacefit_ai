"""Build VLM-judge-oriented images for visual-audit VQA examples.

The human labeling UI images are good for people, but VLMs can over-focus on
small arrows and under-read doors/windows. This script re-renders examples with
stronger target/door/window/reference cues and no rotation arrow.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def _slug(text: str, max_len: int = 140) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")[:max_len] or "item"


def _position_xz(item: Mapping[str, Any]) -> Tuple[float, float]:
    pos = item.get("position", {})
    if isinstance(pos, Mapping):
        return float(pos.get("x", 0.0)), float(pos.get("z", 0.0))
    if isinstance(pos, Sequence) and not isinstance(pos, (str, bytes)):
        return float(pos[0]), float(pos[2] if len(pos) > 2 else pos[1])
    return float(item.get("x", 0.0)), float(item.get("z", 0.0))


def _size_wdh(item: Mapping[str, Any]) -> Tuple[float, float, float]:
    size = item.get("size", {})
    if isinstance(size, Mapping):
        return float(size.get("width", 0.5)), float(size.get("depth", 0.5)), float(size.get("height", 0.5))
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)):
        return float(size[0]), float(size[2] if len(size) > 2 else size[-1]), float(size[1] if len(size) > 1 else 0.5)
    return float(item.get("width", 0.5)), float(item.get("depth", 0.5)), float(item.get("height", 0.5))


def _yaw_deg(item: Mapping[str, Any]) -> float:
    raw = float(item.get("rotation_y", item.get("yaw", 0.0)) or 0.0)
    return math.degrees(raw) if abs(raw) <= 2 * math.pi + 1e-3 else raw


def _corners(cx: float, cz: float, width: float, depth: float, yaw_deg: float) -> List[Tuple[float, float]]:
    hw = max(width, 0.05) * 0.5
    hd = max(depth, 0.05) * 0.5
    yaw = math.radians(yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    local = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
    return [(cx + dx * c - dz * s, cz + dx * s + dz * c) for dx, dz in local]


def _add_box(
    ax: Any,
    cx: float,
    cz: float,
    width: float,
    depth: float,
    yaw_deg: float,
    face: str,
    edge: str,
    alpha: float,
    lw: float,
    zorder: int,
    linestyle: str = "-",
) -> None:
    ax.add_patch(
        mpatches.Polygon(
            _corners(cx, cz, width, depth, yaw_deg),
            closed=True,
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
            alpha=alpha,
            linestyle=linestyle,
            zorder=zorder,
        )
    )


def _reference_categories(case: Mapping[str, Any]) -> set[str]:
    cats: set[str] = set()
    for c in case.get("intent", {}).get("constraints", []):
        if c.get("target_category"):
            cats.add(str(c["target_category"]))
    return cats


def _render(case: Mapping[str, Any], pred: Mapping[str, Any], out_path: Path) -> None:
    scene = case["scene"]
    floor = [(float(x), float(z)) for x, z in scene["floor"]["polygon"]]
    xs = [p[0] for p in floor]
    zs = [p[1] for p in floor]
    room_w = max(xs) - min(xs)
    room_h = max(zs) - min(zs)
    aspect = max(0.75, min(1.6, room_w / max(room_h, 1e-6)))
    ref_cats = _reference_categories(case)

    fig, ax = plt.subplots(figsize=(6.2 * aspect, 6.2), dpi=150)
    ax.set_aspect("equal")
    ax.add_patch(mpatches.Polygon(floor, closed=True, facecolor="#FAF8F0", edgecolor="#111111", linewidth=2.3, zorder=0))

    for obj in scene.get("objects", []):
        ox, oz = _position_xz(obj)
        ow, od, _ = _size_wdh(obj)
        cat = str(obj.get("category", "")).replace("_", " ")
        is_ref = str(obj.get("category", "")) in ref_cats
        _add_box(
            ax,
            ox,
            oz,
            ow,
            od,
            _yaw_deg(obj),
            "#BFC5CC" if not is_ref else "#D8C7FF",
            "#59616A" if not is_ref else "#6D28D9",
            0.86,
            1.1 if not is_ref else 2.2,
            2 if not is_ref else 4,
        )
        ax.text(ox, oz, cat[:12], ha="center", va="center", fontsize=5.2, color="#111827", zorder=6)

    for win in scene.get("windows", []):
        wx, wz = _position_xz(win)
        ax.plot(wx, wz, marker="D", color="#0284C7", markerfacecolor="#7DD3FC", markersize=10, markeredgewidth=1.5, zorder=8)
        ax.text(wx, wz + 0.16, "WINDOW", ha="center", va="bottom", fontsize=5.5, color="#075985", fontweight="bold", zorder=9)

    for door in scene.get("doors", []):
        dx, dz = _position_xz(door)
        ax.plot(dx, dz, marker="s", color="#92400E", markerfacecolor="#F59E0B", markersize=11, markeredgewidth=1.5, zorder=8)
        ax.text(dx, dz + 0.16, "DOOR", ha="center", va="bottom", fontsize=5.5, color="#78350F", fontweight="bold", zorder=9)

    if pred.get("status") == "placed":
        tx, tz = _position_xz(pred)
        tw, td, _ = _size_wdh(pred)
        _add_box(ax, tx, tz, tw, td, _yaw_deg(pred), "#22C55E", "#052E16", 0.93, 2.8, 10)
        target = str(case.get("target_asset", {}).get("category", pred.get("category", "target"))).replace("_", " ")
        ax.text(tx, tz, "TARGET\n" + target[:12], ha="center", va="center", fontsize=6.2, color="white", fontweight="bold", zorder=12)
    else:
        ax.text(0.5, 0.5, "TARGET NOT PLACED", transform=ax.transAxes, ha="center", va="center", fontsize=16, color="#B91C1C")

    ax.text(
        0.01,
        0.99,
        "green=target | gray=existing | cyan=window | brown=door | purple=reference",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#111827",
        bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.85, "pad": 3},
        zorder=20,
    )

    margin = 0.55
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(zs) - margin, max(zs) + margin)
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _add_box3d(
    ax: Any,
    cx: float,
    cz: float,
    width: float,
    depth: float,
    height: float,
    yaw_deg: float,
    color: str,
    edge: str,
    alpha: float,
) -> None:
    base = _corners(cx, cz, width, depth, yaw_deg)
    h = max(float(height), 0.15)
    bottom = [(x, z, 0.0) for x, z in base]
    top = [(x, z, h) for x, z in base]
    faces = [
        top,
        bottom,
        [bottom[0], bottom[1], top[1], top[0]],
        [bottom[1], bottom[2], top[2], top[1]],
        [bottom[2], bottom[3], top[3], top[2]],
        [bottom[3], bottom[0], top[0], top[3]],
    ]
    poly = Poly3DCollection(faces, facecolors=color, edgecolors=edge, linewidths=0.7, alpha=alpha)
    ax.add_collection3d(poly)


def _render_oblique(case: Mapping[str, Any], pred: Mapping[str, Any], out_path: Path) -> None:
    scene = case["scene"]
    floor = [(float(x), float(z)) for x, z in scene["floor"]["polygon"]]
    xs = [p[0] for p in floor]
    zs = [p[1] for p in floor]
    fig = plt.figure(figsize=(6.6, 5.8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    floor3d = [[(x, z, 0.0) for x, z in floor]]
    ax.add_collection3d(Poly3DCollection(floor3d, facecolors="#FAF8F0", edgecolors="#111111", linewidths=1.4, alpha=0.95))
    for obj in scene.get("objects", []):
        ox, oz = _position_xz(obj)
        ow, od, oh = _size_wdh(obj)
        _add_box3d(ax, ox, oz, ow, od, oh, _yaw_deg(obj), "#BFC5CC", "#59616A", 0.82)
    for win in scene.get("windows", []):
        wx, wz = _position_xz(win)
        ax.scatter([wx], [wz], [0.08], c="#0284C7", s=75, marker="D", depthshade=False)
    for door in scene.get("doors", []):
        dx, dz = _position_xz(door)
        ax.scatter([dx], [dz], [0.08], c="#F59E0B", s=90, marker="s", depthshade=False)
    if pred.get("status") == "placed":
        tx, tz = _position_xz(pred)
        tw, td, th = _size_wdh(pred)
        _add_box3d(ax, tx, tz, tw, td, th, _yaw_deg(pred), "#22C55E", "#052E16", 0.96)
        ax.text(tx, tz, max(th, 0.15) + 0.1, "TARGET", ha="center", va="center", color="#052E16", fontsize=8, fontweight="bold")
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_ylim(min(zs) - 0.6, max(zs) + 0.6)
    ax.set_zlim(0, 2.4)
    ax.view_init(elev=38, azim=-55)
    ax.set_box_aspect((max(xs) - min(xs), max(zs) - min(zs), 2.0))
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _render_relation(case: Mapping[str, Any], pred: Mapping[str, Any], out_path: Path) -> None:
    # Same drawing as enhanced top-down, but keep only strong relation/access cues visually prominent.
    _render(case, pred, out_path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build enhanced VQA judge images.")
    p.add_argument("--input", default="spacefit_v2/results/visual_audit_distribution/result/analysis/vqa_seed/vqa_val.jsonl")
    p.add_argument("--cases", default="spacefit_v2/data/single_target_benchmark/gpt_intent_cases_test.json")
    p.add_argument("--predictions", default="spacefit_v2/results/experiment_final/test_gpt_intent/raw_predictions.json")
    p.add_argument("--out_dir", default="spacefit_v2/results/visual_audit_distribution/result/analysis/vqa_seed_enhanced")
    p.add_argument("--output_name", default=None)
    p.add_argument("--multi_view", action="store_true")
    return p


def main(args: argparse.Namespace) -> None:
    rows = _read_jsonl(Path(args.input))
    cases = {str(c["id"]): c for c in _load_json(args.cases)}
    preds = _load_json(args.predictions)
    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    updated: List[Dict[str, Any]] = []
    cache: set[Tuple[str, str]] = set()

    for row in rows:
        cid = str(row["case_id"])
        method = str(row["method"])
        key = (cid, method)
        case = cases[cid]
        pred_list = preds.get(method, {}).get(cid) or []
        pred = pred_list[0] if pred_list else {"status": "missing"}
        stem = f"{_slug(cid, 90)}__{_slug(method, 40)}"
        image_path = image_dir / f"{stem}.png"
        oblique_path = image_dir / f"{stem}__oblique.png"
        relation_path = image_dir / f"{stem}__relation.png"
        if key not in cache:
            _render(case, pred, image_path)
            if args.multi_view:
                _render_oblique(case, pred, oblique_path)
                _render_relation(case, pred, relation_path)
            cache.add(key)
        new_row = dict(row)
        new_row["image_original"] = row.get("image")
        new_row["image"] = str(image_path.as_posix())
        new_row["image_variant"] = "enhanced_noarrow_v1"
        if args.multi_view:
            # Keep the original human-rendered view as the first view, then add VLM-oriented views.
            views = [row.get("image"), str(image_path.as_posix()), str(relation_path.as_posix()), str(oblique_path.as_posix())]
            new_row["images"] = [v for v in views if v]
            new_row["image_variant"] = "multiview_original_enhanced_relation_oblique_v1"
        updated.append(new_row)

    output_name = args.output_name or Path(args.input).name.replace(".jsonl", "_enhanced.jsonl")
    _write_jsonl(out_dir / output_name, updated)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "input": args.input,
                "items": len(updated),
                "unique_images": len(cache),
                "image_variant": "enhanced_noarrow_v1",
                "multi_view": bool(args.multi_view),
                "output": str(out_dir / output_name),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(out_dir / output_name), "items": len(updated), "unique_images": len(cache)}, indent=2))


if __name__ == "__main__":
    main(build_parser().parse_args())
