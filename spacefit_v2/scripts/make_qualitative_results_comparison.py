"""Build a slide-ready qualitative comparison figure for placement methods.

The script first tries to find existing visualization images in method-specific
folders. If an image is missing, it falls back to the saved benchmark
predictions and renders a top-down image automatically.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import font_manager, rcParams
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spacefit_v2.scripts.build_visual_audit import _render_image


OUT_DIR = Path("spacefit_v2/results/presentation_figures")
DEFAULT_CASES = Path("spacefit_v2/data/single_target_benchmark/gpt_intent_cases_test.json")
DEFAULT_PREDICTIONS = Path("spacefit_v2/results/experiment_final/test_gpt_intent/raw_predictions.json")
DEFAULT_RESULTS = Path("spacefit_v2/results/experiment_final/test_gpt_intent/results.json")
DEFAULT_RERANK_PREDICTIONS = Path(
    "spacefit_v2/results/final_constraint_solver_human_rerank/test_gpt_intent/raw_predictions_human_reranked.json"
)
DEFAULT_RERANK_RESULTS = Path("spacefit_v2/results/final_constraint_solver_human_rerank/test_gpt_intent/results.json")


METHODS = [
    {
        "key": "layoutgpt_direct",
        "label": "Direct Coordinate\nPrediction",
        "caption": "좌표 직접 예측",
        "kind": "prediction",
        "dir": "spacefit_v2/results/layoutgpt_direct/visualization",
        "border": "#D97706",
    },
    {
        "key": "proposal_diffopt_constraint",
        "label": "Loss-based Pose\nOptimization",
        "caption": "loss 기반 pose 최적화",
        "kind": "prediction",
        "dir": "spacefit_v2/results/diffopt_constraint/visualization",
        "border": "#B45309",
    },
    {
        "key": "spacefit_gpt_text",
        "label": "LLM Region\nSelection",
        "caption": "LLM 후보 region 선택",
        "kind": "prediction",
        "dir": "spacefit_v2/results/spacefit_gpt_text/visualization",
        "border": "#F59E0B",
    },
    {
        "key": "constraint_solver",
        "label": "Candidate Search\n& Scoring",
        "caption": "후보 탐색 및 점수화",
        "kind": "prediction",
        "dir": "spacefit_v2/results/constraint_solver/visualization",
        "border": "#2563EB",
    },
    {
        "key": "constraint_solver_human_rerank",
        "label": "Preference-Reranked\nCandidate Selection",
        "caption": "사람 선호 기반 재정렬",
        "kind": "rerank_prediction",
        "source_key": "constraint_solver",
        "dir": "spacefit_v2/results/final_constraint_solver_human_rerank/visualization",
        "border": "#16A34A",
        "highlight": True,
    },
]


# Optional manual metadata and annotations. Unknown cases are filled from the benchmark.
CASE_METADATA: Dict[str, Dict[str, str]] = {
    "1c79cb23-f69d-4766-b829-2747eb6152c5__livingroom_5401__furniture_170": {
        "room_type": "거실",
        "target": "식탁 의자",
        "intent": "커피 테이블 주변의 기존 의자 옆에 추가 의자를 배치해, 손님이 앉을 수 있는 좌석을 만든다.",
    },
    "15e31e17-330c-4904-98fd-210770901565__masterbedroom_7197__furniture_103": {
        "room_type": "침실",
        "target": "옷장",
        "intent": "TV 스탠드 근처 벽에 옷장을 배치하되, 창문을 막지 않고 이동 공간을 확보한다.",
    },
    "0c646244-f779-46f2-8723-3cfef6bfc23b__library_12728__furniture_803": {
        "room_type": "서재",
        "target": "안락의자",
        "intent": "책상 근처에 안락의자를 배치하고, 큰 창문을 바라보도록 두어 편하게 독서할 수 있게 한다.",
    },
}

ANNOTATIONS: Dict[str, Dict[str, List[str]]] = {
    # Example:
    # "case_id": {
    #     "layoutgpt_direct": ["relation mismatch"],
    #     "constraint_solver_human_rerank": ["preferred"],
    # },
}


ROOM_KO = {
    "bedroom": "침실",
    "masterbedroom": "침실",
    "secondbedroom": "침실",
    "living_room": "거실",
    "livingroom": "거실",
    "livingdiningroom": "거실/식당",
    "library": "서재",
}


CATEGORY_KO = {
    "armchair": "안락의자",
    "sofa": "소파",
    "bed": "침대",
    "tv_stand": "TV 스탠드",
    "tv stand": "TV 스탠드",
    "coffee_table": "커피 테이블",
    "coffee table": "커피 테이블",
    "dining_chair": "식탁 의자",
    "dining chair": "식탁 의자",
    "desk": "책상",
    "nightstand": "협탁",
    "bookshelf": "책장",
    "wardrobe": "옷장",
}


def setup_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR", "Arial", "DejaVu Sans"]:
        if name in available:
            rcParams["font.family"] = name
            break
    rcParams["axes.unicode_minus"] = False


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def slug(text: str, max_len: int = 140) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")[:max_len] or "case"


def find_image_for_case(method_dir: str | Path | None, case_id: str) -> Path | None:
    if not method_dir:
        return None
    root = Path(method_dir)
    if not root.exists():
        return None
    exts = {".png", ".jpg", ".jpeg"}
    direct = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts and case_id in p.name]
    if direct:
        return sorted(direct, key=lambda p: (len(p.name), str(p)))[0]
    case_slug = slug(case_id)
    fuzzy = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts and case_slug in p.name]
    return sorted(fuzzy, key=lambda p: (len(p.name), str(p)))[0] if fuzzy else None


def load_and_prepare_image(path: Path | None, target_size: tuple[int, int]) -> Image.Image:
    width, height = target_size
    if path is None or not path.exists():
        img = Image.new("RGB", target_size, "#F3F4F6")
        draw = ImageDraw.Draw(img)
        draw.rectangle([1, 1, width - 2, height - 2], outline="#CBD5E1", width=3)
        draw.text((width // 2 - 18, height // 2 - 8), "없음", fill="#64748B")
        return img
    img = Image.open(path).convert("RGB")
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target_size, "white")
    x = (width - img.width) // 2
    y = (height - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def prediction_for_method(
    method: Mapping[str, Any],
    case_id: str,
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    rerank_predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    case: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    kind = method.get("kind")
    if kind == "reference":
        ref = case.get("reference_pose") or {}
        asset = case.get("target_asset") or {}
        return {
            "furniture_id": asset.get("id", "target"),
            "category": asset.get("category", "target"),
            "position": ref.get("position"),
            "rotation_y": ref.get("rotation_y", 0.0),
            "size": asset.get("size"),
            "status": "placed",
        }
    if kind == "rerank_prediction":
        source_key = str(method.get("source_key", "constraint_solver"))
        preds = (rerank_predictions.get(source_key) or {}).get(case_id) or []
    else:
        preds = (predictions.get(str(method["key"])) or {}).get(case_id) or []
    return preds[0] if preds else None


def ensure_rendered_image(
    method: Mapping[str, Any],
    case: Mapping[str, Any],
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    rerank_predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    render_dir: Path,
) -> Path | None:
    case_id = str(case["id"])
    existing = find_image_for_case(method.get("dir"), case_id)
    if existing:
        return existing
    pred = prediction_for_method(method, case_id, predictions, rerank_predictions, case)
    if not pred:
        return None
    out_path = render_dir / str(method["key"]) / f"{slug(case_id)}.png"
    if not out_path.exists():
        _render_image(case, pred, out_path)
    return out_path


def metric_lookup(results_path: Path, method_alias: Mapping[str, str] | None = None) -> Dict[tuple[str, str], Dict[str, Any]]:
    method_alias = method_alias or {}
    if not results_path.exists():
        return {}
    data = load_json(results_path)
    out: Dict[tuple[str, str], Dict[str, Any]] = {}
    for method in data.get("methods", []):
        method_key = str(method.get("source_name") or method.get("method"))
        method_key = method_alias.get(method_key, method_key)
        for scene in method.get("scenes", []):
            cid = str(scene.get("query_id") or scene.get("scene_id"))
            top = (scene.get("candidate_metrics") or [{}])[0]
            out[(method_key, cid)] = {
                "success_at_1": scene.get("success_at_1"),
                "success_at_5": scene.get("success_at_5"),
                "top1": top,
            }
    return out


def auto_caption(method_key: str, case_id: str, metrics: Mapping[tuple[str, str], Mapping[str, Any]]) -> str:
    if method_key == "gt":
        return "원래 배치 위치"
    item = metrics.get((method_key, case_id), {})
    top = item.get("top1", {})
    if not top:
        return ""
    if top.get("cf") == 0:
        return "충돌/겹침"
    if top.get("ib") == 0:
        return "방 밖 배치"
    if top.get("reachability") == 0:
        return "접근성 부족"
    if top.get("cps") == 1:
        return "조건 만족"
    ca = top.get("constraint_accuracy")
    if isinstance(ca, (int, float)):
        if ca >= 0.75:
            return "물리적으로 안정적"
        if ca >= 0.45:
            return "일부 관계 만족"
    return "관계 불일치"


def build_case_metadata(case: Mapping[str, Any]) -> Dict[str, str]:
    cid = str(case["id"])
    if cid in CASE_METADATA:
        return CASE_METADATA[cid]
    intent = str((case.get("intent") or {}).get("text", ""))
    room_raw = str((case.get("scene") or {}).get("room_type", ""))
    target_raw = str((case.get("target_asset") or {}).get("category", ""))
    return {
        "room_type": ROOM_KO.get(room_raw.lower(), room_raw.replace("_", " ")),
        "target": CATEGORY_KO.get(target_raw.lower(), target_raw.replace("_", " ")),
        "intent": intent,
    }


def draw_annotation_box(ax: plt.Axes, labels: Sequence[str], color: str = "#111827") -> None:
    if not labels:
        return
    text = " / ".join(labels[:2])
    ax.text(
        0.03,
        0.92,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="white",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.24", facecolor=color, edgecolor="none", alpha=0.86),
        zorder=10,
    )


def metric_text(method_key: str, case_id: str, metrics: Mapping[tuple[str, str], Mapping[str, Any]]) -> str:
    top = (metrics.get((method_key, case_id), {}) or {}).get("top1", {})
    if not top:
        return ""
    cps = int(top.get("cps") or 0)
    cf = int(top.get("cf") or 0)
    ib = int(top.get("ib") or 0)
    ca = top.get("constraint_accuracy")
    ca_text = f"{float(ca):.2f}" if isinstance(ca, (int, float)) else "-"
    return f"CPS {cps} | CA {ca_text} | CF/IB {cf}/{ib}"


def render_image_panel(
    ax: plt.Axes,
    img: Image.Image,
    method: Mapping[str, Any],
    case_id: str,
    caption: str,
    metric_line: str,
    annotations: Sequence[str],
) -> None:
    ax.imshow(img, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    border = method.get("border", "#D1D5DB")
    lw = 3.0 if method.get("highlight") else 1.8
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(lw)
        spine.set_edgecolor(border)
    badge_color = "#16A34A" if method.get("highlight") else "#1F2937"
    ax.text(
        0.03,
        0.04,
        str(method["label"]),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.4,
        color="white",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=badge_color, edgecolor="none", alpha=0.92),
    )
    if annotations:
        ann_color = "#16A34A" if "preferred" in " ".join(annotations).lower() else "#DC2626"
        draw_annotation_box(ax, annotations, ann_color)
    label = caption if not metric_line else f"{caption}\n{metric_line}"
    ax.set_xlabel(label, fontsize=8.8, color="#334155", labelpad=5, linespacing=1.12)


def render_case_row(
    axes: Sequence[plt.Axes],
    case: Mapping[str, Any],
    methods: Sequence[Mapping[str, Any]],
    image_paths: Mapping[str, Path | None],
    metrics: Mapping[tuple[str, str], Mapping[str, Any]],
    target_size: tuple[int, int],
) -> None:
    case_id = str(case["id"])
    meta = build_case_metadata(case)
    label_ax = axes[0]
    label_ax.axis("off")
    wrapped_intent = "\n".join(textwrap.wrap(meta.get("intent", ""), width=34, max_lines=4, placeholder="..."))
    label_ax.text(0.02, 0.88, "사례", fontsize=13, weight="bold", color="#0F172A", ha="left", va="top")
    label_ax.text(0.02, 0.72, case_id[:8], fontsize=10.5, color="#475569", ha="left", va="top")
    label_ax.text(
        0.02,
        0.55,
        f"방 종류: {meta.get('room_type', '')}\n배치 가구: {meta.get('target', '')}",
        fontsize=10.5,
        color="#0F172A",
        ha="left",
        va="top",
        linespacing=1.25,
    )
    label_ax.text(0.02, 0.24, wrapped_intent, fontsize=8.8, color="#475569", ha="left", va="top", linespacing=1.18)

    for ax, method in zip(axes[1:], methods):
        method_key = str(method["key"])
        img = load_and_prepare_image(image_paths.get(method_key), target_size)
        annotation = (ANNOTATIONS.get(case_id) or {}).get(method_key, [])
        caption = annotation[0] if annotation else auto_caption(method_key, case_id, metrics)
        if not caption:
            caption = str(method.get("caption", ""))
        render_image_panel(ax, img, method, case_id, caption, metric_text(method_key, case_id, metrics), annotation)


def choose_default_cases(
    cases: Sequence[Mapping[str, Any]],
    metrics: Mapping[tuple[str, str], Mapping[str, Any]],
    n: int,
) -> List[str]:
    scored = []
    for case in cases:
        cid = str(case["id"])
        layout = metrics.get(("layoutgpt_direct", cid), {}).get("top1", {})
        solver = metrics.get(("constraint_solver", cid), {}).get("top1", {})
        ours = metrics.get(("spacefit_gpt_text", cid), {}).get("top1", {})
        diff = metrics.get(("proposal_diffopt_constraint", cid), {}).get("top1", {})
        rerank = metrics.get(("constraint_solver_human_rerank", cid), {}).get("top1", {})
        score = 0.0
        score += 2.0 if solver.get("cps") == 1 else 0.0
        score += 2.0 if ours.get("cps") == 1 else 0.0
        score += 2.5 if rerank.get("cps") == 1 else 0.0
        score += 1.5 if layout.get("cps") == 0 else 0.0
        score += 1.2 if diff.get("cps") == 0 else 0.0
        score += float(ours.get("constraint_accuracy") or 0.0)
        scored.append((score, cid))
    chosen = [cid for _score, cid in sorted(scored, reverse=True)[:n]]
    return chosen or [str(case["id"]) for case in cases[:n]]


def build_qualitative_figure(
    cases: Sequence[Mapping[str, Any]],
    case_ids: Sequence[str],
    methods: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    rerank_predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    metrics: Mapping[tuple[str, str], Mapping[str, Any]],
    out_png: Path,
    out_svg: Path,
) -> None:
    case_by_id = {str(case["id"]): case for case in cases}
    selected_cases = [case_by_id[cid] for cid in case_ids if cid in case_by_id]
    render_dir = out_png.parent / "_qualitative_render_cache"
    target_size = (310, 225)

    image_paths: Dict[str, Dict[str, Path | None]] = {}
    for case in selected_cases:
        cid = str(case["id"])
        image_paths[cid] = {}
        for method in methods:
            image_paths[cid][str(method["key"])] = ensure_rendered_image(
                method,
                case,
                predictions,
                rerank_predictions,
                render_dir,
            )

    n_rows = len(selected_cases)
    n_cols = len(methods) + 1
    fig_w = 3.2 + len(methods) * 3.15
    fig_h = 1.55 + n_rows * 2.75
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        facecolor="white",
        gridspec_kw={"width_ratios": [1.32] + [1] * len(methods), "wspace": 0.15, "hspace": 0.36},
    )
    if n_rows == 1:
        axes = [axes]

    fig.suptitle("정성적 시각화 결과", fontsize=21, weight="bold", y=0.965)
    for row_idx, case in enumerate(selected_cases):
        render_case_row(
            list(axes[row_idx]),
            case,
            methods,
            image_paths[str(case["id"])],
            metrics,
            target_size,
        )

    fig.text(
        0.5,
        0.025,
        "정량 지표가 높아도 사람이 보기에 자연스러운 배치를 항상 보장하지는 않는다.",
        ha="center",
        va="bottom",
        fontsize=12,
        color="#334155",
    )
    fig.subplots_adjust(left=0.012, right=0.996, top=0.89, bottom=0.105)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--rerank_predictions", type=Path, default=DEFAULT_RERANK_PREDICTIONS)
    parser.add_argument("--rerank_results", type=Path, default=DEFAULT_RERANK_RESULTS)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--case_ids", nargs="*", default=None)
    parser.add_argument("--n_cases", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    setup_fonts()
    args = parse_args()
    cases = load_json(args.cases)
    predictions = load_json(args.predictions)
    rerank_predictions = load_json(args.rerank_predictions) if args.rerank_predictions.exists() else {}
    metrics = {}
    metrics.update(metric_lookup(args.results))
    metrics.update(metric_lookup(args.rerank_results, {"constraint_solver": "constraint_solver_human_rerank"}))
    case_ids = args.case_ids or choose_default_cases(cases, metrics, args.n_cases)
    case_ids = case_ids[: args.n_cases]
    out_png = args.out_dir / "qualitative_results_comparison.png"
    out_svg = args.out_dir / "qualitative_results_comparison.svg"
    build_qualitative_figure(cases, case_ids, METHODS, predictions, rerank_predictions, metrics, out_png, out_svg)
    print(f"Saved {out_png}")
    print(f"Saved {out_svg}")
    print("Cases:")
    for cid in case_ids:
        print(f"  - {cid}")


if __name__ == "__main__":
    main()
