"""Build a slide-ready visual comparison for open-source LLM backbones."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spacefit_v2.scripts.build_visual_audit import _render_image


OUT_DIR = Path("spacefit_v2/results/open_llm_backbone_comparison")
CASES_PATH = Path("spacefit_v2/data/single_target_benchmark/gpt_intent_cases_test.json")
CASE_METRICS_PATH = OUT_DIR / "qwen3_vs_gemma4_case_metrics.csv"

PREDICTIONS = {
    "qwen3_8b": {
        "label": "Qwen3-8B",
        "model": "Qwen/Qwen3-8B",
        "path": Path("spacefit_v2/results/qwen3_8b_spacefit_gpt_text_sample40_compact/test_gpt_intent/raw_predictions.json"),
        "color": "#2563EB",
    },
    "gemma4_e4b": {
        "label": "Gemma 4 E4B",
        "model": "google/gemma-4-E4B-it",
        "path": Path("spacefit_v2/results/open_llm_gemma4_e4b_compare40/test_gpt_intent/raw_predictions.json"),
        "color": "#16A34A",
    },
}

SELECTED_CASES = [
    "0220df39-8356-4ba5-8f26-4f385afa2cae__livingroom_37659__furniture_393",
    "02f7ad87-8aaf-49a0-bd98-c010e7b84c33__livingdiningroom_10060__furniture_216",
    "050de31b-ced3-47ec-8f47-6ba847d48a2d__livingdiningroom_515__furniture_146",
    "0446844d-66c4-4f8f-8b0f-69048e6c2d8c__livingroom_254465__furniture_216",
]

ROOM_KO = {
    "bedroom": "침실",
    "masterbedroom": "침실",
    "secondbedroom": "침실",
    "living_room": "거실",
    "livingroom": "거실",
    "livingdiningroom": "거실/다이닝",
    "library": "서재",
}

CATEGORY_KO = {
    "armchair": "안락의자",
    "sofa": "소파",
    "bed": "침대",
    "tv_stand": "TV 스탠드",
    "coffee_table": "커피 테이블",
    "dining_chair": "식탁 의자",
    "desk": "책상",
    "nightstand": "협탁",
    "bookshelf": "책장",
    "wardrobe": "옷장",
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_case_metrics(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {row["case_id"]: row for row in csv.DictReader(f)}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            if Path(path).exists():
                return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))[:140]


def first_prediction(predictions: Mapping[str, Sequence[Mapping[str, Any]]], case_id: str) -> Mapping[str, Any]:
    preds = predictions.get(case_id) or []
    return preds[0] if preds else {"status": "unplaced", "reason": "missing prediction"}


def render_prediction_images(
    cases_by_id: Mapping[str, Mapping[str, Any]],
    predictions_by_model: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> Dict[tuple[str, str], Path]:
    image_dir = OUT_DIR / "rendered_llm_predictions"
    image_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[tuple[str, str], Path] = {}
    for case_id in SELECTED_CASES:
        case = cases_by_id[case_id]
        for model_key, by_case in predictions_by_model.items():
            out_path = image_dir / f"{model_key}__{slug(case_id)}.png"
            _render_image(case, first_prediction(by_case, case_id), out_path)
            out[(case_id, model_key)] = out_path
    return out


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    img = Image.open(path).convert("RGB")
    img.thumbnail((target_w - 28, target_h - 12), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (target_w - img.width) // 2
    y = (target_h - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str = "#111827") -> None:
    draw.text(xy, text, font=font, fill=fill)


def metric_text(row: Mapping[str, str], model_key: str) -> str:
    cps = float(row.get(f"{model_key}_top1_cps") or 0.0)
    ca = float(row.get(f"{model_key}_top1_constraint_accuracy") or 0.0)
    cf = float(row.get(f"{model_key}_top1_cf") or 0.0)
    ib = float(row.get(f"{model_key}_top1_ib") or 0.0)
    region = row.get(f"{model_key}_region_id") or "-"
    return f"CPS {cps:.0f} | 제약 {ca:.2f} | CF {cf:.0f} | IB {ib:.0f} | region {region}"


def case_label(case: Mapping[str, Any], row: Mapping[str, str], index: int) -> str:
    room = ROOM_KO.get(str(case["scene"].get("room_type", "")), str(case["scene"].get("room_type", "")))
    target = CATEGORY_KO.get(str(case["target_asset"].get("category", "")), str(case["target_asset"].get("category", "")))
    winner = row.get("winner_top1", "tie")
    winner_label = {
        "gemma4_e4b": "Gemma 우세",
        "qwen3_8b": "Qwen 우세",
        "tie": "동률",
    }.get(winner, winner)
    return f"Case {index}. {room} / Target: {target} / {winner_label}"


def draw_card(
    canvas: Image.Image,
    image_path: Path,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    subtitle: str,
    metric: str,
    color: str,
    highlight: bool,
) -> None:
    draw = ImageDraw.Draw(canvas)
    border = color if highlight else "#CBD5E1"
    line_w = 8 if highlight else 3
    draw.rounded_rectangle([x, y, x + w, y + h], radius=10, fill="#FFFFFF", outline=border, width=line_w)
    img = fit_image(image_path, (w - 28, h - 168))
    canvas.paste(img, (x + 14, y + 98))
    draw_text(draw, (x + 24, y + 18), title, load_font(34, bold=True), color)
    draw_text(draw, (x + 24, y + 58), subtitle, load_font(20), "#4B5563")
    draw.rounded_rectangle([x + 18, y + h - 54, x + w - 18, y + h - 16], radius=8, fill="#F8FAFC", outline="#CBD5E1", width=2)
    draw_text(draw, (x + 34, y + h - 48), metric, load_font(21), "#111827")


def build_png() -> Path:
    cases = load_json(CASES_PATH)[:40]
    cases_by_id = {str(case["id"]): case for case in cases}
    metrics = load_case_metrics(CASE_METRICS_PATH)
    predictions_by_model = {
        key: load_json(info["path"])["spacefit_gpt_text"]
        for key, info in PREDICTIONS.items()
    }
    image_paths = render_prediction_images(cases_by_id, predictions_by_model)

    width = 1800
    top_h = 170
    row_h = 430
    gap = 34
    bottom_h = 80
    height = top_h + len(SELECTED_CASES) * row_h + (len(SELECTED_CASES) - 1) * gap + bottom_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(48, bold=True)
    subtitle_font = load_font(24)
    row_font = load_font(26, bold=True)
    small_font = load_font(22)

    title = "오픈소스 LLM Backbone별 배치 시각화 비교"
    tw = draw.textbbox((0, 0), title, font=title_font)[2]
    draw_text(draw, ((width - tw) // 2, 34), title, title_font)
    subtitle = "동일한 SpaceFit GPT-Text pipeline에서 LLM은 후보 region을 선택하고, geometry module이 최종 배치를 정제한다."
    sw = draw.textbbox((0, 0), subtitle, font=subtitle_font)[2]
    draw_text(draw, ((width - sw) // 2, 96), subtitle, subtitle_font, "#4B5563")

    left_x = 90
    card_w = 780
    right_x = width - left_x - card_w
    card_h = 370
    y = top_h
    for idx, case_id in enumerate(SELECTED_CASES, start=1):
        case = cases_by_id[case_id]
        row = metrics[case_id]
        draw_text(draw, (left_x, y - 6), case_label(case, row, idx), row_font)
        draw_text(draw, (left_x, y + 28), str(case_id), small_font, "#6B7280")

        card_y = y + 64
        for x, model_key in [(left_x, "qwen3_8b"), (right_x, "gemma4_e4b")]:
            info = PREDICTIONS[model_key]
            draw_card(
                canvas,
                image_paths[(case_id, model_key)],
                x,
                card_y,
                card_w,
                card_h,
                info["label"],
                info["model"],
                metric_text(row, model_key),
                info["color"],
                highlight=row.get("winner_top1") == model_key,
            )
        y += row_h + gap

    note = "초록/파랑 굵은 테두리는 해당 case에서 top-1 CPS 기준 더 나은 LLM backbone을 의미한다."
    nw = draw.textbbox((0, 0), note, font=subtitle_font)[2]
    draw_text(draw, ((width - nw) // 2, height - 54), note, subtitle_font, "#374151")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "open_llm_visual_comparison.png"
    canvas.save(out_path)
    return out_path


def build_svg_from_png(png_path: Path) -> None:
    img = Image.open(png_path).convert("RGB")
    fig_w = 16
    fig_h = fig_w * img.height / img.width
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
    ax.imshow(img)
    ax.axis("off")
    fig.savefig(OUT_DIR / "open_llm_visual_comparison.svg", bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)


def main() -> None:
    png_path = build_png()
    build_svg_from_png(png_path)


if __name__ == "__main__":
    main()
