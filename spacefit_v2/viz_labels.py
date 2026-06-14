"""Korean label utilities for visualization."""
from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager


CATEGORY_LABEL_KO = {
    "armchair": "안락의자",
    "bed": "침대",
    "bookshelf": "책장",
    "cabinet": "캐비닛",
    "ceiling_lamp": "천장등",
    "chair": "의자",
    "chinese_chair": "중국식의자",
    "children_cabinet": "아이수납장",
    "coffee_table": "커피테이블",
    "console_table": "콘솔테이블",
    "corner_side_table": "코너협탁",
    "desk": "책상",
    "dining_chair": "식탁의자",
    "dining_table": "식탁",
    "double_bed": "더블침대",
    "dressing_chair": "화장의자",
    "dressing_table": "화장대",
    "floor_lamp": "장스탠드",
    "kids_bed": "아동침대",
    "l_shaped_sofa": "ㄱ자소파",
    "lounge_chair": "라운지의자",
    "loveseat_sofa": "2인소파",
    "multi_seat_sofa": "다인소파",
    "nightstand": "협탁",
    "pendant_lamp": "펜던트등",
    "round_end_table": "원형협탁",
    "shelf": "선반",
    "single_bed": "싱글침대",
    "sofa": "소파",
    "stool": "스툴",
    "table": "테이블",
    "table_lamp": "탁상등",
    "tv_stand": "TV장",
    "wardrobe": "옷장",
    "wine_cabinet": "와인장",
}
_FONT_PATH_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
]
_FONT_FAMILY_CANDIDATES = [
    "Noto Sans CJK KR",
    "Noto Sans CJK JP",
    "Noto Sans CJK SC",
    "Noto Serif CJK KR",
    "NanumGothic",
    "Malgun Gothic",
]


def _resolve_font_family() -> str:
    for font_path in _FONT_PATH_CANDIDATES:
        path = Path(font_path)
        if not path.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(path))
            return font_manager.FontProperties(fname=str(path)).get_name()
        except Exception:
            continue

    available = {font.name for font in font_manager.fontManager.ttflist}
    for family in _FONT_FAMILY_CANDIDATES:
        if family in available:
            return family

    return "DejaVu Sans"


def setup_korean_matplotlib() -> None:
    matplotlib.rcParams["font.family"] = _resolve_font_family()
    matplotlib.rcParams["axes.unicode_minus"] = False


def category_label_ko(category: str, max_chars: int | None = None) -> str:
    key = (category or "").strip().lower()
    label = CATEGORY_LABEL_KO.get(key, category)
    if max_chars is not None:
        return label[:max_chars]
    return label
