"""Build a visual audit set for single-target placement results.

Creates:
  - per-case/per-method top-down PNGs
  - labels.jsonl with VQA-style questions and empty human labels
  - index.html for quick browser-based labeling

Example:
    python -m spacefit_v2.scripts.build_visual_audit \
        --cases spacefit_v2/data/single_target_benchmark/gpt_intent_cases_test.json \
        --predictions spacefit_v2/results/experiment_final/test_gpt_intent/raw_predictions.json \
        --results spacefit_v2/results/experiment_final/test_gpt_intent/results.json \
        --out_dir spacefit_v2/results/visual_audit_v1 \
        --methods layoutgpt_direct spacefit_gpt_text constraint_solver proposal_diffopt_constraint \
        --n 30 --mode high_metric
"""
from __future__ import annotations

import argparse
import html
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


DEFAULT_METHODS = [
    "layoutgpt_direct",
    "spacefit_gpt_text",
    "constraint_solver",
    "proposal_diffopt_constraint",
    "proposal_heuristic",
    "heuristic_baseline",
]

METHOD_LABELS = {
    "heuristic_baseline": "Heuristic",
    "proposal_heuristic": "Proposal+Heuristic",
    "proposal_diffopt_basic": "DiffOpt-Basic",
    "proposal_diffopt_constraint": "DiffOpt-Constraint",
    "constraint_solver": "Constraint Solver",
    "layoutgpt_direct": "LayoutGPT Direct",
    "spacefit_gpt_text": "SpaceFit+GPT-Text",
}

QUESTIONS = [
    {
        "id": "physical_ok",
        "text": "대상 가구가 방 안에 완전히 들어가 있고 기존 가구와 겹치지 않나요?",
        "kind": "binary",
    },
    {
        "id": "relation_ok",
        "text": "요청한 공간 관계를 만족하나요? 예: 소파 근처, 창문 쪽, 벽에 붙임 등",
        "kind": "binary",
    },
    {
        "id": "orientation_ok",
        "text": "가구의 방향이나 바라보는 방향이 지시에 맞고 자연스러운가요?",
        "kind": "binary",
    },
    {
        "id": "access_ok",
        "text": "가구 주변에 접근하거나 지나갈 공간이 충분한가요?",
        "kind": "binary",
    },
    {
        "id": "naturalness_ok",
        "text": "실제 방에서 봤을 때 배치가 자연스럽고 쓸 만해 보이나요?",
        "kind": "binary",
    },
    {
        "id": "overall_ok",
        "text": "종합적으로 이 지시에 대해 괜찮은 배치인가요?",
        "kind": "binary",
    },
]

FAILURE_TAGS = [
    "충돌_겹침",
    "방_밖으로_나감",
    "관계_틀림",
    "방향_틀림",
    "너무_멀다",
    "너무_가깝다",
    "문을_막음",
    "창문을_막음",
    "접근공간_부족",
    "가구군_어색함",
    "혼자_동떨어짐",
    "크기_좌표_오류",
    "그림이_불명확함",
]

INSTRUCTION_TRANSLATIONS = {
    "Place the desk against the wall, near the window for natural light. Make sure it's not blocking any doorways.": "책상을 벽에 붙이고 자연광을 받을 수 있도록 창문 근처에 배치하세요. 출입구를 막지 않도록 해주세요.",
    "Place the new sofa directly across from the existing sofa to create a cozy seating area for conversations.": "새 소파를 기존 소파의 정면 맞은편에 배치해서 대화하기 좋은 아늑한 좌석 공간을 만드세요.",
    "Please place the new desk against the wall and near the existing desk, ensuring it's positioned to make good use of the natural light from the windows for working.": "새 책상을 벽에 붙이고 기존 책상 근처에 배치하세요. 작업할 때 창문에서 들어오는 자연광을 잘 활용할 수 있는 위치여야 합니다.",
    "Please place the new bookshelf against the wall beside the existing bookshelves, creating a continuous line of storage along that wall. Make sure it's not blocking the desk area, so there's still plenty of room to work.": "새 책장을 기존 책장 옆 벽에 붙여 배치해서 그 벽을 따라 수납장이 이어지도록 해주세요. 책상 공간을 막지 않아 작업할 공간이 충분히 남아야 합니다.",
    "Place the bed against the wall and make sure it is near the window so you can enjoy the natural light in the mornings.": "침대를 벽에 붙이고 아침 자연광을 즐길 수 있도록 창문 근처에 배치하세요.",
    "Place the bed against the wall, positioned near the armchair so that it creates a cozy corner for reading or relaxing.": "침대를 벽에 붙이고 암체어 근처에 배치해서 독서하거나 쉬기 좋은 아늑한 코너를 만드세요.",
    "Please place the nightstand beside the existing nightstand, making sure it's on the same side as the bed for easy access when lying down.": "새 협탁을 기존 협탁 옆에 배치하세요. 침대에 누웠을 때 쉽게 닿을 수 있도록 침대와 같은 쪽에 있어야 합니다.",
    "Please place the bed against the longest wall in the room, with enough space on either side for easy access. Position the bed so that it is near one of the armchairs, creating a cozy reading nook.": "침대를 방에서 가장 긴 벽에 붙여 배치하고, 양쪽에 접근하기 쉬운 충분한 공간을 남겨주세요. 암체어 중 하나 근처에 두어 아늑한 독서 공간이 되게 하세요.",
    "Please place the dining chair near the coffee table, so it can be used for extra seating when we have guests over. Make sure it's not blocking the way to the sofa.": "손님이 왔을 때 추가 좌석으로 쓸 수 있도록 식탁 의자를 커피 테이블 근처에 배치하세요. 소파로 가는 길을 막지 않게 해주세요.",
    "Place the armchair beside the existing armchair to create a cozy reading nook.": "암체어를 기존 암체어 옆에 배치해서 아늑한 독서 공간을 만드세요.",
    "Place the new sofa against the wall beside the existing sofa, so they form a continuous seating area.": "새 소파를 기존 소파 옆 벽에 붙여 배치해서 좌석 공간이 이어지도록 하세요.",
    "Place the armchair near the window so I can sit and enjoy the view or read a book.": "앉아서 경치를 보거나 책을 읽을 수 있도록 암체어를 창문 근처에 배치하세요.",
    "Place the armchair near the existing armchair, creating a cozy reading area, but make sure it's not blocking the window so we can still enjoy the natural light.": "암체어를 기존 암체어 근처에 배치해 아늑한 독서 공간을 만드세요. 단, 자연광을 계속 받을 수 있도록 창문을 막지 않아야 합니다.",
    "Please place the additional dining chair beside one of the existing dining chairs around the coffee table, so we can have extra seating for guests when they are relaxing or having a coffee.": "추가 식탁 의자를 커피 테이블 주변의 기존 식탁 의자 중 하나 옆에 배치하세요. 손님이 쉬거나 커피를 마실 때 추가 좌석으로 사용할 수 있게 해주세요.",
    "Please place the armchair near the desk so I can sit comfortably while reading. Make sure it's positioned to face the large window to enjoy the natural light.": "편하게 앉아 책을 읽을 수 있도록 암체어를 책상 근처에 배치하세요. 자연광을 즐길 수 있도록 큰 창문을 바라보는 방향이어야 합니다.",
    "Please place the coffee table in front of one of the sofas, making sure it's close enough for easy access to set down drinks or snacks while sitting on the sofa. Ensure that it doesn't block any doorways.": "커피 테이블을 소파 중 하나 앞에 배치하세요. 소파에 앉아 음료나 간식을 내려놓기 쉽게 충분히 가까워야 하며, 출입구를 막지 않아야 합니다.",
    "Place the TV stand directly in front of the bed so that it is easy to watch TV while lying down.": "침대에 누워 TV를 보기 쉽도록 TV 스탠드를 침대 바로 앞에 배치하세요.",
    "Please place the coffee table in front of the sofas so that it's easily accessible from both. Make sure it does not block walking paths between the sofas and the TV stand.": "두 소파에서 모두 쉽게 닿을 수 있도록 커피 테이블을 소파들 앞에 배치하세요. 소파와 TV 스탠드 사이의 이동 동선을 막지 않게 해주세요.",
    "Please place the bed against the wall opposite the TV stand so I can easily watch TV from bed. Make sure it doesn't block the door.": "침대에서 TV를 쉽게 볼 수 있도록 TV 스탠드 맞은편 벽에 침대를 붙여 배치하세요. 문을 막지 않도록 해주세요.",
    "Place the coffee table directly in front of one of the sofas, ensuring it's close enough for easy access from the sofa but not blocking any of the doors.": "커피 테이블을 소파 중 하나의 바로 앞에 배치하세요. 소파에서 쉽게 닿을 만큼 가까워야 하지만 어떤 문도 막으면 안 됩니다.",
    "Place the coffee table in front of one of the sofas so that it is easily reachable from the seating area. Make sure it's positioned in a way that it doesn't block the door or any windows.": "좌석 공간에서 쉽게 닿을 수 있도록 커피 테이블을 소파 중 하나 앞에 배치하세요. 문이나 창문을 막지 않는 위치여야 합니다.",
    "Please place the coffee table in front of one of the sofas, making sure it's close enough for someone sitting on the sofa to easily reach. It should be centered with the sofa and ensure there is enough space to walk around it without blocking access to the armchairs.": "커피 테이블을 소파 중 하나 앞에 배치하세요. 소파에 앉은 사람이 쉽게 닿을 만큼 가까워야 합니다. 소파 중심에 맞추고, 암체어로 가는 길을 막지 않으면서 주변을 걸어 다닐 공간을 충분히 남겨주세요.",
    "Place the coffee table directly in front of the sofa so that it's easy to reach when sitting.": "앉았을 때 쉽게 닿을 수 있도록 커피 테이블을 소파 바로 앞에 배치하세요.",
    "Place the TV stand against the wall directly facing the bed, so I can comfortably watch TV while lying in bed.": "침대에 누워 편하게 TV를 볼 수 있도록 TV 스탠드를 침대를 정면으로 마주보는 벽에 붙여 배치하세요.",
    "Please place the tv stand directly across from the bed so I can comfortably watch TV while lying down.": "누워서 편하게 TV를 볼 수 있도록 TV 스탠드를 침대 바로 맞은편에 배치하세요.",
    "Place the sofa against the wall facing the TV stand, ensuring that it's close enough for comfortable viewing. Make sure there's enough space to walk around it without blocking any of the doors or windows.": "소파를 TV 스탠드를 바라보는 벽에 붙여 배치하고, 편하게 볼 수 있을 만큼 가까이 두세요. 문이나 창문을 막지 않으면서 주변을 걸어 다닐 공간을 충분히 남겨주세요.",
    "Place the armchair beside one of the existing armchairs, creating a cozy seating area. Make sure it's facing the coffee table for easy conversation or relaxing while enjoying a drink.": "암체어를 기존 암체어 중 하나 옆에 배치해서 아늑한 좌석 공간을 만드세요. 대화하거나 음료를 마시며 쉬기 쉽도록 커피 테이블을 바라보게 해주세요.",
    "Place the TV stand against the wall opposite the sofa, so that it faces the sofa and is easily viewable from there. Ensure it does not block access to any of the doors.": "TV 스탠드를 소파 맞은편 벽에 붙여 배치해서 소파에서 보기 쉽게 소파를 바라보게 하세요. 어떤 문으로 가는 길도 막지 않아야 합니다.",
    "Place the new sofa against the wall opposite the TV stand, so that anyone sitting on it can easily watch TV. Ensure it is beside one of the existing sofas to create a cozy seating area.": "새 소파를 TV 스탠드 맞은편 벽에 붙여 배치해서 앉은 사람이 TV를 쉽게 볼 수 있게 하세요. 기존 소파 중 하나 옆에 두어 아늑한 좌석 공간을 만들도록 해주세요.",
    "Place the bed against the wall opposite the TV stand, making sure it faces the TV stand so that it's convenient for watching TV. Ensure not to block the door or the window.": "침대를 TV 스탠드 맞은편 벽에 붙여 배치하고, TV 보기 편하도록 TV 스탠드를 바라보게 하세요. 문이나 창문을 막지 않아야 합니다.",
    "Place the new sofa right beside the existing sofa, ensuring it's directly adjacent to it. This way, it forms a cozy seating area for watching TV.": "새 소파를 기존 소파 바로 옆에 붙여 배치하세요. 이렇게 해서 TV를 보기 좋은 아늑한 좌석 공간을 만드세요.",
    "Place the new sofa against the wall opposite the window, facing the existing coffee table, so it's easy to have conversations and enjoy the view outside.": "새 소파를 창문 맞은편 벽에 붙여 배치하고 기존 커피 테이블을 바라보게 하세요. 대화하기 쉽고 바깥 경치도 즐길 수 있어야 합니다.",
    "Place the bed against the wall opposite the TV stand so that it's easy to watch TV from the bed. Ensure it's not blocking the door or the window.": "침대에서 TV를 보기 쉽도록 침대를 TV 스탠드 맞은편 벽에 붙여 배치하세요. 문이나 창문을 막지 않아야 합니다.",
    "Place the coffee table in front of the sofa so it's easy to reach when sitting down. Make sure it's aligned with the center of the sofa for a balanced look.": "앉았을 때 쉽게 닿을 수 있도록 커피 테이블을 소파 앞에 배치하세요. 균형 잡힌 느낌이 나도록 소파 중심에 맞춰주세요.",
    "Place the bed against the longest wall in the bedroom, ensuring it's near the window for natural light, but not blocking any of the windows. Position it beside the armchair so that the chair can serve as a convenient spot for reading or relaxing before bedtime.": "침대를 침실에서 가장 긴 벽에 붙여 배치하세요. 자연광을 받을 수 있도록 창문 근처에 두되, 어떤 창문도 막지 않아야 합니다. 잠들기 전 독서나 휴식을 위한 자리로 쓸 수 있도록 암체어 옆에 배치하세요.",
    "Place the TV stand against the wall, directly facing the bed, so that it’s easy to watch TV while lying in bed.": "침대에 누워 TV를 보기 쉽도록 TV 스탠드를 침대를 정면으로 바라보는 벽에 붙여 배치하세요.",
    "Place the TV stand against the wall directly opposite the sofa so that you can comfortably watch TV from the sofa without any obstruction. Make sure it's centered with the sofa and leave enough space for access from both sides.": "소파에서 방해 없이 편하게 TV를 볼 수 있도록 TV 스탠드를 소파 정면 맞은편 벽에 붙여 배치하세요. 소파 중심에 맞추고 양쪽에서 접근할 수 있는 공간을 충분히 남겨주세요.",
    "Place the new sofa beside one of the existing sofas, creating an L-shape for a cozy conversation area.": "새 소파를 기존 소파 중 하나 옆에 배치해서 아늑한 대화 공간이 되도록 L자 형태를 만드세요.",
    "Place the armchair near the window so that it faces the bed, providing a cozy spot to sit and read with natural light.": "자연광을 받으며 앉아 책을 읽기 좋은 공간이 되도록 암체어를 창문 근처에 배치하고 침대를 바라보게 하세요.",
    "Please place the nightstand beside the bed, so it's convenient for reaching things while lying in bed. Make sure it's not blocking any doors or windows.": "침대에 누웠을 때 물건을 집기 편하도록 협탁을 침대 옆에 배치하세요. 문이나 창문을 막지 않아야 합니다.",
}


def _translate_instruction(text: str) -> str:
    return INSTRUCTION_TRANSLATIONS.get(text, text)


def _load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _slug(text: str, max_len: int = 120) -> str:
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
        return (
            float(size.get("width", 0.5)),
            float(size.get("depth", 0.5)),
            float(size.get("height", 0.5)),
        )
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
    alpha: float = 0.9,
    lw: float = 1.0,
    linestyle: str = "-",
    zorder: int = 2,
) -> None:
    patch = mpatches.Polygon(
        _corners(cx, cz, width, depth, yaw_deg),
        closed=True,
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        alpha=alpha,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)


def _add_heading(ax: Any, cx: float, cz: float, width: float, depth: float, yaw_deg: float) -> None:
    yaw = math.radians(yaw_deg)
    length = max(0.18, min(width, depth) * 0.35)
    ax.annotate(
        "",
        xy=(cx + math.cos(yaw) * length, cz + math.sin(yaw) * length),
        xytext=(cx, cz),
        arrowprops={"arrowstyle": "->", "color": "#1F1F1F", "lw": 1.4},
        zorder=7,
    )


def _render_image(case: Mapping[str, Any], pred: Mapping[str, Any], out_path: Path) -> None:
    scene = case["scene"]
    floor = [(float(x), float(z)) for x, z in scene["floor"]["polygon"]]
    xs = [p[0] for p in floor]
    zs = [p[1] for p in floor]
    room_w = max(xs) - min(xs)
    room_h = max(zs) - min(zs)
    aspect = max(0.75, min(1.6, room_w / max(room_h, 1e-6)))

    fig, ax = plt.subplots(figsize=(6.0 * aspect, 6.0), dpi=150)
    ax.set_aspect("equal")
    ax.add_patch(
        mpatches.Polygon(
            floor,
            closed=True,
            facecolor="#F7F4EC",
            edgecolor="#2E2E2E",
            linewidth=1.8,
            zorder=0,
        )
    )

    for obj in scene.get("objects", []):
        ox, oz = _position_xz(obj)
        ow, od, _ = _size_wdh(obj)
        _add_box(ax, ox, oz, ow, od, _yaw_deg(obj), "#C9CED3", "#6E7378", alpha=0.82, zorder=2)
        label = str(obj.get("category", "")).replace("_", " ")[:12]
        ax.text(ox, oz, label, ha="center", va="center", fontsize=4.5, color="#1F2933", zorder=5)

    for door in scene.get("doors", []):
        dx, dz = _position_xz(door)
        ax.scatter([dx], [dz], marker="s", s=48, color="#A65B2A", zorder=4)
        ax.text(dx, dz, "D", ha="center", va="center", fontsize=6, color="white", weight="bold", zorder=5)

    for win in scene.get("windows", []):
        wx, wz = _position_xz(win)
        ax.scatter([wx], [wz], marker="D", s=42, color="#2F80C1", zorder=4)
        ax.text(wx, wz, "W", ha="center", va="center", fontsize=5.5, color="white", weight="bold", zorder=5)

    ref = case.get("reference_pose")
    if ref:
        tx = float(ref["position"]["x"])
        tz = float(ref["position"]["z"])
        tw, td, _ = _size_wdh(case["target_asset"])
        _add_box(ax, tx, tz, tw, td, float(ref.get("rotation_y", 0.0)), "none", "#E0B422", alpha=1.0, lw=1.6, linestyle="--", zorder=5)

    if pred.get("status") == "placed":
        px, pz = _position_xz(pred)
        pw, pd, _ = _size_wdh(pred)
        pyaw = _yaw_deg(pred)
        _add_box(ax, px, pz, pw, pd, pyaw, "#43B77A", "#0B5D3B", alpha=0.88, lw=1.8, zorder=6)
        _add_heading(ax, px, pz, pw, pd, pyaw)
        ax.text(px, pz, "TARGET", ha="center", va="center", fontsize=5.0, color="#062B1D", weight="bold", zorder=8)
    else:
        ax.text(0.5, 0.5, f"NOT PLACED\n{str(pred.get('reason', ''))[:80]}", transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#8A1F11")

    margin = max(room_w, room_h) * 0.08
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(zs) - margin, max(zs) + margin)
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _metrics_by_case(results_path: Optional[str | Path]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not results_path:
        return {}
    data = _load_json(results_path)
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for method in data.get("methods", []):
        name = method.get("source_name") or method.get("method")
        for scene in method.get("scenes", []):
            cid = scene.get("query_id") or scene.get("scene_id")
            if not cid:
                continue
            candidate_metrics = scene.get("candidate_metrics") or []
            top = candidate_metrics[0] if candidate_metrics else {}
            out[(str(name), str(cid))] = {
                "success_at_1": scene.get("success_at_1"),
                "success_at_5": scene.get("success_at_5"),
                "top1": top,
            }
    return out


def _case_score(case: Mapping[str, Any], methods: Sequence[str], metrics: Mapping[Tuple[str, str], Mapping[str, Any]]) -> float:
    cid = str(case["id"])
    best = 0.0
    for method in methods:
        item = metrics.get((method, cid), {})
        top = item.get("top1", {})
        score = 0.0
        score += 2.0 * float(item.get("success_at_1") or 0.0)
        score += 1.0 * float(item.get("success_at_5") or 0.0)
        score += float(top.get("constraint_accuracy") or 0.0)
        score += 0.5 * float(top.get("cf") or 0.0)
        score += 0.5 * float(top.get("ib") or 0.0)
        best = max(best, score)
    return best


def _select_cases(
    cases: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    methods: Sequence[str],
    metrics: Mapping[Tuple[str, str], Mapping[str, Any]],
    mode: str,
    n: int,
    offset: int,
    seed: int,
) -> List[Mapping[str, Any]]:
    rng = random.Random(seed)
    available = [c for c in cases if any(str(c["id"]) in predictions.get(m, {}) for m in methods)]
    if mode == "all":
        return list(available)[offset:offset + n]
    if mode == "random":
        pool = list(available)
        rng.shuffle(pool)
        return pool[offset:offset + n]
    if mode == "high_metric":
        ranked = sorted(available, key=lambda c: _case_score(c, methods, metrics), reverse=True)
        return ranked[offset:offset + n]
    if mode == "method_disagreement":
        def disagreement(case: Mapping[str, Any]) -> Tuple[int, float]:
            cid = str(case["id"])
            statuses = []
            for method in methods:
                pred = (predictions.get(method, {}).get(cid) or [{}])[0]
                statuses.append(pred.get("status") == "placed")
            return (len(set(statuses)), _case_score(case, methods, metrics))

        ranked = sorted(available, key=disagreement, reverse=True)
        return ranked[offset:offset + n]
    return [c for c in available if mode in str(c["id"])][offset:offset + n]


def _record_for(
    case: Mapping[str, Any],
    method: str,
    pred: Mapping[str, Any],
    rel_image: str,
    metrics: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> Dict[str, Any]:
    cid = str(case["id"])
    raw_instruction = str(case.get("intent", {}).get("text", ""))
    return {
        "item_id": f"{cid}__{method}",
        "case_id": cid,
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "image_path": rel_image.replace("\\", "/"),
        "instruction": _translate_instruction(raw_instruction),
        "instruction_original": raw_instruction,
        "target_category": case.get("target_asset", {}).get("category"),
        "room_type": case.get("scene", {}).get("room_type"),
        "prediction_status": pred.get("status"),
        "prediction_reason": pred.get("reason"),
        "metrics": metrics.get((method, cid), {}),
        "questions": QUESTIONS,
        "human_labels": {q["id"]: None for q in QUESTIONS},
        "failure_tags": [],
        "notes": "",
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_html(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = json.dumps(rows, ensure_ascii=False)
    tags_payload = json.dumps(FAILURE_TAGS)
    questions_payload = json.dumps(QUESTIONS)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SpaceFit 시각 평가</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f7f8; color: #20242a; }}
    header {{ position: sticky; top: 0; z-index: 2; background: #ffffff; border-bottom: 1px solid #d9dde2; padding: 12px 18px; display: flex; gap: 12px; align-items: center; }}
    button {{ border: 1px solid #b9c0c8; background: #ffffff; padding: 7px 10px; border-radius: 6px; cursor: pointer; }}
    main {{ padding: 18px; display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 18px; }}
    article {{ background: white; border: 1px solid #d9dde2; border-radius: 8px; overflow: hidden; }}
    img {{ width: 100%; display: block; background: #fff; }}
    .body {{ padding: 12px; }}
    .meta {{ font-size: 12px; color: #5c6670; line-height: 1.4; }}
    .instruction {{ margin: 8px 0 10px; font-size: 13px; line-height: 1.35; }}
    .q {{ display: grid; grid-template-columns: 1fr auto auto auto; gap: 6px; align-items: center; font-size: 12px; padding: 4px 0; border-top: 1px solid #eef0f2; }}
    .tags {{ margin-top: 8px; display: flex; flex-wrap: wrap; gap: 5px; }}
    .tag {{ font-size: 11px; border: 1px solid #c8ced6; border-radius: 999px; padding: 3px 7px; }}
    textarea {{ width: 100%; min-height: 54px; margin-top: 8px; box-sizing: border-box; }}
  </style>
</head>
<body>
<header>
  <strong>SpaceFit 시각 평가</strong>
  <span id="count"></span>
  <button onclick="exportJsonl()">라벨 JSONL 내보내기</button>
  <button onclick="localStorage.removeItem('spacefit_visual_audit_v1'); location.reload()">라벨 초기화</button>
</header>
<main id="app"></main>
<script>
const rows = {payload};
const questions = {questions_payload};
const tags = {tags_payload};
const storageKey = 'spacefit_visual_audit_v1';
let labels = JSON.parse(localStorage.getItem(storageKey) || '{{}}');
function save() {{ localStorage.setItem(storageKey, JSON.stringify(labels)); }}
function ensure(id) {{
  if (!labels[id]) labels[id] = {{human_labels: {{}}, failure_tags: [], notes: ''}};
  return labels[id];
}}
function setAnswer(id, q, val) {{ ensure(id).human_labels[q] = val; save(); }}
function toggleTag(id, tag, checked) {{
  const item = ensure(id);
  const set = new Set(item.failure_tags || []);
  if (checked) set.add(tag); else set.delete(tag);
  item.failure_tags = Array.from(set);
  save();
}}
function setNotes(id, val) {{ ensure(id).notes = val; save(); }}
function render() {{
  document.getElementById('count').textContent = rows.length + '개 항목';
  const app = document.getElementById('app');
  app.innerHTML = rows.map(r => {{
    const state = ensure(r.item_id);
    const qHtml = questions.map(q => {{
      const current = state.human_labels[q.id];
      return `<div class="q"><span>${{q.text}}</span>
        <label><input type="radio" name="${{r.item_id}}_${{q.id}}" ${{current===true?'checked':''}} onchange="setAnswer('${{r.item_id}}','${{q.id}}',true)"> 예</label>
        <label><input type="radio" name="${{r.item_id}}_${{q.id}}" ${{current===false?'checked':''}} onchange="setAnswer('${{r.item_id}}','${{q.id}}',false)"> 아니오</label>
        <label><input type="radio" name="${{r.item_id}}_${{q.id}}" ${{current===null||current===undefined?'checked':''}} onchange="setAnswer('${{r.item_id}}','${{q.id}}',null)"> 모름</label>
      </div>`;
    }}).join('');
    const tagHtml = tags.map(t => `<label class="tag"><input type="checkbox" ${{(state.failure_tags||[]).includes(t)?'checked':''}} onchange="toggleTag('${{r.item_id}}','${{t}}',this.checked)"> ${{t}}</label>`).join('');
    return `<article>
      <img src="${{r.image_path}}" alt="${{r.item_id}}">
      <div class="body">
        <div class="meta"><b>${{r.method_label}}</b> | ${{r.target_category}} | ${{r.room_type}}<br>${{r.case_id}}</div>
        <div class="instruction">${{r.instruction}}</div>
        ${{qHtml}}
        <div class="tags">${{tagHtml}}</div>
        <textarea placeholder="메모" oninput="setNotes('${{r.item_id}}', this.value)">${{state.notes || ''}}</textarea>
      </div>
    </article>`;
  }}).join('');
}}
function exportJsonl() {{
  const merged = rows.map(r => {{
    const state = ensure(r.item_id);
    return {{...r, human_labels: state.human_labels || {{}}, failure_tags: state.failure_tags || [], notes: state.notes || ''}};
  }});
  const text = merged.map(r => JSON.stringify(r)).join('\\n') + '\\n';
  const blob = new Blob([text], {{type: 'application/jsonl'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'visual_audit_labels_export.jsonl';
  a.click();
}}
render();
</script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a visual audit dataset for placement results.")
    p.add_argument("--cases", default="spacefit_v2/data/single_target_benchmark/gpt_intent_cases_test.json")
    p.add_argument("--predictions", default="spacefit_v2/results/experiment_final/test_gpt_intent/raw_predictions.json")
    p.add_argument("--results", default=None)
    p.add_argument("--out_dir", default="spacefit_v2/results/visual_audit_v1")
    p.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--offset", type=int, default=0, help="Skip this many selected cases before taking --n.")
    p.add_argument("--mode", choices=["high_metric", "method_disagreement", "random", "all"], default="high_metric")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(args: argparse.Namespace) -> None:
    cases = _load_json(args.cases)
    predictions = _load_json(args.predictions)
    methods = [m for m in args.methods if m in predictions]
    if not methods:
        raise SystemExit("No requested methods are present in predictions.")

    metrics = _metrics_by_case(args.results)
    selected = _select_cases(cases, predictions, methods, metrics, args.mode, args.n, args.offset, args.seed)

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    rows: List[Dict[str, Any]] = []

    for case_idx, case in enumerate(selected, start=1):
        cid = str(case["id"])
        for method in methods:
            preds = predictions.get(method, {}).get(cid) or []
            pred = preds[0] if preds else {"status": "missing", "reason": "no prediction"}
            image_name = f"{case_idx:03d}_{_slug(cid, 80)}__{_slug(method, 40)}.png"
            image_path = image_dir / image_name
            _render_image(case, pred, image_path)
            rel_image = str(image_path.relative_to(out_dir))
            rows.append(_record_for(case, method, pred, rel_image, metrics))

    _write_jsonl(out_dir / "labels.jsonl", rows)
    (out_dir / "questions.json").write_text(json.dumps({"questions": QUESTIONS, "failure_tags": FAILURE_TAGS}, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "cases": len(selected),
                "items": len(rows),
                "methods": methods,
                "mode": args.mode,
                "offset": args.offset,
                "source_cases": args.cases,
                "source_predictions": args.predictions,
                "source_results": args.results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_html(out_dir / "index.html", rows)

    print(json.dumps({"out_dir": str(out_dir), "cases": len(selected), "items": len(rows), "html": str(out_dir / "index.html")}, indent=2))


if __name__ == "__main__":
    main(build_parser().parse_args())
