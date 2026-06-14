"""Run rubric-based VLM-as-judge evaluation for placement quality.

This judge is intended to extend human qualitative criteria to more cases.
It should be reported as an auxiliary visual-quality evaluator, not as ground
truth.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]


SYSTEM_PROMPT = """\
You are a calibrated visual judge for indoor single-target furniture placement.
Your goal is to extend human qualitative evaluation with a consistent rubric.

You will see a top-down floor plan:
- gray/blue rectangles: existing fixed furniture
- green rectangle: newly placed target furniture
- white arrow on green rectangle: target facing direction
- brown marks: doors
- blue marks: windows
- yellow dashed outline, when present: original/reference target pose

Evaluate the target placement with respect to the user intent and visual quality.
Do not over-penalize minor grid-level offsets. Be strict for visible collision,
outside-room placement, blocked doors/main access, clearly wrong relations, and
obviously unnatural isolated or cramped placement.

Respond with ONLY valid JSON:
{
  "physical_validity": 0 or 1,
  "accessibility": 0 or 1,
  "relation_satisfaction": 0 or 1,
  "orientation_naturalness": 0 or 1,
  "grouping_naturalness": 0 or 1,
  "overall_naturalness": 0 or 1,
  "quality_score": <integer 0-10>,
  "main_issue": "none|collision|out_of_boundary|access_blocked|relation_mismatch|orientation_issue|awkward_grouping|isolated|cramped|ambiguous",
  "reason": "<one short sentence>"
}

Rubric:
- physical_validity: target is inside the room and not visibly overlapping existing furniture.
- accessibility: target does not block doors, windows, main walking paths, or expected use space.
- relation_satisfaction: target roughly satisfies the requested relation, such as near, beside, in front of, against wall, near window, or facing another item.
- orientation_naturalness: target direction is natural for the furniture and intent. If direction is irrelevant or visually ambiguous, judge from the layout and do not be overly strict.
- grouping_naturalness: target belongs to a plausible furniture group instead of being isolated, randomly placed, or incoherent.
- overall_naturalness: a human would accept this placement as usable and visually plausible.
- quality_score: 0-2 unusable, 3-4 poor, 5-6 acceptable but awkward, 7-8 good, 9-10 excellent.
"""


def load_dotenv(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def pos_xz(item: Mapping[str, Any]) -> Tuple[float, float]:
    pos = item.get("position", {})
    if isinstance(pos, Mapping):
        return float(pos.get("x", 0.0)), float(pos.get("z", 0.0))
    if isinstance(pos, Sequence) and not isinstance(pos, (str, bytes)):
        return float(pos[0]), float(pos[2] if len(pos) > 2 else pos[1])
    return float(item.get("x", 0.0)), float(item.get("z", 0.0))


def size_wd(item: Mapping[str, Any]) -> Tuple[float, float]:
    size = item.get("size", {})
    if isinstance(size, Mapping):
        return float(size.get("width", 0.5)), float(size.get("depth", 0.5))
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)):
        return float(size[0]), float(size[2] if len(size) > 2 else size[-1])
    return float(item.get("width", 0.5)), float(item.get("depth", 0.5))


def yaw_deg(item: Mapping[str, Any]) -> float:
    raw = float(item.get("rotation_y", item.get("yaw", 0.0)) or 0.0)
    return math.degrees(raw) if abs(raw) <= 2 * math.pi + 1e-3 else raw


def corners(cx: float, cz: float, width: float, depth: float, yaw: float) -> List[Tuple[float, float]]:
    hw = max(width, 0.05) * 0.5
    hd = max(depth, 0.05) * 0.5
    rad = math.radians(yaw)
    c, s = math.cos(rad), math.sin(rad)
    return [(cx + dx * c - dz * s, cz + dx * s + dz * c) for dx, dz in [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]]


def add_box(ax: Any, cx: float, cz: float, w: float, d: float, yaw: float, face: str, edge: str, alpha: float, lw: float, z: int, label: str = "") -> None:
    ax.add_patch(patches.Polygon(corners(cx, cz, w, d, yaw), closed=True, facecolor=face, edgecolor=edge, alpha=alpha, linewidth=lw, zorder=z))
    if label:
        ax.text(cx, cz, label, ha="center", va="center", fontsize=5.8, color="#111827", zorder=z + 1)


def add_arrow(ax: Any, cx: float, cz: float, yaw: float, length: float) -> None:
    rad = math.radians(yaw)
    dx, dz = math.cos(rad) * length, math.sin(rad) * length
    ax.annotate("", xy=(cx + dx, cz + dz), xytext=(cx, cz), arrowprops={"arrowstyle": "-|>", "color": "white", "lw": 1.8}, zorder=12)


def render_case_prediction(case: Mapping[str, Any], prediction: Mapping[str, Any], out_path: Path) -> Path:
    scene = case["scene"]
    floor = [(float(x), float(z)) for x, z in scene["floor"]["polygon"]]
    xs = [p[0] for p in floor]
    zs = [p[1] for p in floor]
    width = max(xs) - min(xs)
    depth = max(zs) - min(zs)
    aspect = max(0.75, min(1.65, width / max(depth, 1e-6)))
    fig, ax = plt.subplots(figsize=(6.0 * aspect, 6.0), dpi=150)
    ax.set_aspect("equal")
    ax.add_patch(patches.Polygon(floor, closed=True, facecolor="#f8fafc", edgecolor="#111827", linewidth=2.0, zorder=0))

    for obj in scene.get("objects", []):
        x, z = pos_xz(obj)
        w, d = size_wd(obj)
        label = str(obj.get("category", "")).replace("_", " ")[:12]
        add_box(ax, x, z, w, d, yaw_deg(obj), "#dbeafe", "#64748b", 0.82, 1.1, 2, label=label)

    for door in scene.get("doors", []):
        x, z = pos_xz(door)
        ax.plot(x, z, marker="s", color="#92400e", markersize=10, zorder=8)
        ax.text(x, z + 0.15, "DOOR", ha="center", va="bottom", fontsize=6, color="#92400e", weight="bold", zorder=9)

    for win in scene.get("windows", []):
        x, z = pos_xz(win)
        ax.plot(x, z, marker="D", color="#0284c7", markersize=9, zorder=8)
        ax.text(x, z + 0.15, "WINDOW", ha="center", va="bottom", fontsize=6, color="#0284c7", weight="bold", zorder=9)

    ref = case.get("reference_pose")
    if ref:
        x, z = pos_xz(ref)
        w = float(case["target_asset"]["size"]["width"])
        d = float(case["target_asset"]["size"]["depth"])
        ax.add_patch(patches.Polygon(corners(x, z, w, d, yaw_deg(ref)), closed=True, facecolor="none", edgecolor="#eab308", linewidth=2.0, linestyle="--", zorder=5))

    if prediction.get("status") == "placed":
        x, z = pos_xz(prediction)
        w, d = size_wd(prediction)
        yaw = yaw_deg(prediction)
        add_box(ax, x, z, w, d, yaw, "#22c55e", "#052e16", 0.92, 2.4, 10)
        add_arrow(ax, x, z, yaw, max(0.18, min(w, d) * 0.36))
        target = str(case.get("target_asset", {}).get("category", prediction.get("category", "target"))).replace("_", " ")
        ax.text(x, z, "TARGET\n" + target[:12], ha="center", va="center", fontsize=6.2, color="white", weight="bold", zorder=13)
    else:
        ax.text(0.5, 0.5, "TARGET NOT PLACED", transform=ax.transAxes, ha="center", va="center", fontsize=16, color="#b91c1c", weight="bold")

    ax.text(
        0.01,
        0.99,
        "green=target | arrow=facing | gray/blue=fixed | brown=door | blue=window | yellow dashed=GT",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.88},
        zorder=20,
    )
    margin = 0.55
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(zs) - margin, max(zs) + margin)
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def image_data_url(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def parse_json_response(text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_judgment(raw: Mapping[str, Any]) -> Dict[str, Any]:
    keys = [
        "physical_validity",
        "accessibility",
        "relation_satisfaction",
        "orientation_naturalness",
        "grouping_naturalness",
        "overall_naturalness",
    ]
    out: Dict[str, Any] = {}
    for key in keys:
        value = raw.get(key, 0)
        if isinstance(value, bool):
            out[key] = int(value)
        else:
            out[key] = 1 if str(value).strip().lower() in {"1", "true", "yes"} else 0
    try:
        out["quality_score"] = max(0, min(10, int(round(float(raw.get("quality_score", 0))))))
    except Exception:
        out["quality_score"] = 0
    out["main_issue"] = str(raw.get("main_issue", "ambiguous"))
    out["reason"] = str(raw.get("reason", ""))[:500]
    return out


def judge_one(client: Any, model: str, case: Mapping[str, Any], prediction: Mapping[str, Any], image_path: Path) -> Dict[str, Any]:
    target = str(case.get("target_asset", {}).get("category", prediction.get("category", "target"))).replace("_", " ")
    intent = case.get("intent", {}).get("text", "")
    user_text = (
        f"Target furniture: {target}\n"
        f"User intent: {intent}\n"
        "Please judge the placement in the image using the rubric."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path), "detail": "low"}},
                ],
            },
        ],
        temperature=0.0,
        max_tokens=320,
    )
    content = response.choices[0].message.content or "{}"
    return normalize_judgment(parse_json_response(content))


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("error"):
            by_method[row["method"]].append(row)
    summary: List[Dict[str, Any]] = []
    fields = [
        "physical_validity",
        "accessibility",
        "relation_satisfaction",
        "orientation_naturalness",
        "grouping_naturalness",
        "overall_naturalness",
        "quality_score",
    ]
    for method, items in by_method.items():
        entry: Dict[str, Any] = {"method": method, "n": len(items)}
        for field in fields:
            entry[field] = sum(float(x["judgment"][field]) for x in items) / len(items) if items else 0.0
        summary.append(entry)
    summary.sort(key=lambda r: (r["overall_naturalness"], r["quality_score"]), reverse=True)
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(out_dir: Path, args: argparse.Namespace, summary: List[Dict[str, Any]], dry_run: bool) -> None:
    lines = [
        "# VLM Judge Evaluation",
        "",
        "## 목적",
        "",
        "정량 지표(CF, IB, CPS)가 설명하지 못하는 방향성, 접근성, 가구군 조화, 전체 자연스러움을 VLM judge로 보조 평가한다.",
        "이 평가는 사람 라벨을 대체하는 정답이 아니라, 제한된 human qualitative 기준을 더 많은 케이스로 확장하기 위한 보조 평가자이다.",
        "",
        "## Rubric",
        "",
        "- `physical_validity`: 방 안에 있고 기존 가구와 시각적으로 충돌하지 않는가",
        "- `accessibility`: 문, 창문, 주요 이동 동선, 사용 공간을 막지 않는가",
        "- `relation_satisfaction`: 사용자 의도상의 near/beside/facing/against wall 등 관계를 만족하는가",
        "- `orientation_naturalness`: 목표 가구 방향이 의도와 가구 특성상 자연스러운가",
        "- `grouping_naturalness`: 기존 가구군과 어울리며 고립/랜덤 배치처럼 보이지 않는가",
        "- `overall_naturalness`: 사람이 보기에도 사용 가능하고 자연스러운 배치인가",
        "- `quality_score`: 0-10 종합 품질 점수",
        "",
        "## 실행 설정",
        "",
        f"- model: `{args.model}`",
        f"- predictions: `{args.predictions}`",
        f"- cases: `{args.cases}`",
        f"- methods: `{', '.join(args.methods) if args.methods else 'all available methods'}`",
        f"- max_cases: `{args.max_cases}`",
        f"- dry_run: `{dry_run}`",
        "",
    ]
    if summary:
        lines += [
            "## Summary",
            "",
            "| Method | n | Quality | Overall | Physical | Access | Relation | Orientation | Grouping |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summary:
            lines.append(
                f"| {row['method']} | {row['n']} | {row['quality_score']:.2f} | "
                f"{row['overall_naturalness']:.3f} | {row['physical_validity']:.3f} | "
                f"{row['accessibility']:.3f} | {row['relation_satisfaction']:.3f} | "
                f"{row['orientation_naturalness']:.3f} | {row['grouping_naturalness']:.3f} |"
            )
    else:
        lines += [
            "## Summary",
            "",
            "아직 API judge 결과는 생성하지 않았다. `--dry_run`을 제거하고 API key/model을 설정하면 같은 스크립트로 실행할 수 있다.",
        ]
    lines += [
        "",
        "## 해석상 주의",
        "",
        "- VLM judge 결과는 모델/프롬프트/시각화 방식에 민감하므로 human-labeled subset과 agreement를 확인해야 한다.",
        "- 논문/발표에서는 `auxiliary VLM-based visual quality evaluation`으로 표현하는 것이 안전하다.",
        "- top-down만으로 방향이 애매한 가구는 과도하게 감점하지 않도록 rubric에 명시했다.",
    ]
    (out_dir / "VLM_JUDGE_EVALUATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rubric-based VLM-as-judge evaluation.")
    parser.add_argument("--cases", default="spacefit_v2/data/single_target_benchmark/gpt_intent_cases_test.json")
    parser.add_argument("--predictions", default="spacefit_v2/results/final_constraint_solver_human_rerank/test_gpt_intent/raw_predictions_human_reranked.json")
    parser.add_argument("--out_dir", default="spacefit_v2/results/vlm_quality_judge")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--openai_base_url", default=None)
    parser.add_argument("--max_cases", type=int, default=5)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv([ROOT / ".env", ROOT / "spacefit_v2" / ".env"])
    cases = load_json(ROOT / args.cases)
    preds = load_json(ROOT / args.predictions)
    methods = args.methods or list(preds.keys())
    if args.max_cases:
        cases = cases[: args.max_cases]
    out_dir = ROOT / args.out_dir
    image_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    client = None
    if not args.dry_run:
        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("ERROR: OPENAI_API_KEY is required unless --dry_run is set.")
        from openai import OpenAI

        kwargs: Dict[str, Any] = {"api_key": api_key}
        base_url = args.openai_base_url or os.environ.get("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)

    rows: List[Dict[str, Any]] = []
    prompt_sample_written = False
    for ci, case in enumerate(cases, 1):
        cid = case["id"]
        for method in methods:
            pred_list = preds.get(method, {}).get(cid, [])
            pred = pred_list[0] if pred_list else {"status": "missing", "category": case.get("target_asset", {}).get("category", "target")}
            image_path = image_dir / method / f"{cid}.png"
            render_case_prediction(case, pred, image_path)
            row: Dict[str, Any] = {
                "case_id": cid,
                "method": method,
                "image": str(image_path.relative_to(ROOT)),
                "target": case.get("target_asset", {}).get("category", ""),
                "intent": case.get("intent", {}).get("text", ""),
                "error": "",
            }
            if args.dry_run:
                row["judgment"] = {
                    "physical_validity": 0,
                    "accessibility": 0,
                    "relation_satisfaction": 0,
                    "orientation_naturalness": 0,
                    "grouping_naturalness": 0,
                    "overall_naturalness": 0,
                    "quality_score": 0,
                    "main_issue": "dry_run",
                    "reason": "Dry run only; no API call was made.",
                }
                if not prompt_sample_written:
                    (out_dir / "prompt_sample.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
                    prompt_sample_written = True
            else:
                try:
                    assert client is not None
                    row["judgment"] = judge_one(client, args.model, case, pred, image_path)
                except Exception as exc:
                    row["error"] = str(exc)
                    row["judgment"] = {}
            rows.append(row)
            status = "dry" if args.dry_run else ("err" if row["error"] else row["judgment"].get("quality_score", "?"))
            print(f"[{ci}/{len(cases)}|{method}] {cid[:48]} score={status}")
            if args.sleep and not args.dry_run:
                time.sleep(args.sleep)

    summary = [] if args.dry_run else summarize(rows)
    write_json(out_dir / "vlm_judge_predictions.json", rows)
    write_json(out_dir / "vlm_judge_summary.json", summary)
    flat_summary = [{k: v for k, v in row.items()} for row in summary]
    write_csv(out_dir / "vlm_judge_summary.csv", flat_summary)
    write_report(out_dir, args, summary, dry_run=args.dry_run)
    print(out_dir / "VLM_JUDGE_EVALUATION.md")


if __name__ == "__main__":
    main()
