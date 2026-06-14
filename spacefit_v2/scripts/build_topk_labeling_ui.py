"""Build a top-k preference labeling UI for human-aligned reranker training."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from spacefit_v2.scripts.build_visual_audit import _render_image, _slug, _write_jsonl


ISSUE_TAGS = [
    "충돌/겹침",
    "방 밖으로 나감",
    "요청 관계가 틀림",
    "방향이 어색함",
    "접근공간 부족",
    "문/창문을 막음",
    "가구군이 어색함",
    "너무 멀다",
    "너무 가깝다",
    "크기/좌표 오류",
]

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
    "window": "창문",
    "door": "문",
    "center": "방 중앙",
}

ROOM_KO = {
    "bedroom": "침실",
    "masterbedroom": "안방",
    "secondbedroom": "작은 침실",
    "livingroom": "거실",
    "livingdiningroom": "거실/식당",
    "library": "서재",
}


def _load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ko_category(value: Any) -> str:
    raw = str(value or "").strip()
    key = raw.lower().replace("-", "_")
    return CATEGORY_KO.get(key, CATEGORY_KO.get(raw.lower(), raw.replace("_", " ")))


def _ko_room(value: Any) -> str:
    raw = str(value or "").strip()
    return ROOM_KO.get(raw.lower(), raw)


def _object_marker(text: str) -> str:
    if not text:
        return "을/를"
    code = ord(text[-1])
    if 0xAC00 <= code <= 0xD7A3:
        return "을" if (code - 0xAC00) % 28 else "를"
    return "을/를"


def _constraint_target_ko(constraint: Mapping[str, Any]) -> str:
    if constraint.get("target_category"):
        return _ko_category(constraint.get("target_category"))
    if constraint.get("target_kind"):
        return _ko_category(constraint.get("target_kind"))
    if constraint.get("target_id"):
        return "지정된 기존 가구"
    return "관련 대상"


def _korean_instruction(case: Mapping[str, Any]) -> str:
    target = _ko_category((case.get("target_asset") or {}).get("category"))
    constraints = list((case.get("intent") or {}).get("constraints") or [])
    phrases: List[str] = []
    seen: set[str] = set()
    for constraint in constraints:
        ctype = str(constraint.get("constraint_type", ""))
        ref = _constraint_target_ko(constraint)
        if ctype == "against_wall":
            phrase = "벽에 붙여 배치"
        elif ctype == "near":
            phrase = f"{ref} 근처에 배치"
        elif ctype == "beside":
            phrase = f"{ref} 옆에 배치"
        elif ctype == "facing":
            phrase = f"{ref}을/를 바라보게 배치"
        elif ctype == "not_block_door":
            phrase = "문과 출입 동선을 막지 않기"
        elif ctype == "keep_window_clear":
            phrase = "창문을 가리지 않기"
        elif ctype == "access_zone":
            phrase = "주변 접근공간 확보"
        elif ctype == "in_front_of":
            phrase = f"{ref} 앞쪽에 배치"
        elif ctype == "left_of":
            phrase = f"{ref}의 왼쪽에 배치"
        elif ctype == "right_of":
            phrase = f"{ref}의 오른쪽에 배치"
        else:
            phrase = ctype.replace("_", " ")
        if phrase and phrase not in seen:
            phrases.append(phrase)
            seen.add(phrase)
    if not phrases:
        return f"{target}{_object_marker(target)} 방 안에 자연스럽고 사용하기 좋은 위치에 배치하세요."
    return f"{target}{_object_marker(target)} " + ", ".join(phrases) + "하도록 배치하세요."


def _select_cases(
    cases: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    method: str,
    mode: str,
    n: int,
    offset: int,
    seed: int,
) -> List[Mapping[str, Any]]:
    available = [case for case in cases if predictions.get(method, {}).get(str(case["id"]))]

    def changed(case: Mapping[str, Any]) -> bool:
        preds = predictions.get(method, {}).get(str(case["id"])) or []
        return bool(preds and int(preds[0].get("human_aligned_original_rank", 0) or 0) != 0)

    if mode == "rerank_changed":
        ranked = [case for case in available if changed(case)]
    elif mode == "rerank_unchanged":
        ranked = [case for case in available if not changed(case)]
    elif mode == "random":
        ranked = list(available)
        random.Random(seed).shuffle(ranked)
    else:
        changed_cases = [case for case in available if changed(case)]
        unchanged_cases = [case for case in available if not changed(case)]
        random.Random(seed).shuffle(unchanged_cases)
        ranked = changed_cases + unchanged_cases
    return ranked[offset : offset + n]


def _candidate_record(
    case: Mapping[str, Any],
    method: str,
    pred: Mapping[str, Any],
    rank: int,
    rel_image: str,
) -> Dict[str, Any]:
    metrics = pred.get("human_aligned_metrics") or {}
    return {
        "candidate_id": f"{case['id']}__{method}__rank{rank}",
        "rank": rank,
        "original_rank": pred.get("human_aligned_original_rank", rank),
        "human_aligned_score": pred.get("human_aligned_score"),
        "image_path": rel_image.replace("\\", "/"),
        "status": pred.get("status"),
        "category": pred.get("category") or (case.get("target_asset") or {}).get("category"),
        "position": pred.get("position"),
        "rotation_y": pred.get("rotation_y"),
        "metrics": metrics,
        "prediction": pred,
    }


def _case_record(
    case: Mapping[str, Any],
    method: str,
    candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    instruction_original = str((case.get("intent") or {}).get("text", ""))
    instruction = _korean_instruction(case)
    return {
        "item_id": f"{case['id']}__{method}__topk",
        "case_id": str(case["id"]),
        "method": method,
        "instruction": instruction,
        "instruction_original": instruction_original,
        "target_category": (case.get("target_asset") or {}).get("category"),
        "target_category_ko": _ko_category((case.get("target_asset") or {}).get("category")),
        "room_type": (case.get("scene") or {}).get("room_type"),
        "room_type_ko": _ko_room((case.get("scene") or {}).get("room_type")),
        "candidates": list(candidates),
        "human_label": {
            "best_candidate_id": None,
            "acceptable_candidate_ids": [],
            "candidate_issue_tags": {},
            "candidate_notes": {},
            "case_notes": "",
        },
    }


def _write_html(path: Path, rows: Sequence[Mapping[str, Any]], title: str) -> None:
    payload = json.dumps(rows, ensure_ascii=False)
    tags_payload = json.dumps(ISSUE_TAGS, ensure_ascii=False)
    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ margin: 0; font-family: Arial, "Malgun Gothic", sans-serif; background: #f4f6f8; color: #20242a; }}
    header {{ position: sticky; top: 0; z-index: 10; background: #111b26; color: #fff; padding: 12px 18px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
    header strong {{ font-size: 18px; }}
    button {{ border: 1px solid #b9c0c8; background: #fff; color: #111b26; padding: 8px 11px; border-radius: 6px; cursor: pointer; font-weight: 600; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 18px; }}
    .case {{ background: #fff; border: 1px solid #d7dde4; border-radius: 8px; margin-bottom: 18px; overflow: hidden; }}
    .case-head {{ padding: 14px 16px; border-bottom: 1px solid #e6eaf0; }}
    .meta {{ color: #5b6673; font-size: 12px; line-height: 1.45; }}
    .instruction {{ margin-top: 8px; font-size: 14px; line-height: 1.45; }}
    .hint {{ margin-top: 8px; padding: 8px 10px; background: #fff7e0; border: 1px solid #ecc861; border-radius: 6px; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px; padding: 14px; }}
    .cand {{ border: 2px solid #d7dde4; border-radius: 8px; overflow: hidden; background: #fbfcfd; }}
    .cand.selected {{ border-color: #238b45; box-shadow: 0 0 0 2px rgba(35,139,69,0.14); }}
    .cand img {{ width: 100%; display: block; background: #fff; }}
    .cand-body {{ padding: 10px; }}
    .rank {{ font-weight: 700; }}
    .score {{ color: #5b6673; font-size: 12px; margin-top: 3px; }}
    .controls {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 9px; align-items: center; }}
    .tags {{ display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }}
    .tag {{ font-size: 11px; border: 1px solid #c7ced6; border-radius: 999px; padding: 3px 7px; background: #fff; }}
    textarea {{ width: 100%; box-sizing: border-box; min-height: 44px; margin-top: 8px; border: 1px solid #c7ced6; border-radius: 6px; padding: 7px; font-family: inherit; }}
    .case-note {{ padding: 0 14px 14px; }}
    .progress {{ margin-left: auto; color: #dbe6f2; }}
  </style>
</head>
<body>
<header>
  <strong>{title}</strong>
  <span id="count"></span>
  <span class="progress" id="progress"></span>
  <button onclick="exportJsonl()">라벨 내보내기</button>
  <button onclick="resetLabels()">초기화</button>
</header>
<main id="app"></main>
<script>
const rows = {payload};
const issueTags = {tags_payload};
const storageKey = 'spacefit_topk_labeling_v1_' + location.pathname;
let labels = JSON.parse(localStorage.getItem(storageKey) || '{{}}');
function save() {{ localStorage.setItem(storageKey, JSON.stringify(labels)); updateProgress(); }}
function defaultLabel() {{ return {{best_candidate_id: null, acceptable_candidate_ids: [], candidate_issue_tags: {{}}, candidate_notes: {{}}, case_notes: ''}}; }}
function ensure(id) {{
  if (!labels[id]) labels[id] = defaultLabel();
  return labels[id];
}}
function setBest(itemId, candId) {{ ensure(itemId).best_candidate_id = candId; save(); render(); }}
function toggleAccept(itemId, candId, checked) {{
  const label = ensure(itemId);
  const set = new Set(label.acceptable_candidate_ids || []);
  if (checked) set.add(candId); else set.delete(candId);
  label.acceptable_candidate_ids = Array.from(set);
  save();
}}
function toggleIssue(itemId, candId, tag, checked) {{
  const label = ensure(itemId);
  if (!label.candidate_issue_tags) label.candidate_issue_tags = {{}};
  const set = new Set(label.candidate_issue_tags[candId] || []);
  if (checked) set.add(tag); else set.delete(tag);
  label.candidate_issue_tags[candId] = Array.from(set);
  save();
}}
function setCandNote(itemId, candId, value) {{
  const label = ensure(itemId);
  if (!label.candidate_notes) label.candidate_notes = {{}};
  label.candidate_notes[candId] = value;
  save();
}}
function setCaseNote(itemId, value) {{ ensure(itemId).case_notes = value; save(); }}
function fmt(value) {{ return value === null || value === undefined ? '-' : Number(value).toFixed(3); }}
function updateProgress() {{
  const done = rows.filter(r => ensure(r.item_id).best_candidate_id).length;
  document.getElementById('progress').textContent = `${{done}} / ${{rows.length}} 완료`;
}}
function render() {{
  document.getElementById('count').textContent = rows.length + '개 케이스';
  const app = document.getElementById('app');
  app.innerHTML = rows.map((r, idx) => {{
    const label = ensure(r.item_id);
    const cards = r.candidates.map(c => {{
      const selected = label.best_candidate_id === c.candidate_id;
      const acceptable = (label.acceptable_candidate_ids || []).includes(c.candidate_id);
      const issues = (label.candidate_issue_tags || {{}})[c.candidate_id] || [];
      const note = ((label.candidate_notes || {{}})[c.candidate_id] || '');
      const m = c.metrics || {{}};
      const tagHtml = issueTags.map(t => `<label class="tag"><input type="checkbox" ${{issues.includes(t)?'checked':''}} onchange="toggleIssue('${{r.item_id}}','${{c.candidate_id}}','${{t}}',this.checked)"> ${{t}}</label>`).join('');
      return `<section class="cand ${{selected ? 'selected' : ''}}">
        <img src="${{c.image_path}}" alt="${{c.candidate_id}}">
        <div class="cand-body">
          <div class="rank">후보 ${{c.rank + 1}} <span class="meta">(원래 rank: ${{Number(c.original_rank) + 1}})</span></div>
          <div class="score">score=${{fmt(c.human_aligned_score)}} / CPS=${{m.cps ?? '-'}} / 관계=${{fmt(m.constraint_accuracy)}} / 접근=${{fmt(m.reachability)}}</div>
          <div class="controls">
            <label><input type="radio" name="best_${{r.item_id}}" ${{selected?'checked':''}} onchange="setBest('${{r.item_id}}','${{c.candidate_id}}')"> 가장 좋음</label>
            <label><input type="checkbox" ${{acceptable?'checked':''}} onchange="toggleAccept('${{r.item_id}}','${{c.candidate_id}}',this.checked)"> 사용 가능</label>
          </div>
          <div class="tags">${{tagHtml}}</div>
          <textarea placeholder="이 후보에 대한 메모" oninput="setCandNote('${{r.item_id}}','${{c.candidate_id}}',this.value)">${{note}}</textarea>
        </div>
      </section>`;
    }}).join('');
    return `<article class="case">
      <div class="case-head">
        <div class="meta">#${{idx + 1}} | ${{r.case_id}} | ${{r.room_type_ko || r.room_type}} | 배치 대상: ${{r.target_category_ko || r.target_category}}</div>
        <div class="instruction">${{r.instruction}}</div>
        <div class="hint">판단 기준: GT 위치는 참고만 하세요. 사용자 요청, 충돌/경계, 접근성, 실제 방처럼 자연스러운지를 우선해서 가장 좋은 후보를 고르면 됩니다.</div>
      </div>
      <div class="grid">${{cards}}</div>
      <div class="case-note"><textarea placeholder="케이스 전체 메모" oninput="setCaseNote('${{r.item_id}}', this.value)">${{label.case_notes || ''}}</textarea></div>
    </article>`;
  }}).join('');
  updateProgress();
}}
function exportJsonl() {{
  const merged = rows.map(r => ({{...r, human_label: ensure(r.item_id)}}));
  const text = merged.map(r => JSON.stringify(r)).join('\\n') + '\\n';
  const blob = new Blob([text], {{type: 'application/jsonl;charset=utf-8'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'topk_labels_export.jsonl';
  a.click();
}}
function resetLabels() {{
  if (confirm('현재 브라우저에 저장된 라벨을 모두 지울까요?')) {{
    localStorage.removeItem(storageKey);
    labels = {{}};
    render();
  }}
}}
render();
</script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build top-k preference labeling UI.")
    p.add_argument("--cases", default="spacefit_v2/data/single_target_benchmark/gpt_intent_cases_test.json")
    p.add_argument("--predictions", default="spacefit_v2/results/final_constraint_solver_human_rerank/test_gpt_intent/raw_predictions_human_reranked.json")
    p.add_argument("--method", default="constraint_solver")
    p.add_argument("--out_dir", default="spacefit_v2/results/topk_labeling_v1")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--mode", choices=["mixed", "rerank_changed", "rerank_unchanged", "random"], default="mixed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--title", default="SpaceFit Top-k 후보 라벨링")
    return p


def main(args: argparse.Namespace) -> None:
    cases = _load_json(args.cases)
    predictions = _load_json(args.predictions)
    if args.method not in predictions:
        raise SystemExit(f"Method not found in predictions: {args.method}")

    selected = _select_cases(cases, predictions, args.method, args.mode, args.n, args.offset, args.seed)
    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    rows: List[Dict[str, Any]] = []

    for case_idx, case in enumerate(selected, start=1):
        cid = str(case["id"])
        candidates: List[Dict[str, Any]] = []
        preds = list(predictions.get(args.method, {}).get(cid) or [])[: args.topk]
        for rank, pred in enumerate(preds):
            image_name = f"{case_idx:03d}_{_slug(cid, 70)}__rank{rank + 1}.png"
            image_path = image_dir / image_name
            _render_image(case, pred, image_path)
            candidates.append(_candidate_record(case, args.method, pred, rank, str(image_path.relative_to(out_dir))))
        if candidates:
            rows.append(_case_record(case, args.method, candidates))

    _write_jsonl(out_dir / "topk_items.jsonl", rows)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "cases": len(rows),
                "method": args.method,
                "topk": args.topk,
                "mode": args.mode,
                "offset": args.offset,
                "source_cases": args.cases,
                "source_predictions": args.predictions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_html(out_dir / "index.html", rows, args.title)
    print(json.dumps({"out_dir": str(out_dir), "cases": len(rows), "html": str(out_dir / "index.html")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(build_parser().parse_args())
