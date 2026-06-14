"""Run a VLM yes/no judge baseline on the visual-audit VQA seed dataset.

Example:
    python -m spacefit_v2.scripts.run_vqa_judge_baseline \
        --input spacefit_v2/results/visual_audit_distribution/result/analysis/vqa_seed/vqa_val.jsonl \
        --out_dir spacefit_v2/results/visual_audit_distribution/result/analysis/vqa_judge_gpt4o_mini_val \
        --model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SYSTEM_PROMPT_V1 = """\
You are a strict visual evaluator for indoor furniture placement.
You will see a top-down floor plan image and a yes/no question.

Visual legend:
- Gray rectangles: existing furniture
- Green rectangle: newly placed target furniture
- Arrow on green rectangle: predicted rotation only; use it as weak evidence because front direction can be ambiguous
- Brown marks: doors
- Blue marks: windows

Answer the question using the image and the user's placement instruction.
Respond with ONLY valid JSON:
{
  "answer": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "short reason"
}
"""

SYSTEM_PROMPT_V2 = """\
You are a calibrated VQA evaluator for indoor furniture placement.
Your goal is to match a reasonable human annotator, not to be a perfectionist.

Visual legend:
- Gray rectangles: existing furniture
- Green rectangle: newly placed target furniture
- Arrow on green rectangle: predicted rotation only; use it as weak evidence because furniture front direction can be ambiguous
- Brown marks: doors
- Blue marks: windows

General policy:
- Answer ONLY the asked question.
- Do not mark "no" for unrelated issues.
- Use the user's placement instruction as the main intent reference.
- If the image is ambiguous but the placement seems reasonably acceptable for the asked question, answer "yes".
- Be stricter for visible collision, outside-room placement, blocked doors, and clearly wrong requested relations.

Question-specific rubric:
- physical_ok: answer yes if the target appears inside the room and not visibly overlapping existing furniture. Ignore relation, aesthetics, and rotation.
- relation_ok: answer yes if the target roughly satisfies the requested spatial relation, such as near, beside, in front of, against wall, near window, or facing another item. Minor distance errors are acceptable.
- orientation_ok: answer no only when the instruction explicitly requires facing/being oriented and the arrow is clearly inconsistent. For desks, beds, shelves, and ambiguous assets, treat the arrow as weak evidence.
- access_ok: answer yes if doors and main walking paths are not obviously blocked. Do not require perfect clearance.
- naturalness_ok: answer yes if the layout would look usable in a real room, even if not ideal. Answer no for isolated, awkward, cramped, or nonsensical placements.
- overall_ok: answer yes if the placement is acceptable overall for the user's instruction. It does not need to be perfect.

Respond with ONLY valid JSON:
{
  "answer": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "short reason"
}
"""

SYSTEM_PROMPT_V3 = """\
You are a calibrated VQA evaluator for indoor furniture placement.
Your goal is to match a reasonable human annotator.

Visual legend:
- Gray rectangles: existing furniture
- Green rectangle: newly placed target furniture
- Arrow on green rectangle: predicted rotation only; it can be noisy
- Brown marks: doors
- Blue marks: windows

General policy:
- Answer ONLY the asked question.
- Do not mark "yes" unless the asked property is visibly or logically satisfied.
- Do not mark "no" for unrelated issues.
- Use the user's placement instruction as the main intent reference.
- If the question cannot be verified from the top-down image, lean "no" for relation/naturalness/overall, but lean "yes" for physical validity when the object is visibly inside and not overlapping.

Question-specific rubric:
- physical_ok: yes if the target is visibly inside the room and not visibly overlapping existing furniture. Ignore relation, aesthetics, and rotation.
- relation_ok: yes only if the requested relation is reasonably clear in the image. For near/beside/in front of/facing/against wall/near window, check that relation directly.
- orientation_ok: yes only if the direction seems consistent when the instruction requires direction or facing. If there is no meaningful direction requirement, answer yes unless the arrow is obviously nonsensical. For ambiguous furniture front directions, reduce confidence but still decide from the visible layout.
- access_ok: yes if doors and main walking paths are not obviously blocked and there appears to be usable clearance.
- naturalness_ok: yes if the placement looks plausible and usable in a real room. No for isolated, cramped, awkward, or furniture-group-incoherent placements.
- overall_ok: yes only if the placement is broadly acceptable for the instruction, including physical validity, relation, access, and naturalness.

Respond with ONLY valid JSON:
{
  "answer": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "short reason"
}
"""

SYSTEM_PROMPT_NOARROW = """\
You are a strict visual evaluator for indoor furniture placement.
You will see a top-down floor plan image and a yes/no question.

Visual legend:
- Gray rectangles: existing furniture
- Bright green rectangle: newly placed target furniture
- Cyan marks: windows
- Brown marks: doors
- Purple outlines, when present: reference furniture related to the instruction

Evaluate only the asked question. The image may omit facing-direction arrows, so do not invent a precise front direction unless it is visually obvious from the layout and the user's instruction.

Respond with ONLY valid JSON:
{
  "answer": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "short reason"
}
"""


QUESTION_HINTS = {
    "physical_ok": (
        "This question is only about physical validity: inside the room and no visible overlap. "
        "Ignore relation quality, naturalness, and rotation."
    ),
    "relation_ok": (
        "This question is about whether the requested spatial relation is roughly satisfied. "
        "Ignore minor aesthetic issues if the relation is met."
    ),
    "orientation_ok": (
        "This question is about direction only when direction is explicitly relevant. "
        "The arrow is weak evidence; do not over-penalize ambiguous furniture front directions."
    ),
    "access_ok": (
        "This question is about usability of access and walking paths. "
        "Answer no only if doors, windows, or obvious paths are blocked or clearance is clearly poor."
    ),
    "naturalness_ok": (
        "This question is about whether the placement looks plausible and usable in a real room. "
        "It can be yes even when the placement is not perfect."
    ),
    "overall_ok": (
        "This question is the final human acceptability judgment. "
        "Answer yes if the placement is broadly acceptable for the instruction."
    ),
}


def _load_dotenv(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _image_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _image_paths(row: Dict[str, Any]) -> List[Path]:
    images = row.get("images")
    if isinstance(images, list) and images:
        return [Path(str(p)) for p in images]
    return [Path(row["image"])]


def _parse_response(text: str) -> Dict[str, Any]:
    content = text.strip()
    content = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise
        result = json.loads(match.group(0))
    answer = str(result.get("answer", "")).strip().lower()
    if answer not in {"yes", "no"}:
        raise ValueError(f"Invalid answer: {answer!r}")
    result["answer"] = answer
    try:
        result["confidence"] = float(result.get("confidence", 0.0))
    except Exception:
        result["confidence"] = 0.0
    result["reason"] = str(result.get("reason", ""))
    return result


def _judge_one(
    client: Any,
    row: Dict[str, Any],
    model: str,
    temperature: float,
    prompt_version: str,
) -> Dict[str, Any]:
    image_paths = _image_paths(row)
    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(str(image_path))

    question_id = str(row.get("question_id", ""))
    hint = QUESTION_HINTS.get(question_id, "")
    if prompt_version == "hybrid":
        system_prompt = SYSTEM_PROMPT_V2 if question_id in {"physical_ok", "access_ok"} else SYSTEM_PROMPT_V1
    elif prompt_version == "noarrow":
        system_prompt = SYSTEM_PROMPT_NOARROW
    elif prompt_version == "v3":
        system_prompt = SYSTEM_PROMPT_V3
    elif prompt_version == "v2":
        system_prompt = SYSTEM_PROMPT_V2
    else:
        system_prompt = SYSTEM_PROMPT_V1
    if prompt_version == "v1":
        user_text = (
            f"User placement instruction:\n{row.get('instruction', '')}\n\n"
            f"Target furniture category: {row.get('target_category', '')}\n"
            f"Room type: {row.get('room_type', '')}\n"
            f"Method that produced this placement: {row.get('method', '')}\n\n"
            f"Question:\n{row.get('question', '')}\n\n"
            "Answer yes or no. Be strict about physical placement and intent satisfaction, "
            "but treat the rotation arrow as weak evidence when furniture front direction is ambiguous."
        )
    else:
        user_text = (
            f"User placement instruction:\n{row.get('instruction', '')}\n\n"
            f"Target furniture category: {row.get('target_category', '')}\n"
            f"Room type: {row.get('room_type', '')}\n"
            f"Method that produced this placement: {row.get('method', '')}\n\n"
            f"Question ID: {question_id}\n"
            f"Question:\n{row.get('question', '')}\n\n"
            f"Rubric hint:\n{hint}\n\n"
            "Answer yes or no according to the rubric. Match a reasonable human annotator."
        )

    content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    for idx, image_path in enumerate(image_paths, 1):
        content.append({"type": "text", "text": f"View {idx} of {len(image_paths)}"})
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(image_path), "detail": "low"}})

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": content,
            },
        ],
        temperature=temperature,
        max_tokens=180,
    )
    content = response.choices[0].message.content or ""
    parsed = _parse_response(content)
    usage = getattr(response, "usage", None)
    return {
        "raw_response": content,
        "pred_answer": parsed["answer"],
        "pred_bool": parsed["answer"] == "yes",
        "confidence": parsed["confidence"],
        "reason": parsed["reason"],
        "usage": usage.model_dump() if hasattr(usage, "model_dump") else None,
    }


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in rows if r.get("error") is None]
    total = len(valid)
    correct = sum(r.get("pred_bool") == r.get("answer_bool") for r in valid)

    def grouped(key: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in valid:
            groups[str(r.get(key, ""))].append(r)
        for name, items in sorted(groups.items()):
            n = len(items)
            c = sum(r.get("pred_bool") == r.get("answer_bool") for r in items)
            yes_gold = sum(r.get("answer_bool") is True for r in items)
            yes_pred = sum(r.get("pred_bool") is True for r in items)
            out[name] = {
                "n": n,
                "accuracy": c / n if n else None,
                "gold_yes_rate": yes_gold / n if n else None,
                "pred_yes_rate": yes_pred / n if n else None,
            }
        return out

    confusion = Counter()
    for r in valid:
        gold = "yes" if r.get("answer_bool") else "no"
        pred = "yes" if r.get("pred_bool") else "no"
        confusion[f"{gold}->{pred}"] += 1

    return {
        "n": len(rows),
        "valid_n": total,
        "errors": len(rows) - total,
        "accuracy": correct / total if total else None,
        "confusion": dict(confusion),
        "by_question": grouped("question_id"),
        "by_method": grouped("method"),
        "by_answer": grouped("answer"),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run VLM yes/no judge on visual-audit VQA examples.")
    p.add_argument("--input", default="spacefit_v2/results/visual_audit_distribution/result/analysis/vqa_seed/vqa_val.jsonl")
    p.add_argument("--out_dir", default="spacefit_v2/results/visual_audit_distribution/result/analysis/vqa_judge_gpt4o_mini_val")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--openai_api_key", default=None)
    p.add_argument("--openai_base_url", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--prompt_version", choices=["v1", "v2", "v3", "hybrid", "noarrow"], default="hybrid")
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--resume", action="store_true")
    return p


def main(args: argparse.Namespace) -> None:
    _load_dotenv([Path(".env"), Path("spacefit_v2/.env")])
    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY is not set.")

    from openai import OpenAI

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    base_url = args.openai_base_url or os.environ.get("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    examples = _read_jsonl(Path(args.input))
    if args.sample is not None:
        rng = random.Random(args.seed)
        examples = rng.sample(examples, min(args.sample, len(examples)))
    if args.limit is not None:
        examples = examples[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "vqa_judge_predictions.jsonl"
    done: Dict[str, Dict[str, Any]] = {}
    if args.resume and pred_path.exists():
        for row in _read_jsonl(pred_path):
            done[str(row["id"])] = row

    with pred_path.open("a", encoding="utf-8") as f:
        for idx, row in enumerate(examples, 1):
            row_id = str(row["id"])
            if row_id in done:
                print(f"[{idx}/{len(examples)}] skip {row_id}")
                continue
            print(f"[{idx}/{len(examples)}] judge {row_id}")
            result = dict(row)
            try:
                judged = _judge_one(client, row, args.model, args.temperature, args.prompt_version)
                result.update(judged)
                result["error"] = None
            except Exception as exc:
                result["pred_answer"] = None
                result["pred_bool"] = None
                result["confidence"] = None
                result["reason"] = ""
                result["raw_response"] = ""
                result["usage"] = None
                result["error"] = str(exc)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            if args.sleep > 0:
                time.sleep(args.sleep)

    predictions = _read_jsonl(pred_path)
    wanted = {str(r["id"]) for r in examples}
    predictions = [r for r in predictions if str(r.get("id")) in wanted]
    summary = _summarize(predictions)
    summary["model"] = args.model
    summary["prompt_version"] = args.prompt_version
    summary["input"] = args.input
    summary["out_dir"] = str(out_dir)
    (out_dir / "vqa_judge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nVQA Judge Summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(build_parser().parse_args())
