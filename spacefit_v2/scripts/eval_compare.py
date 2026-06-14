"""Compare SpaceFit v3 output against v2 diff-opt output."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.eval.eval_3dfront import compute_kl, compute_oob_metrics, compute_overlap_metrics
from experiments.adapters.threedfront_adapter import load_3dfront_scenes


def _load(path: str | Path):
    with open(path, "r") as f:
        return json.load(f)


def _summarize(preds: Iterable[Dict[str, Any]], gt_scenes: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    preds = list(preds)
    gt_scenes = list(gt_scenes)
    summary: Dict[str, Any] = {}
    summary.update(compute_oob_metrics(preds))
    summary.update(compute_overlap_metrics(preds))
    summary.update(compute_kl(preds, gt_scenes))
    times = [float(item["time"]) for item in preds if "time" in item]
    placed = [int(item.get("num_placed", 0)) for item in preds]
    summary["avg_time_sec"] = sum(times) / len(times) if times else None
    summary["avg_num_placed"] = sum(placed) / len(placed) if placed else None
    return summary


def main(args: argparse.Namespace) -> None:
    v3 = _load(args.v3_file)
    v2 = _load(args.v2_file)
    gt = load_3dfront_scenes(args.data_dir, room_type=args.room, split=args.split, limit=None, rect_only=not args.all_test)
    result = {
        "v3": _summarize(v3, gt),
        "v2": _summarize(v2, gt),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3_file", required=True)
    parser.add_argument("--v2_file", required=True)
    parser.add_argument("--data_dir", default="LayoutGPT/ATISS/data_output")
    parser.add_argument("--room", default="bedroom")
    parser.add_argument("--split", default="test")
    parser.add_argument("--all_test", action="store_true")
    parser.add_argument("--output", default=None)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
