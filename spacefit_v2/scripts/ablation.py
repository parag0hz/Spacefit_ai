"""Simple ablation runner for loss-component variants."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spacefit_v2.scripts.run_diffopt import build_parser as build_run_parser, run as run_diffopt


def main(args: argparse.Namespace) -> None:
    variants = [
        ("all", args.scorer_path),
        ("no_scorer", None),
    ]
    results = {}
    for name, scorer_path in variants:
        run_args = build_run_parser().parse_args([])
        run_args.data_dir = args.data_dir
        run_args.room = args.room
        run_args.split = args.split
        run_args.max_scenes = args.max_scenes
        run_args.iters = args.iters
        run_args.restarts = args.restarts
        run_args.lr = args.lr
        run_args.device = args.device
        run_args.scorer_path = scorer_path
        run_args.output = str(Path(args.output_dir) / f"{name}.json")
        outputs = run_diffopt(run_args)
        results[name] = {
            "num_scenes": len(outputs),
            "avg_num_placed": sum(item.get("num_placed", 0) for item in outputs) / max(len(outputs), 1),
            "avg_time_sec": sum(item.get("time", 0.0) for item in outputs) / max(len(outputs), 1),
        }

    out_path = Path(args.output_dir) / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="LayoutGPT/ATISS/data_output")
    parser.add_argument("--room", default="bedroom")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_scenes", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scorer_path", default=None)
    parser.add_argument("--output_dir", default="spacefit_v2/results/ablation")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
