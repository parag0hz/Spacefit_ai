"""Build the raw-3D-FRONT single-target benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from spacefit_v2.single_target.benchmark import generate_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset/3D-FRONT")
    parser.add_argument("--out_dir", default="spacefit_v2/data/single_target_benchmark")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--max_cases", type=int, default=None)
    return parser


def main(args: argparse.Namespace) -> None:
    summary = generate_benchmark(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        max_scenes=args.max_scenes,
        max_cases=args.max_cases,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(build_parser().parse_args())

