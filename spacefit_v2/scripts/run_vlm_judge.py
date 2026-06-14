"""Run VLM-as-judge evaluation on saved raw predictions.

Usage:
    python -m spacefit_v2.scripts.run_vlm_judge \
        --cases spacefit_v2/data/single_target_benchmark/gpt_intent_cases_test.json \
        --predictions spacefit_v2/results/experiment_gpt_intent/test_gpt_intent/raw_predictions.json \
        --out spacefit_v2/results/experiment_gpt_intent/test_gpt_intent/vlm_judgments.json \
        --methods layoutgpt_direct spacefit_gpt_text \
        --max_cases 30
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from spacefit_v2.single_target.vlm_judge import aggregate_vlm_scores, judge_all_methods


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VLM-as-judge evaluation")
    p.add_argument("--cases", default="spacefit_v2/data/single_target_benchmark/gpt_intent_cases_test.json")
    p.add_argument("--predictions", default="spacefit_v2/results/experiment_gpt_intent/test_gpt_intent/raw_predictions.json")
    p.add_argument("--out", default="spacefit_v2/results/experiment_gpt_intent/test_gpt_intent/vlm_judgments.json")
    p.add_argument("--methods", nargs="*", default=None, help="Methods to judge (default: all)")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--openai_api_key", default=None)
    p.add_argument("--max_cases", type=int, default=None)
    return p


def main(args: argparse.Namespace) -> None:
    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: No OpenAI API key.")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    with open(args.cases) as f:
        cases = json.load(f)
    with open(args.predictions) as f:
        preds = json.load(f)

    if args.max_cases:
        cases = cases[: args.max_cases]

    methods = args.methods or list(preds.keys())
    print(f"VLM judging {len(cases)} cases × {len(methods)} methods ({args.model})")

    judgments = judge_all_methods(cases, preds, client, model=args.model, methods=methods)
    summary = aggregate_vlm_scores(judgments)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"judgments": judgments, "summary": summary}, f, indent=2)

    print("\n── VLM Judge Summary ──")
    for method, s in summary.items():
        print(f"  {method:<35} score={s['vlm_score']:.2f}/10  "
              f"phys={s['vlm_physical_valid']:.2f}  intent={s['vlm_intent_satisfied']:.2f}  (n={s['n']})")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main(build_parser().parse_args())
