"""Run single-target placement method variants and evaluate them."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env from repo root (OPENAI_API_KEY etc.) if present
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[2] / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

from spacefit_v2.single_target.eval import aggregate_results, save_results
from spacefit_v2.single_target.human_aligned_reranker import rerank_predictions, train_human_aligned_model
from spacefit_v2.single_target.intent_grounder import IntentGrounder
from spacefit_v2.single_target.methods import run_method
from spacefit_v2.single_target.noise_aug import augment_cases


# ── Method groups ─────────────────────────────────────────────────────────────

NON_GPT_METHODS = [
    "heuristic_baseline",
    "proposal_heuristic",
    "proposal_diffopt_basic",
    "proposal_diffopt_constraint",
    "constraint_solver",
]

GPT_METHODS = [
    "layoutgpt_direct",
    "spacefit_gpt_text",
    "proposal_llm_grounded_diffopt",
]

ALL_METHODS = NON_GPT_METHODS + GPT_METHODS

ABLATION_METHODS = [
    "proposal_diffopt_basic",
    "proposal_diffopt_constraint",
    "no_proposal_diffopt_basic",
    "no_proposal_diffopt_constraint",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-target benchmark runner")
    p.add_argument("--benchmark_dir", default="spacefit_v2/data/single_target_benchmark")
    p.add_argument("--split", default="test")
    p.add_argument("--max_cases", type=int, default=None,
                   help="Limit number of cases (useful for smoke-tests)")
    p.add_argument("--methods", nargs="*", default=NON_GPT_METHODS,
                   help="Methods to run. Use 'all' to include GPT methods.")
    p.add_argument("--ablation", action="store_true",
                   help="Run proposal-ablation methods instead of defaults")

    # GPT options
    p.add_argument("--openai_model", default="gpt-4o",
                   help="OpenAI model for GPT-based methods (layoutgpt_direct, spacefit_gpt_text)")
    p.add_argument("--openai_api_key", default=None,
                   help="OpenAI API key (falls back to OPENAI_API_KEY env var)")
    p.add_argument("--openai_base_url", default=None,
                   help="Optional OpenAI-compatible base URL, e.g. DashScope, vLLM, or Ollama.")
    p.add_argument("--local_hf_model", default=None,
                   help="Run GPT-style methods with a local HuggingFace chat model instead of OpenAI, e.g. Qwen/Qwen3-8B.")
    p.add_argument("--local_hf_max_new_tokens", type=int, default=512)
    p.add_argument("--local_hf_torch_dtype", default="bfloat16")
    p.add_argument("--grounder_provider", default="deterministic",
                   choices=["deterministic", "openai"],
                   help="Provider for proposal_llm_grounded_diffopt intent grounding.")
    p.add_argument("--grounder_model", default=None,
                   help="Model name for --grounder_provider openai. Defaults to --openai_model.")

    # Solver options
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--constraint_grid_step", type=float, default=0.18,
                   help="x/z pose-search spacing in meters for constraint_solver.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--diffopt_iters", type=int, default=100)
    p.add_argument("--diffopt_lr", type=float, default=1e-2)

    # Sim-to-real noise
    p.add_argument("--noise", default="none", choices=["none", "roomplan"],
                   help="Noise preset for sim-to-real evaluation. "
                        "'roomplan' injects ±2cm wall / ±5cm furniture noise.")
    p.add_argument("--noise_seed", type=int, default=42)

    # GPT intent evaluation
    p.add_argument("--use_gpt_intents", action="store_true",
                   help="Load GPT-generated user intents instead of template-based intents. "
                        "Requires gpt_intent_cases_<split>.json to exist in benchmark_dir. "
                        "Generate it first with scripts/generate_gpt_intents.py.")
    p.add_argument("--gpt_intent_file", default=None,
                   help="Explicit path to GPT intent cases JSON. "
                        "If omitted and --use_gpt_intents is set, auto-resolves to "
                        "<benchmark_dir>/gpt_intent_cases_<split>.json.")

    # Human-aligned candidate reranking
    p.add_argument("--human_rerank_labels", default=None,
                   help="Visual-audit JSONL labels used to train a human-aligned top-k reranker. "
                        "If set, predictions are reranked before evaluation.")
    p.add_argument("--human_rerank_train_predictions",
                   default="spacefit_v2/results/experiment_final/test_gpt_intent/raw_predictions.json",
                   help="Raw predictions used to extract training features for the human labels.")
    p.add_argument("--human_rerank_train_cases",
                   default="spacefit_v2/data/single_target_benchmark/gpt_intent_cases_test.json",
                   help="Cases JSON used to extract training features for the human labels.")
    p.add_argument("--human_rerank_model", choices=["rf", "logreg"], default="rf",
                   help="Seed scorer model used for human-aligned reranking.")
    p.add_argument("--human_rerank_methods", nargs="*", default=None,
                   help="Methods to rerank. Defaults to all currently executed methods.")

    p.add_argument("--out_dir", default="spacefit_v2/results/single_target_gpt")
    return p


def _load_json_list(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_cases(benchmark_dir: Path, split: str, max_cases: Optional[int],
                use_gpt_intents: bool = False,
                gpt_intent_file: Optional[str] = None) -> tuple[Dict, List[Dict]]:
    with open(benchmark_dir / "manifest.json") as f:
        manifest = json.load(f)

    if use_gpt_intents:
        intent_path = Path(gpt_intent_file) if gpt_intent_file else (
            benchmark_dir / f"gpt_intent_cases_{split}.json"
        )
        if not intent_path.exists():
            raise FileNotFoundError(
                f"GPT intent file not found: {intent_path}\n"
                f"Run: python -m spacefit_v2.scripts.generate_gpt_intents --split {split}"
            )
        with open(intent_path) as f:
            cases = json.load(f)
        # These are already filtered to the target split
        selected = cases
    else:
        with open(benchmark_dir / "cases.json") as f:
            cases = json.load(f)
        selected = [c for c in cases if c["split"] == split]

    if max_cases is not None:
        selected = selected[:int(max_cases)]
    return manifest, selected


def _make_openai_client(api_key: Optional[str], model: str, base_url: Optional[str] = None) -> Optional[Any]:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    url = base_url or os.environ.get("OPENAI_BASE_URL")
    if not key:
        return None
    try:
        from openai import OpenAI
        kwargs = {"api_key": key}
        if url:
            kwargs["base_url"] = url
        return OpenAI(**kwargs)
    except ImportError:
        print("WARNING: openai package not installed. GPT methods will fail.")
        return None


def _make_local_hf_client(model_id: str, max_new_tokens: int, torch_dtype: str) -> Any:
    from spacefit_v2.single_target.local_hf_chat import LocalHFChatClient, LocalHFChatConfig

    return LocalHFChatClient(
        LocalHFChatConfig(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            torch_dtype=torch_dtype,
        )
    )


def main(args: argparse.Namespace) -> None:
    benchmark_dir = Path(args.benchmark_dir)
    manifest, cases = _load_cases(
        benchmark_dir, args.split, args.max_cases,
        use_gpt_intents=args.use_gpt_intents,
        gpt_intent_file=args.gpt_intent_file,
    )

    # Resolve method list
    methods = args.methods
    if "all" in methods:
        methods = ALL_METHODS
    if args.ablation:
        methods = ABLATION_METHODS

    # Noise augmentation (sim-to-real)
    if args.noise != "none":
        print(f"[noise] Applying '{args.noise}' augmentation to {len(cases)} cases (seed={args.noise_seed})")
        cases = augment_cases(cases, preset=args.noise, seed=args.noise_seed)

    # Build OpenAI client if needed
    gpt_methods_requested = [m for m in methods if m in GPT_METHODS]
    openai_client = None
    if gpt_methods_requested:
        if args.local_hf_model:
            print(f"[local_hf] Loading {args.local_hf_model} for GPT-style methods: {gpt_methods_requested}")
            openai_client = _make_local_hf_client(args.local_hf_model, args.local_hf_max_new_tokens, args.local_hf_torch_dtype)
            args.openai_model = args.local_hf_model
        else:
            openai_client = _make_openai_client(args.openai_api_key, args.openai_model, args.openai_base_url)
        if openai_client is None:
            print(f"WARNING: OpenAI client unavailable. Skipping: {gpt_methods_requested}")
            methods = [m for m in methods if m not in GPT_METHODS]

    if args.grounder_provider == "openai":
        if not (args.openai_api_key or os.environ.get("OPENAI_API_KEY")):
            print("WARNING: OpenAI API key unavailable. Falling back to deterministic grounder.")
            grounder = IntentGrounder(provider="deterministic")
        else:
            grounder = IntentGrounder(
                provider="openai",
                model_name=args.grounder_model or args.openai_model,
                api_key=args.openai_api_key,
                base_url=args.openai_base_url,
            )
    else:
        grounder = IntentGrounder(provider="deterministic")

    predictions_by_method: Dict[str, Dict[str, List[Dict]]] = {}
    for method in methods:
        per_case: Dict[str, List[Dict]] = {}
        for idx, case in enumerate(cases, start=1):
            try:
                preds = run_method(
                    method=method,
                    case=case,
                    grounder=grounder,
                    topk=args.topk,
                    constraint_grid_step=args.constraint_grid_step,
                    device=args.device,
                    diffopt_iters=args.diffopt_iters,
                    diffopt_lr=args.diffopt_lr,
                    openai_client=openai_client,
                    openai_model=args.openai_model,
                )
            except Exception as exc:
                preds = [{"status": "error", "reason": str(exc),
                          "furniture_id": case["target_asset"]["id"],
                          "category": case["target_asset"]["category"]}]
            per_case[case["id"]] = preds
            placed = preds[0].get("status") == "placed" if preds else False
            print(f"[{method}] {idx}/{len(cases)} placed={placed}  {case['id'][:60]}")
        predictions_by_method[method] = per_case

    suffixes = []
    if args.noise != "none":
        suffixes.append(args.noise)
    if args.use_gpt_intents:
        suffixes.append("gpt_intent")
    out_suffix = ("_" + "_".join(suffixes)) if suffixes else ""
    out_dir = Path(args.out_dir) / (args.split + out_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write raw predictions first so data is never lost even if aggregation crashes
    raw_path = out_dir / "raw_predictions.json"
    with open(raw_path, "w") as f:
        json.dump(predictions_by_method, f, indent=2)

    rerank_outputs: Dict[str, Any] = {}
    eval_predictions = predictions_by_method
    if args.human_rerank_labels:
        try:
            model, train_summary = train_human_aligned_model(
                labels_path=args.human_rerank_labels,
                cases=_load_json_list(args.human_rerank_train_cases),
                train_predictions=args.human_rerank_train_predictions,
                model_kind=args.human_rerank_model,
            )
            rerank_methods = args.human_rerank_methods or methods
            eval_predictions, score_rows = rerank_predictions(
                cases=cases,
                predictions_by_method=predictions_by_method,
                model=model,
                methods=rerank_methods,
            )
            reranked_path = out_dir / "raw_predictions_human_reranked.json"
            with open(reranked_path, "w") as f:
                json.dump(eval_predictions, f, indent=2)
            scores_path = out_dir / "human_rerank_candidate_scores.csv"
            if score_rows:
                with open(scores_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=list(score_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(score_rows)
            rerank_outputs = {
                "enabled": True,
                "training": train_summary,
                "methods": list(rerank_methods),
                "raw_predictions_human_reranked": str(reranked_path),
                "candidate_scores": str(scores_path),
            }
            print(f"[human_rerank] enabled model={args.human_rerank_model} methods={list(rerank_methods)}")
        except Exception as exc:
            print(f"WARNING: human-aligned rerank failed; evaluating original predictions: {exc}")
            rerank_outputs = {"enabled": False, "error": str(exc)}

    try:
        aggregates = aggregate_results(cases, eval_predictions)
        outputs = save_results(str(out_dir), manifest, aggregates)
    except Exception as exc:
        print(f"WARNING: aggregate_results/save_results failed: {exc}")
        outputs = {}

    print(json.dumps({
        "cases": len(cases),
        "methods": methods,
        "noise": args.noise,
        "human_rerank": rerank_outputs,
        "outputs": outputs,
        "raw_predictions": str(raw_path),
    }, indent=2))


if __name__ == "__main__":
    main(build_parser().parse_args())
