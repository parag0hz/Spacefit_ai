# SpaceFit

SpaceFit is a single-target furniture placement pipeline for 3D indoor scenes. It combines language-based intent grounding with geometric candidate generation, physical filtering, constraint scoring, and optional human-aligned reranking.

This public repository contains the core implementation under `spacefit_v2/`. Datasets, generated results, model weights, API keys, PDFs, presentation files, and large visualization assets are intentionally excluded.

## Method Overview

Given:

1. an existing room scene,
2. one target furniture item,
3. a natural-language placement request,

SpaceFit performs:

1. **Scene normalization**: parse the floor polygon, existing furniture, doors, and windows.
2. **Intent grounding**: convert the user request into structured relations such as `near`, `facing`, `against_wall`, and `not_block`.
3. **Free-space extraction**: identify candidate regions not occupied by existing objects or protected access areas.
4. **Candidate generation**: sample target positions and orientations on a configurable spatial grid.
5. **Physical filtering**: reject collision, out-of-boundary, and access-blocking candidates.
6. **Constraint scoring**: score feasible candidates using geometric and relational rules.
7. **Top-k selection**: retain the highest-scoring placements.
8. **Optional preference reranking**: reorder top-k candidates using a Random Forest scorer trained from human visual-audit labels.

The LLM does not need to generate final coordinates directly. It can be used only for semantic region selection, while the geometric solver computes and validates the final pose.

## Repository Structure

```text
spacefit_v2/
├── cross_method/   # Adapted comparison methods
├── data/           # Dataset loader and sampling code only
├── grounding/      # Intent and relation grounding
├── legacy/         # Small compatibility layer for the original geometry/solver helpers
├── model/          # Geometry features, losses, and learned scorers
├── optim/          # Loss-based pose refinement
├── scripts/        # Benchmark, evaluation, visualization, and analysis tools
├── single_target/  # Main single-target placement pipeline
└── tests/          # Lightweight unit tests
```

## Installation

```bash
git clone https://github.com/parag0hz/Spacefit_ai.git
cd Spacefit_ai

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
```

For local Qwen/Gemma experiments, install a CUDA-compatible PyTorch build appropriate for the host GPU.

## Dataset Setup

The dataset is not distributed in this repository. Prepare licensed copies locally using this structure:

```text
dataset/
├── 3D-FRONT/
├── 3D-FRONT-texture/
└── 3D-FUTURE-model/
```

Generated benchmark JSON files should be placed under:

```text
spacefit_v2/data/single_target_benchmark/
```

Both paths are ignored by Git.

## API Configuration

Never commit API keys. Copy `.env.example` to `.env` locally:

```text
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=
HF_TOKEN=
```

`.env` is excluded by `.gitignore`.

## Example Commands

Run the lightweight tests:

```bash
python -m spacefit_v2.tests.test_features
python -m spacefit_v2.tests.test_scorer
python -m spacefit_v2.tests.test_diffopt
```

Run the code-based constraint solver:

```bash
python spacefit_v2/scripts/run_single_target_benchmark.py \
  --methods constraint_solver \
  --use_gpt_intents \
  --constraint_grid_step 0.18 \
  --out_dir spacefit_v2/results/constraint_solver
```

Run an OpenAI-backed region-selection experiment:

```bash
python spacefit_v2/scripts/run_single_target_benchmark.py \
  --methods spacefit_gpt_text \
  --openai_model gpt-4o \
  --use_gpt_intents \
  --out_dir spacefit_v2/results/gpt4o
```

Run a local Hugging Face backbone:

```bash
python spacefit_v2/scripts/run_single_target_benchmark.py \
  --methods spacefit_gpt_text \
  --local_hf_model Qwen/Qwen3-8B \
  --local_hf_torch_dtype bfloat16 \
  --use_gpt_intents \
  --out_dir spacefit_v2/results/qwen3_8b
```

## Main Results

All tables below use the same 181-case GPT-intent test split. Raw outputs are not included in the public repository.

### LLM Backbone Comparison

The backbone is used for semantic region selection within the same SpaceFit GPT-Text pipeline.

| Backbone | Placed | CF | IB | Constraint Accuracy | CPS | Success@5 |
|---|---:|---:|---:|---:|---:|---:|
| GPT-4o | 173/181 | 0.950 | 0.890 | **0.643** | **0.381** | **0.530** |
| Qwen3-8B | 173/181 | 0.945 | 0.873 | 0.598 | 0.282 | 0.475 |
| Qwen3.5-9B | 173/181 | 0.950 | 0.873 | 0.579 | 0.260 | 0.436 |
| Gemma 4 E4B | 173/181 | 0.950 | 0.851 | 0.585 | 0.282 | 0.453 |

The larger Qwen3.5-9B backbone did not outperform Qwen3-8B in this placement-intent grounding setup, showing that a newer or larger backbone does not automatically improve region selection.

### Grid-Step Ablation

| Grid step | Runtime | CF | IB | Constraint Accuracy | CPS | Success@5 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 m | 1360.6 s | 0.956 | 0.956 | 0.719 | 0.486 | **0.735** |
| 0.18 m | 444.2 s | **0.961** | **0.961** | **0.725** | **0.503** | 0.702 |
| 0.25 m | **252.7 s** | 0.950 | 0.950 | 0.713 | **0.503** | 0.696 |

The 0.18 m grid provides the strongest overall balance between runtime, physical validity, and strict success.

### Incremental Ablation

| Step | Variant | CPS | Constraint Accuracy | CF | IB |
|---:|---|---:|---:|---:|---:|
| 0 | Direct coordinate prediction | 0.094 | 0.657 | 0.398 | 0.635 |
| 1 | + Candidate search | 0.188 | 0.502 | 0.956 | 0.939 |
| 2 | + Local refinement | 0.254 | 0.556 | 0.945 | 0.840 |
| 3 | + Constraint solver | 0.503 | 0.725 | 0.961 | 0.961 |
| 4 | + Human-aligned rerank | **0.674** | **0.774** | 0.961 | 0.961 |

The reranking result is from the saved human-aligned experiment and depends on the available human visual-audit labels.

### GPT-4o VLM Judge

| Metric | Score |
|---|---:|
| Physical validity | 0.961 |
| Accessibility | 0.950 |
| Relation satisfaction | 0.652 |
| Orientation naturalness | 0.845 |
| Grouping naturalness | 0.702 |
| Overall naturalness | 0.646 |
| Mean quality score | 7.19 / 10 |

The VLM evaluation highlights a remaining gap: physical validity is high, while relation satisfaction and overall visual naturalness still leave room for improvement.

## Metric Definitions

- **CF**: collision-free placement rate.
- **IB**: in-boundary placement rate.
- **Constraint Accuracy**: average satisfaction of requested geometric and relational constraints.
- **CPS / Success@1**: strict top-1 placement success.
- **Success@5**: whether at least one valid candidate exists among the top five.
- **Reachability**: target accessibility under the geometry-based reachability test.
- **Walkability**: preservation of navigable free space.

## Privacy and Distribution Policy

The following must not be committed:

- 3D-FRONT and 3D-FUTURE datasets,
- generated benchmark cases and raw predictions,
- API keys or `.env` files,
- model weights and checkpoints,
- PDFs, presentation files, and large rendered assets.

The repository `.gitignore` enforces these exclusions.
