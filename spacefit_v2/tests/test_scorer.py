"""Unit tests for the learned scorer."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from spacefit_v2.model.features import FEATURE_DIM
from spacefit_v2.model.scorer import PlacementScorer


def main() -> int:
    model = PlacementScorer(numeric_dim=FEATURE_DIM, num_categories=8)
    features = torch.randn(4, FEATURE_DIM, requires_grad=True)
    category_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    logits = model(features, category_idx=category_idx)
    assert logits.shape == (4, 1), logits.shape

    loss = logits.sum()
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()

    full_features = torch.randn(2, FEATURE_DIM + model.embed_dim)
    logits2 = model(full_features)
    assert logits2.shape == (2, 1)

    print("test_scorer: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
