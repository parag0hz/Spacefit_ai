"""Learned placement scoring MLP."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from spacefit_v2 import config


class PlacementScorer(nn.Module):
    """Scores a single placement from numeric features + category id."""

    def __init__(
        self,
        numeric_dim: int = config.SCORER_NUMERIC_DIM,
        embed_dim: int = config.SCORER_EMBED_DIM,
        hidden_dim: int = config.SCORER_HIDDEN_DIM,
        num_categories: int = 32,
    ) -> None:
        super().__init__()
        self.numeric_dim = int(numeric_dim)
        self.embed_dim = int(embed_dim)
        self.num_categories = int(num_categories)

        self.category_embedding = nn.Embedding(self.num_categories, self.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.numeric_dim + self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def prepare_input(
        self,
        features: torch.Tensor,
        category_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if features.dim() == 1:
            features = features.unsqueeze(0)

        if category_idx is None:
            if features.shape[-1] != self.numeric_dim + self.embed_dim:
                raise ValueError(
                    "category_idx is required when passing numeric-only features. "
                    f"Expected {self.numeric_dim + self.embed_dim}-dim full input, "
                    f"got {features.shape[-1]}."
                )
            return features

        if category_idx.dim() == 0:
            category_idx = category_idx.unsqueeze(0)
        category_idx = category_idx.long()
        if category_idx.shape[0] != features.shape[0]:
            if category_idx.shape[0] == 1:
                category_idx = category_idx.expand(features.shape[0])
            else:
                raise ValueError("batch size mismatch between features and category_idx")

        emb = self.category_embedding(category_idx)
        return torch.cat([features, emb], dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        category_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.prepare_input(features, category_idx=category_idx)
        return self.mlp(x)


def load_scorer_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[PlacementScorer, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata: Dict[str, Any] = {}
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        metadata = dict(checkpoint.get("metadata") or {})
        model = PlacementScorer(
            numeric_dim=int(metadata.get("numeric_dim", config.SCORER_NUMERIC_DIM)),
            embed_dim=int(metadata.get("embed_dim", config.SCORER_EMBED_DIM)),
            hidden_dim=int(metadata.get("hidden_dim", config.SCORER_HIDDEN_DIM)),
            num_categories=int(metadata.get("num_categories", 32)),
        )
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = PlacementScorer()
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model, metadata
