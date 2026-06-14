"""Dataset utilities for scorer training."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import torch
from torch.utils.data import Dataset


def build_training_tensors(samples: List[Dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.stack([sample["features"].float() for sample in samples])
    category_idx = torch.tensor([int(sample["category_idx"]) for sample in samples], dtype=torch.long)
    labels = torch.tensor([float(sample["label"]) for sample in samples], dtype=torch.float32)
    return features, category_idx, labels


class PlacementDataset(Dataset):
    def __init__(self, samples: List[Dict]) -> None:
        self.samples = list(samples)
        self.features, self.category_idx, self.labels = build_training_tensors(samples)
        self.category_names = [sample.get("category", "unknown") for sample in samples]

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        return self.features[idx], self.category_idx[idx], self.labels[idx]

    def make_sample_weights(
        self,
        balance_labels: bool = False,
        balance_categories: bool = False,
    ) -> torch.Tensor:
        weights = torch.ones(len(self.samples), dtype=torch.float32)

        if balance_labels:
            label_counts = Counter(float(x) for x in self.labels.tolist())
            for idx, label in enumerate(self.labels.tolist()):
                weights[idx] *= 1.0 / max(label_counts.get(float(label), 1), 1)

        if balance_categories:
            category_counts = Counter(self.category_names)
            for idx, category in enumerate(self.category_names):
                weights[idx] *= 1.0 / max(category_counts.get(category, 1), 1)

        return weights
