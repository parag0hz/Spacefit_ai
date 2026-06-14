"""Train the learned placement scorer on 3D-FRONT GT + negatives."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spacefit_v2 import config
from spacefit_v2.data.dataset import PlacementDataset
from spacefit_v2.data.gt_loader import generate_training_data, load_category_mapping
from spacefit_v2.device import resolve_torch_device
from spacefit_v2.model.scorer import PlacementScorer


def _move_dataset_to_device(dataset: PlacementDataset, device: torch.device) -> None:
    dataset.features = dataset.features.to(device, non_blocking=True)
    dataset.category_idx = dataset.category_idx.to(device, non_blocking=True)
    dataset.labels = dataset.labels.to(device, non_blocking=True)


def train(args: argparse.Namespace) -> None:
    requested_device = "cuda" if args.use_gpu and args.device == "auto" else args.device
    device = resolve_torch_device(requested_device)
    print(f"Training on {device}")

    samples = generate_training_data(
        data_dir=args.data_dir,
        room_type=args.room,
        split=args.split,
        limit=args.max_scenes,
        seed=args.seed,
    )
    dataset = PlacementDataset(samples)

    val_size = max(1, int(round(len(dataset) * args.val_ratio)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    if args.preload_gpu and device.type == "cuda":
        _move_dataset_to_device(dataset, device)

    pin_memory = device.type == "cuda" and not args.preload_gpu
    sampler = None
    shuffle = True
    if args.balance_labels or args.balance_categories:
        full_weights = dataset.make_sample_weights(
            balance_labels=args.balance_labels,
            balance_categories=args.balance_categories,
        )
        train_weights = full_weights[train_ds.indices]
        sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=pin_memory,
        num_workers=0 if args.preload_gpu else args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=0 if args.preload_gpu else args.num_workers,
    )

    category_to_idx = load_category_mapping(args.data_dir, args.room)
    model = PlacementScorer(num_categories=max(1, len(category_to_idx))).to(device)
    pos_weight = None
    if args.pos_weight is not None:
        pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for features, cat_idx, labels in train_loader:
            features = features.to(device, non_blocking=True)
            cat_idx = cat_idx.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(features, category_idx=cat_idx).squeeze(-1)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * labels.shape[0]
            correct += int(((logits > 0.0).float() == labels).sum().item())
            total += int(labels.shape[0])

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == args.epochs:
            model.eval()
            with torch.no_grad():
                val_correct = 0
                val_total = 0
                for features, cat_idx, labels in val_loader:
                    features = features.to(device, non_blocking=True)
                    cat_idx = cat_idx.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    logits = model(features, category_idx=cat_idx).squeeze(-1)
                    val_correct += int(((logits > 0.0).float() == labels).sum().item())
                    val_total += int(labels.shape[0])
            print(
                f"Epoch {epoch + 1:03d}/{args.epochs} "
                f"loss={total_loss / max(total, 1):.4f} "
                f"train_acc={correct / max(total, 1):.4f} "
                f"val_acc={val_correct / max(val_total, 1):.4f}"
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "numeric_dim": config.SCORER_NUMERIC_DIM,
        "embed_dim": config.SCORER_EMBED_DIM,
        "hidden_dim": config.SCORER_HIDDEN_DIM,
        "num_categories": len(category_to_idx),
        "category_to_idx": category_to_idx,
        "room": args.room,
        "split": args.split,
    }
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, out_path)
    print(f"Saved scorer checkpoint to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(config.DATA_DIR))
    parser.add_argument("--room", default="bedroom")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.TRAIN_LR)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preload_gpu", action="store_true")
    parser.add_argument("--balance_labels", action="store_true")
    parser.add_argument("--balance_categories", action="store_true")
    parser.add_argument("--pos_weight", type=float, default=None)
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "scorer_model.pt"))
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)
