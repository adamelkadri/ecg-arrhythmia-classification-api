"""
PyTorch Dataset + DataLoader helpers for the MIT-BIH CSV files.

The CSVs have 188 columns: 187 signal samples + 1 label. A 1D CNN in PyTorch
expects input shaped (batch, channels, length). Each heartbeat has a single
channel, so one sample is shaped (1, 187).
"""
from typing import Tuple

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split

from src import config


class ECGDataset(Dataset):
    """Wraps ECG signals + labels as tensors for a 1D CNN.

    Parameters
    ----------
    signals : np.ndarray, shape (N, 187)
        One row per heartbeat.
    labels : np.ndarray, shape (N,)
        Integer class labels 0..4.
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        # float32 is what nn.Conv1d expects; add a channel dim -> (N, 1, 187).
        self.signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.signals[idx], self.labels[idx]


def _load_csv(csv_path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a MIT-BIH CSV -> (signals (N,187), labels (N,)).

    The Kaggle files have NO header row, so we pass header=None.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}\n"
            "Download the MIT-BIH 'Heartbeat' CSVs from Kaggle and place "
            "mitbih_train.csv and mitbih_test.csv in the data/ folder.\n"
            "See data/README.md for download instructions."
        )
    df = pd.read_csv(csv_path, header=None)
    data = df.values
    signals = data[:, :-1].astype(np.float32)   # all but last column
    labels = data[:, -1].astype(np.int64)        # last column = label
    return signals, labels


def load_datasets() -> Tuple[ECGDataset, ECGDataset]:
    """Load the full train and test sets as ECGDataset objects."""
    train_signals, train_labels = _load_csv(config.TRAIN_CSV)
    test_signals, test_labels = _load_csv(config.TEST_CSV)
    return (
        ECGDataset(train_signals, train_labels),
        ECGDataset(test_signals, test_labels),
    )


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss.

    Rare classes get a larger weight so the model is penalised more for getting
    them wrong. Weights are normalised to average 1.0 for stable loss scale.
    """
    counts = torch.bincount(labels, minlength=config.NUM_CLASSES).float()
    counts = torch.clamp(counts, min=1.0)            # avoid divide-by-zero
    weights = counts.sum() / (config.NUM_CLASSES * counts)
    return weights


def _make_weighted_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    """Sampler that draws minority-class beats more often, balancing batches."""
    class_weights = compute_class_weights(labels)
    sample_weights = class_weights[labels]           # weight per sample
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    val_split: float = config.VAL_SPLIT,
    imbalance_strategy: str = config.IMBALANCE_STRATEGY,
    seed: int = config.RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Build train / val / test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, class_weights
        class_weights is always returned so train.py can use it for a weighted
        loss even when the sampler strategy is selected.
    """
    full_train, test_ds = load_datasets()

    # Split the training set into train + validation (deterministic via seed).
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(
        full_train, [train_size, val_size], generator=generator
    )

    # Class weights are computed from the TRAIN split only (no val/test leakage).
    train_labels = full_train.labels[train_ds.indices]
    class_weights = compute_class_weights(train_labels)

    if imbalance_strategy == "weighted_sampler":
        sampler = _make_weighted_sampler(train_labels)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler
        )
    else:  # "class_weights" (or anything else) -> plain shuffling
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True
        )

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    # Quick smoke test: python -m src.dataset
    tr, va, te, w = build_dataloaders()
    xb, yb = next(iter(tr))
    print(f"Batch signals shape: {tuple(xb.shape)}  (expect (B, 1, 187))")
    print(f"Batch labels shape:  {tuple(yb.shape)}")
    print(f"Class weights:       {w.numpy().round(3)}")
    print(f"Train/val/test batches: {len(tr)}/{len(va)}/{len(te)}")
