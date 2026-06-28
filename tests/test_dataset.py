"""
Tests for src/dataset.py — no CSV files or training required.

We build ECGDataset straight from dummy NumPy arrays, and for the DataLoader
test we monkeypatch the CSV loader so nothing touches disk.
"""
import numpy as np
import torch

from src import config
from src.dataset import ECGDataset, build_dataloaders, compute_class_weights


def _make_dummy_arrays(n=50):
    """n random 'heartbeats' of the right length + labels spanning all classes."""
    rng = np.random.default_rng(0)
    signals = rng.random((n, config.SIGNAL_LENGTH)).astype(np.float32)
    labels = rng.integers(0, config.NUM_CLASSES, size=n)
    return signals, labels


def test_dataset_length():
    signals, labels = _make_dummy_arrays(n=37)
    ds = ECGDataset(signals, labels)
    assert len(ds) == 37


def test_dataset_item_shape_and_dtype():
    signals, labels = _make_dummy_arrays()
    ds = ECGDataset(signals, labels)

    signal, label = ds[0]

    # One beat must be (1, 187) float32 so it feeds straight into nn.Conv1d.
    assert signal.shape == (1, config.SIGNAL_LENGTH)
    assert signal.dtype == torch.float32

    # Label is a scalar long tensor.
    assert label.dtype == torch.long
    assert label.ndim == 0


def test_compute_class_weights_balanced_is_near_one():
    # Perfectly balanced labels -> every weight should be ~1.0.
    labels = torch.tensor([0, 1, 2, 3, 4] * 10)
    weights = compute_class_weights(labels)

    assert weights.shape == (config.NUM_CLASSES,)
    assert torch.allclose(weights, torch.ones(config.NUM_CLASSES), atol=1e-5)


def test_compute_class_weights_rare_class_gets_higher_weight():
    # Class 0 is common, class 4 is rare -> class 4 must weigh more.
    labels = torch.tensor([0] * 90 + [4] * 10 + [1, 2, 3])
    weights = compute_class_weights(labels)
    assert weights[4] > weights[0]


def test_build_dataloaders_batch_shapes(monkeypatch):
    """Monkeypatch the CSV reader so build_dataloaders needs no files."""
    train = _make_dummy_arrays(n=200)
    test = _make_dummy_arrays(n=60)

    def fake_load_csv(path):
        # Return train or test arrays depending on which path was requested.
        return train if str(path) == str(config.TRAIN_CSV) else test

    monkeypatch.setattr("src.dataset._load_csv", fake_load_csv)

    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        batch_size=16
    )

    xb, yb = next(iter(train_loader))
    assert xb.shape == (16, 1, config.SIGNAL_LENGTH)
    assert yb.shape == (16,)
    assert class_weights.shape == (config.NUM_CLASSES,)

    # Train/val/test loaders are all non-empty.
    assert len(train_loader) > 0 and len(val_loader) > 0 and len(test_loader) > 0
