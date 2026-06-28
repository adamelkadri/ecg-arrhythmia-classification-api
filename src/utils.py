"""
Small shared helpers used across training, evaluation and inference.

Keeping these in one place means train.py / evaluate.py / predict.py don't each
re-implement seeding or device selection.
"""
import random

import numpy as np
import torch

from src import config


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Seed Python, NumPy and PyTorch so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the best available device: CUDA > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon GPUs (M1/M2/M3...) via Metal.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Number of trainable parameters — handy for the README/CV."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
