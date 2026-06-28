"""
Tests for src/model.py — pure dummy tensors, no data or training.
"""
import torch

from src import config
from src.model import ECGCNN
from src.utils import count_parameters


def test_forward_output_shape():
    model = ECGCNN()
    model.eval()

    batch = torch.randn(8, 1, config.SIGNAL_LENGTH)
    with torch.no_grad():
        out = model(batch)

    # 8 inputs -> 8 rows of NUM_CLASSES logits.
    assert out.shape == (8, config.NUM_CLASSES)


def test_forward_single_sample():
    # Batch size 1 must work (BatchNorm in eval mode handles this fine).
    model = ECGCNN()
    model.eval()

    with torch.no_grad():
        out = model(torch.randn(1, 1, config.SIGNAL_LENGTH))

    assert out.shape == (1, config.NUM_CLASSES)


def test_output_is_finite():
    model = ECGCNN()
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(4, 1, config.SIGNAL_LENGTH))
    assert torch.isfinite(out).all()


def test_model_has_trainable_parameters():
    model = ECGCNN()
    assert count_parameters(model) > 0
