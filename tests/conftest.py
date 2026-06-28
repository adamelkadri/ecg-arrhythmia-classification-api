"""
Shared pytest fixtures and path setup.

Putting a conftest.py at the repo root makes pytest insert the project root on
sys.path, so `from src import ...` and `from app ...` work no matter where you
launch pytest from.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402  (import after sys.path tweak)
from src.model import ECGCNN  # noqa: E402
from src.utils import get_device  # noqa: E402


@pytest.fixture
def dummy_signal():
    """A valid-length ECG signal (all zeros) for API/inference tests."""
    return [0.0] * config.SIGNAL_LENGTH


@pytest.fixture
def untrained_model():
    """A real ECGCNN with random weights — no training, no saved file needed.

    This lets prediction tests exercise the *real* forward + softmax path
    without depending on models/ecg_cnn.pt existing.

    The model is moved to the inference device (CPU/MPS/CUDA) to mirror what the
    production `load_model()` does — otherwise the input (which `_prepare_signal`
    puts on that device) and the weights would be on different devices.
    """
    model = ECGCNN().to(get_device())
    model.eval()
    return model
