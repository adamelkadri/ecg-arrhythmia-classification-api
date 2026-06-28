"""
Inference helpers: load the trained model once and classify a single heartbeat.

Used by both the CLI (`python -m src.predict`) and the FastAPI app.

The model is cached at module level so the API doesn't reload weights on every
request.
"""
from functools import lru_cache
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from src import config
from src.model import ECGCNN
from src.utils import get_device


@lru_cache(maxsize=1)
def load_model() -> torch.nn.Module:
    """Load weights once and return an eval-mode model (cached)."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {config.MODEL_PATH}. "
            "Run `python -m src.train` first."
        )
    device = get_device()
    model = ECGCNN().to(device)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.eval()
    return model


def _prepare_signal(signal: List[float]) -> torch.Tensor:
    """Validate a raw signal list and shape it to (1, 1, 187) on the right device.

    Pads with zeros or truncates to SIGNAL_LENGTH so the API is forgiving about
    slightly-off lengths (the MIT-BIH beats are zero-padded anyway).
    """
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("Signal must be a 1D list of numbers.")
    if arr.size == 0:
        raise ValueError("Signal is empty.")

    if arr.size < config.SIGNAL_LENGTH:
        arr = np.pad(arr, (0, config.SIGNAL_LENGTH - arr.size))
    elif arr.size > config.SIGNAL_LENGTH:
        arr = arr[: config.SIGNAL_LENGTH]

    tensor = torch.tensor(arr, dtype=torch.float32).view(1, 1, config.SIGNAL_LENGTH)
    return tensor.to(get_device())


@torch.no_grad()
def predict_signal(signal: List[float]) -> Dict:
    """Classify one ECG heartbeat.

    Returns a dict with the predicted class id, its name/description, the
    confidence (max softmax probability) and the full probability vector.
    """
    model = load_model()
    x = _prepare_signal(signal)

    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_id = int(probs.argmax())
    return {
        "predicted_class": pred_id,
        "predicted_class_name": config.CLASS_NAMES[pred_id],
        "predicted_class_description": config.CLASS_DESCRIPTIONS[pred_id],
        "confidence": float(probs[pred_id]),
        "class_probabilities": {
            config.CLASS_NAMES[i]: float(probs[i])
            for i in range(config.NUM_CLASSES)
        },
    }


if __name__ == "__main__":
    # Demo: classify the first row of the test CSV.
    import pandas as pd

    df = pd.read_csv(config.TEST_CSV, header=None)
    row = df.iloc[0]
    signal = row.values[:-1].tolist()
    true_label = int(row.values[-1])

    result = predict_signal(signal)
    print(f"True label     : {true_label} ({config.CLASS_NAMES[true_label]})")
    print(f"Predicted      : {result['predicted_class']} "
          f"({result['predicted_class_name']})")
    print(f"Confidence     : {result['confidence']:.4f}")
    print(f"Probabilities  : "
          f"{ {k: round(v, 4) for k, v in result['class_probabilities'].items()} }")
