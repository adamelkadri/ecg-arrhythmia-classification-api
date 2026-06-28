"""
Tests for app/main.py using FastAPI's TestClient.

No trained weights file is needed:

- /health and the input-validation tests don't touch the model at all.
- The successful /predict test swaps in an untrained model by monkeypatching
  `src.predict.load_model`. `predict_signal` looks that name up as a module
  global *at call time*, so patching it there is enough even though app.main
  imported `predict_signal` by reference. This is the cleanest bypass: we still
  exercise the real forward + softmax + response-shaping code.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from src import config

client = TestClient(app)


@pytest.fixture
def patch_model(monkeypatch, untrained_model):
    """Make load_model() return a random-weight model instead of reading a file."""
    monkeypatch.setattr("src.predict.load_model", lambda: untrained_model)
    return untrained_model


# --------------------------------------------------------------------------- #
# Health / root
# --------------------------------------------------------------------------- #
def test_health_returns_200():
    resp = client.get("/health")
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)


def test_root_returns_200():
    resp = client.get("/")
    assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# /predict — happy path (with mocked model)
# --------------------------------------------------------------------------- #
def test_predict_valid_signal(patch_model, dummy_signal):
    resp = client.post("/predict", json={"signal": dummy_signal})
    assert resp.status_code == 200

    body = resp.json()
    # All documented response fields are present.
    for key in (
        "predicted_class",
        "predicted_class_name",
        "predicted_class_description",
        "confidence",
        "class_probabilities",
    ):
        assert key in body

    # Predicted class is a valid label, name matches config.
    assert body["predicted_class"] in range(config.NUM_CLASSES)
    assert body["predicted_class_name"] in config.CLASS_NAMES
    assert 0.0 <= body["confidence"] <= 1.0

    # Probabilities cover every class and sum to ~1.
    probs = body["class_probabilities"]
    assert set(probs.keys()) == set(config.CLASS_NAMES)
    assert abs(sum(probs.values()) - 1.0) < 1e-4


def test_predict_accepts_short_signal_via_padding(patch_model):
    # predict._prepare_signal zero-pads short inputs, so this should still 200.
    resp = client.post("/predict", json={"signal": [0.1, 0.2, 0.3]})
    assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# /predict — input validation (no model needed; Pydantic rejects first)
# --------------------------------------------------------------------------- #
def test_predict_empty_signal_is_422():
    resp = client.post("/predict", json={"signal": []})
    assert resp.status_code == 422


def test_predict_missing_field_is_422():
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_predict_wrong_type_is_422():
    resp = client.post("/predict", json={"signal": "not-a-list"})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# /predict — model missing -> graceful 503
# --------------------------------------------------------------------------- #
def test_predict_without_model_returns_503(monkeypatch, dummy_signal):
    def _raise():
        raise FileNotFoundError("no weights")

    monkeypatch.setattr("src.predict.load_model", _raise)

    resp = client.post("/predict", json={"signal": dummy_signal})
    assert resp.status_code == 503
