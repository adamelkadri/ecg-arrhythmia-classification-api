"""
Evaluate the trained model on the held-out test set and save reports.

Run from the repo root (after training):

    python -m src.evaluate

Produces:
- results/classification_report.txt  (precision/recall/F1 per class + macro avg)
- results/confusion_matrix.png       (normalised confusion matrix heatmap)

Also prints overall accuracy and macro-F1 to the console.
"""
import matplotlib

matplotlib.use("Agg")  # headless backend so it works inside Docker / no display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src import config
from src.dataset import build_dataloaders
from src.model import ECGCNN
from src.utils import get_device, set_seed


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run the model over a loader and return (y_true, y_pred) arrays."""
    model.eval()
    all_preds, all_targets = [], []
    for signals, labels in loader:
        signals = signals.to(device)
        logits = model(signals)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_targets.append(labels)
    return torch.cat(all_targets).numpy(), torch.cat(all_preds).numpy()


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Save a row-normalised confusion matrix heatmap (recall per class)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        vmin=0.0,
        vmax=1.0,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (row-normalised = recall per class)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate() -> None:
    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {config.MODEL_PATH}. "
            "Run `python -m src.train` first."
        )

    _, _, test_loader, _ = build_dataloaders()

    model = ECGCNN().to(device)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))

    y_true, y_pred = collect_predictions(model, test_loader, device)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(config.NUM_CLASSES)),
        target_names=config.CLASS_NAMES,
        digits=4,
    )

    # ---- Console summary ---------------------------------------------------- #
    print(f"\nTest accuracy : {accuracy:.4f}")
    print(f"Test macro-F1 : {macro_f1:.4f}\n")
    print(report)

    # ---- Save text report --------------------------------------------------- #
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORT_PATH, "w") as f:
        f.write("ECG Arrhythmia Classification — Test Set Evaluation\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Overall accuracy : {accuracy:.4f}\n")
        f.write(f"Macro-F1         : {macro_f1:.4f}\n\n")
        f.write(report + "\n")
    print(f"\nSaved report  -> {config.REPORT_PATH}")

    # ---- Save confusion matrix ---------------------------------------------- #
    plot_confusion_matrix(y_true, y_pred, config.CONFUSION_MATRIX_PATH)
    print(f"Saved heatmap -> {config.CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    evaluate()
