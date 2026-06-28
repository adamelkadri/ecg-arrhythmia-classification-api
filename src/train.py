"""
Train the 1D CNN and save the best model to models/ecg_cnn.pt.

Run from the repo root:

    python -m src.train

Key choices:
- Loss     : CrossEntropyLoss, optionally weighted by inverse class frequency.
- Optimiser: Adam with weight decay.
- Scheduler: ReduceLROnPlateau on validation macro-F1.
- Selection: we keep the checkpoint with the best *validation macro-F1*, which
             matters more than raw accuracy on an imbalanced dataset.
"""
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src import config
from src.dataset import build_dataloaders
from src.model import ECGCNN
from src.utils import count_parameters, get_device, set_seed


@torch.no_grad()
def evaluate_loader(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float, float]:
    """Return (average loss, accuracy, macro-F1) over a DataLoader."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        logits = model(signals)
        loss = criterion(logits, labels)
        total_loss += loss.item() * signals.size(0)

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = (preds == targets).mean()
    macro_f1 = f1_score(targets, preds, average="macro")
    return avg_loss, accuracy, macro_f1


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch; return the average training loss."""
    model.train()
    total_loss = 0.0

    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * signals.size(0)

    return total_loss / len(loader.dataset)


def train() -> Dict[str, float]:
    """Full training loop. Saves the best model and returns its val metrics."""
    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, _, class_weights = build_dataloaders()

    model = ECGCNN().to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Use class-weighted loss unless the sampler strategy is selected.
    if config.IMBALANCE_STRATEGY == "class_weights":
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(f"Class weights: {class_weights.numpy().round(3)}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using WeightedRandomSampler (unweighted loss).")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    best_metrics: Dict[str, float] = {}

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1 = evaluate_loader(
            model, val_loader, criterion, device
        )
        scheduler.step(val_f1)

        print(
            f"Epoch {epoch:02d}/{config.NUM_EPOCHS} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | "
            f"val_acc {val_acc:.4f} | "
            f"val_macroF1 {val_f1:.4f}"
        )

        # Keep the checkpoint with the best validation macro-F1.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
            }
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"  -> saved new best model (val_macroF1={val_f1:.4f})")

    print(
        f"\nBest model: epoch {best_metrics['epoch']} | "
        f"val_acc {best_metrics['val_acc']:.4f} | "
        f"val_macroF1 {best_metrics['val_macro_f1']:.4f}"
    )
    print(f"Saved to: {config.MODEL_PATH}")
    return best_metrics


if __name__ == "__main__":
    train()
