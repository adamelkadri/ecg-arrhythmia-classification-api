"""
1D CNN for ECG heartbeat classification.

Input  : (batch, 1, 187)  — one channel, 187 signal samples per beat.
Output : (batch, 5)       — raw logits, one per class (N, S, V, F, Q).

Architecture (3 conv blocks -> global pooling -> classifier):

    Conv1d(1  -> 32, k=5) -> BN -> ReLU -> MaxPool(2)
    Conv1d(32 -> 64, k=5) -> BN -> ReLU -> MaxPool(2)
    Conv1d(64 ->128, k=3) -> BN -> ReLU -> AdaptiveAvgPool(1)
    Flatten -> Dropout -> Linear(128 -> 64) -> ReLU -> Dropout -> Linear(64 -> 5)

Using AdaptiveAvgPool1d(1) at the end means the classifier size doesn't depend
on the exact signal length, which keeps the model robust to small input changes.
"""
import torch
import torch.nn as nn

from src import config


class ECGCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()

        # ---- Feature extractor: stacked 1D conv blocks ---------------------- #
        self.features = nn.Sequential(
            # Block 1: (B, 1, 187) -> (B, 32, 93)
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 2: (B, 32, 93) -> (B, 64, 46)
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 3: (B, 64, 46) -> (B, 128, 46)
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # Squash the time axis to length 1 -> (B, 128, 1)
            nn.AdaptiveAvgPool1d(1),
        )

        # ---- Classifier head ----------------------------------------------- #
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # (B, 128, 1) -> (B, 128)
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),   # raw logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Quick shape check: python -m src.model
    from src.utils import count_parameters

    model = ECGCNN()
    dummy = torch.randn(8, 1, config.SIGNAL_LENGTH)   # batch of 8 beats
    out = model(dummy)
    print(model)
    print(f"\nInput shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}  (expect (8, {config.NUM_CLASSES}))")
    print(f"Trainable parameters: {count_parameters(model):,}")
