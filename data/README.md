# Data

The dataset CSVs are **not** committed to git (they are ~100 MB). Download them
yourself and place them in this folder so you end up with:

```
data/
├── mitbih_train.csv
└── mitbih_test.csv
```

## How to download (MIT-BIH Arrhythmia — Kaggle "Heartbeat" version)

The dataset is "ECG Heartbeat Categorization Dataset" by Shayan Fazeli on Kaggle.

### Option A — Manual (no account setup beyond a Kaggle login)
1. Go to https://www.kaggle.com/datasets/shayanfazeli/heartbeat
2. Click **Download** (you need a free Kaggle account).
3. Unzip it and copy `mitbih_train.csv` and `mitbih_test.csv` into this `data/` folder.

### Option B — Kaggle CLI
```bash
pip install kaggle
# Put your kaggle.json API token in ~/.kaggle/ first (Kaggle > Account > Create New API Token)
kaggle datasets download -d shayanfazeli/heartbeat -p data/ --unzip
```

## Format

- Each row is **one heartbeat**.
- Columns `0..186` are the ECG signal samples (187 values), already
  amplitude-normalized to roughly `[0, 1]` and zero-padded to a fixed length.
- The **last column (187)** is the integer class label `0..4`.

| Label | Symbol | Meaning                                   |
|-------|--------|-------------------------------------------|
| 0     | N      | Normal beat                               |
| 1     | S      | Supraventricular ectopic beat             |
| 2     | V      | Ventricular ectopic beat                  |
| 3     | F      | Fusion beat                               |
| 4     | Q      | Unknown / unclassifiable beat             |

The dataset is **heavily imbalanced** — class 0 (Normal) dominates. This is why
the project uses a class-weighted loss / weighted sampler (see `src/train.py`).
