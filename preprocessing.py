"""
preprocessing.py
----------------
Downloads MIT-BIH Arrhythmia records, segments individual heartbeats,
labels them Normal vs Abnormal, normalizes signals, and saves:
  - data/train_data.npz      (multi-patient training set)
  - data/patient_<id>.npz   (per-patient test sets)
"""

import os
import numpy as np
import wfdb
from sklearn.model_selection import train_test_split

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR        = "data"
WINDOW_SIZE     = 180          # samples per beat segment (±90 around R-peak)
ABNORMAL_LABELS = {"V", "A", "L", "R", "E", "f", "j", "a", "F", "Q"}
NORMAL_LABEL    = "N"

# First 30 records for training general model; last 10 kept for per-patient test
TRAIN_RECORDS = [
    "100","101","103","105","106","107","108","109","111","112",
    "113","114","115","116","118","119","121","122","123","124",
    "200","201","202","203","205","207","208","209","210","212",
]
TEST_RECORDS = ["213","214","215","217","219","220","221","222","223","228"]
# ──────────────────────────────────────────────────────────────────────────────


def download_record(record_id: str) -> tuple:
    """Download a single MIT-BIH record and return (signal, annotation)."""
    path = os.path.join(DATA_DIR, record_id)
    wfdb.dl_database("mitdb", dl_dir=DATA_DIR, records=[record_id])
    record = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, "atr")
    return record.p_signal[:, 0], annotation   # use lead II only


def segment_beats(signal: np.ndarray, annotation) -> tuple:
    """
    Slice fixed-length windows around each annotated R-peak.
    Returns (segments, labels) where label 0=Normal, 1=Abnormal.
    """
    segments, labels = [], []
    half = WINDOW_SIZE // 2

    for idx, sym in zip(annotation.sample, annotation.symbol):
        if sym not in ([NORMAL_LABEL] + list(ABNORMAL_LABELS)):
            continue
        start, end = idx - half, idx + half
        if start < 0 or end > len(signal):
            continue
        beat = signal[start:end]
        beat = (beat - beat.mean()) / (beat.std() + 1e-8)   # z-score normalize
        segments.append(beat)
        labels.append(0 if sym == NORMAL_LABEL else 1)

    return np.array(segments, dtype=np.float32), np.array(labels, dtype=np.int32)


def build_general_dataset():
    """Build multi-patient train/val set from TRAIN_RECORDS."""
    os.makedirs(DATA_DIR, exist_ok=True)
    all_X, all_y = [], []

    print("=== Building general training dataset ===")
    for rec in TRAIN_RECORDS:
        print(f"  Processing record {rec} ...", end=" ")
        try:
            sig, ann = download_record(rec)
            X, y = segment_beats(sig, ann)
            all_X.append(X)
            all_y.append(y)
            print(f"{len(y)} beats  (Normal: {(y==0).sum()}, Abnormal: {(y==1).sum()})")
        except Exception as e:
            print(f"SKIPPED ({e})")

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)

    # Add channel dim for CNN: (N, 180, 1)
    X = X[..., np.newaxis]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    np.savez_compressed(
        os.path.join(DATA_DIR, "train_data.npz"),
        X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val
    )
    print(f"\n✅ Saved train_data.npz  —  Train: {len(y_train)}  Val: {len(y_val)}")


def build_patient_datasets():
    """Build per-patient datasets from TEST_RECORDS."""
    print("\n=== Building per-patient test datasets ===")
    for rec in TEST_RECORDS:
        print(f"  Processing record {rec} ...", end=" ")
        try:
            sig, ann = download_record(rec)
            X, y = segment_beats(sig, ann)
            X = X[..., np.newaxis]
            np.savez_compressed(
                os.path.join(DATA_DIR, f"patient_{rec}.npz"),
                X=X, y=y
            )
            print(f"{len(y)} beats saved.")
        except Exception as e:
            print(f"SKIPPED ({e})")
    print("✅ Per-patient files saved.")


if __name__ == "__main__":
    build_general_dataset()
    build_patient_datasets()
