"""
personalize.py
--------------
Fine-tunes the general model for each test patient using small fractions
(5%, 10%, 20%) of that patient's data.

Strategy (two-phase fine-tuning):
  Phase 1 – Freeze conv layers, train only dense head (5 epochs, higher LR)
  Phase 2 – Unfreeze all layers, full fine-tune (10 epochs, low LR)

Saves:
  - results/personalized_<patient>_frac<pct>.keras
  - results/personalization_metrics.npz   (for evaluate.py)
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score

from model import freeze_feature_extractor, unfreeze_all

DATA_DIR     = "data"
RESULTS_DIR  = "results"
MODEL_PATH   = os.path.join(RESULTS_DIR, "general_model.keras")

FRACTIONS    = [0.05, 0.10, 0.20]
TEST_RECORDS = ["213","214","215","217","219","220","221","222","223","228"]

PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 10
BATCH_SIZE    = 32


def load_patient(record_id: str):
    path = os.path.join(DATA_DIR, f"patient_{record_id}.npz")
    d = np.load(path)
    return d["X"], d["y"]


def fine_tune_patient(patient_id: str, fraction: float, general_model_path: str):
    """
    Fine-tune for one patient at one data fraction.
    Returns (accuracy, f1) on the held-out test portion.
    """
    X, y = load_patient(patient_id)

    # ── Split: fraction for fine-tuning, rest for testing ──────────────────
    # Ensure we don't use entire patient data
    ft_size = max(int(len(y) * fraction), 10)   # at least 10 samples
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    ft_idx   = idx[:ft_size]
    test_idx = idx[ft_size:]

    if len(test_idx) < 10:
        return None, None   # not enough data

    X_ft, y_ft     = X[ft_idx], y[ft_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Class weights for fine-tune set
    classes = np.unique(y_ft)
    if len(classes) < 2:
        return None, None   # only one class present
    weights = compute_class_weight("balanced", classes=classes, y=y_ft)
    cw = dict(zip(classes, weights))

    # ── Load fresh copy of general model ───────────────────────────────────
    model = tf.keras.models.load_model(general_model_path)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=3, restore_best_weights=True, monitor="loss"
        )
    ]

    # Phase 1: head-only training
    model = freeze_feature_extractor(model)
    model.fit(X_ft, y_ft, epochs=PHASE1_EPOCHS, batch_size=BATCH_SIZE,
              class_weight=cw, callbacks=callbacks, verbose=0)

    # Phase 2: full fine-tuning with lower LR
    model = unfreeze_all(model, lr=5e-5)
    model.fit(X_ft, y_ft, epochs=PHASE2_EPOCHS, batch_size=BATCH_SIZE,
              class_weight=cw, callbacks=callbacks, verbose=0)

    # ── Evaluate on held-out test set ──────────────────────────────────────
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Save model
    pct = int(fraction * 100)
    save_path = os.path.join(RESULTS_DIR, f"personalized_{patient_id}_frac{pct}.keras")
    model.save(save_path)

    return acc, f1


def evaluate_general_on_patient(patient_id: str, general_model):
    """Evaluate the general model (no fine-tuning) on full patient data."""
    X, y = load_patient(patient_id)
    y_pred = np.argmax(general_model.predict(X, verbose=0), axis=1)
    acc = accuracy_score(y, y_pred)
    f1  = f1_score(y, y_pred, average="weighted", zero_division=0)
    return acc, f1


def run_personalization():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.random.seed(42)

    print("=== Loading General Model ===")
    general_model = tf.keras.models.load_model(MODEL_PATH)

    results = {}   # {patient_id: {"general": (acc,f1), "0.05": (acc,f1), ...}}

    for pid in TEST_RECORDS:
        patient_file = os.path.join(DATA_DIR, f"patient_{pid}.npz")
        if not os.path.exists(patient_file):
            print(f"  ⚠️  Skipping patient {pid} — file not found")
            continue

        print(f"\n── Patient {pid} ──────────────────────────────────────────")
        results[pid] = {}

        # General model baseline
        g_acc, g_f1 = evaluate_general_on_patient(pid, general_model)
        results[pid]["general"] = {"accuracy": g_acc, "f1": g_f1}
        print(f"  General model   → Acc: {g_acc:.4f}  F1: {g_f1:.4f}")

        # Fine-tuned at each fraction
        for frac in FRACTIONS:
            pct = int(frac * 100)
            acc, f1 = fine_tune_patient(pid, frac, MODEL_PATH)
            if acc is None:
                print(f"  Frac {pct:2d}%         → Skipped (insufficient data)")
                continue
            results[pid][f"frac_{pct}"] = {"accuracy": acc, "f1": f1}
            delta = acc - g_acc
            print(f"  Frac {pct:2d}%         → Acc: {acc:.4f}  F1: {f1:.4f}  "
                  f"(Δ={delta:+.4f})")

    # Save results dict as JSON
    metrics_path = os.path.join(RESULTS_DIR, "personalization_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Metrics saved → {metrics_path}")


if __name__ == "__main__":
    run_personalization()
