"""
train.py
--------
Trains the general CNN model on the multi-patient dataset produced
by preprocessing.py.  Saves:
  - results/general_model.keras
  - results/training_history.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from model import build_ecg_cnn

RESULTS_DIR  = "results"
FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures")
DATA_PATH    = os.path.join("data", "train_data.npz")
MODEL_PATH   = os.path.join(RESULTS_DIR, "general_model.keras")
EPOCHS       = 30
BATCH_SIZE   = 64


def load_data():
    d = np.load(DATA_PATH)
    return d["X_train"], d["X_val"], d["y_train"], d["y_val"]


def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("General Model – Training History", fontsize=14, fontweight="bold")

    axes[0].plot(history.history["loss"],     label="Train Loss",  color="#E74C3C")
    axes[0].plot(history.history["val_loss"], label="Val Loss",    color="#3498DB", linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["accuracy"],     label="Train Acc",  color="#2ECC71")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc",    color="#F39C12", linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved training history → {save_path}")


def train_general_model():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=== Training General ECG Classification Model ===\n")
    X_train, X_val, y_train, y_val = load_data()
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}")
    print(f"  Class distribution – Train: Normal={( y_train==0).sum()}, "
          f"Abnormal={(y_train==1).sum()}\n")

    # ── Handle class imbalance ──────────────────────────────────────────────
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))
    print(f"  Class weights: {class_weight}\n")

    model = build_ecg_cnn(input_length=X_train.shape[1])
    model.summary()

    # ── Callbacks ───────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=7, restore_best_weights=True, monitor="val_loss"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Save plot
    plot_history(history, os.path.join(FIGURES_DIR, "training_history.png"))

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n✅ General model – Val Accuracy: {val_acc:.4f}  Val Loss: {val_loss:.4f}")
    print(f"   Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train_general_model()
