"""
model.py
--------
Defines the 1-D CNN architecture used for ECG beat classification.
The same architecture is reused for both the general model and fine-tuning.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_ecg_cnn(input_length: int = 180, num_classes: int = 2) -> tf.keras.Model:
    """
    1-D CNN for ECG beat classification.

    Architecture:
        Conv1D → BN → ReLU → MaxPool  (x3 blocks)
        GlobalAvgPool
        Dense(128) → Dropout
        Dense(num_classes, softmax)

    Parameters
    ----------
    input_length : int  – number of time-steps per beat segment
    num_classes  : int  – 2 for binary (Normal / Abnormal)

    Returns
    -------
    Compiled Keras Model
    """
    inp = layers.Input(shape=(input_length, 1), name="ecg_input")

    # ── Block 1 ──────────────────────────────────────────────────────────────
    x = layers.Conv1D(32, kernel_size=7, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4), name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    # ── Block 2 ──────────────────────────────────────────────────────────────
    x = layers.Conv1D(64, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4), name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    # ── Block 3 ──────────────────────────────────────────────────────────────
    x = layers.Conv1D(128, kernel_size=3, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4), name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool3")(x)

    # ── Classifier head ──────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4), name="dense1")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="ECG_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def freeze_feature_extractor(model: tf.keras.Model) -> tf.keras.Model:
    """
    Freeze all convolutional layers so only the dense head is trained
    during early fine-tuning epochs.
    """
    for layer in model.layers:
        if layer.name.startswith("conv") or layer.name.startswith("bn"):
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_all(model: tf.keras.Model, lr: float = 1e-4) -> tf.keras.Model:
    """Unfreeze all layers for full fine-tuning with a lower learning rate."""
    for layer in model.layers:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    m = build_ecg_cnn()
    m.summary()
