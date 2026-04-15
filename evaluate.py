"""
evaluate.py
-----------
Loads personalization_metrics.json and generates:
  1. results/figures/per_patient_accuracy.png  – bar chart per patient
  2. results/figures/fine_tune_fraction.png    – effect of data fraction
  3. results/figures/confusion_matrix_general.png
  4. results/figures/summary_table.png         – tabular comparison
  5. Prints a formatted summary to console
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

DATA_DIR    = "data"
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_PATH = os.path.join(RESULTS_DIR, "personalization_metrics.json")
MODEL_PATH   = os.path.join(RESULTS_DIR, "general_model.keras")

COLORS = {
    "general": "#E74C3C",
    "frac_5":  "#F39C12",
    "frac_10": "#2ECC71",
    "frac_20": "#3498DB",
}
LABELS = {
    "general": "General Model",
    "frac_5":  "Personalized 5%",
    "frac_10": "Personalized 10%",
    "frac_20": "Personalized 20%",
}


def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


# ── Plot 1: Per-patient accuracy bar chart ───────────────────────────────────
def plot_per_patient_accuracy(metrics):
    patients = list(metrics.keys())
    model_keys = ["general", "frac_5", "frac_10", "frac_20"]
    present_keys = [k for k in model_keys if any(k in metrics[p] for p in patients)]

    n_groups = len(patients)
    n_bars   = len(present_keys)
    x = np.arange(n_groups)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, key in enumerate(present_keys):
        vals = [metrics[p].get(key, {}).get("accuracy", np.nan) for p in patients]
        ax.bar(x + i * width, vals, width=width,
               label=LABELS[key], color=COLORS[key], alpha=0.85, edgecolor="white")

    ax.set_xlabel("Patient ID", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Per-Patient Accuracy: General vs Personalized Models", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * (n_bars - 1) / 2)
    ax.set_xticklabels([f"P{p}" for p in patients], rotation=30)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5, label="90% threshold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "per_patient_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved → {path}")


# ── Plot 2: Effect of fine-tuning data fraction ───────────────────────────────
def plot_fraction_effect(metrics):
    fracs    = [0, 5, 10, 20]
    frac_keys = ["general", "frac_5", "frac_10", "frac_20"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for pid, pdata in metrics.items():
        acc_series = []
        for key in frac_keys:
            if key in pdata:
                acc_series.append(pdata[key]["accuracy"])
            else:
                acc_series.append(np.nan)
        if not all(np.isnan(acc_series)):
            ax.plot(fracs, acc_series, marker="o", alpha=0.55, linewidth=1.5,
                    label=f"Patient {pid}")

    # Average line
    all_vals = {k: [] for k in frac_keys}
    for pid, pdata in metrics.items():
        for key in frac_keys:
            if key in pdata:
                all_vals[key].append(pdata[key]["accuracy"])
    avg = [np.mean(all_vals[k]) if all_vals[k] else np.nan for k in frac_keys]
    ax.plot(fracs, avg, marker="D", color="black", linewidth=2.5,
            markersize=8, label="Average", zorder=5)

    ax.set_xlabel("Fine-tuning Data Fraction (%)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Effect of Fine-Tuning Data Size on Accuracy", fontsize=13, fontweight="bold")
    ax.set_xticks(fracs)
    ax.set_xticklabels(["0% (General)", "5%", "10%", "20%"])
    ax.set_ylim(0.6, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "fine_tune_fraction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved → {path}")


# ── Plot 3: Confusion matrix for general model ────────────────────────────────
def plot_general_confusion_matrix():
    model = tf.keras.models.load_model(MODEL_PATH)

    all_true, all_pred = [], []
    test_records = ["213","214","215","217","219","220","221","222","223","228"]

    for pid in test_records:
        fpath = os.path.join(DATA_DIR, f"patient_{pid}.npz")
        if not os.path.exists(fpath):
            continue
        d = np.load(fpath)
        X, y = d["X"], d["y"]
        preds = np.argmax(model.predict(X, verbose=0), axis=1)
        all_true.extend(y)
        all_pred.extend(preds)

    if not all_true:
        print("  ⚠️  No patient data found for confusion matrix.")
        return

    cm = confusion_matrix(all_true, all_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal","Abnormal"],
                yticklabels=["Normal","Abnormal"], ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("General Model – Confusion Matrix\n(All Test Patients Combined)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "confusion_matrix_general.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved → {path}")

    print("\n  Classification Report (General Model):")
    print(classification_report(all_true, all_pred,
                                 target_names=["Normal","Abnormal"], digits=4))


# ── Plot 4: Summary F1 comparison ────────────────────────────────────────────
def plot_f1_comparison(metrics):
    keys   = ["general", "frac_5", "frac_10", "frac_20"]
    avgs   = []
    stdevs = []
    for key in keys:
        vals = [v[key]["f1"] for v in metrics.values() if key in v]
        avgs.append(np.mean(vals) if vals else 0)
        stdevs.append(np.std(vals) if vals else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([LABELS[k] for k in keys], avgs,
                  color=[COLORS[k] for k in keys], alpha=0.85,
                  edgecolor="white", yerr=stdevs, capsize=5)
    ax.set_ylabel("Average Weighted F1-Score", fontsize=12)
    ax.set_title("F1-Score Comparison: General vs Personalized Models",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "f1_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved → {path}")


# ── Console summary ───────────────────────────────────────────────────────────
def print_summary(metrics):
    keys = ["general", "frac_5", "frac_10", "frac_20"]
    header = f"{'Patient':<12}" + "".join(f"{LABELS[k]:>22}" for k in keys)
    print("\n" + "=" * len(header))
    print("  ACCURACY SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for pid, pdata in metrics.items():
        row = f"{pid:<12}"
        for key in keys:
            val = pdata.get(key, {}).get("accuracy", None)
            row += f"{'N/A':>22}" if val is None else f"{val:>21.4f} "
        print(row)

    print("-" * len(header))
    row = f"{'AVERAGE':<12}"
    for key in keys:
        vals = [v[key]["accuracy"] for v in metrics.values() if key in v]
        row += f"{np.mean(vals):>21.4f} " if vals else f"{'N/A':>22}"
    print(row)
    print("=" * len(header))


def run_evaluation():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=== Running Evaluation ===\n")
    metrics = load_metrics()

    print_summary(metrics)

    print("\nGenerating plots …")
    plot_per_patient_accuracy(metrics)
    plot_fraction_effect(metrics)
    plot_f1_comparison(metrics)
    plot_general_confusion_matrix()

    print("\n✅ All figures saved to results/figures/")


if __name__ == "__main__":
    run_evaluation()
