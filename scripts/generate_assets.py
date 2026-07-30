"""
Generate README and documentation assets from project results.
Uses summary statistics from the notebook (no raw data required).
Run: python scripts/generate_assets.py
Output: assets/*.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")
sns.set_palette("husl")

# --- Data from notebook (summary statistics) ---
CLASS_COUNTS = [569_877, 20_663]  # Legitimate, Fraud
CLASS_LABELS = ["Legitimate (0)", "Fraud (1)"]
PRODUCT_FRAUD = {"C": 11.69, "S": 5.90, "H": 4.77, "R": 3.78, "W": 2.04}
DEVICE_FRAUD = {"mobile": 10.17, "desktop": 6.52, "null": 2.10}
MODEL_METRICS = [
    ("Logistic Regression", 0.7779, 0.1776, 0.0679, 0.8035, 0.1253, 80.4),
    ("Random Forest", 0.9164, 0.5815, 0.1528, 0.8362, 0.2583, 83.6),
    ("GBT (Champion)", 0.9460, 0.6753, 0.2052, 0.8667, 0.3319, 86.7),
]
TOP_FEATURES = [
    ("addr2", 0.161), ("card3", 0.128), ("addr1", 0.107),
    ("C2", 0.037), ("card5", 0.033), ("C8", 0.032), ("C9", 0.032),
    ("C5", 0.031), ("C1", 0.031), ("C4", 0.030),
]


def plot_class_distribution():
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(CLASS_LABELS, CLASS_COUNTS, color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Transaction count", fontsize=11)
    ax.set_title("Class Distribution (Highly Imbalanced)", fontsize=13, fontweight="bold")
    for b in bars:
        h = b.get_height()
        pct = 100 * h / sum(CLASS_COUNTS)
        ax.annotate(f"{h:,}\n({pct:.1f}%)", xy=(b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=10)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved class_distribution.png")


def plot_product_fraud_rate():
    fig, ax = plt.subplots(figsize=(8, 5))
    prods = list(PRODUCT_FRAUD.keys())
    rates = list(PRODUCT_FRAUD.values())
    colors = sns.color_palette("Reds_r", len(prods))
    bars = ax.bar(prods, rates, color=colors, edgecolor="white")
    ax.set_ylabel("Fraud rate (%)", fontsize=11)
    ax.set_title("Fraud Rate by Product Category (ProductCD)", fontsize=13, fontweight="bold")
    for b in bars:
        ax.annotate(f"{b.get_height():.2f}%", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fraud_by_product.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fraud_by_product.png")


def plot_device_fraud_rate():
    fig, ax = plt.subplots(figsize=(7, 5))
    devs = list(DEVICE_FRAUD.keys())
    rates = list(DEVICE_FRAUD.values())
    bars = ax.bar(devs, rates, color=["#3498db", "#9b59b6", "#95a5a6"], edgecolor="white")
    ax.set_ylabel("Fraud rate (%)", fontsize=11)
    ax.set_title("Fraud Rate by Device Type", fontsize=13, fontweight="bold")
    for b in bars:
        ax.annotate(f"{b.get_height():.2f}%", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fraud_by_device.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fraud_by_device.png")


def plot_model_comparison():
    names = [m[0] for m in MODEL_METRICS]
    aucs = [m[1] for m in MODEL_METRICS]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, aucs, color=["#e74c3c", "#f39c12", "#27ae60"], edgecolor="white")
    ax.axhline(y=0.90, color="gray", linestyle="--", label="Target AUC-ROC (0.90)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("AUC-ROC", fontsize=11)
    ax.set_title("Model Comparison — Validation AUC-ROC (Champion: GBT > 0.90)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.0)
    for i, (bar, v) in enumerate(zip(bars, aucs)):
        ax.annotate(f"{v:.4f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "model_comparison_auc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved model_comparison_auc.png")


def plot_feature_importance():
    features = [f[0] for f in TOP_FEATURES][::-1]
    imps = [f[1] for f in TOP_FEATURES][::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, imps, color=sns.color_palette("viridis", len(features)))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel("|Correlation with isFraud|", fontsize=11)
    ax.set_title("Top 10 Features by Correlation with Fraud (EDA)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved feature_importance_correlation.png")


def plot_precision_recall_by_threshold():
    # Champion GBT metrics at 0.3, 0.4, 0.5
    thresh = [0.3, 0.4, 0.5]
    prec = [0.1496, 0.2052, 0.2730]
    rec = [0.9095, 0.8667, 0.8270]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(thresh))
    w = 0.35
    ax.bar(x - w/2, prec, w, label="Precision", color="#1f77b4")
    ax.bar(x + w/2, rec, w, label="Recall", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in thresh])
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Score")
    ax.set_title("Champion Model (GBT): Precision & Recall by Threshold", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "precision_recall_by_threshold.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved precision_recall_by_threshold.png")


def main():
    plot_class_distribution()
    plot_product_fraud_rate()
    plot_device_fraud_rate()
    plot_model_comparison()
    plot_feature_importance()
    plot_precision_recall_by_threshold()
    print(f"\nAll assets written to {OUT_DIR}")


if __name__ == "__main__":
    main()
