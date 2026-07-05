#!/usr/bin/env python3
"""Generate README visualizations → docs/images/"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "images")
DATA = os.path.join(ROOT, "dataset_balanced_filtered")
EXP = os.path.join(ROOT, "outputs", "experiments_9class")

# Short labels for charts
SHORT = {
    "Online and Social Media Related Crime": "Social Media",
    "Online Financial Fraud": "Financial Fraud",
    "Cyber Attack/ Dependent Crimes": "Cyber Attack",
    "Hacking  Damage to computercomputer system etc": "Hacking",
    "Sexually Obscene material": "Obscene Material",
    "Any Other Cyber Crime": "Other",
    "Cryptocurrency Crime": "Crypto Crime",
    "Child Pornography CPChild Sexual Abuse Material CSAM": "CSAM",
    "RapeGang Rape RGRSexually Abusive Content": "Sexual Abuse",
}

# Published showcase scores (README)
SHOWCASE = [
    ("LoRA Qwen2.5-0.5B", 0.5880, "llm"),
    ("MuRIL + NER + calibrated", 0.5826, "neural"),
    ("MuRIL + NER + focal", 0.5813, "neural"),
    ("XLM-R-large + NER", 0.5692, "neural"),
    ("LightGBM + TF-IDF", 0.5013, "classical"),
    ("LinearSVM + TF-IDF + NER", 0.4813, "classical"),
    ("LLM zero-shot prompt", 0.0227, "llm"),
]

COLORS = {
    "llm": "#8b5cf6",
    "neural": "#3b82f6",
    "classical": "#94a3b8",
}

plt.rcParams.update(
    {
        "figure.facecolor": "#0f1419",
        "axes.facecolor": "#1a2332",
        "axes.edgecolor": "#2d3a4f",
        "axes.labelcolor": "#e7ecf3",
        "text.color": "#e7ecf3",
        "xtick.color": "#8b9cb3",
        "ytick.color": "#8b9cb3",
        "grid.color": "#2d3a4f",
        "font.family": "DejaVu Sans",
        "font.size": 10,
    }
)


def save(fig, name: str):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  {path}")


def short_label(cat: str) -> str:
    return SHORT.get(cat, cat[:22])


def plot_class_distribution():
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    counts = train["category"].value_counts().sort_values()
    labels = [short_label(c) for c in counts.index]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(labels, counts.values, color="#3b82f6", height=0.7)
    for bar, n in zip(bars, counts.values):
        ax.text(
            bar.get_width() + 60,
            bar.get_y() + bar.get_height() / 2,
            f"{n:,}",
            va="center",
            fontsize=9,
            color="#e7ecf3",
        )
    ax.set_xlabel("Training samples")
    ax.set_title(
        f"9-class distribution (train n={len(train):,} · test n={len(test):,})",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlim(0, counts.max() * 1.15)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save(fig, "dataset_class_distribution.png")


def plot_text_length():
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    lengths = train["crimeaditionalinfo"].fillna("").str.split().map(len)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(lengths.clip(upper=400), bins=50, color="#22c55e", alpha=0.85, edgecolor="#0f1419")
    med = int(lengths.median())
    ax.axvline(med, color="#f59e0b", linestyle="--", linewidth=1.5, label=f"median = {med} words")
    ax.set_xlabel("Complaint length (words)")
    ax.set_ylabel("Count")
    ax.set_title("Complaint text length distribution (train)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(facecolor="#1a2332", edgecolor="#2d3a4f")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "dataset_text_length.png")


def plot_split_pie():
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].pie(
        [len(train), len(test)],
        labels=["Train", "Test"],
        autopct="%1.1f%%",
        colors=["#3b82f6", "#6366f1"],
        textprops={"color": "#e7ecf3"},
        startangle=90,
    )
    axes[0].set_title(f"Split\n({len(train)+len(test):,} total)", fontweight="bold")

    top = train["category"].value_counts().head(5)
    other = len(train) - top.sum()
    pie_labels = [short_label(c) for c in top.index] + ["Other 4 classes"]
    pie_vals = list(top.values) + [other]
    axes[1].pie(
        pie_vals,
        labels=pie_labels,
        autopct="%1.0f%%",
        colors=plt.cm.Blues(np.linspace(0.35, 0.9, len(pie_vals))),
        textprops={"color": "#e7ecf3", "fontsize": 8},
        startangle=140,
    )
    axes[1].set_title("Train volume share (top 5 + rest)", fontweight="bold")
    fig.suptitle("Dataset overview", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "dataset_overview.png")


def plot_model_comparison():
    methods = [m for m, _, _ in reversed(SHOWCASE)]
    scores = [s * 100 for _, s, _ in reversed(SHOWCASE)]
    families = [f for _, _, f in reversed(SHOWCASE)]
    colors = [COLORS[f] for f in families]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(methods, scores, color=colors, height=0.65)
    for bar, sc in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{sc:.1f}%",
            va="center",
            fontsize=9,
            color="#e7ecf3",
        )
    ax.set_xlabel("Test macro-F1 (%)")
    ax.set_xlim(0, 68)
    ax.set_title("Model comparison (9-class, n=29,978 test)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.3)
    legend = [
        mpatches.Patch(color=COLORS["llm"], label="LLM"),
        mpatches.Patch(color=COLORS["neural"], label="Neural encoder"),
        mpatches.Patch(color=COLORS["classical"], label="Classical TF-IDF"),
    ]
    ax.legend(handles=legend, loc="lower right", facecolor="#1a2332", edgecolor="#2d3a4f")
    fig.tight_layout()
    save(fig, "model_comparison.png")


def plot_training_curves():
    path = os.path.join(EXP, "muril_focal_results.json")
    with open(path) as f:
        data = json.load(f)
    hist = data["history"]
    epochs = [h["epoch"] for h in hist]
    val_f1 = [h["cat_macro_f1"] * 100 for h in hist]
    train_loss = [h["train_loss"] for h in hist]
    val_loss = [h["val_loss"] for h in hist]
    best_ep = data["best_epoch"]
    best_f1 = data["best_val_cat_macro_f1"] * 100

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(epochs, val_f1, color="#22c55e", linewidth=2, marker="o", markersize=3, label="Val macro-F1")
    ax1.axhline(best_f1, color="#22c55e", linestyle=":", alpha=0.5)
    ax1.axvline(best_ep, color="#f59e0b", linestyle="--", alpha=0.7, label=f"Best epoch {best_ep}")
    ax2.plot(epochs, train_loss, color="#ef4444", alpha=0.7, linewidth=1.5, label="Train loss")
    ax2.plot(epochs, val_loss, color="#f59e0b", alpha=0.7, linewidth=1.5, linestyle="--", label="Val loss")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val macro-F1 (%)", color="#22c55e")
    ax2.set_ylabel("Loss", color="#ef4444")
    ax1.set_title("MuRIL + NER training (focal loss)", fontsize=12, fontweight="bold", pad=12)
    ax1.grid(alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", facecolor="#1a2332", edgecolor="#2d3a4f", fontsize=8)
    fig.tight_layout()
    save(fig, "training_curves_muril.png")


def plot_encoder_ablation():
    items = [
        ("MuRIL-base", 0.5813),
        ("XLM-R-large", 0.5692),
        ("LightGBM", 0.5013),
        ("LLM zero-shot", 0.0227),
    ]
    names = [i[0] for i in items]
    vals = [i[1] * 100 for i in items]
    cols = ["#3b82f6", "#6366f1", "#94a3b8", "#8b5cf6"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, vals, color=cols, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("Test macro-F1 (%)")
    ax.set_title("Encoder & paradigm comparison", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, 65)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=12, ha="right")
    fig.tight_layout()
    save(fig, "encoder_comparison.png")


def main():
    print("Generating README plots → docs/images/")
    plot_class_distribution()
    plot_text_length()
    plot_split_pie()
    plot_model_comparison()
    plot_training_curves()
    plot_encoder_ablation()
    print("Done.")


if __name__ == "__main__":
    main()
