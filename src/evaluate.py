"""Métricas y gráficas."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score)

ORANGE, DARK, GREY = "#F04E23", "#1F1F1F", "#9A9A9A"


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_normal": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.6), dpi=160)
    ax.imshow(cm, cmap="Oranges")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else DARK, fontsize=11)
    ax.set_xticks([0, 1], ["Normal", "Ataque"])
    ax.set_yticks([0, 1], ["Normal", "Ataque"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, transparent=False)
    plt.close(fig)


def plot_convergence(history, path):
    fig, ax = plt.subplots(figsize=(5, 3), dpi=160)
    ax.plot(range(1, len(history) + 1), history, color=ORANGE, lw=2)
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Mejor fitness")
    ax.set_title("Convergencia del Firefly Algorithm", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_history(hist, path):
    fig, ax = plt.subplots(figsize=(5, 3), dpi=160)
    ax.plot(hist["accuracy"], color=ORANGE, lw=2, label="train")
    if "val_accuracy" in hist:
        ax.plot(hist["val_accuracy"], color=DARK, lw=2, label="validación")
    ax.set_xlabel("Época")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_comparison(rows, path):
    """rows: lista de (nombre, valor_paper|None, valor_replica)."""
    names = [r[0] for r in rows]
    paper = [r[1] if r[1] is not None else np.nan for r in rows]
    ours = [r[2] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=160)
    b1 = ax.bar(x - 0.2, paper, 0.38, color=GREY, label="Paper")
    b2 = ax.bar(x + 0.2, ours, 0.38, color=ORANGE, label="Réplica")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if not np.isnan(h):
                ax.text(b.get_x() + b.get_width() / 2, h + 0.005, f"{h:.1%}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, names, fontsize=8)
    lo = np.nanmin(paper + ours)
    ax.set_ylim(max(0, lo - 0.08), 1.02)
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
