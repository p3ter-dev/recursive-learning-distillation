"""
src/plotting.py
─────────────────────────────────────────────────────────────────────────────
Visualisation utilities for training curves, confusion matrices, pseudo-label
diagnostics, and comparison bar charts.

All functions save figures to disk (``save_path``) and return the figure for
optional display in notebooks / scripts.  Matplotlib is used in non-interactive
mode (``Agg`` backend compatible) so figures render correctly on headless CPUs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")   # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger("rld.plotting")

CLASS_NAMES: List[str] = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

_STYLE = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "axes.titlecolor": "#e0e0e0",
    "grid.color": "#0f3460",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
}


def _apply_style() -> None:
    plt.rcParams.update(_STYLE)


# ─── Training curves ──────────────────────────────────────────────────────────

def plot_training_curves(
    history: Dict,
    save_path: str,
    title: str = "Training Curves",
) -> plt.Figure:
    """Plot loss and accuracy curves for a single training run."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)
    palette = {"train": "#e94560", "val": "#0f9b8e"}

    # Loss subplot
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color=palette["train"], label="Train loss", linewidth=2)
    ax.plot(epochs, history["val_loss"],   color=palette["val"],   label="Val loss",   linewidth=2)
    if "best_epoch" in history:
        ax.axvline(history["best_epoch"], color="#f5a623", linestyle=":", linewidth=1.5,
                   label=f"Best (ep {history['best_epoch']})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend(fontsize=9)
    ax.grid(True)

    # Accuracy subplot
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], color=palette["train"], label="Train acc", linewidth=2)
    ax.plot(epochs, history["val_acc"],   color=palette["val"],   label="Val acc",   linewidth=2)
    if "best_epoch" in history:
        ax.axvline(history["best_epoch"], color="#f5a623", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend(fontsize=9)
    ax.grid(True)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    log.info("Saved training curves → %s", save_path)
    return fig


# ─── Confusion matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: List[List[int]],
    save_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> plt.Figure:
    """Plot a 10×10 confusion matrix with Fashion-MNIST class labels."""
    _apply_style()
    cm_arr = np.array(cm, dtype=float)
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_arr = cm_arr / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    short = [n.split("/")[0][:8] for n in CLASS_NAMES]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=12, fontweight="bold")

    thresh = 0.5 if normalize else cm_arr.max() / 2
    for i in range(10):
        for j in range(10):
            val = cm_arr[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center", fontsize=6,
                    color="white" if val > thresh else "#cccccc")

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    log.info("Saved confusion matrix → %s", save_path)
    return fig


# ─── Per-class F1 bar chart ───────────────────────────────────────────────────

def plot_per_class_f1(
    metrics_list: List[Dict],
    labels: List[str],
    save_path: str,
    title: str = "Per-class F1",
) -> plt.Figure:
    """Compare per-class F1 scores across multiple models."""
    _apply_style()
    n_models = len(metrics_list)
    n_classes = 10
    x = np.arange(n_classes)
    width = 0.8 / n_models
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (m, lab) in enumerate(zip(metrics_list, labels)):
        f1_values = []
        for name in CLASS_NAMES:
            safe = name.replace("/", "_").replace(" ", "_").replace("-", "_")
            f1_values.append(m.get(f"f1_{safe}", 0.0))
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, f1_values, width, label=lab, color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([n.split("/")[0] for n in CLASS_NAMES], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="y")

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    log.info("Saved per-class F1 chart → %s", save_path)
    return fig


# ─── Self-training round diagnostics ─────────────────────────────────────────

def plot_self_training_rounds(
    round_results: List[Dict],
    save_path: str,
    title: str = "Self-training Diagnostics",
) -> plt.Figure:
    """Plot round-by-round val accuracy, acceptance rate, and pseudo-label accuracy."""
    _apply_style()
    rounds = [r["round"] for r in round_results]
    val_accs = [r["val_acc"] for r in round_results]
    acceptance = [r["acceptance_rate"] * 100 for r in round_results]
    pl_acc = [
        r["pseudo_label_accuracy"] if r.get("pseudo_label_accuracy") is not None else float("nan")
        for r in round_results
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    palette = ["#e94560", "#0f9b8e", "#f5a623"]

    for ax, values, ylabel, col in zip(
        axes,
        [val_accs, acceptance, pl_acc],
        ["Validation accuracy", "Acceptance rate (%)", "Pseudo-label accuracy (diagnostic)"],
        palette,
    ):
        ax.plot(rounds, values, "o-", color=col, linewidth=2, markersize=6)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=10)
        ax.set_xticks(rounds)
        ax.grid(True)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    log.info("Saved self-training diagnostics → %s", save_path)
    return fig


# ─── Model comparison bar chart ───────────────────────────────────────────────

def plot_comparison(
    metrics_dict: Dict[str, Dict],
    metric_key: str = "accuracy",
    save_path: str = "results/figures/comparison.png",
    title: str = "Model Comparison",
    ylabel: str = "Accuracy",
) -> plt.Figure:
    """Bar chart comparing a single metric across multiple models."""
    _apply_style()
    names = list(metrics_dict.keys())
    values = [metrics_dict[n].get(metric_key, 0.0) for n in names]
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(names)))

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.6), 5))
    bars = ax.bar(names, values, color=colors, edgecolor="#e0e0e0", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, min(1.1, max(values) * 1.12) if values else 1.0)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=9)
    ax.grid(True, axis="y")

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    log.info("Saved comparison chart → %s", save_path)
    return fig


# ─── Temperature ablation ─────────────────────────────────────────────────────

def plot_ablation(
    param_values: Sequence,
    metric_values: Sequence[float],
    param_name: str,
    save_path: str,
    title: str = "Ablation",
    ylabel: str = "Val accuracy",
) -> plt.Figure:
    """Line plot for single-parameter ablation (e.g. temperature sweep)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(param_values, metric_values, "o-", color="#e94560", linewidth=2, markersize=7)
    ax.set_xlabel(param_name)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    log.info("Saved ablation plot → %s", save_path)
    return fig
