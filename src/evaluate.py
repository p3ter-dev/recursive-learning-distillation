"""
src/evaluate.py
─────────────────────────────────────────────────────────────────────────────
Evaluation metrics, inference-latency measurement, and result formatting.

Metrics returned
────────────────
accuracy, macro_precision, macro_recall, macro_f1,
per-class F1 for each of the 10 Fashion-MNIST classes,
confusion matrix (as a nested list for JSON / CSV compatibility),
n_parameters, model_size_mb, latency_ms_per_sample.

Latency methodology
────────────────────
Single-sample inference (batch size = 1) measured on CPU using
``time.perf_counter``.  The first ``warmup`` calls are discarded.
The reported value is the **median** over ``n`` calls.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .utils import count_parameters, model_size_mb

log = logging.getLogger("rld.evaluate")

CLASS_NAMES: List[str] = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ─── Prediction collection ────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the model over an entire DataLoader.

    Returns
    -------
    y_true : (N,) int array of ground-truth labels
    y_pred : (N,) int array of predicted class indices
    probs  : (N, C) float array of softmax probabilities
    """
    model.eval()
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_prob: List[np.ndarray] = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        pred = logits.argmax(1).cpu().numpy()
        all_true.append(labels.numpy())
        all_pred.append(pred)
        all_prob.append(p)

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
    )


# ─── Metrics computation ─────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model: Optional[nn.Module] = None,
    loader: Optional[DataLoader] = None,
    device: Optional[torch.device] = None,
    latency_n: int = 1000,
    latency_warmup: int = 50,
) -> Dict:
    """
    Compute the full set of evaluation metrics.

    Parameters
    ----------
    y_true, y_pred : arrays returned by ``get_predictions``
    model          : if provided, n_parameters and model_size_mb are added
    loader         : if provided alongside model and device, latency is measured
    device         : torch device for latency measurement
    latency_n      : number of single-sample calls for median latency
    latency_warmup : warm-up calls discarded before timing

    Returns
    -------
    Flat ``dict`` suitable for a CSV row.
    Confusion matrix is stored as a nested list (JSON-serialisable).
    """
    m: Dict = {}
    m["accuracy"] = float(accuracy_score(y_true, y_pred))
    m["macro_precision"] = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    m["macro_recall"] = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    m["macro_f1"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(CLASS_NAMES):
        safe_name = name.replace("/", "_").replace(" ", "_").replace("-", "_")
        m[f"f1_{safe_name}"] = float(per_class_f1[i])

    m["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    if model is not None:
        m["n_parameters"] = count_parameters(model)
        m["model_size_mb"] = model_size_mb(model)

    if model is not None and loader is not None and device is not None:
        m["latency_ms_per_sample"] = measure_latency(
            model, loader, device, n=latency_n, warmup=latency_warmup
        )

    return m


# ─── Inference latency ────────────────────────────────────────────────────────

def measure_latency(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n: int = 1000,
    warmup: int = 50,
) -> float:
    """
    Measure single-sample CPU inference latency.

    Procedure
    ---------
    1. Collect up to ``n + warmup`` individual images from *loader*.
    2. Repeat images cyclically if the loader provides fewer samples.
    3. Run each image through the model individually (batch size = 1).
    4. Discard the first ``warmup`` timings.
    5. Return the **median** of the remaining ``n`` timings in milliseconds.

    All calls are wrapped in ``torch.no_grad()`` and ``model.eval()``.
    """
    model.eval()
    total_needed = n + warmup

    # Collect individual sample tensors
    samples: List[torch.Tensor] = []
    for imgs, _ in loader:
        for img in imgs:
            samples.append(img.unsqueeze(0))
            if len(samples) >= total_needed:
                break
        if len(samples) >= total_needed:
            break

    # Repeat if dataset is smaller than total_needed
    if len(samples) < total_needed:
        reps = (total_needed // len(samples)) + 1
        samples = (samples * reps)[:total_needed]

    latencies: List[float] = []
    with torch.no_grad():
        for i, img in enumerate(samples):
            img = img.to(device)
            t0 = time.perf_counter()
            _ = model(img)
            dt_ms = (time.perf_counter() - t0) * 1_000
            if i >= warmup:
                latencies.append(dt_ms)

    return float(np.median(latencies))


# ─── Human-readable report ────────────────────────────────────────────────────

def print_report(metrics: Dict, label: str = "") -> None:
    """Print a formatted one-page summary of metrics to stdout."""
    width = 56
    header = f" {label} " if label else " Results "
    print("\n" + "═" * width)
    print(header.center(width))
    print("─" * width)
    print(f"  Accuracy          {metrics.get('accuracy', float('nan')):.4f}")
    print(f"  Macro-F1          {metrics.get('macro_f1', float('nan')):.4f}")
    print(f"  Macro-Precision   {metrics.get('macro_precision', float('nan')):.4f}")
    print(f"  Macro-Recall      {metrics.get('macro_recall', float('nan')):.4f}")
    if "n_parameters" in metrics:
        print(f"  Parameters        {metrics['n_parameters']:,}")
    if "model_size_mb" in metrics:
        print(f"  Model size        {metrics['model_size_mb']:.3f} MB")
    if "latency_ms_per_sample" in metrics:
        print(f"  Latency (median)  {metrics['latency_ms_per_sample']:.3f} ms/sample")
    if "training_time_s" in metrics:
        print(f"  Training time     {metrics['training_time_s']:.1f} s")
    print("═" * width + "\n")
