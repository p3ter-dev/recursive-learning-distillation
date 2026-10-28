"""
src/recursive_learning.py
─────────────────────────────────────────────────────────────────────────────
Iterative confidence-filtered self-training (pseudo-labelling).

Algorithm per round
────────────────────
1. Run the current model over the unlabeled pool → class probabilities.
2. Accept examples whose max probability >= confidence_threshold.
3. Assign the predicted class as pseudo-label.
4. Build a combined dataset: original labeled examples ∪ accepted pseudo-labeled.
5. Fine-tune (or retrain) on the combined dataset.
6. Evaluate on the validation set.
7. Retain the model with the best validation accuracy across all rounds.
8. Repeat for max_rounds or until fewer than min_pseudo_labels are accepted.

Design safeguards
─────────────────
* Withheld true labels are fetched only through
  ``split.get_withheld_labels(use_for_diagnostic=True)`` and are used
  **exclusively** to compute ``pseudo_label_accuracy`` in the diagnostics dict.
  They are never passed to any DataLoader or loss function.
* ``generate_pseudo_labels`` returns LOCAL indices into the unlabeled Subset so
  that ``PseudoLabeledDataset`` maps them correctly without index aliasing.
* The original labeled dataset is re-included in every fine-tuning round
  (no catastrophic forgetting of ground-truth annotations).
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import (
    PseudoLabeledDataset,
    SplitDataManager,
    build_combined_loader,
)
from .train import eval_epoch, train_model
from .utils import Timer

log = logging.getLogger("rld.recursive")


# ─── Pseudo-label generation ──────────────────────────────────────────────────

@torch.no_grad()
def generate_pseudo_labels(
    model: nn.Module,
    unlabeled_loader: DataLoader,
    confidence_threshold: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on the unlabeled pool and apply the confidence filter.

    The loader MUST NOT shuffle (stable ordering required for index tracking).

    Returns
    -------
    accepted_indices  : LOCAL indices (into the Subset) of accepted examples
    pseudo_labels     : predicted class for each accepted example
    accepted_max_prob : max probability for each accepted example
    all_max_prob      : max probability for every unlabeled example (diagnostics)
    """
    model.eval()
    all_probs: List[np.ndarray] = []

    for imgs, _ in unlabeled_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)

    probs_arr = np.concatenate(all_probs, axis=0)       # (N, C)
    max_probs = probs_arr.max(axis=1)                    # (N,)
    pred_classes = probs_arr.argmax(axis=1)              # (N,)

    accepted_mask = max_probs >= confidence_threshold
    accepted_idx = np.where(accepted_mask)[0]

    return (
        accepted_idx,
        pred_classes[accepted_idx],
        max_probs[accepted_idx],
        max_probs,
    )


# ─── Diagnostics ─────────────────────────────────────────────────────────────

def pseudo_label_diagnostics(
    accepted_idx: np.ndarray,
    pseudo_labels: np.ndarray,
    accepted_max_prob: np.ndarray,
    all_max_prob: np.ndarray,
    withheld_labels: Optional[np.ndarray],
    n_unlabeled: int,
) -> Dict:
    """
    Compute pseudo-label quality statistics.

    Parameters
    ----------
    withheld_labels : true labels of the FULL unlabeled pool (indexed by LOCAL
                      unlabeled-Subset indices), or ``None`` when not available.
                      ⚠ DIAGNOSTIC-ONLY — must never be used in training.

    Returns
    -------
    dict with acceptance rate, class distribution, confidence statistics,
    and (when withheld_labels is not None) pseudo-label accuracy.
    """
    n_accepted = len(accepted_idx)
    diag: Dict = {
        "n_accepted": n_accepted,
        "n_unlabeled": n_unlabeled,
        "acceptance_rate": n_accepted / max(n_unlabeled, 1),
        "mean_confidence_accepted": float(accepted_max_prob.mean()) if n_accepted else 0.0,
        "mean_confidence_all": float(all_max_prob.mean()),
        "pseudo_label_class_dist": {
            int(c): int((pseudo_labels == c).sum()) for c in range(10)
        },
        # Diagnostic accuracy — filled below if withheld labels available
        "pseudo_label_accuracy": None,
        "pseudo_label_per_class_accuracy": None,
    }

    if withheld_labels is not None and n_accepted > 0:
        true_for_accepted = withheld_labels[accepted_idx]
        correct = pseudo_labels == true_for_accepted
        diag["pseudo_label_accuracy"] = float(correct.mean())

        per_class: Dict[int, Optional[float]] = {}
        for c in range(10):
            mask = pseudo_labels == c
            if mask.sum() > 0:
                per_class[c] = float(correct[mask].mean())
            else:
                per_class[c] = None
        diag["pseudo_label_per_class_accuracy"] = per_class

    return diag


# ─── Main self-training loop ──────────────────────────────────────────────────

def run_self_training(
    model: nn.Module,
    split: SplitDataManager,
    cfg: dict,
    device: torch.device,
    val_loader: DataLoader,
    checkpoint_dir: str = "./results/checkpoints",
    use_diagnostic_labels: bool = True,
) -> Dict:
    """
    Execute the full iterative self-training loop.

    Parameters
    ----------
    model               : initial model (already trained on labeled data).
                          Not modified in-place; deep copies are used per round.
    split               : SplitDataManager providing labeled / unlabeled subsets
    cfg                 : full experiment config dict
    device              : torch device
    val_loader          : validation DataLoader for model selection
    checkpoint_dir      : directory for round checkpoints
    use_diagnostic_labels : if True, compute pseudo-label accuracy diagnostics.
                            Set False when running without ground truth (realistic).

    Returns
    -------
    dict with keys:
      ``round_results``     — list of per-round metric dicts
      ``best_val_acc``      — validation accuracy of the selected model
      ``best_model_state``  — state_dict of the selected model (CPU)
    """
    rec_cfg = cfg["recursive"]
    max_rounds = int(rec_cfg.get("max_rounds", 5))
    threshold = float(rec_cfg.get("confidence_threshold", 0.95))
    min_pseudo = int(rec_cfg.get("min_pseudo_labels", 10))
    ft_epochs = int(rec_cfg.get("fine_tune_epochs", 5))
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["data"]["num_workers"])
    seed = int(cfg["seed"])

    # Fine-tune config: inherit from training but override epochs and disable
    # early stopping within rounds (short rounds; use patience = ft_epochs+1).
    ft_cfg = {**cfg["training"], "epochs": ft_epochs, "patience": ft_epochs + 1}

    # Fetch withheld labels (diagnostic-only gate)
    withheld_labels = split.get_withheld_labels(
        use_for_diagnostic=use_diagnostic_labels
    )
    unlabeled_subset = split.get_unlabeled_dataset()  # Subset
    unlabeled_loader = DataLoader(
        unlabeled_subset,
        batch_size=batch_size,
        shuffle=False,          # MUST NOT shuffle — index tracking depends on order
        num_workers=num_workers,
    )

    current_model = copy.deepcopy(model)
    best_val_acc = -1.0
    best_model_state = copy.deepcopy(model.state_dict())

    round_results: List[Dict] = []

    for rnd in range(1, max_rounds + 1):
        log.info(
            "━━━ Self-training round %d / %d  (threshold=%.2f) ━━━",
            rnd, max_rounds, threshold,
        )

        # ── 1. Generate pseudo-labels ───────────────────────────────────────
        with Timer() as t_pl:
            accepted_idx, pseudo_labels, accepted_probs, all_probs = (
                generate_pseudo_labels(current_model, unlabeled_loader, threshold, device)
            )

        # ── 2. Diagnostics ──────────────────────────────────────────────────
        diag = pseudo_label_diagnostics(
            accepted_idx, pseudo_labels, accepted_probs, all_probs,
            withheld_labels, len(unlabeled_subset),
        )
        pseudo_acc_str = (
            f"{diag['pseudo_label_accuracy']:.4f}"
            if diag["pseudo_label_accuracy"] is not None
            else "N/A (no ground truth)"
        )
        log.info(
            "Round %d: accepted %d / %d (%.1f%%) | "
            "mean conf=%.3f | pseudo-label accuracy=%s",
            rnd,
            diag["n_accepted"], diag["n_unlabeled"],
            diag["acceptance_rate"] * 100,
            diag["mean_confidence_accepted"],
            pseudo_acc_str,
        )

        # ── 3. Stopping criterion ───────────────────────────────────────────
        if diag["n_accepted"] < min_pseudo:
            log.warning(
                "Round %d: only %d pseudo-labels accepted (min=%d). Stopping.",
                rnd, diag["n_accepted"], min_pseudo,
            )
            break

        # ── 4. Build combined dataset ───────────────────────────────────────
        labeled_ds = split.get_labeled_dataset(augment=True)
        pseudo_ds = PseudoLabeledDataset(
            unlabeled_subset, accepted_idx, pseudo_labels
        )
        combined_loader = build_combined_loader(
            labeled_ds, pseudo_ds,
            batch_size=batch_size, seed=seed + rnd, num_workers=num_workers,
        )

        # ── 5. Fine-tune ────────────────────────────────────────────────────
        round_model = copy.deepcopy(current_model)
        ckpt = str(Path(checkpoint_dir) / f"recursive_round{rnd:02d}.pt")

        with Timer() as t_train:
            history = train_model(
                round_model, combined_loader, val_loader, ft_cfg, device,
                checkpoint_path=ckpt,
                label=f"recursive-r{rnd}",
            )

        # ── 6. Evaluate ─────────────────────────────────────────────────────
        from torch.nn import CrossEntropyLoss
        criterion = CrossEntropyLoss()
        val_loss, val_acc = eval_epoch(round_model, val_loader, criterion, device)

        log.info(
            "Round %d finished: val acc=%.4f | "
            "pseudo-label time=%.1fs | fine-tune time=%.1fs",
            rnd, val_acc, t_pl.elapsed, t_train.elapsed,
        )

        round_result = {
            "round": rnd,
            "val_acc": float(val_acc),
            "val_loss": float(val_loss),
            "fine_tune_best_val_acc": float(history["best_val_acc"]),
            "pseudo_gen_time_s": float(t_pl.elapsed),
            "fine_tune_time_s": float(t_train.elapsed),
            **diag,
        }
        round_results.append(round_result)

        # ── 7. Best-model selection ─────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                k: v.cpu().clone() for k, v in round_model.state_dict().items()
            }
            log.info("Round %d is new best (val acc=%.4f).", rnd, val_acc)

        current_model = round_model

    log.info(
        "Self-training complete over %d round(s). Best val acc=%.4f.",
        len(round_results), best_val_acc,
    )
    return {
        "round_results": round_results,
        "best_val_acc": best_val_acc,
        "best_model_state": best_model_state,
    }
