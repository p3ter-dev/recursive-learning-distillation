"""
src/train.py
─────────────────────────────────────────────────────────────────────────────
Generic supervised training loop with early stopping, best-model
checkpointing, and optional learning-rate scheduling.

Key design choices
──────────────────
* ``train_model`` returns a ``history`` dict containing per-epoch loss /
  accuracy values and restores the best-validation-accuracy weights before
  returning, so callers receive the best model, not just the final one.
* Early stopping monitors *validation loss* (less noisy than accuracy).
* Checkpointing saves a CPU state_dict (no device-specific tensors).
* Scheduler options: cosine annealing (default), step-decay, or none.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from .utils import Timer

log = logging.getLogger("rld.train")


# ─── Optimizer / scheduler builders ──────────────────────────────────────────

def _build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    lr = float(cfg.get("learning_rate", 1e-3))
    wd = float(cfg.get("weight_decay", 1e-4))
    name = str(cfg.get("optimizer", "adam")).lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=wd, momentum=0.9
        )
    raise ValueError(f"Unknown optimizer '{name}'. Choose 'adam' or 'sgd'.")


def _build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict, epochs: int
) -> Optional[object]:
    name = str(cfg.get("scheduler", "cosine")).lower()
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    return None


# ─── Per-epoch routines ────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch.  Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = correct = n = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)

    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a DataLoader.  Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = correct = n = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)

    return total_loss / n, correct / n


# ─── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stop training when validation loss fails to improve for ``patience`` epochs.

    Parameters
    ----------
    patience  : number of consecutive non-improving epochs before stopping
    min_delta : minimum improvement (absolute) to count as an improvement
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, val_loss: float, epoch: int) -> bool:
        """Return ``True`` if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ─── Main training function ────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    label: str = "model",
) -> Dict:
    """
    Train a model with early stopping and best-checkpoint saving.

    Parameters
    ----------
    model           : model to train (modified in-place; best weights restored)
    train_loader    : labeled training DataLoader
    val_loader      : validation DataLoader
    cfg             : training sub-config dict (keys: epochs, patience,
                      learning_rate, weight_decay, optimizer, scheduler)
    device          : torch device
    checkpoint_path : if given, best state_dict is saved here (CPU tensors)
    label           : string identifier for log messages

    Returns
    -------
    history : dict with keys ``train_loss``, ``val_loss``, ``train_acc``,
              ``val_acc`` (all lists), plus scalar ``best_epoch``,
              ``best_val_acc``, and ``total_time``.
    """
    epochs = int(cfg.get("epochs", 20))
    patience = int(cfg.get("patience", 7))

    model = model.to(device)
    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg, epochs)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=patience)

    history: Dict[str, List] = {
        "train_loss": [], "val_loss": [],
        "train_acc": [],  "val_acc": [],
    }
    best_val_acc = -1.0
    best_epoch = 0
    best_state: Optional[Dict] = None

    log.info(
        "Training [%s] on %s — up to %d epochs, patience=%d.",
        label, device, epochs, patience,
    )
    overall = Timer()
    overall.__enter__()

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc = eval_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        is_best = v_acc > best_val_acc
        if is_best:
            best_val_acc = v_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)

        log.info(
            "[%s] epoch %3d/%d | "
            "train loss=%.4f acc=%.4f | "
            "val loss=%.4f acc=%.4f%s",
            label, epoch, epochs,
            t_loss, t_acc,
            v_loss, v_acc,
            " ★ best" if is_best else "",
        )

        if stopper(v_loss, epoch):
            log.info(
                "[%s] Early stopping at epoch %d (patience=%d).",
                label, epoch, patience,
            )
            break

    overall.__exit__(None, None, None)

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        log.info(
            "[%s] Restored best weights from epoch %d (val acc=%.4f).",
            label, best_epoch, best_val_acc,
        )

    history["best_epoch"] = best_epoch
    history["best_val_acc"] = best_val_acc
    history["total_time"] = overall.elapsed
    return history
