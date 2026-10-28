"""
src/distillation.py
─────────────────────────────────────────────────────────────────────────────
Knowledge distillation: loss implementation and training loop.

Loss function (Hinton et al., 2015)
─────────────────────────────────────
    L = α · L_CE(student_logits, y_true)
      + (1 − α) · T² · KL( softmax(teacher_logits / T)
                            ‖ log_softmax(student_logits / T) )

where:
  L_CE  — cross-entropy of student logits against hard ground-truth labels
          (computed at T=1; sharpens the student's hard-label predictions)
  L_KD  — KL divergence between teacher soft targets and student log-probs,
          both computed at temperature T
  T²    — compensates for the 1/T² factor that appears in the gradients when
          softmax inputs are scaled by 1/T, so the KL term contributes at the
          same effective magnitude regardless of temperature
  α     — interpolation weight in [0, 1]; α=1 → pure hard-label CE,
          α=0 → pure soft-label distillation

Teacher is unconditionally frozen before the training loop begins.
No teacher gradient should ever be computed; this is verified in the unit tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .train import EarlyStopping, _build_optimizer, _build_scheduler, eval_epoch
from .utils import Timer, freeze

log = logging.getLogger("rld.distillation")


# ─── Loss module ─────────────────────────────────────────────────────────────

class DistillationLoss(nn.Module):
    """
    Combined hard-label cross-entropy + soft-label KL divergence loss.

    Parameters
    ----------
    temperature : T ≥ 1.  Divides logits before softmax/log-softmax.
                  Higher T → softer (more uniform) teacher distribution.
    alpha       : weight on L_CE ∈ [0, 1].
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}.")
        self.T = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Compute the combined distillation loss.

        Parameters
        ----------
        student_logits : (B, C) raw logits from student
        teacher_logits : (B, C) raw logits from teacher — must have no grad
        labels         : (B,) integer class labels

        Returns
        -------
        total_loss     : scalar tensor (with grad for student parameters)
        ce_loss_item   : float value of L_CE (for logging)
        kl_loss_item   : float value of L_KD (for logging, before T² scaling)
        """
        # Hard-label term (full temperature; T cancels for CE)
        l_ce = F.cross_entropy(student_logits, labels)

        # Soft-label term at temperature T
        # teacher_logits.detach() is a safety measure on top of requires_grad=False
        p_teacher = F.softmax(teacher_logits.detach() / self.T, dim=1)
        log_p_student = F.log_softmax(student_logits / self.T, dim=1)
        # KL(P_teacher ‖ P_student) = Σ P_teacher · (log P_teacher − log P_student)
        l_kl = F.kl_div(log_p_student, p_teacher, reduction="batchmean")

        total = self.alpha * l_ce + (1.0 - self.alpha) * (self.T ** 2) * l_kl
        return total, l_ce.item(), l_kl.item()

    def extra_repr(self) -> str:  # shown in print(model)
        return f"temperature={self.T}, alpha={self.alpha}"


# ─── Distillation training loop ───────────────────────────────────────────────

def train_distillation(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    label: str = "student-distilled",
) -> Dict:
    """
    Train a student model via knowledge distillation from a frozen teacher.

    Parameters
    ----------
    teacher         : fully trained teacher (frozen before training begins)
    student         : student model to train (modified in-place; best weights
                      restored before returning)
    train_loader    : labeled training DataLoader (images, hard labels)
    val_loader      : validation DataLoader
    cfg             : full experiment config dict
    device          : torch device
    checkpoint_path : path to save best student state_dict

    Returns
    -------
    history dict with loss/accuracy curves, timing, temperature, and alpha.
    """
    dist_cfg = cfg["distillation"]
    epochs = int(dist_cfg.get("epochs", cfg["training"]["epochs"]))
    patience = int(cfg["training"].get("patience", 7))
    temperature = float(dist_cfg.get("temperature", 4.0))
    alpha = float(dist_cfg.get("alpha", 0.5))

    # Merge distillation overrides into a training sub-config
    train_cfg = {
        **cfg["training"],
        "epochs": epochs,
        "learning_rate": float(dist_cfg.get("learning_rate", cfg["training"]["learning_rate"])),
        "weight_decay": float(dist_cfg.get("weight_decay", cfg["training"]["weight_decay"])),
    }

    # ── Freeze teacher ──────────────────────────────────────────────────────
    teacher = teacher.to(device)
    freeze(teacher)           # sets requires_grad=False on all params
    teacher.eval()

    student = student.to(device)
    optimizer = _build_optimizer(student, train_cfg)
    scheduler = _build_scheduler(optimizer, train_cfg, epochs)
    dist_criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    ce_criterion = nn.CrossEntropyLoss()   # for val evaluation only
    stopper = EarlyStopping(patience=patience)

    history: Dict[str, List] = {
        "train_loss": [], "train_ce_loss": [], "train_kl_loss": [],
        "train_acc": [], "val_loss": [], "val_acc": [],
    }
    best_val_acc = -1.0
    best_epoch = 0
    best_state: Optional[Dict] = None

    log.info(
        "Distillation [%s]: T=%.1f, α=%.2f, %d epochs on %s.",
        label, temperature, alpha, epochs, device,
    )
    total_timer = Timer()
    total_timer.__enter__()

    for epoch in range(1, epochs + 1):
        student.train()
        tot_loss = ce_sum = kl_sum = correct = n = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # Teacher inference — no grad tracked
            with torch.no_grad():
                t_logits = teacher(imgs)

            optimizer.zero_grad()
            s_logits = student(imgs)
            loss, ce_val, kl_val = dist_criterion(s_logits, t_logits, labels)
            loss.backward()
            optimizer.step()

            b = len(labels)
            tot_loss += loss.item() * b
            ce_sum += ce_val * b
            kl_sum += kl_val * b
            correct += (s_logits.argmax(1) == labels).sum().item()
            n += b

        if scheduler is not None:
            scheduler.step()

        t_loss = tot_loss / n
        t_acc = correct / n
        v_loss, v_acc = eval_epoch(student, val_loader, ce_criterion, device)

        history["train_loss"].append(t_loss)
        history["train_ce_loss"].append(ce_sum / n)
        history["train_kl_loss"].append(kl_sum / n)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        is_best = v_acc > best_val_acc
        if is_best:
            best_val_acc = v_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            if checkpoint_path:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)

        log.info(
            "[%s] epoch %3d/%d | "
            "loss=%.4f (ce=%.4f kl=%.4f) acc=%.4f | "
            "val loss=%.4f acc=%.4f%s",
            label, epoch, epochs,
            t_loss, ce_sum / n, kl_sum / n, t_acc,
            v_loss, v_acc,
            " ★ best" if is_best else "",
        )

        if stopper(v_loss, epoch):
            log.info("[%s] Early stopping at epoch %d.", label, epoch)
            break

    total_timer.__exit__(None, None, None)

    if best_state is not None:
        student.load_state_dict(best_state)
        log.info(
            "[%s] Restored best weights from epoch %d (val acc=%.4f).",
            label, best_epoch, best_val_acc,
        )

    history["best_epoch"] = best_epoch
    history["best_val_acc"] = best_val_acc
    history["total_time"] = total_timer.elapsed
    history["temperature"] = temperature
    history["alpha"] = alpha
    return history
