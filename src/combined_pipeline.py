"""
src/combined_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Combined recursive learning + knowledge distillation pipeline (Experiment 3).

Pipeline steps
──────────────
1. Train initial teacher on labeled data.
2. Apply iterative self-training to produce an improved teacher.
3. Select the best teacher via validation-based model selection.
4. Freeze the selected teacher.
5. Train a hard-label student baseline (for fair comparison).
6. Distill from the original teacher.
7. Distill from the recursive teacher.
8. Evaluate every model on the test set.

All models are evaluated on the same test set, which is never touched during
any training, self-training, threshold-tuning, or model-selection step.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import SplitDataManager, make_loaders, make_test_loader
from .distillation import train_distillation
from .evaluate import compute_metrics, get_predictions, print_report
from .models import build_model
from .recursive_learning import run_self_training
from .train import train_model
from .utils import Timer

log = logging.getLogger("rld.pipeline")


def run_combined_pipeline(
    cfg: dict,
    split: SplitDataManager,
    test_ds,
    device: torch.device,
) -> Dict[str, Dict]:
    """
    Execute the full combined pipeline and return metrics for all models.

    Parameters
    ----------
    cfg      : full experiment config dict
    split    : SplitDataManager with labeled/unlabeled/val split
    test_ds  : raw test FashionMNIST dataset (never used for selection)
    device   : torch device

    Returns
    -------
    dict mapping model_name → metrics dict (from compute_metrics).
    """
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["data"]["num_workers"])
    seed = int(cfg["seed"])
    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    lat_n = int(cfg["evaluation"]["latency_n_samples"])
    lat_wu = int(cfg["evaluation"]["latency_warmup"])

    loaders = make_loaders(split, batch_size=batch_size, seed=seed, num_workers=num_workers)
    val_loader = loaders["val"]
    labeled_loader = loaders["labeled"]
    test_loader = make_test_loader(test_ds, batch_size=batch_size, num_workers=num_workers)

    dist_cfg = cfg["distillation"]
    student_train_cfg = {
        **cfg["training"],
        "epochs": int(dist_cfg.get("epochs", cfg["training"]["epochs"])),
    }

    all_metrics: Dict[str, Dict] = {}

    # ── 1. Initial teacher (labeled data only) ────────────────────────────────
    log.info("━━━ Step 1: Initial teacher (labeled only) ━━━")
    teacher_init = build_model(cfg["teacher"], role="teacher-initial").to(device)
    with Timer() as t:
        train_model(
            teacher_init, labeled_loader, val_loader, cfg["training"], device,
            checkpoint_path=f"{ckpt_dir}/combined_teacher_initial.pt",
            label="combined/teacher-initial",
        )
    y_true, y_pred, _ = get_predictions(teacher_init, test_loader, device)
    m = compute_metrics(y_true, y_pred, teacher_init, test_loader, device,
                        latency_n=lat_n, latency_warmup=lat_wu)
    m["training_time_s"] = t.elapsed
    all_metrics["teacher_initial"] = m
    print_report(m, "Teacher — initial (labeled only)")

    # ── 2. Recursive teacher (labeled + pseudo-labeled) ───────────────────────
    log.info("━━━ Step 2: Recursive self-training of teacher ━━━")
    rec_results = run_self_training(
        model=copy.deepcopy(teacher_init),
        split=split,
        cfg=cfg,
        device=device,
        val_loader=val_loader,
        checkpoint_dir=ckpt_dir,
        use_diagnostic_labels=bool(
            cfg["recursive"].get("use_true_labels_for_diagnostic", True)
        ),
    )
    teacher_rec = build_model(cfg["teacher"], role="teacher-recursive").to(device)
    teacher_rec.load_state_dict(rec_results["best_model_state"])
    y_true, y_pred, _ = get_predictions(teacher_rec, test_loader, device)
    m = compute_metrics(y_true, y_pred, teacher_rec, test_loader, device,
                        latency_n=lat_n, latency_warmup=lat_wu)
    m["recursive_round_results"] = rec_results["round_results"]
    m["recursive_best_val_acc"] = rec_results["best_val_acc"]
    all_metrics["teacher_recursive"] = m
    print_report(m, "Teacher — recursive self-training")

    # ── 3. Hard-label student (scratch baseline) ──────────────────────────────
    log.info("━━━ Step 3: Hard-label student (scratch) ━━━")
    student_scratch = build_model(cfg["student"], role="student-scratch").to(device)
    with Timer() as t:
        train_model(
            student_scratch, labeled_loader, val_loader, student_train_cfg, device,
            checkpoint_path=f"{ckpt_dir}/combined_student_scratch.pt",
            label="combined/student-scratch",
        )
    y_true, y_pred, _ = get_predictions(student_scratch, test_loader, device)
    m = compute_metrics(y_true, y_pred, student_scratch, test_loader, device,
                        latency_n=lat_n, latency_warmup=lat_wu)
    m["training_time_s"] = t.elapsed
    all_metrics["student_scratch"] = m
    print_report(m, "Student — hard-label (scratch)")

    # ── 4. Student distilled from initial teacher ─────────────────────────────
    log.info("━━━ Step 4: Distill from initial teacher ━━━")
    student_dist_init = build_model(cfg["student"], role="student-distill-init").to(device)
    with Timer() as t:
        train_distillation(
            teacher=copy.deepcopy(teacher_init),
            student=student_dist_init,
            train_loader=labeled_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            checkpoint_path=f"{ckpt_dir}/combined_student_distill_initial.pt",
            label="combined/student-distill-initial",
        )
    y_true, y_pred, _ = get_predictions(student_dist_init, test_loader, device)
    m = compute_metrics(y_true, y_pred, student_dist_init, test_loader, device,
                        latency_n=lat_n, latency_warmup=lat_wu)
    m["training_time_s"] = t.elapsed
    all_metrics["student_distill_initial"] = m
    print_report(m, "Student — distilled from initial teacher")

    # ── 5. Student distilled from recursive teacher ───────────────────────────
    log.info("━━━ Step 5: Distill from recursive teacher ━━━")
    student_dist_rec = build_model(cfg["student"], role="student-distill-rec").to(device)
    with Timer() as t:
        train_distillation(
            teacher=copy.deepcopy(teacher_rec),
            student=student_dist_rec,
            train_loader=labeled_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            checkpoint_path=f"{ckpt_dir}/combined_student_distill_recursive.pt",
            label="combined/student-distill-recursive",
        )
    y_true, y_pred, _ = get_predictions(student_dist_rec, test_loader, device)
    m = compute_metrics(y_true, y_pred, student_dist_rec, test_loader, device,
                        latency_n=lat_n, latency_warmup=lat_wu)
    m["training_time_s"] = t.elapsed
    all_metrics["student_distill_recursive"] = m
    print_report(m, "Student — distilled from recursive teacher")

    return all_metrics
