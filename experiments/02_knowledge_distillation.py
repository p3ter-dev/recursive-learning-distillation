"""
experiments/02_knowledge_distillation.py
─────────────────────────────────────────────────────────────────────────────
Experiment 2: Knowledge Distillation Alone.

Comparisons
───────────
  A. Teacher (SmallCNN) trained on all available labeled data.
  B. Student (TinyCNN) trained directly on ground-truth labels (hard-label).
  C. Student trained via knowledge distillation from the teacher.

Ablation sub-experiment
───────────────────────
If ``--run-ablation`` is passed, sweeps temperature ∈ {1,2,4,6,8} and
α ∈ {0.1,0.3,0.5,0.7,0.9} independently (one axis at a time), selecting the
best config on the validation set.

NOTE: For this experiment, p_labeled=1.0 is used by default (all 54k training
      examples are treated as labeled) to give distillation the fairest
      comparison.  Use --p-labeled to change this.

Outputs saved
─────────────
  results/metrics.csv
  results/figures/distill_*.png
  results/checkpoints/teacher.pt / student_scratch.pt / student_distilled.pt

Usage
─────
  python -m experiments.02_knowledge_distillation
  python -m experiments.02_knowledge_distillation --run-ablation
  python -m experiments.02_knowledge_distillation --temperature 6 --alpha 0.3
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data import SplitDataManager, load_raw_datasets, make_loaders, make_test_loader
from src.distillation import train_distillation
from src.evaluate import compute_metrics, get_predictions, print_report
from src.models import build_model
from src.plotting import (
    plot_ablation,
    plot_comparison,
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_training_curves,
)
from src.train import train_model
from src.utils import Timer, ensure_dirs, load_config, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 2 — Knowledge Distillation")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--p-labeled", type=float, default=None,
                   help="Label fraction (default: 1.0 for Exp 2)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-ablation", action="store_true",
                   help="Sweep temperature and alpha grids")
    return p.parse_args()


def append_metrics_csv(metrics: dict, tag: str, csv_path: str) -> None:
    row = {"experiment": tag, **{
        k: v for k, v in metrics.items()
        if not isinstance(v, (list, dict))
    }}
    path = Path(csv_path)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # For Exp 2, default to full supervision unless overridden
    if args.p_labeled is not None:
        cfg["data"]["p_labeled"] = args.p_labeled
    else:
        cfg["data"]["p_labeled"] = 1.0

    if args.temperature is not None:
        cfg["distillation"]["temperature"] = args.temperature
    if args.alpha is not None:
        cfg["distillation"]["alpha"] = args.alpha
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
        cfg["distillation"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed

    log = setup_logging()
    set_seed(cfg["seed"])
    ensure_dirs(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(
        "Experiment 2 — Knowledge Distillation "
        "(T=%.1f, α=%.2f, p_labeled=%.2f)",
        cfg["distillation"]["temperature"],
        cfg["distillation"]["alpha"],
        cfg["data"]["p_labeled"],
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, test_ds = load_raw_datasets(cfg["paths"]["data_dir"])
    split = SplitDataManager(
        train_ds,
        val_fraction=cfg["data"]["val_fraction"],
        p_labeled=cfg["data"]["p_labeled"],
        seed=cfg["seed"],
    )
    loaders = make_loaders(
        split,
        batch_size=cfg["training"]["batch_size"],
        seed=cfg["seed"],
        num_workers=cfg["data"]["num_workers"],
    )
    test_loader = make_test_loader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    figures_dir = cfg["paths"]["figures_dir"]
    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    metrics_csv = str(Path(cfg["paths"]["results_dir"]) / "metrics.csv")
    lat_n = cfg["evaluation"]["latency_n_samples"]
    lat_wu = cfg["evaluation"]["latency_warmup"]
    val_loader = loaders["val"]
    labeled_loader = loaders["labeled"]

    all_metrics: Dict[str, dict] = {}

    # ══════════════════════════════════════════════════════════════════════════
    # A: Teacher
    # ══════════════════════════════════════════════════════════════════════════
    log.info("A: Training teacher (SmallCNN) on %d labeled examples.", len(split.labeled_indices))
    teacher = build_model(cfg["teacher"], role="teacher").to(device)
    set_seed(cfg["seed"])
    with Timer() as t:
        history_t = train_model(
            teacher, labeled_loader, val_loader, cfg["training"], device,
            checkpoint_path=f"{ckpt_dir}/teacher.pt",
            label="teacher",
        )
    y_true, y_pred, _ = get_predictions(teacher, test_loader, device)
    metrics_t = compute_metrics(y_true, y_pred, teacher, test_loader, device,
                                latency_n=lat_n, latency_warmup=lat_wu)
    metrics_t["training_time_s"] = t.elapsed
    metrics_t["p_labeled"] = cfg["data"]["p_labeled"]
    all_metrics["teacher"] = metrics_t
    print_report(metrics_t, "Teacher (SmallCNN)")
    plot_training_curves(history_t, f"{figures_dir}/distill_teacher_curves.png",
                         "Teacher Training Curves")
    append_metrics_csv(metrics_t, "distill_teacher", metrics_csv)

    # ══════════════════════════════════════════════════════════════════════════
    # B: Hard-label student (scratch)
    # ══════════════════════════════════════════════════════════════════════════
    log.info("B: Training hard-label student (TinyCNN, scratch).")
    student_cfg_train = {**cfg["training"],
                         "epochs": cfg["distillation"].get("epochs", cfg["training"]["epochs"])}
    student_scratch = build_model(cfg["student"], role="student-scratch").to(device)
    set_seed(cfg["seed"])
    with Timer() as t:
        history_s = train_model(
            student_scratch, labeled_loader, val_loader, student_cfg_train, device,
            checkpoint_path=f"{ckpt_dir}/student_scratch.pt",
            label="student-scratch",
        )
    y_true, y_pred, _ = get_predictions(student_scratch, test_loader, device)
    metrics_s = compute_metrics(y_true, y_pred, student_scratch, test_loader, device,
                                latency_n=lat_n, latency_warmup=lat_wu)
    metrics_s["training_time_s"] = t.elapsed
    metrics_s["p_labeled"] = cfg["data"]["p_labeled"]
    all_metrics["student_scratch"] = metrics_s
    print_report(metrics_s, "Student — Hard-label (scratch)")
    plot_training_curves(history_s, f"{figures_dir}/distill_student_scratch_curves.png",
                         "Student (hard-label) Training Curves")
    append_metrics_csv(metrics_s, "distill_student_scratch", metrics_csv)

    # ══════════════════════════════════════════════════════════════════════════
    # C: Distilled student
    # ══════════════════════════════════════════════════════════════════════════
    log.info("C: Training distilled student (T=%.1f, α=%.2f).",
             cfg["distillation"]["temperature"], cfg["distillation"]["alpha"])
    student_dist = build_model(cfg["student"], role="student-distilled").to(device)
    set_seed(cfg["seed"])
    with Timer() as t:
        history_d = train_distillation(
            teacher=copy.deepcopy(teacher),
            student=student_dist,
            train_loader=labeled_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            checkpoint_path=f"{ckpt_dir}/student_distilled.pt",
            label="student-distilled",
        )
    y_true, y_pred, _ = get_predictions(student_dist, test_loader, device)
    metrics_d = compute_metrics(y_true, y_pred, student_dist, test_loader, device,
                                latency_n=lat_n, latency_warmup=lat_wu)
    metrics_d["training_time_s"] = t.elapsed
    metrics_d["temperature"] = cfg["distillation"]["temperature"]
    metrics_d["alpha"] = cfg["distillation"]["alpha"]
    metrics_d["p_labeled"] = cfg["data"]["p_labeled"]
    all_metrics["student_distilled"] = metrics_d
    print_report(metrics_d, f"Student — Distilled (T={cfg['distillation']['temperature']}, α={cfg['distillation']['alpha']})")
    plot_training_curves(history_d, f"{figures_dir}/distill_student_distilled_curves.png",
                         "Distilled Student Training Curves")
    append_metrics_csv(metrics_d, "distill_student_distilled", metrics_csv)

    # ══════════════════════════════════════════════════════════════════════════
    # Comparison plots
    # ══════════════════════════════════════════════════════════════════════════
    plot_comparison(
        all_metrics, metric_key="accuracy",
        save_path=f"{figures_dir}/distill_accuracy_comparison.png",
        title="Distillation — Test Accuracy Comparison",
        ylabel="Test Accuracy",
    )
    plot_per_class_f1(
        list(all_metrics.values()),
        labels=list(all_metrics.keys()),
        save_path=f"{figures_dir}/distill_per_class_f1.png",
        title="Distillation — Per-class F1 Comparison",
    )
    for name, m in all_metrics.items():
        plot_confusion_matrix(
            m["confusion_matrix"],
            save_path=f"{figures_dir}/distill_cm_{name}.png",
            title=f"Confusion Matrix — {name}",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Ablation sweep (optional)
    # ══════════════════════════════════════════════════════════════════════════
    if args.run_ablation:
        log.info("Running ablation sweeps (temperature and alpha).")
        T_grid = cfg["distillation"]["temperature_grid"]
        alpha_grid = cfg["distillation"]["alpha_grid"]

        # Temperature sweep (α fixed at default)
        T_accs: List[float] = []
        for T in T_grid:
            s_abl = build_model(cfg["student"], role=f"abl-T{T}").to(device)
            abl_cfg = copy.deepcopy(cfg)
            abl_cfg["distillation"]["temperature"] = T
            set_seed(cfg["seed"])
            h = train_distillation(
                teacher=copy.deepcopy(teacher),
                student=s_abl,
                train_loader=labeled_loader,
                val_loader=val_loader,
                cfg=abl_cfg,
                device=device,
                label=f"abl-T{T}",
            )
            T_accs.append(h["best_val_acc"])
            log.info("Temperature=%.1f → val acc=%.4f", T, h["best_val_acc"])

        plot_ablation(T_grid, T_accs, "Temperature (T)",
                      f"{figures_dir}/ablation_temperature.png",
                      title="Temperature Ablation (val accuracy)")

        # Alpha sweep (T fixed at default)
        alpha_accs: List[float] = []
        for alpha in alpha_grid:
            s_abl = build_model(cfg["student"], role=f"abl-a{alpha}").to(device)
            abl_cfg = copy.deepcopy(cfg)
            abl_cfg["distillation"]["alpha"] = alpha
            set_seed(cfg["seed"])
            h = train_distillation(
                teacher=copy.deepcopy(teacher),
                student=s_abl,
                train_loader=labeled_loader,
                val_loader=val_loader,
                cfg=abl_cfg,
                device=device,
                label=f"abl-alpha{alpha}",
            )
            alpha_accs.append(h["best_val_acc"])
            log.info("Alpha=%.2f → val acc=%.4f", alpha, h["best_val_acc"])

        plot_ablation(alpha_grid, alpha_accs, "Alpha (α)",
                      f"{figures_dir}/ablation_alpha.png",
                      title="Alpha Ablation (val accuracy)")

        best_T = T_grid[int(torch.tensor(T_accs).argmax())]
        best_alpha = alpha_grid[int(torch.tensor(alpha_accs).argmax())]
        log.info("Ablation best T=%.1f, best α=%.2f", best_T, best_alpha)

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    log.info(
        "Accuracy comparison — teacher: %.4f | student scratch: %.4f | "
        "student distilled: %.4f",
        metrics_t["accuracy"], metrics_s["accuracy"], metrics_d["accuracy"],
    )
    gap = metrics_d["accuracy"] - metrics_s["accuracy"]
    log.info(
        "Distillation gain vs. hard-label student: %+.4f  (%s)",
        gap, "H3 supported ✓" if gap > 0 else "H3 not supported ✗",
    )
    log.info("Experiment 2 complete.  Metrics appended to %s", metrics_csv)


if __name__ == "__main__":
    main()
