"""
experiments/03_combined_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Experiment 3: Combined Recursive Learning and Knowledge Distillation.

Delegates to ``src.combined_pipeline.run_combined_pipeline``, which runs:
  1. Initial teacher (labeled only)
  2. Recursive teacher (labeled + pseudo-labeled)
  3. Hard-label student (scratch)
  4. Student distilled from initial teacher
  5. Student distilled from recursive teacher

Outputs saved
─────────────
  results/metrics.csv
  results/figures/combined_*.png
  results/checkpoints/combined_*.pt

Usage
─────
  python -m experiments.03_combined_pipeline
  python -m experiments.03_combined_pipeline --seed 7
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.combined_pipeline import run_combined_pipeline
from src.data import SplitDataManager, load_raw_datasets
from src.plotting import (
    plot_comparison,
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_self_training_rounds,
)
from src.utils import ensure_dirs, load_config, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 3 — Combined Pipeline")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override training.epochs")
    p.add_argument("--rounds", type=int, default=None,
                   help="Override recursive.max_rounds")
    p.add_argument("--fine-tune-epochs", type=int, default=None,
                   help="Override recursive.fine_tune_epochs")
    p.add_argument("--p-labeled", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
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

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
        cfg["distillation"]["epochs"] = args.epochs
    if args.rounds is not None:
        cfg["recursive"]["max_rounds"] = args.rounds
    if args.fine_tune_epochs is not None:
        cfg["recursive"]["fine_tune_epochs"] = args.fine_tune_epochs
    if args.p_labeled is not None:
        cfg["data"]["p_labeled"] = args.p_labeled
    if args.seed is not None:
        cfg["seed"] = args.seed

    log = setup_logging()
    set_seed(cfg["seed"])
    ensure_dirs(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(
        "Experiment 3 — Combined Pipeline (p_labeled=%.2f, seed=%d)",
        cfg["data"]["p_labeled"], cfg["seed"],
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, test_ds = load_raw_datasets(cfg["paths"]["data_dir"])
    split = SplitDataManager(
        train_ds,
        val_fraction=cfg["data"]["val_fraction"],
        p_labeled=cfg["data"]["p_labeled"],
        seed=cfg["seed"],
    )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    all_metrics = run_combined_pipeline(cfg, split, test_ds, device)

    # ── Save metrics ──────────────────────────────────────────────────────────
    figures_dir = cfg["paths"]["figures_dir"]
    metrics_csv = str(Path(cfg["paths"]["results_dir"]) / "metrics.csv")

    for model_name, m in all_metrics.items():
        append_metrics_csv(m, f"combined_{model_name}", metrics_csv)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_comparison(
        {k: v for k, v in all_metrics.items() if not isinstance(v.get("accuracy"), type(None))},
        metric_key="accuracy",
        save_path=f"{figures_dir}/combined_accuracy_comparison.png",
        title="Combined Pipeline — Test Accuracy Comparison",
        ylabel="Test Accuracy",
    )
    plot_per_class_f1(
        list(all_metrics.values()),
        labels=list(all_metrics.keys()),
        save_path=f"{figures_dir}/combined_per_class_f1.png",
        title="Combined Pipeline — Per-class F1",
    )
    for name, m in all_metrics.items():
        if "confusion_matrix" in m:
            plot_confusion_matrix(
                m["confusion_matrix"],
                save_path=f"{figures_dir}/combined_cm_{name}.png",
                title=f"Confusion Matrix — {name}",
            )

    # Self-training diagnostic plot (if recursive results available)
    if "teacher_recursive" in all_metrics:
        rec_rounds = all_metrics["teacher_recursive"].get("recursive_round_results", [])
        if rec_rounds:
            plot_self_training_rounds(
                rec_rounds,
                save_path=f"{figures_dir}/combined_recursive_diagnostics.png",
                title="Combined Pipeline — Recursive Teacher Diagnostics",
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Final accuracy comparison:")
    for name, m in all_metrics.items():
        log.info("  %-35s %.4f", name, m.get("accuracy", float("nan")))
    log.info("=" * 60)

    # H5: Does recursive teacher help the distilled student?
    acc_dist_init = all_metrics.get("student_distill_initial", {}).get("accuracy", None)
    acc_dist_rec = all_metrics.get("student_distill_recursive", {}).get("accuracy", None)
    if acc_dist_init is not None and acc_dist_rec is not None:
        delta = acc_dist_rec - acc_dist_init
        verdict = "H5 supported ✓" if delta > 0 else "H5 not supported ✗"
        log.info(
            "H5 (distill-rec > distill-init): %+.4f → %s", delta, verdict
        )

    # H6: If recursive teacher ≤ initial teacher, combined pipeline should not win
    acc_t_init = all_metrics.get("teacher_initial", {}).get("accuracy", None)
    acc_t_rec = all_metrics.get("teacher_recursive", {}).get("accuracy", None)
    if acc_t_init is not None and acc_t_rec is not None:
        teacher_delta = acc_t_rec - acc_t_init
        log.info(
            "Recursive teacher gain vs. initial teacher: %+.4f  (%s)",
            teacher_delta,
            "teacher improved" if teacher_delta > 0 else "teacher did NOT improve (H6 test active)",
        )

    log.info("Experiment 3 complete.  Metrics appended to %s", metrics_csv)


if __name__ == "__main__":
    main()
