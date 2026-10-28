"""
experiments/01_recursive_learning.py
─────────────────────────────────────────────────────────────────────────────
Experiment 1: Recursive / Iterative Self-training Alone.

Procedure
─────────
1. Train an initial classifier on the labeled subset.
2. Run iterative self-training (configurable rounds, confidence threshold).
3. Select the best model via validation accuracy.
4. Evaluate on the test set.
5. Compare against the supervised baseline trained on identical data.

Diagnostic information
──────────────────────
When ``recursive.use_true_labels_for_diagnostic=true``, pseudo-label accuracy
(true accuracy of assigned pseudo-labels) is computed each round and logged
as a diagnostic metric.  These labels are NEVER passed to any training code.

Outputs saved
─────────────
  results/metrics.csv                    — appended rows
  results/figures/recursive_curves.png   — training curves for initial model
  results/figures/recursive_diagnostics.png — round-by-round diagnostics
  results/figures/recursive_cm.png       — confusion matrix of best model
  results/checkpoints/recursive_initial.pt
  results/checkpoints/recursive_roundNN.pt  (one per round)

Usage
─────
  python -m experiments.01_recursive_learning
  python -m experiments.01_recursive_learning --threshold 0.90 --rounds 3
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data import SplitDataManager, load_raw_datasets, make_loaders, make_test_loader
from src.evaluate import compute_metrics, get_predictions, print_report
from src.models import build_model
from src.plotting import (
    plot_confusion_matrix,
    plot_self_training_rounds,
    plot_training_curves,
)
from src.recursive_learning import run_self_training
from src.train import train_model
from src.utils import Timer, ensure_dirs, load_config, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 1 — Recursive Learning")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--threshold", type=float, default=None,
                   help="Override recursive.confidence_threshold")
    p.add_argument("--rounds", type=int, default=None,
                   help="Override recursive.max_rounds")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override training.epochs for initial model")
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

    if args.threshold is not None:
        cfg["recursive"]["confidence_threshold"] = args.threshold
    if args.rounds is not None:
        cfg["recursive"]["max_rounds"] = args.rounds
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
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
        "Experiment 1 — Recursive Learning "
        "(threshold=%.2f, max_rounds=%d, p_labeled=%.2f)",
        cfg["recursive"]["confidence_threshold"],
        cfg["recursive"]["max_rounds"],
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

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1: Initial model (supervised baseline for this experiment)
    # ══════════════════════════════════════════════════════════════════════════
    log.info("Step 1: Training initial model on labeled data (%d examples).",
             len(split.labeled_indices))
    initial_model = build_model(cfg["teacher"], role="recursive-initial").to(device)
    set_seed(cfg["seed"])

    with Timer() as t_init:
        history_init = train_model(
            initial_model,
            loaders["labeled"],
            loaders["val"],
            cfg["training"],
            device,
            checkpoint_path=f"{ckpt_dir}/recursive_initial.pt",
            label="recursive-initial",
        )

    y_true, y_pred, _ = get_predictions(initial_model, test_loader, device)
    metrics_init = compute_metrics(
        y_true, y_pred, initial_model, test_loader, device,
        latency_n=lat_n, latency_warmup=lat_wu,
    )
    metrics_init["training_time_s"] = t_init.elapsed
    metrics_init["p_labeled"] = cfg["data"]["p_labeled"]
    metrics_init["seed"] = cfg["seed"]
    print_report(metrics_init, "Initial Model (before self-training)")
    plot_training_curves(
        history_init,
        save_path=f"{figures_dir}/recursive_initial_curves.png",
        title="Recursive Learning — Initial Model Training",
    )
    append_metrics_csv(metrics_init, "recursive_initial", metrics_csv)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2: Iterative self-training
    # ══════════════════════════════════════════════════════════════════════════
    log.info("Step 2: Iterative self-training.")
    import copy
    rec_results = run_self_training(
        model=copy.deepcopy(initial_model),
        split=split,
        cfg=cfg,
        device=device,
        val_loader=loaders["val"],
        checkpoint_dir=ckpt_dir,
        use_diagnostic_labels=bool(
            cfg["recursive"].get("use_true_labels_for_diagnostic", True)
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3: Evaluate best self-trained model on test set
    # ══════════════════════════════════════════════════════════════════════════
    log.info("Step 3: Evaluating best self-trained model on test set.")
    best_model = build_model(cfg["teacher"], role="recursive-best").to(device)
    best_model.load_state_dict(rec_results["best_model_state"])

    y_true, y_pred, _ = get_predictions(best_model, test_loader, device)
    metrics_best = compute_metrics(
        y_true, y_pred, best_model, test_loader, device,
        latency_n=lat_n, latency_warmup=lat_wu,
    )
    metrics_best["recursive_best_val_acc"] = rec_results["best_val_acc"]
    metrics_best["n_rounds_completed"] = len(rec_results["round_results"])
    metrics_best["p_labeled"] = cfg["data"]["p_labeled"]
    metrics_best["seed"] = cfg["seed"]
    metrics_best["confidence_threshold"] = cfg["recursive"]["confidence_threshold"]

    print_report(metrics_best, "Best Self-trained Model (test set)")

    plot_confusion_matrix(
        metrics_best["confusion_matrix"],
        save_path=f"{figures_dir}/recursive_cm.png",
        title="Recursive Learning — Best Model Confusion Matrix",
    )
    if rec_results["round_results"]:
        plot_self_training_rounds(
            rec_results["round_results"],
            save_path=f"{figures_dir}/recursive_diagnostics.png",
            title="Self-training Round Diagnostics",
        )
    append_metrics_csv(metrics_best, "recursive_best", metrics_csv)

    # ══════════════════════════════════════════════════════════════════════════
    # Summary comparison
    # ══════════════════════════════════════════════════════════════════════════
    delta_acc = metrics_best["accuracy"] - metrics_init["accuracy"]
    log.info(
        "Accuracy change vs initial model: %+.4f  (%s)",
        delta_acc,
        "improvement ↑" if delta_acc > 0 else "regression ↓" if delta_acc < 0 else "no change",
    )
    log.info(
        "Round-by-round val accuracy: %s",
        [f"{r['val_acc']:.4f}" for r in rec_results["round_results"]],
    )

    log.info("Experiment 1 complete.  Metrics appended to %s", metrics_csv)


if __name__ == "__main__":
    main()
