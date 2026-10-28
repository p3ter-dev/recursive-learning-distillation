"""
experiments/00_baseline.py
─────────────────────────────────────────────────────────────────────────────
Experiment 0: Supervised Baseline.

Trains a SmallCNN exclusively on the labeled training subset (default: 10 %
of the 54k training pool, i.e. ~5 400 examples) and evaluates it on the
held-out test set.  Also reports a full-supervision reference (p_labeled=1.0)
to establish the ceiling accuracy achievable by the architecture.

Outputs saved
─────────────
  results/metrics.csv              — append row(s) with experiment tag
  results/figures/baseline_curves.png
  results/figures/baseline_cm.png
  results/checkpoints/baseline.pt  — best checkpoint (low-label run)

Usage
─────
  # From project root:
  python -m experiments.00_baseline
  python -m experiments.00_baseline --config configs/experiment.yaml
  python -m experiments.00_baseline --p-labeled 0.20
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Allow running from project root as a module or directly as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data import SplitDataManager, load_raw_datasets, make_loaders, make_test_loader
from src.evaluate import compute_metrics, get_predictions, print_report
from src.models import build_model
from src.plotting import plot_confusion_matrix, plot_training_curves
from src.train import train_model
from src.utils import (
    Timer,
    ensure_dirs,
    load_config,
    set_seed,
    setup_logging,
)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 0 — Supervised Baseline")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--p-labeled", type=float, default=None,
                   help="Override data.p_labeled from config")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override training.epochs")
    p.add_argument("--seed", type=int, default=None,
                   help="Override global seed")
    p.add_argument("--skip-ceiling", action="store_true",
                   help="Skip the full-supervision ceiling run")
    return p.parse_args()


# ─── Results persistence ──────────────────────────────────────────────────────

def append_metrics_csv(metrics: dict, tag: str, csv_path: str) -> None:
    """Append a flat metrics dict as one row in a CSV, adding 'experiment' column."""
    row = {"experiment": tag, **{
        k: v for k, v in metrics.items()
        if not isinstance(v, (list, dict))  # skip non-scalar fields
    }}
    path = Path(csv_path)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.p_labeled is not None:
        cfg["data"]["p_labeled"] = args.p_labeled
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed

    log = setup_logging()
    set_seed(cfg["seed"])
    ensure_dirs(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Experiment 0 — Supervised Baseline (device=%s, seed=%d)", device, cfg["seed"])

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
    lat_n = cfg["evaluation"]["latency_n_samples"]
    lat_wu = cfg["evaluation"]["latency_warmup"]
    figures_dir = cfg["paths"]["figures_dir"]
    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    metrics_csv = str(Path(cfg["paths"]["results_dir"]) / "metrics.csv")

    # ══════════════════════════════════════════════════════════════════════════
    # Run A — Low-label baseline (p_labeled as configured)
    # ══════════════════════════════════════════════════════════════════════════
    log.info(
        "Run A: low-label baseline (p_labeled=%.2f, %d labeled examples)",
        cfg["data"]["p_labeled"], len(split.labeled_indices),
    )
    model_a = build_model(cfg["teacher"], role="baseline").to(device)
    set_seed(cfg["seed"])   # re-seed before training for strict reproducibility

    with Timer() as t_train:
        history_a = train_model(
            model_a,
            loaders["labeled"],
            loaders["val"],
            cfg["training"],
            device,
            checkpoint_path=f"{ckpt_dir}/baseline.pt",
            label="baseline",
        )

    y_true, y_pred, _ = get_predictions(model_a, test_loader, device)
    metrics_a = compute_metrics(
        y_true, y_pred, model_a, test_loader, device,
        latency_n=lat_n, latency_warmup=lat_wu,
    )
    metrics_a["training_time_s"] = t_train.elapsed
    metrics_a["p_labeled"] = cfg["data"]["p_labeled"]
    metrics_a["n_labeled"] = len(split.labeled_indices)
    metrics_a["seed"] = cfg["seed"]

    print_report(metrics_a, f"Baseline (p_labeled={cfg['data']['p_labeled']:.0%})")

    # Plots
    plot_training_curves(
        history_a,
        save_path=f"{figures_dir}/baseline_curves.png",
        title=f"Baseline Training Curves (p_labeled={cfg['data']['p_labeled']:.0%})",
    )
    plot_confusion_matrix(
        metrics_a["confusion_matrix"],
        save_path=f"{figures_dir}/baseline_cm.png",
        title=f"Baseline Confusion Matrix (p_labeled={cfg['data']['p_labeled']:.0%})",
    )
    append_metrics_csv(metrics_a, f"baseline_p{cfg['data']['p_labeled']:.2f}", metrics_csv)

    # ══════════════════════════════════════════════════════════════════════════
    # Run B — Full-supervision ceiling (p_labeled=1.0, all 54k examples)
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_ceiling:
        log.info("Run B: full-supervision ceiling (p_labeled=1.00)")
        cfg_ceil = dict(cfg)
        data_ceil = dict(cfg["data"])
        data_ceil["p_labeled"] = 1.0
        cfg_ceil["data"] = data_ceil

        split_ceil = SplitDataManager(
            train_ds,
            val_fraction=cfg["data"]["val_fraction"],
            p_labeled=1.0,
            seed=cfg["seed"],
        )
        loaders_ceil = make_loaders(
            split_ceil,
            batch_size=cfg["training"]["batch_size"],
            seed=cfg["seed"],
            num_workers=cfg["data"]["num_workers"],
        )
        model_b = build_model(cfg["teacher"], role="baseline-ceiling").to(device)
        set_seed(cfg["seed"])

        with Timer() as t_ceil:
            history_b = train_model(
                model_b,
                loaders_ceil["labeled"],
                loaders_ceil["val"],
                cfg["training"],
                device,
                checkpoint_path=f"{ckpt_dir}/baseline_ceiling.pt",
                label="baseline-ceiling",
            )

        y_true, y_pred, _ = get_predictions(model_b, test_loader, device)
        metrics_b = compute_metrics(
            y_true, y_pred, model_b, test_loader, device,
            latency_n=lat_n, latency_warmup=lat_wu,
        )
        metrics_b["training_time_s"] = t_ceil.elapsed
        metrics_b["p_labeled"] = 1.0
        metrics_b["n_labeled"] = len(split_ceil.labeled_indices)
        metrics_b["seed"] = cfg["seed"]

        print_report(metrics_b, "Baseline — Full Supervision Ceiling (p_labeled=100%)")
        plot_training_curves(
            history_b,
            save_path=f"{figures_dir}/baseline_ceiling_curves.png",
            title="Baseline Ceiling (100% labeled)",
        )
        append_metrics_csv(metrics_b, "baseline_ceiling", metrics_csv)

        log.info(
            "Accuracy gap (ceiling − low-label): %.4f",
            metrics_b["accuracy"] - metrics_a["accuracy"],
        )

    log.info("Experiment 0 complete.  Metrics appended to %s", metrics_csv)


if __name__ == "__main__":
    main()
