# Recursive Learning and Knowledge Distillation for Efficient Image Classification

A reproducible ML research project on Fashion-MNIST investigating:

1. **Recursive learning** — iterative self-training with confidence-filtered pseudo-labels
2. **Knowledge distillation** — training a small student from a larger teacher's soft outputs
3. **Combined pipeline** — recursive teacher improvement followed by distillation

All conclusions are supported by measured results. Negative outcomes are reported with equal rigour.

---

## Repository Structure

```
recursive-learning-distillation/
├── configs/experiment.yaml        # All hyperparameters (single source of truth)
├── src/
│   ├── utils.py                   # Seeds, config loading, timing
│   ├── models.py                  # SmallCNN (teacher) & TinyCNN (student)
│   ├── data.py                    # FashionMNIST loading, stratified splits
│   ├── train.py                   # Training loop, early stopping, checkpointing
│   ├── evaluate.py                # Metrics, latency measurement
│   ├── plotting.py                # Training curves, confusion matrices, comparisons
│   ├── recursive_learning.py      # Iterative self-training
│   ├── distillation.py            # DistillationLoss + training loop
│   └── combined_pipeline.py      # Full combined experiment
├── notebooks/
│   ├── recursive_learning_and_distillation_colab.ipynb  # End-to-end interactive Colab notebook
│   └── README.md                  # Colab execution guide & fast demo tips
├── experiments/
│   ├── 00_baseline.py             # Experiment 0: supervised baseline
│   ├── 01_recursive_learning.py   # Experiment 1: self-training alone
│   ├── 02_knowledge_distillation.py  # Experiment 2: distillation alone
│   └── 03_combined_pipeline.py   # Experiment 3: combined
├── tests/
│   ├── test_data.py               # Data split & pseudo-label dataset tests
│   ├── test_models.py             # Architecture shape & parameter count tests
│   └── test_losses.py             # Distillation loss correctness tests
├── results/
│   ├── metrics.csv                # Appended after each experiment run
│   ├── figures/                   # Auto-generated PNGs
│   └── checkpoints/               # Best model state_dicts
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🚀 Run in Google Colab

You can run the entire experiment suite, generate interactive plots, and benchmark models directly in **Google Colab** with zero local setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

1. Open `notebooks/recursive_learning_and_distillation_colab.ipynb` in Colab.
2. Under **Runtime $\rightarrow$ Change runtime type**, select **T4 GPU** (or standard CPU).
3. Toggle between `FAST_DEMO = True` (interactive 3-minute run with 5 epochs & 2 self-training rounds) and `FAST_DEMO = False` (full publication-grade run).
4. Run all cells (`Ctrl+F9`). All metrics, reliability diagrams, confusion matrices, and Pareto frontier plots are visualized directly inline.

---

---

## Quick Start

### 1. Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install
pip install -e .
# or: pip install -r requirements.txt
```

> **CPU only.** No CUDA or GPU required. All experiments are designed for
> a standard Linux CPU machine.

### 2. Verify tests pass

```bash
pytest tests/ -v
```

All tests run offline using mock datasets. Expected output: ≥ 20 tests passed.

### 3. Run experiments (in order)

```bash
# Experiment 0 — Supervised baseline
python -m experiments.00_baseline

# Experiment 1 — Recursive / self-training
python -m experiments.01_recursive_learning

# Experiment 2 — Knowledge distillation
python -m experiments.02_knowledge_distillation

# Optional: distillation ablation sweep (temperature & alpha)
python -m experiments.02_knowledge_distillation --run-ablation

# Experiment 3 — Combined pipeline
python -m experiments.03_combined_pipeline
```

Results are appended to `results/metrics.csv`. Plots are saved to
`results/figures/`.

---

## Key Configuration (`configs/experiment.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `seed` | 42 | Global random seed |
| `data.p_labeled` | 0.10 | Fraction of training pool treated as labeled |
| `data.val_fraction` | 0.10 | Fraction held out for validation |
| `training.epochs` | 20 | Max training epochs |
| `training.patience` | 7 | Early-stopping patience |
| `recursive.max_rounds` | 5 | Self-training rounds |
| `recursive.confidence_threshold` | 0.95 | Min confidence to accept pseudo-label |
| `recursive.fine_tune_epochs` | 5 | Epochs per self-training round |
| `distillation.temperature` | 4.0 | Distillation temperature T |
| `distillation.alpha` | 0.5 | CE weight (1−α = KL weight) |

All values can be overridden via CLI flags. See `python -m experiments.XX --help`.

---

## Models

| Model | Role | Architecture | ~Parameters |
|-------|------|-------------|-------------|
| `SmallCNN` | Teacher | Conv(32)→Conv(32)→FC(128)→FC(10) | ~212K |
| `TinyCNN` | Student | Conv(8)→FC(64)→FC(10) | ~26K |

Both architectures run comfortably on CPU; typical epoch time on a 4-core
machine is **< 30 s** for the labeled subset (5 400 examples).

---

## Data Protocol

```
60 000 training images (Fashion-MNIST)
├── 6 000  → validation  (stratified, fixed, seed=42)
└── 54 000 → training pool
    ├── ~5 400  → labeled   (p_labeled=0.10)
    └── ~48 600 → unlabeled (true labels withheld)

10 000 test images → evaluated ONCE per model, AFTER all decisions frozen
```

**Data leakage safeguards:**
- Withheld labels for the unlabeled pool are stored behind a boolean gate
  (`use_for_diagnostic`). Any call with `use_for_diagnostic=False` returns
  `None`. Training code must always pass `False`.
- The test set is never loaded during training, pseudo-label generation,
  threshold selection, or model selection.
- The confidence threshold is tuned on the validation set only.

---

## Distillation Loss

```
L = α · CrossEntropy(student_logits, y)
  + (1 − α) · T² · KL( softmax(teacher_logits / T) ‖ log_softmax(student_logits / T) )
```

The T² factor compensates for the 1/T² reduction in gradient magnitude that
arises from temperature-scaling the logits before softmax.

The teacher is frozen (`.requires_grad_(False)`) before student training begins.

---

## Reproducing Results with a Different Seed

```bash
python -m experiments.00_baseline --seed 7
python -m experiments.01_recursive_learning --seed 7
python -m experiments.02_knowledge_distillation --seed 7
python -m experiments.03_combined_pipeline --seed 7
```

---

## Using an Existing Local Copy of Fashion-MNIST

If the dataset is already present in `./data/FashionMNIST/`, `torchvision`
will detect it automatically and skip the download. No additional
configuration is required.

---

## Research Questions

| # | Question |
|---|---------|
| RQ1 | Does iterative self-training improve accuracy at low label fractions? |
| RQ2 | Does knowledge distillation let a small student outperform a scratch-trained student? |
| RQ3 | Does distilling from a recursively-improved teacher yield a better student? |

---

## Hypotheses

| ID | Hypothesis | Falsifiable? |
|----|-----------|-------------|
| H1 | Recursive self-training improves accuracy when ≤50% of labels are available | Yes |
| H2 | Gains diminish / reverse after 2–4 rounds (confirmation bias) | Yes |
| H3 | Distilled student outperforms equally-sized scratch-trained student | Yes |
| H4 | Temperature T ∈ [3,6] outperforms T=1 for distillation on Fashion-MNIST | Yes |
| H5 | Distilling from a recursively-improved teacher further improves the student | Yes |
| H6 | If recursive learning fails the teacher, the combined pipeline also fails | Yes |

---

## License

MIT
