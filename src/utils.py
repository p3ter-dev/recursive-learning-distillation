"""
src/utils.py
─────────────────────────────────────────────────────────────────────────────
Reproducibility utilities, configuration loading, timing, and model helpers.

All random seeds are set via set_seed() before any dataset, model, or sampler
is constructed.  Worker processes in DataLoaders are seeded via worker_init_fn.
"""

from __future__ import annotations

import io
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml


# ─── Reproducibility ────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker process (passed as worker_init_fn)."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


# ─── Configuration ──────────────────────────────────────────────────────────

def load_config(path: str = "configs/experiment.yaml") -> Dict[str, Any]:
    """Load and return the YAML experiment configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Logging ────────────────────────────────────────────────────────────────

def setup_logging(name: str = "rld", level: int = logging.INFO) -> logging.Logger:
    """Configure root logger and return a named child logger."""
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    return logging.getLogger(name)


# ─── Directory helpers ───────────────────────────────────────────────────────

def ensure_dirs(cfg: Dict[str, Any]) -> None:
    """Create all output directories listed in cfg['paths'] if absent."""
    for key in ("results_dir", "figures_dir", "checkpoints_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


# ─── Timing ─────────────────────────────────────────────────────────────────

class Timer:
    """Wall-clock timer usable as a context manager.

    Example
    -------
    >>> with Timer() as t:
    ...     do_something()
    >>> print(f"Elapsed: {t.elapsed:.2f}s")
    """

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed: float = time.perf_counter() - self._start

    def __str__(self) -> str:
        return f"{self.elapsed:.2f}s"


# ─── Model helpers ───────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    """Return the serialised size of a model's state_dict in megabytes."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1_048_576  # bytes → MB


def freeze(model: nn.Module) -> nn.Module:
    """Disable gradient computation for all parameters (in-place, returns model)."""
    for p in model.parameters():
        p.requires_grad_(False)
    return model
