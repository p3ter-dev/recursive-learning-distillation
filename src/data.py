"""
src/data.py
─────────────────────────────────────────────────────────────────────────────
Data loading, stratified splitting, and labeled / unlabeled pool management
for Fashion-MNIST.

Key design decisions
────────────────────
1.  Stratified splits are used at every level (train-pool / val, labeled /
    unlabeled) to preserve the 10-class distribution even at low label fractions.

2.  Withheld labels for the unlabeled pool are stored in a private attribute
    and exposed only through ``get_withheld_labels(use_for_diagnostic=True)``.
    Every code path that touches training must call
    ``get_withheld_labels(use_for_diagnostic=False)`` and receive ``None``.

3.  ``PseudoLabeledDataset`` accepts LOCAL indices into an unlabeled Subset so
    that ``__getitem__`` maps correctly through the Subset → base dataset chain.

Fashion-MNIST mean / std (per-channel, pixel values in [0, 1]):
    mean = 0.2860   std = 0.3530
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit

from .utils import worker_init_fn

log = logging.getLogger("rld.data")

# Display names for the 10 Fashion-MNIST classes (index → name)
CLASS_NAMES: List[str] = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# ─── Transforms ──────────────────────────────────────────────────────────────

def _get_transform(augment: bool = False) -> transforms.Compose:
    """
    Return a composed transform for Fashion-MNIST.

    augment=True  adds RandomHorizontalFlip + RandomCrop (training only).
    augment=False returns normalisation-only transform (val, test, pseudo-label
    inference).
    """
    norm = transforms.Normalize((0.2860,), (0.3530,))
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([transforms.ToTensor(), norm])


# ─── Raw dataset loading ─────────────────────────────────────────────────────

def load_raw_datasets(
    data_dir: str = "./data",
) -> Tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
    """
    Download (or load from cache) Fashion-MNIST.

    Parameters
    ----------
    data_dir : local directory; dataset is downloaded if absent.

    Returns
    -------
    train_ds, test_ds  — both use the **non-augmented** transform.
    Augmentation for training is applied later via ``_augmented_copy``.
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    train_ds = datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=_get_transform(augment=False)
    )
    test_ds = datasets.FashionMNIST(
        data_dir, train=False, download=True, transform=_get_transform(augment=False)
    )
    log.info(
        "Fashion-MNIST loaded: %d training, %d test images.",
        len(train_ds), len(test_ds),
    )
    return train_ds, test_ds


def _augmented_copy(ds: datasets.FashionMNIST) -> datasets.FashionMNIST:
    """Return a shallow copy of a FashionMNIST dataset with augmentation enabled."""
    ds_aug = copy.copy(ds)
    ds_aug.transform = _get_transform(augment=True)
    return ds_aug


# ─── Split manager ───────────────────────────────────────────────────────────

class SplitDataManager:
    """
    Manages the three-way stratified data split used across all experiments.

    After construction the following integer-index arrays are available:

    ``val_indices``        — indices into *train_ds* used for validation
    ``labeled_indices``    — indices into *train_ds* used as labeled training data
    ``unlabeled_indices``  — indices into *train_ds* comprising the unlabeled pool

    The true class labels for the unlabeled pool are stored privately and can
    only be retrieved via :meth:`get_withheld_labels`.  This gate must be
    respected by all training code.

    Parameters
    ----------
    train_ds     : full FashionMNIST training dataset (60 000 images)
    val_fraction : fraction of training data held out for validation
    p_labeled    : fraction of the remaining pool treated as labeled
    seed         : master random seed
    """

    def __init__(
        self,
        train_ds: datasets.FashionMNIST,
        val_fraction: float = 0.10,
        p_labeled: float = 0.10,
        seed: int = 42,
    ) -> None:
        self.train_ds = train_ds
        self.val_fraction = val_fraction
        self.p_labeled = p_labeled
        self.seed = seed
        self._split()

    # ── Internal split logic ────────────────────────────────────────────────

    def _split(self) -> None:
        all_targets = np.array(self.train_ds.targets)
        n = len(all_targets)
        indices = np.arange(n)

        # Step 1 — stratified validation split
        sss1 = StratifiedShuffleSplit(
            n_splits=1, test_size=self.val_fraction, random_state=self.seed
        )
        pool_idx, val_idx = next(sss1.split(indices, all_targets))

        # Step 2 — stratified labeled / unlabeled split within the training pool
        pool_targets = all_targets[pool_idx]
        if self.p_labeled >= 1.0:
            labeled_idx = pool_idx
            unlabeled_idx = np.array([], dtype=int)
        elif self.p_labeled <= 0.0:
            labeled_idx = np.array([], dtype=int)
            unlabeled_idx = pool_idx
        else:
            sss2 = StratifiedShuffleSplit(
                n_splits=1,
                test_size=(1.0 - self.p_labeled),
                random_state=self.seed + 1,
            )
            labeled_local, unlabeled_local = next(
                sss2.split(pool_idx, pool_targets)
            )
            labeled_idx = pool_idx[labeled_local]
            unlabeled_idx = pool_idx[unlabeled_local]

        self.val_indices: np.ndarray = val_idx
        self.labeled_indices: np.ndarray = labeled_idx
        self.unlabeled_indices: np.ndarray = unlabeled_idx

        # PRIVATE — true labels for unlabeled pool.
        # ⚠  Never pass this to any training or pseudo-label selection code path.
        self._withheld_labels: np.ndarray = all_targets[unlabeled_idx]

        log.info(
            "Data split (seed=%d, val=%.0f%%, p_labeled=%.0f%%): "
            "%d labeled | %d unlabeled | %d val",
            self.seed,
            self.val_fraction * 100,
            self.p_labeled * 100,
            len(labeled_idx),
            len(unlabeled_idx),
            len(val_idx),
        )
        unique, counts = np.unique(all_targets[labeled_idx], return_counts=True)
        log.info(
            "Labeled class distribution: %s",
            {int(k): int(v) for k, v in zip(unique, counts)},
        )

    # ── Public accessors ────────────────────────────────────────────────────

    def get_withheld_labels(
        self, use_for_diagnostic: bool = True
    ) -> Optional[np.ndarray]:
        """
        Return true labels for the unlabeled pool.

        Parameters
        ----------
        use_for_diagnostic : if ``True``, returns the label array.
                             if ``False``, returns ``None``.

        ⚠ Training code MUST call this with ``use_for_diagnostic=False``.
        Diagnostic / evaluation code may call with ``True``.
        """
        if not use_for_diagnostic:
            return None
        return self._withheld_labels.copy()

    def get_labeled_dataset(self, augment: bool = True) -> Subset:
        """Return the labeled training subset (with optional augmentation)."""
        ds = _augmented_copy(self.train_ds) if augment else self.train_ds
        return Subset(ds, self.labeled_indices)

    def get_unlabeled_dataset(self) -> Subset:
        """
        Return the unlabeled pool as a Subset.

        The ground-truth labels stored in the underlying FashionMNIST tensors
        are never explicitly suppressed here because callers must discard them.
        The SplitDataManager contract, enforced through get_withheld_labels(),
        is the primary safeguard.
        """
        return Subset(self.train_ds, self.unlabeled_indices)

    def get_val_dataset(self) -> Subset:
        """Return the validation subset (no augmentation)."""
        return Subset(self.train_ds, self.val_indices)


# ─── DataLoader factory ──────────────────────────────────────────────────────

def make_loaders(
    split: SplitDataManager,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for the labeled training set, unlabeled pool, and
    validation set.

    The labeled loader shuffles with a seeded Generator.
    The unlabeled loader does NOT shuffle so that per-sample indices in
    ``generate_pseudo_labels`` remain stable.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    labeled_loader = DataLoader(
        split.get_labeled_dataset(augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    unlabeled_loader = DataLoader(
        split.get_unlabeled_dataset(),
        batch_size=batch_size,
        shuffle=False,        # stable ordering required for index mapping
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        split.get_val_dataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return {
        "labeled": labeled_loader,
        "unlabeled": unlabeled_loader,
        "val": val_loader,
    }


def make_test_loader(
    test_ds: Dataset,
    batch_size: int = 64,
    num_workers: int = 0,
) -> DataLoader:
    """Create a non-shuffling DataLoader for the held-out test set."""
    return DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )


# ─── Pseudo-labeled dataset ──────────────────────────────────────────────────

class PseudoLabeledDataset(Dataset):
    """
    Wraps a subset of the unlabeled pool with assigned pseudo-labels.

    Parameters
    ----------
    unlabeled_subset : the Subset returned by ``split.get_unlabeled_dataset()``.
                       Indices are LOCAL (0-based within the Subset).
    accepted_indices : LOCAL indices (into ``unlabeled_subset``) of accepted
                       examples.
    pseudo_labels    : integer array of length ``len(accepted_indices)`` with
                       the assigned class for each accepted example.

    The FashionMNIST dataset's own transform (ToTensor + Normalize) is applied
    when the base Subset is indexed, so images arrive as tensors.
    """

    def __init__(
        self,
        unlabeled_subset: Subset,
        accepted_indices: np.ndarray,
        pseudo_labels: np.ndarray,
    ) -> None:
        self.unlabeled_subset = unlabeled_subset
        self.accepted_indices = accepted_indices
        self.pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.accepted_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Map local PseudoLabeledDataset index → local unlabeled Subset index
        local_idx = int(self.accepted_indices[idx])
        img, _ = self.unlabeled_subset[local_idx]   # true label discarded
        return img, self.pseudo_labels[idx]


def _combined_collate(batch):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([int(item[1]) for item in batch], dtype=torch.long)
    return imgs, labels


def build_combined_loader(
    labeled_ds: Subset,
    pseudo_ds: PseudoLabeledDataset,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """Concatenate labeled and pseudo-labeled sets and return a shuffled loader."""
    combined = ConcatDataset([labeled_ds, pseudo_ds])
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=g,
        collate_fn=_combined_collate,
    )
