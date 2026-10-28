"""
tests/test_data.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for data loading, splitting, and pseudo-label dataset utilities.

A lightweight MockFashionMNIST is used instead of downloading real data so
the test suite runs fully offline and in seconds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset


# ─── Mock dataset ─────────────────────────────────────────────────────────────

class MockFashionMNIST(Dataset):
    """Mimics the FashionMNIST API (data, targets, __len__, __getitem__)."""

    def __init__(self, n: int = 1000, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        # Simulate pre-normalised tensor images (FashionMNIST stores uint8 but
        # after ToTensor + Normalize we get float32 tensors)
        self.data = torch.zeros(n, 1, 28, 28)
        self.targets = torch.tensor(
            rng.integers(0, 10, size=n), dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.data[idx], self.targets[idx]


# ─── SplitDataManager ────────────────────────────────────────────────────────

class TestSplitDataManager:
    def _make_split(self, **kwargs):
        from src.data import SplitDataManager
        ds = MockFashionMNIST(n=1000)
        return SplitDataManager(ds, **kwargs)

    def test_index_counts_sum_to_dataset_size(self):
        split = self._make_split(val_fraction=0.10, p_labeled=0.10, seed=42)
        n = 1000
        total = len(split.labeled_indices) + len(split.unlabeled_indices) + len(split.val_indices)
        assert total == n, f"Expected {n}, got {total}"

    def test_no_index_overlap(self):
        split = self._make_split(val_fraction=0.10, p_labeled=0.10, seed=42)
        labeled = set(split.labeled_indices.tolist())
        unlabeled = set(split.unlabeled_indices.tolist())
        val = set(split.val_indices.tolist())

        assert labeled & unlabeled == set(), "labeled ∩ unlabeled must be empty"
        assert labeled & val == set(), "labeled ∩ val must be empty"
        assert unlabeled & val == set(), "unlabeled ∩ val must be empty"

    def test_val_fraction_respected(self):
        split = self._make_split(val_fraction=0.10, p_labeled=0.10, seed=42)
        # Allow ±5 % tolerance for stratification rounding
        assert abs(len(split.val_indices) - 100) <= 10

    def test_labeled_fraction_respected(self):
        split = self._make_split(val_fraction=0.10, p_labeled=0.10, seed=42)
        n_pool = 1000 - len(split.val_indices)
        expected_labeled = int(n_pool * 0.10)
        assert abs(len(split.labeled_indices) - expected_labeled) <= 15

    def test_withheld_labels_gating_true(self):
        split = self._make_split(val_fraction=0.10, p_labeled=0.10, seed=42)
        labels = split.get_withheld_labels(use_for_diagnostic=True)
        assert labels is not None
        assert len(labels) == len(split.unlabeled_indices)

    def test_withheld_labels_gating_false(self):
        split = self._make_split(val_fraction=0.10, p_labeled=0.10, seed=42)
        labels = split.get_withheld_labels(use_for_diagnostic=False)
        assert labels is None

    def test_withheld_labels_independent_copy(self):
        """Mutating the returned copy must not affect the private store."""
        split = self._make_split(val_fraction=0.10, p_labeled=0.10, seed=42)
        labels_a = split.get_withheld_labels(use_for_diagnostic=True)
        labels_a[:] = -1
        labels_b = split.get_withheld_labels(use_for_diagnostic=True)
        assert (labels_b != -1).all(), "Withheld labels must not be mutated by the caller"

    def test_different_seeds_produce_different_splits(self):
        split_a = self._make_split(val_fraction=0.10, p_labeled=0.10, seed=0)
        split_b = self._make_split(val_fraction=0.10, p_labeled=0.10, seed=99)
        # Extremely unlikely to be identical at N=1000
        assert not np.array_equal(
            np.sort(split_a.labeled_indices),
            np.sort(split_b.labeled_indices),
        ), "Different seeds should (almost certainly) produce different splits"

    def test_same_seed_is_deterministic(self):
        split_a = self._make_split(val_fraction=0.10, p_labeled=0.10, seed=42)
        split_b = self._make_split(val_fraction=0.10, p_labeled=0.10, seed=42)
        np.testing.assert_array_equal(
            np.sort(split_a.labeled_indices),
            np.sort(split_b.labeled_indices),
        )


# ─── PseudoLabeledDataset ────────────────────────────────────────────────────

class TestPseudoLabeledDataset:
    def _make_pseudo_ds(self, n_unlabeled=100, n_accepted=10):
        from src.data import SplitDataManager, PseudoLabeledDataset

        ds = MockFashionMNIST(n=500)
        split = SplitDataManager(ds, val_fraction=0.10, p_labeled=0.10, seed=42)
        unlabeled_subset = split.get_unlabeled_dataset()

        rng = np.random.default_rng(0)
        # Sample accepted LOCAL indices into the unlabeled subset
        n_avail = len(unlabeled_subset)
        accepted = rng.choice(n_avail, size=min(n_accepted, n_avail), replace=False)
        pseudo_labels = rng.integers(0, 10, size=len(accepted))
        return PseudoLabeledDataset(unlabeled_subset, accepted, pseudo_labels)

    def test_length(self):
        pds = self._make_pseudo_ds(n_accepted=10)
        assert len(pds) == 10

    def test_getitem_returns_tensor_and_label(self):
        pds = self._make_pseudo_ds(n_accepted=5)
        img, lbl = pds[0]
        assert isinstance(img, torch.Tensor), f"Expected Tensor, got {type(img)}"
        assert isinstance(lbl, torch.Tensor), f"Expected Tensor label, got {type(lbl)}"

    def test_image_shape(self):
        pds = self._make_pseudo_ds(n_accepted=5)
        img, _ = pds[0]
        assert img.shape == (1, 28, 28), f"Expected (1,28,28), got {img.shape}"

    def test_label_range(self):
        pds = self._make_pseudo_ds(n_accepted=20)
        for i in range(len(pds)):
            _, lbl = pds[i]
            assert 0 <= lbl.item() < 10, f"Label {lbl.item()} out of range [0,9]"
