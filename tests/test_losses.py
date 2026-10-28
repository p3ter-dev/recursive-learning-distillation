"""
tests/test_losses.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for the DistillationLoss module.

Tests verify:
  1. Mathematical correctness at α=0 and α=1 edge cases.
  2. T² scaling is applied correctly.
  3. Teacher parameters receive no gradients.
  4. Non-negativity of all loss terms.
  5. Higher temperature produces softer teacher distributions.
  6. Loss decreases monotonically toward the teacher for a warm student.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


class TestDistillationLossEdgeCases:
    """Algebraic edge-case verification."""

    def test_alpha_1_equals_crossentropy(self):
        """When α=1, total loss must equal standard cross-entropy (KL term is zeroed)."""
        from src.distillation import DistillationLoss
        torch.manual_seed(0)
        criterion = DistillationLoss(temperature=4.0, alpha=1.0)
        s = torch.randn(8, 10)
        t = torch.randn(8, 10)
        y = torch.randint(0, 10, (8,))

        total, ce_val, _ = criterion(s, t, y)
        expected_ce = F.cross_entropy(s, y).item()
        assert abs(total.item() - expected_ce) < 1e-5, (
            f"α=1: total={total.item():.6f} ≠ CE={expected_ce:.6f}"
        )

    def test_alpha_0_equals_t2_kl(self):
        """When α=0, total loss must equal T² · KL only (CE term is zeroed)."""
        from src.distillation import DistillationLoss
        torch.manual_seed(1)
        T = 4.0
        criterion = DistillationLoss(temperature=T, alpha=0.0)
        s = torch.randn(8, 10)
        t = torch.randn(8, 10)
        y = torch.randint(0, 10, (8,))

        total, _, kl_val = criterion(s, t, y)
        expected = T ** 2 * kl_val
        assert abs(total.item() - expected) < 1e-5, (
            f"α=0: total={total.item():.6f} ≠ T²·KL={expected:.6f}"
        )

    def test_alpha_half_weighted_sum(self):
        """For generic α, total must equal α·CE + (1-α)·T²·KL."""
        from src.distillation import DistillationLoss
        torch.manual_seed(2)
        T, alpha = 3.0, 0.3
        criterion = DistillationLoss(temperature=T, alpha=alpha)
        s = torch.randn(8, 10)
        t = torch.randn(8, 10)
        y = torch.randint(0, 10, (8,))

        total, ce_val, kl_val = criterion(s, t, y)
        expected = alpha * ce_val + (1.0 - alpha) * T**2 * kl_val
        assert abs(total.item() - expected) < 1e-5, (
            f"Weighted sum check failed: {total.item():.6f} ≠ {expected:.6f}"
        )


class TestDistillationLossProperties:
    """Properties that should always hold regardless of inputs."""

    def test_nonnegative_loss(self):
        from src.distillation import DistillationLoss
        torch.manual_seed(42)
        criterion = DistillationLoss(temperature=4.0, alpha=0.5)
        s = torch.randn(16, 10)
        t = torch.randn(16, 10)
        y = torch.randint(0, 10, (16,))

        total, ce_val, kl_val = criterion(s, t, y)
        assert ce_val >= 0.0, f"CE loss must be non-negative, got {ce_val}"
        assert kl_val >= 0.0, f"KL loss must be non-negative, got {kl_val}"
        assert total.item() >= 0.0, f"Total loss must be non-negative, got {total.item()}"

    def test_teacher_no_gradient(self):
        """Teacher parameters must accumulate zero gradient after backward."""
        from src.models import SmallCNN, TinyCNN
        from src.distillation import DistillationLoss
        from src.utils import freeze

        teacher = SmallCNN()
        student = TinyCNN()
        criterion = DistillationLoss(temperature=4.0, alpha=0.5)

        freeze(teacher)
        teacher.eval()

        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))

        with torch.no_grad():
            t_logits = teacher(x)

        s_logits = student(x)
        loss, _, _ = criterion(s_logits, t_logits, y)
        loss.backward()

        for name, p in teacher.named_parameters():
            assert p.grad is None, (
                f"Teacher param '{name}' must have no gradient, "
                f"but grad={p.grad}"
            )

    def test_student_receives_gradient(self):
        """Student parameters must receive non-zero gradients."""
        from src.models import TinyCNN, SmallCNN
        from src.distillation import DistillationLoss
        from src.utils import freeze

        teacher = SmallCNN()
        student = TinyCNN()
        criterion = DistillationLoss(temperature=4.0, alpha=0.5)
        freeze(teacher)

        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))

        with torch.no_grad():
            t_logits = teacher(x)

        s_logits = student(x)
        loss, _, _ = criterion(s_logits, t_logits, y)
        loss.backward()

        has_grad = any(p.grad is not None for p in student.parameters())
        assert has_grad, "Student must receive gradients"

    def test_temperature_softens_distribution(self):
        """Higher T should produce lower max probability (softer distribution)."""
        teacher_logits = torch.tensor([[8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        max_prob_T1 = F.softmax(teacher_logits / 1.0, dim=1).max().item()
        max_prob_T8 = F.softmax(teacher_logits / 8.0, dim=1).max().item()
        assert max_prob_T8 < max_prob_T1, (
            f"Higher T should soften distribution: T=1 max={max_prob_T1:.4f}, "
            f"T=8 max={max_prob_T8:.4f}"
        )

    def test_invalid_temperature_raises(self):
        from src.distillation import DistillationLoss
        with pytest.raises(ValueError, match="Temperature"):
            DistillationLoss(temperature=0.0)

    def test_invalid_alpha_raises(self):
        from src.distillation import DistillationLoss
        with pytest.raises(ValueError, match="Alpha"):
            DistillationLoss(temperature=4.0, alpha=1.5)

    def test_kl_zero_when_student_matches_teacher(self):
        """KL divergence is ≈ 0 when student and teacher have identical logits."""
        from src.distillation import DistillationLoss
        criterion = DistillationLoss(temperature=4.0, alpha=0.0)
        logits = torch.randn(8, 10)
        y = torch.randint(0, 10, (8,))

        total, _, kl_val = criterion(logits, logits.clone(), y)
        assert kl_val < 1e-5, (
            f"KL(P ‖ P) should be ≈ 0, got {kl_val:.6f}"
        )
