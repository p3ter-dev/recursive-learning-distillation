"""
tests/test_models.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for SmallCNN and TinyCNN architectures and the build_model factory.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class TestSmallCNN:
    def _make(self, **kwargs):
        from src.models import SmallCNN
        return SmallCNN(**kwargs)

    def test_output_shape_default(self):
        model = self._make()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"

    def test_output_shape_batch1(self):
        model = self._make()
        x = torch.randn(1, 1, 28, 28)
        out = model(x)
        assert out.shape == (1, 10)

    def test_is_differentiable(self):
        model = self._make()
        x = torch.randn(2, 1, 28, 28)
        loss = model(x).sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        assert has_grad, "SmallCNN should have computable gradients"

    def test_parameter_count_default(self):
        from src.utils import count_parameters
        model = self._make()
        n = count_parameters(model)
        # Default config yields ~212K params; check it is within a reasonable band
        assert 100_000 < n < 500_000, f"Unexpected param count: {n:,}"

    def test_custom_channels(self):
        model = self._make(conv_channels=[16, 16], fc_hidden=64)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)

    def test_extra_kwargs_ignored(self):
        """build_model passes the full YAML dict; extra keys must not raise."""
        from src.models import SmallCNN
        # 'name' is an extra kwarg that comes from the YAML config
        model = SmallCNN(name="SmallCNN", conv_channels=[32, 32], fc_hidden=128, dropout=0.3)
        assert model is not None

    def test_no_nan_in_forward(self):
        model = self._make()
        x = torch.randn(8, 1, 28, 28)
        out = model(x)
        assert not torch.isnan(out).any(), "Forward pass contains NaN"


class TestTinyCNN:
    def _make(self, **kwargs):
        from src.models import TinyCNN
        return TinyCNN(**kwargs)

    def test_output_shape_default(self):
        model = self._make()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)

    def test_parameter_count_default(self):
        from src.utils import count_parameters
        model = self._make()
        n = count_parameters(model)
        assert 10_000 < n < 100_000, f"Unexpected student param count: {n:,}"

    def test_student_smaller_than_teacher(self):
        from src.models import SmallCNN, TinyCNN
        from src.utils import count_parameters
        teacher = SmallCNN()
        student = TinyCNN()
        assert count_parameters(student) < count_parameters(teacher), (
            "Student must have fewer parameters than teacher"
        )

    def test_is_differentiable(self):
        model = self._make()
        x = torch.randn(2, 1, 28, 28)
        loss = model(x).sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        assert has_grad


class TestBuildModel:
    def test_build_smallcnn_from_config(self):
        from src.models import build_model
        cfg = {"name": "SmallCNN", "conv_channels": [32, 32], "fc_hidden": 128, "dropout": 0.3}
        model = build_model(cfg, "teacher")
        assert isinstance(model, nn.Module)

    def test_build_tinycnn_from_config(self):
        from src.models import build_model
        cfg = {"name": "TinyCNN", "conv_channels": [8], "fc_hidden": 64, "dropout": 0.3}
        model = build_model(cfg, "student")
        assert isinstance(model, nn.Module)

    def test_build_resnet18_from_config(self):
        from src.models import build_model
        cfg = {"name": "ResNet18"}
        model = build_model(cfg, "teacher")
        assert isinstance(model, nn.Module)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)

    def test_build_resnet8_from_config(self):
        from src.models import build_model
        cfg = {"name": "ResNet8"}
        model = build_model(cfg, "student")
        assert isinstance(model, nn.Module)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)

    def test_unknown_name_raises(self):
        from src.models import build_model
        with pytest.raises(ValueError, match="Unknown model"):
            build_model({"name": "DoesNotExist"})

