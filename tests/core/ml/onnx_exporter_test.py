"""Tests for ``OnnxExporter``."""

from pathlib import Path

import numpy as np
import onnxruntime
import pytest
import torch
from torch import nn

from baseline_core.core.ml.onnx_exporter import OnnxExporter


class TestOnnxExporter:
    """Tests for ``OnnxExporter`` on a plain linear model."""

    @pytest.fixture
    def model(self) -> nn.Linear:
        """A fixed linear map from four features to two outputs."""
        torch.manual_seed(0)
        return nn.Linear(4, 2)

    @pytest.fixture
    def sample_input(self) -> dict[str, torch.Tensor]:
        """One batch of three feature vectors."""
        return {"input": torch.randn(3, 4)}

    @pytest.fixture
    def exporter(self, tmp_path: Path) -> OnnxExporter:
        """Exporter writing under ``tmp_path / "bundle"``."""
        return OnnxExporter(tmp_path / "bundle")

    def test__export__should_write_a_runnable_model_with_named_io(
        self,
        exporter: OnnxExporter,
        model: nn.Linear,
        sample_input: dict[str, torch.Tensor],
        tmp_path: Path,
    ) -> None:
        """The file runs under ONNX runtime and reproduces the PyTorch outputs."""
        path = exporter.export(model, sample_input, output_names=["output"])

        assert path == tmp_path / "bundle" / "model.onnx"
        assert [p.name for p in path.parent.iterdir()] == ["model.onnx"]
        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        assert [i.name for i in session.get_inputs()] == ["input"]
        assert [o.name for o in session.get_outputs()] == ["output"]
        (output,) = session.run(["output"], {"input": sample_input["input"].numpy()})

        with torch.no_grad():
            expected = model(**sample_input).numpy()

        assert np.allclose(output, expected, atol=1e-6)

    def test__export__should_leave_the_model_in_eval_mode(
        self, exporter: OnnxExporter, model: nn.Linear, sample_input: dict[str, torch.Tensor]
    ) -> None:
        """Tracing a training-mode model would bake training behavior into the file."""
        model.train()

        exporter.export(model, sample_input, output_names=["output"])

        assert not model.training
