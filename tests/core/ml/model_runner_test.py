"""Tests for ``TorchModelRunner`` and ``OnnxModelRunner``."""

from pathlib import Path

import pytest
import torch
from torch import nn

from baseline_core.core.ml.model_runner import OnnxModelRunner, TorchModelRunner
from baseline_core.core.ml.onnx_exporter import OnnxExporter


class MappingModel(nn.Module):
    """Model returning its outputs by name, as Hugging Face models do, with a ``None`` entry."""

    def __init__(self) -> None:
        """One linear layer."""
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, input: torch.Tensor) -> dict[str, torch.Tensor | None]:  # noqa: A002
        """Logits under ``logits`` and an absent loss; the argument named as in ``nn.Linear``."""
        return {"loss": None, "logits": self.linear(input)}


class TestTorchModelRunner:
    """Tests for ``TorchModelRunner``."""

    @pytest.fixture
    def features(self) -> dict[str, torch.Tensor]:
        """One batch of three feature vectors, keyed by the models' argument name."""
        return {"input": torch.randn(3, 4)}

    def test__call__when_mapping_output__should_keep_tensors_by_name(
        self, features: dict[str, torch.Tensor]
    ) -> None:
        """Names come from the model; ``None`` entries are dropped."""
        model = MappingModel()
        runner = TorchModelRunner(model)

        outputs = runner(features)

        assert set(outputs) == {"logits"}
        assert outputs["logits"].shape == (3, 2)
        assert not model.training

    def test__call__when_tensor_output__should_use_the_given_names(
        self, features: dict[str, torch.Tensor]
    ) -> None:
        """A bare tensor needs a name from the caller."""
        model = nn.Linear(4, 2)
        runner = TorchModelRunner(model, output_names=["logits"])

        outputs = runner(features)

        assert set(outputs) == {"logits"}
        assert not outputs["logits"].requires_grad

    def test__call__when_tensor_output_without_names__should_raise_type_error(
        self, features: dict[str, torch.Tensor]
    ) -> None:
        """Unnamed outputs cannot be returned by name."""
        runner = TorchModelRunner(nn.Linear(4, 2))

        with pytest.raises(TypeError, match="give output_names"):
            runner(features)

    def test__call__when_call_given__should_run_it_with_the_fixed_kwargs(
        self, features: dict[str, torch.Tensor]
    ) -> None:
        """Another bound method of the module, e.g. ``generate``, runs instead of ``forward``."""
        model = GeneratingModel()
        runner = TorchModelRunner(
            model, output_names=["sequences"], call=model.generate, call_kwargs={"repeats": 3}
        )

        outputs = runner(features)

        assert outputs["sequences"].shape == (3, 12)

    def test__device__when_module_has_no_parameter__should_be_cpu(self) -> None:
        """A parameterless module runs on the CPU."""
        assert TorchModelRunner(nn.Identity()).device == torch.device("cpu")


class GeneratingModel(nn.Module):
    """Model with a ``generate`` method, as Hugging Face causal models have."""

    def __init__(self) -> None:
        """One linear layer, so the module has a device."""
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def generate(self, input: torch.Tensor, repeats: int) -> torch.Tensor:  # noqa: A002
        """Repeat the input ``repeats`` times along the last axis, as a bare tensor."""
        return input.repeat(1, repeats)


class TestOnnxModelRunner:
    """Tests for ``OnnxModelRunner`` on an exported linear model."""

    @pytest.fixture
    def model(self) -> nn.Linear:
        """A fixed linear map."""
        torch.manual_seed(0)
        return nn.Linear(4, 2)

    @pytest.fixture
    def features(self) -> dict[str, torch.Tensor]:
        """One batch of three feature vectors, keyed by the models' argument name."""
        return {"input": torch.randn(3, 4)}

    @pytest.fixture
    def runner(
        self, model: nn.Linear, features: dict[str, torch.Tensor], tmp_path: Path
    ) -> OnnxModelRunner:
        """The model exported and reopened."""
        path = OnnxExporter(tmp_path).export(model, features, output_names=["logits"])
        return OnnxModelRunner.load(path)

    def test__load__should_read_the_graph_signature(self, runner: OnnxModelRunner) -> None:
        """Input and output names come from the file."""
        assert runner.input_names == ["input"]
        assert runner.output_names == ["logits"]

    def test__call__should_match_the_torch_model_and_ignore_extra_inputs(
        self, runner: OnnxModelRunner, model: nn.Linear, features: dict[str, torch.Tensor]
    ) -> None:
        """Same outputs as PyTorch; inputs the graph does not declare are left out."""
        expected = TorchModelRunner(model, output_names=["logits"])(features)

        outputs = runner({**features, "mask": torch.ones(3)})

        assert set(outputs) == {"logits"}
        assert torch.allclose(outputs["logits"], expected["logits"], atol=1e-6)

    def test__call__when_input_missing__should_raise_key_error(
        self, runner: OnnxModelRunner
    ) -> None:
        """The graph's inputs are mandatory."""
        with pytest.raises(KeyError, match=r"\['input'\] are missing"):
            runner({"pixels": torch.randn(3, 4)})
