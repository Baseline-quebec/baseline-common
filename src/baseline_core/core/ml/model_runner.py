"""Running a model for inference, whatever its format: named tensors in, named tensors out."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar

import onnxruntime
import torch
from torch import nn


class AbstractModelRunner(ABC):
    """One inference call on a model, independent of how the model is stored."""

    @abstractmethod
    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the model on ``inputs`` and return its outputs by name."""


class TorchModelRunner(AbstractModelRunner):
    """Runs a PyTorch module in evaluation mode, without gradients, on the module's device.

    Attributes:
        model: The module, in evaluation mode.
        output_names: Names of tensor or tuple outputs; ``None`` when the call returns a mapping.
        call: What takes the inputs as keyword arguments: the module, or e.g. ``model.generate``.
        call_kwargs: Fixed keyword arguments added to every call, e.g. ``max_new_tokens``.
    """

    def __init__(
        self,
        model: nn.Module,
        output_names: list[str] | None = None,
        call: Callable[..., Any] | None = None,
        call_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Switch ``model`` to evaluation mode and keep how to call it.

        Args:
            model: The module to run.
            output_names: Names of tensor or tuple outputs, in order.
            call: Bound method taking the inputs, or ``None`` for the module's ``forward``.
            call_kwargs: Fixed keyword arguments added to every call.
        """
        self.model = model.eval()
        self.output_names = output_names
        self.call = call if call is not None else self.model
        self.call_kwargs = dict(call_kwargs or {})

    @property
    def device(self) -> torch.device:
        """Device of the module's first parameter, the CPU for a module without any."""
        parameter = next(self.model.parameters(), None)
        return torch.device("cpu") if parameter is None else parameter.device

    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run ``call`` on ``inputs`` moved to the module's device, plus ``call_kwargs``.

        Returns:
            The tensor outputs by name; non-tensor entries of a mapping output are dropped.

        Raises:
            TypeError: If the output is a tensor or a tuple and ``output_names`` is ``None``.
        """
        device = self.device

        with torch.no_grad():
            outputs = self.call(
                **{name: tensor.to(device) for name, tensor in inputs.items()}, **self.call_kwargs
            )

        if isinstance(outputs, Mapping):
            return {
                name: value for name, value in outputs.items() if isinstance(value, torch.Tensor)
            }

        if self.output_names is None:
            error_msg = (
                f"{type(self.model).__name__} returns {type(outputs).__name__}; "
                "give output_names to name its outputs."
            )
            raise TypeError(error_msg)

        values = outputs if isinstance(outputs, tuple) else (outputs,)
        return dict(zip(self.output_names, values, strict=True))


class OnnxModelRunner(AbstractModelRunner):
    """Runs an ONNX model through ``onnxruntime``, feeding it only the inputs it declares.

    Attributes:
        session: The runtime session.
        input_names: Inputs the graph declares.
        output_names: Outputs the graph declares.
    """

    DEFAULT_PROVIDERS: ClassVar[tuple[str, ...]] = ("CPUExecutionProvider",)

    def __init__(self, session: onnxruntime.InferenceSession) -> None:
        """Keep the session and read its signature.

        Args:
            session: The runtime session.
        """
        self.session = session
        self.input_names = [item.name for item in session.get_inputs()]
        self.output_names = [item.name for item in session.get_outputs()]

    @classmethod
    def load(cls, path: Path, providers: list[str] | None = None) -> "OnnxModelRunner":
        """Open the ONNX file at ``path``.

        Args:
            path: The ONNX file.
            providers: ``onnxruntime`` execution providers; CPU only by default.

        Returns:
            The runner.
        """
        session = onnxruntime.InferenceSession(
            str(path), providers=list(providers or cls.DEFAULT_PROVIDERS)
        )
        return cls(session)

    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the graph on the entries of ``inputs`` it declares; extra entries are ignored.

        Raises:
            KeyError: If a declared input is missing.
        """
        missing = [name for name in self.input_names if name not in inputs]

        if missing:
            error_msg = f"Inputs {missing} are missing; the model takes {self.input_names}."
            raise KeyError(error_msg)

        feeds = {name: inputs[name].detach().cpu().numpy() for name in self.input_names}
        values = self.session.run(self.output_names, feeds)
        return {
            name: torch.from_numpy(value)
            for name, value in zip(self.output_names, values, strict=True)
        }
