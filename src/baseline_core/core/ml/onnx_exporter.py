"""Export of any PyTorch model to ONNX."""

from pathlib import Path

import torch
from torch import nn


class OnnxExporter:
    """Writes a model as one ONNX file under a folder.

    Attributes:
        out_dir: Folder the ONNX file is written to.
        file_name: Name of the ONNX file.
    """

    def __init__(self, out_dir: Path, file_name: str = "model.onnx") -> None:
        """Keep the destination.

        Args:
            out_dir: Folder the ONNX file is written to; created if needed.
            file_name: Name of the ONNX file.
        """
        self.out_dir = out_dir
        self.file_name = file_name

    def export(
        self, model: nn.Module, sample_input: dict[str, torch.Tensor], output_names: list[str]
    ) -> Path:
        """Trace ``model`` in evaluation mode and write it as one ONNX file.

        Args:
            model: The model; its ``forward`` takes ``sample_input`` as keyword arguments.
            sample_input: One batch of inputs; the keys name the ONNX inputs.
            output_names: Names of the first ONNX outputs, in ``forward``'s output order.

        Returns:
            The ONNX file.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / self.file_name
        model.eval()
        torch.onnx.export(
            model,
            (),
            path,
            kwargs=sample_input,
            input_names=list(sample_input),
            output_names=output_names,
            dynamo=True,
            optimize=False,
            external_data=False,
        )
        return path
