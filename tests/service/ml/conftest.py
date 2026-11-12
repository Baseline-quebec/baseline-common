"""Fixtures for the detector service: the tiny RF-DETR behind each runner."""

import pytest
from PIL import Image
from transformers import RfDetrForObjectDetection, RfDetrImageProcessor

from baseline_core.core.ml.huggingface.detection.detection_adapter import DetectionAdapter
from baseline_core.core.ml.model_runner import OnnxModelRunner, TorchModelRunner
from baseline_core.core.ml.onnx_exporter import OnnxExporter
from tests.core.ml.huggingface.dummy_classes import TinyRfDetr


@pytest.fixture(scope="session")
def labels() -> list[str]:
    """Label names of the tiny RF-DETR, in category order."""
    return ["box", "bag"]


@pytest.fixture(scope="session")
def rf_detr_model(labels: list[str]) -> RfDetrForObjectDetection:
    """The tiny RF-DETR over ``labels``, in eval mode."""
    return TinyRfDetr.model(labels)


@pytest.fixture(scope="session")
def rf_detr_processor() -> RfDetrImageProcessor:
    """The RF-DETR processor, resizing to the tiny model's input."""
    return TinyRfDetr.processor()


@pytest.fixture
def detection_adapter(labels: list[str]) -> DetectionAdapter:
    """Box adapter over ``labels``."""
    return DetectionAdapter(labels)


@pytest.fixture(scope="session")
def torch_runner(rf_detr_model: RfDetrForObjectDetection) -> TorchModelRunner:
    """The tiny RF-DETR run through PyTorch."""
    return TorchModelRunner(rf_detr_model)


@pytest.fixture(scope="session")
def onnx_runner(
    rf_detr_model: RfDetrForObjectDetection,
    rf_detr_processor: RfDetrImageProcessor,
    tmp_path_factory: pytest.TempPathFactory,
) -> OnnxModelRunner:
    """The tiny RF-DETR exported to ONNX once and run through onnxruntime."""
    sample_input = rf_detr_processor(images=[Image.new("RGB", (64, 64))], return_tensors="pt")
    path = OnnxExporter(tmp_path_factory.mktemp("onnx")).export(
        rf_detr_model, {"pixel_values": sample_input["pixel_values"]}, ["logits", "pred_boxes"]
    )
    return OnnxModelRunner.load(path)
