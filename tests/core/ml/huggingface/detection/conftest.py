"""Fixtures for the detection tests: the tiny RF-DETR, as a head, its module and a batch."""

from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from torchmetrics import MetricCollection
from torchmetrics.detection import MeanAveragePrecision
from transformers import RfDetrForObjectDetection, RfDetrImageProcessor

from baseline_core.core.ml.data.batch import Batch
from baseline_core.core.ml.huggingface.detection.detection_adapter import DetectionAdapter
from baseline_core.core.ml.huggingface.detection.detection_processor import DetectionProcessor
from baseline_core.core.ml.huggingface.detection.hf_detection_head import (
    HFDetectionConfig,
    HFDetectionHead,
)
from baseline_core.core.ml.huggingface.passthrough_backbone import PassthroughBackbone
from baseline_core.core.ml.modality.image.annotation import ImageAnnotation
from baseline_core.core.ml.model.head_container import HeadContainer
from baseline_core.core.ml.training.backbone_heads_module import BackboneHeadsModule
from baseline_core.core.ml.training.optimizer import OptimizerFactory
from tests.core.ml.huggingface.dummy_classes import TinyRfDetr


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
    """Box adapter over the two labels."""
    return DetectionAdapter(labels)


@pytest.fixture(scope="session")
def rf_detr_dir(
    rf_detr_model: RfDetrForObjectDetection,
    rf_detr_processor: RfDetrImageProcessor,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Folder holding the tiny RF-DETR and its processor, as a hub repository does."""
    directory = tmp_path_factory.mktemp("rf_detr")
    rf_detr_model.save_pretrained(directory)
    rf_detr_processor.save_pretrained(directory)
    return directory


@pytest.fixture
def detection_config(rf_detr_dir: Path) -> HFDetectionConfig:
    """The tiny RF-DETR keeping every prediction."""
    return HFDetectionConfig(model_id=str(rf_detr_dir), threshold=0.0)


@pytest.fixture
def detection_head(detection_config: HFDetectionConfig) -> HFDetectionHead:
    """The tiny RF-DETR as a detection head."""
    return HFDetectionHead(detection_config)


@pytest.fixture
def detection_module(
    detection_head: HFDetectionHead,
    detection_config: HFDetectionConfig,
    optimizer_factory: OptimizerFactory,
) -> BackboneHeadsModule:
    """The detection head alone in a module evaluated with mean average precision."""
    return BackboneHeadsModule(
        PassthroughBackbone(),
        HeadContainer([detection_head]),
        optimizer_factory,
        {detection_config.target: MetricCollection([MeanAveragePrecision(box_format="xyxy")])},
    )


@pytest.fixture
def collated_batch(
    rf_detr_processor: RfDetrImageProcessor,
    box_image: ImageAnnotation,
    mixed_image: ImageAnnotation,
    detection_adapter: DetectionAdapter,
) -> Batch[Any]:
    """The box and mixed images, blank, collated the way the service does it."""
    images = [box_image, mixed_image]
    processor = DetectionProcessor(rf_detr_processor, detection_adapter)
    processed = dict(
        processor(
            [Image.new("RGB", (image.width, image.height)) for image in images],
            images,
            return_tensors="pt",
        )
    )
    targets = processed.pop("labels")
    return Batch(x=processed, targets=targets, metadata={"annotations": images})
