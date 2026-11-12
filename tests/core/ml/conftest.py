"""Fixtures shared across the ML framework tests: transform module, dummy model, images."""

from types import ModuleType

import pytest
import torch
from torch import nn

from baseline_core.core.ml.data.batch import Batch, TensorTargets
from baseline_core.core.ml.modality.image.annotation import (
    BoundingBox,
    BoxAnnotation,
    ImageAnnotation,
)
from baseline_core.core.ml.modality.image.format.coco_codec import CocoCodec
from baseline_core.core.ml.model.head_container import HeadContainer
from baseline_core.core.ml.training.optimizer import (
    AdamOptimizer,
    OptimizerFactory,
    PlateauScheduler,
)
from tests.core.ml.dummy_classes import (
    AddOne,
    DummyBackbone,
    DummyBackboneConfig,
    DummyHead,
    DummyHeadConfig,
    Multiply,
    ScaleValues,
    SumValues,
    ToText,
)


@pytest.fixture
def transform_module() -> ModuleType:
    """A module exposing the transform classes, as a real transforms module would."""
    module = ModuleType("fake_transforms")
    module.__dict__.update({
        "AddOne": AddOne,
        "Multiply": Multiply,
        "ToText": ToText,
        "SumValues": SumValues,
        "ScaleValues": ScaleValues,
    })
    return module


@pytest.fixture
def hidden_dim() -> int:
    """Feature and embedding size of the dummy model."""
    return 4


@pytest.fixture
def user_count() -> int:
    """Samples in the ``user`` group of the batch."""
    return 2


@pytest.fixture
def flight_count() -> int:
    """Samples in the ``flight`` group of the batch."""
    return 3


@pytest.fixture
def batch(hidden_dim: int, user_count: int, flight_count: int) -> Batch[TensorTargets]:
    """A batch with two inputs (``user``, ``flight``) and one target per input."""
    return Batch(
        x={
            "user": torch.randn(user_count, hidden_dim),
            "flight": torch.randn(flight_count, hidden_dim),
        },
        targets={
            "y_user": torch.randn(user_count, 1),
            "y_flight": torch.rand(flight_count, 1).round(),
        },
    )


@pytest.fixture
def backbone_config(hidden_dim: int) -> DummyBackboneConfig:
    """Backbone config for the two inputs."""
    return DummyBackboneConfig(hidden_dim=hidden_dim, groups=["user", "flight"])


@pytest.fixture
def mse_head_config(hidden_dim: int) -> DummyHeadConfig:
    """Regression head on the ``user`` group, weight 0.5."""
    return DummyHeadConfig(
        target="y_user", weight=0.5, group="user", hidden_dim=hidden_dim, output_dim=1
    )


@pytest.fixture
def bce_head_config(hidden_dim: int) -> DummyHeadConfig:
    """Binary classification head on the ``flight`` group, weight 2.0."""
    return DummyHeadConfig(
        target="y_flight", weight=2.0, group="flight", hidden_dim=hidden_dim, output_dim=1
    )


@pytest.fixture
def mse_head(mse_head_config: DummyHeadConfig) -> DummyHead:
    """Regression head scored with the mean squared error."""
    return DummyHead(mse_head_config, nn.MSELoss())


@pytest.fixture
def bce_head(bce_head_config: DummyHeadConfig) -> DummyHead:
    """Binary classification head scored with the binary cross-entropy on logits."""
    return DummyHead(bce_head_config, nn.BCEWithLogitsLoss())


@pytest.fixture
def backbone(backbone_config: DummyBackboneConfig) -> DummyBackbone:
    """Dummy backbone."""
    return DummyBackbone(backbone_config)


@pytest.fixture
def heads(mse_head: DummyHead, bce_head: DummyHead) -> HeadContainer:
    """Both dummy heads in a container."""
    return HeadContainer([mse_head, bce_head])


@pytest.fixture
def optimizer_factory() -> OptimizerFactory:
    """Adam with a plateau scheduler monitoring the validation loss."""
    return OptimizerFactory(
        optimizer=AdamOptimizer(lr=0.01), scheduler=PlateauScheduler(metric="val_loss")
    )


@pytest.fixture(scope="session")
def labels() -> list[str]:
    """Label names, in category order."""
    return ["box", "bag"]


@pytest.fixture
def box_image() -> ImageAnnotation:
    """A 64x48 image with one occluded box."""
    return ImageAnnotation(
        file_name="box.png",
        width=64,
        height=48,
        boxes=[
            BoxAnnotation(label="box", box=BoundingBox(x1=4, y1=4, x2=36, y2=28), occluded=True)
        ],
    )


@pytest.fixture
def mixed_image() -> ImageAnnotation:
    """A 64x48 image with one box and one bag."""
    return ImageAnnotation(
        file_name="mixed.png",
        width=64,
        height=48,
        boxes=[
            BoxAnnotation(label="box", box=BoundingBox(x1=0, y1=0, x2=20, y2=20)),
            BoxAnnotation(label="bag", box=BoundingBox(x1=30, y1=10, x2=60, y2=40)),
        ],
    )


@pytest.fixture
def empty_image() -> ImageAnnotation:
    """A 64x48 image without any box."""
    return ImageAnnotation(file_name="empty.png", width=64, height=48)


@pytest.fixture
def annotated_images(
    box_image: ImageAnnotation, mixed_image: ImageAnnotation, empty_image: ImageAnnotation
) -> list[ImageAnnotation]:
    """Three annotated images over ``labels``, one of them without any box."""
    return [box_image, mixed_image, empty_image]


@pytest.fixture
def coco_codec() -> CocoCodec:
    """The annotation format CVAT exports and imports."""
    return CocoCodec()
