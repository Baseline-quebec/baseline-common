"""Fixtures for the Hugging Face adapter: a dummy hub model, its processor, head and module."""

from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import AutoConfig, AutoModel, ViTImageProcessor

from baseline_core.core.ml.data.batch import Batch
from baseline_core.core.ml.huggingface.hf_head import HFHead, HFHeadConfig
from baseline_core.core.ml.huggingface.passthrough_backbone import PassthroughBackbone
from baseline_core.core.ml.model.head_container import HeadContainer
from baseline_core.core.ml.training.backbone_heads_module import BackboneHeadsModule
from baseline_core.core.ml.training.optimizer import OptimizerFactory
from tests.core.ml.huggingface.dummy_classes import DummyHFConfig, DummyHFModel


@pytest.fixture
def feature_count() -> int:
    """Features per regression sample, which is also the dummy model's input size."""
    return 4


@pytest.fixture(scope="session")
def dummy_hf_registered() -> None:
    """``DummyHFModel`` known to the ``transformers`` auto classes, as a hub model is."""
    AutoConfig.register(DummyHFConfig.model_type, DummyHFConfig, exist_ok=True)
    AutoModel.register(DummyHFConfig, DummyHFModel, exist_ok=True)


@pytest.fixture
def dummy_hf_model(feature_count: int) -> DummyHFModel:
    """Randomly initialized dummy Hugging Face model."""
    torch.manual_seed(0)
    return DummyHFModel(DummyHFConfig(input_size=feature_count))


@pytest.fixture
def pretrained_dir(
    tmp_path: Path, dummy_hf_model: DummyHFModel, dummy_hf_registered: None
) -> Path:
    """Folder holding the dummy model and an image processor, as a hub repository does."""
    del dummy_hf_registered
    directory = tmp_path / "pretrained"
    dummy_hf_model.save_pretrained(directory)
    ViTImageProcessor().save_pretrained(directory)
    return directory


@pytest.fixture
def hf_head_config(pretrained_dir: Path) -> HFHeadConfig:
    """Config loading the dummy model and its processor from ``pretrained_dir``."""
    return HFHeadConfig(
        target="regression",
        model_id=str(pretrained_dir),
        auto_class="AutoModel",
        processor_class="AutoImageProcessor",
    )


@pytest.fixture
def hf_head(hf_head_config: HFHeadConfig) -> HFHead:
    """``HFHead`` loaded from ``pretrained_dir``."""
    return HFHead(hf_head_config)


@pytest.fixture
def hf_module(hf_head: HFHead, optimizer_factory: OptimizerFactory) -> BackboneHeadsModule:
    """The dummy Hugging Face model as the only head of a module, without metrics."""
    return BackboneHeadsModule(PassthroughBackbone(), HeadContainer([hf_head]), optimizer_factory)


@pytest.fixture
def regression_batch(feature_count: int) -> Batch[Any]:
    """Three samples for the dummy Hugging Face model, with targets."""
    return Batch(x={"pixel_values": torch.randn(3, feature_count)}, targets=torch.zeros(3, 1))
