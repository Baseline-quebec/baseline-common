"""Tests for the callbacks of Hugging Face detection heads."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import pytorch_lightning as pl

from baseline_core.core.ml.data.batch import Batch
from baseline_core.core.ml.huggingface.detection.detection_adapter import DetectionAdapter
from baseline_core.core.ml.huggingface.detection.hf_detection_head import HFDetectionHead
from baseline_core.core.ml.huggingface.detection.validation_prediction_writer import (
    ValidationPredictionWriter,
)
from baseline_core.core.ml.modality.image.format.coco_codec import CocoCodec
from baseline_core.core.ml.training.backbone_heads_module import BackboneHeadsModule


class TestValidationPredictionWriter:
    """Tests for ``ValidationPredictionWriter`` around the tiny RF-DETR."""

    @pytest.fixture
    def writer(
        self,
        tmp_path: Path,
        detection_head: HFDetectionHead,
        detection_adapter: DetectionAdapter,
        coco_codec: CocoCodec,
    ) -> ValidationPredictionWriter:
        """Writer of the detection head's predictions, as COCO, into a temporary folder."""
        return ValidationPredictionWriter(
            tmp_path / "predictions", detection_head, detection_adapter, coco_codec
        )

    @pytest.fixture
    def step_outputs(
        self, detection_module: BackboneHeadsModule, collated_batch: Batch[Any]
    ) -> dict[str, Any]:
        """What a validation step hands to the callbacks."""
        return {"outputs": detection_module.step(collated_batch).outputs}

    @pytest.fixture
    def trainer(self) -> pl.Trainer:
        """The trainer attributes the writer reads, on one process at epoch 2."""
        return cast(
            "pl.Trainer",
            SimpleNamespace(sanity_checking=False, current_epoch=2, world_size=1, global_rank=0),
        )

    @pytest.fixture
    def sanity_trainer(self) -> pl.Trainer:
        """The same trainer while Lightning runs its sanity check."""
        return cast(
            "pl.Trainer",
            SimpleNamespace(sanity_checking=True, current_epoch=2, world_size=1, global_rank=0),
        )

    def test__on_validation_epoch_end__should_write_the_epoch_predictions_as_coco(
        self,
        writer: ValidationPredictionWriter,
        trainer: pl.Trainer,
        detection_module: BackboneHeadsModule,
        collated_batch: Batch[Any],
        step_outputs: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """One file per epoch, one image per validated image, every box scored."""
        writer.on_validation_batch_end(trainer, detection_module, step_outputs, collated_batch, 0)

        writer.on_validation_epoch_end(trainer, detection_module)

        predictions = CocoCodec.read(tmp_path / "predictions" / "epoch_002_val.json")
        assert [image.file_name for image in predictions] == ["box.png", "mixed.png"]
        assert all(box.score is not None for image in predictions for box in image.boxes)

    def test__on_validation_batch_end__when_metadata_lacks_annotations__should_raise_value_error(
        self,
        writer: ValidationPredictionWriter,
        trainer: pl.Trainer,
        detection_module: BackboneHeadsModule,
        collated_batch: Batch[Any],
        step_outputs: dict[str, Any],
    ) -> None:
        """The collator must carry the annotations; the error says how."""
        bare_batch = collated_batch.model_copy(update={"metadata": None})

        with pytest.raises(ValueError, match=r"metadata_keys=\['annotations'\]"):
            writer.on_validation_batch_end(trainer, detection_module, step_outputs, bare_batch, 0)

    def test__on_validation_epoch_end__when_sanity_check__should_write_nothing(
        self,
        writer: ValidationPredictionWriter,
        sanity_trainer: pl.Trainer,
        detection_module: BackboneHeadsModule,
        collated_batch: Batch[Any],
        step_outputs: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """The batches Lightning validates before training are not predictions of an epoch."""
        writer.on_validation_batch_end(
            sanity_trainer, detection_module, step_outputs, collated_batch, 0
        )

        writer.on_validation_epoch_end(sanity_trainer, detection_module)

        assert not (tmp_path / "predictions").exists()
