"""Lightning callbacks of Hugging Face detection heads."""

from pathlib import Path
from typing import Any, ClassVar, override

import pytorch_lightning as pl

from baseline_core.core.ml.data.batch import Batch
from baseline_core.core.ml.huggingface.detection.detection_adapter import DetectionAdapter
from baseline_core.core.ml.huggingface.detection.hf_detection_head import HFDetectionHead
from baseline_core.core.ml.modality.image.annotation import ImageAnnotation
from baseline_core.core.ml.modality.image.format.abstract_annotation_codec import (
    AbstractAnnotationCodec,
)


class ValidationPredictionWriter(pl.Callback):
    """Write the predictions of every validation epoch in the format of ``codec``.

    Attributes:
        out_dir: Folder receiving ``epoch_<epoch>_val[_rank_<rank>]<codec.SUFFIX>`` files.
        head: Detection head whose predictions are written.
        adapter: Decodes the predicted boxes.
        codec: Format the predictions are written in.
    """

    ANNOTATIONS_KEY: ClassVar[str] = "annotations"

    def __init__(
        self,
        out_dir: Path,
        head: HFDetectionHead,
        adapter: DetectionAdapter,
        codec: AbstractAnnotationCodec,
    ) -> None:
        """Keep the destination folder, the head, the adapter and the codec.

        Args:
            out_dir: Folder receiving one file per validation epoch; created when written.
            head: Detection head whose predictions are written.
            adapter: Decodes the predicted boxes.
            codec: Format the predictions are written in.
        """
        self.out_dir = out_dir
        self.head = head
        self.adapter = adapter
        self.codec = codec
        self._predictions: list[ImageAnnotation] = []

    @override
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Decode the predictions of the batch and keep them for the end of the epoch."""
        del pl_module, batch_idx, dataloader_idx

        if trainer.sanity_checking:
            return

        images = self.annotations_of(batch)
        results = self.head.postprocess(outputs["outputs"][self.head.config.target], batch)
        self._predictions.extend(
            image.model_copy(update={"boxes": self.adapter.decode(result)})
            for image, result in zip(images, results, strict=True)
        )

    @override
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Write the predictions of the epoch, then start over."""
        del pl_module

        if trainer.sanity_checking:
            self._predictions.clear()
            return

        rank = f"_rank_{trainer.global_rank}" if trainer.world_size > 1 else ""
        path = self.out_dir / f"epoch_{trainer.current_epoch:03d}_val{rank}{self.codec.SUFFIX}"
        self.codec.write(path, self._predictions, self.adapter.labels)
        self._predictions.clear()

    @classmethod
    def annotations_of(cls, batch: Batch[Any]) -> list[ImageAnnotation]:
        """The ``ImageAnnotation`` the collator put in the metadata of ``batch``.

        Raises:
            ValueError: If the metadata has none.
        """
        if batch.metadata is None or cls.ANNOTATIONS_KEY not in batch.metadata:
            error_msg = (
                f"Batch metadata lacks '{cls.ANNOTATIONS_KEY}'; configure the collator with "
                f"metadata_keys=['{cls.ANNOTATIONS_KEY}']."
            )
            raise ValueError(error_msg)

        annotations: list[ImageAnnotation] = batch.metadata[cls.ANNOTATIONS_KEY]
        return annotations
