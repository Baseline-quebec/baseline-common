"""Every head run on the same embeddings."""

from typing import Any, cast

import torch
from torch import nn

from baseline_core.core.ml.data.batch import AbstractBatch
from baseline_core.core.ml.model.abstract_head import AbstractHead, LossResult


class HeadContainer(nn.Module):
    """Run every head on the same embeddings and gather their predictions.

    Attributes:
        heads: Heads keyed by their ``config.target``.
    """

    def __init__(self, heads: list[AbstractHead[Any, Any]]) -> None:
        """Register the heads under their target.

        Args:
            heads: Heads; their targets must be unique.

        Raises:
            ValueError: If two heads share a target.
        """
        super().__init__()
        targets = [head.config.target for head in heads]
        duplicates = sorted({target for target in targets if targets.count(target) > 1})

        if duplicates:
            error_msg = f"Head targets must be unique, got duplicates: {duplicates}"
            raise ValueError(error_msg)

        self.heads = nn.ModuleDict(dict(zip(targets, heads, strict=True)))

    @property
    def targets(self) -> list[str]:
        """Targets of the heads, in registration order."""
        return list(self.heads.keys())

    def __call__(self, features: Any, batch: AbstractBatch) -> dict[str, Any]:  # noqa: ANN401
        """Run ``forward`` through the ``nn.Module`` hooks, with a typed result."""
        preds: dict[str, Any] = super().__call__(features, batch)
        return preds

    def forward(self, features: Any, batch: AbstractBatch) -> dict[str, Any]:  # noqa: ANN401
        """Run every head on the embeddings.

        Args:
            features: Embeddings from the backbone.
            batch: Batch the embeddings come from.

        Returns:
            Predictions keyed by target.
        """
        return {target: head(features, batch) for target, head in self.heads.items()}

    def head(self, target: str) -> AbstractHead[Any, Any]:
        """The head predicting ``target``."""
        return cast("AbstractHead[Any, Any]", self.heads[target])

    def loss(self, preds: dict[str, Any], batch: AbstractBatch) -> dict[str, torch.Tensor]:
        """Compute every head's loss and their weighted sum.

        Args:
            preds: Predictions keyed by target, as ``forward`` returns them.
            batch: Batch the predictions come from.

        Returns:
            ``<target>_loss``, the weighted ``loss`` and each ``LossResult`` extra, detached.

        Raises:
            ValueError: If there is no prediction, or a head reports a value that is not scalar.
        """
        losses: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None

        for target, head_preds in preds.items():
            head = self.head(target)
            result = head.loss(head_preds, batch)

            if isinstance(result, LossResult):
                losses.update({
                    f"{target}_{name}": value.detach()
                    for name, value in (result.model_extra or {}).items()
                })
                result = result.loss

            losses[f"{target}_loss"] = result
            weighted = head.config.weight * result
            total = weighted if total is None else total + weighted

        if total is None:
            error_msg = "Cannot compute a loss without predictions."
            raise ValueError(error_msg)

        not_scalar = sorted(name for name, value in losses.items() if value.ndim != 0)

        if not_scalar:
            error_msg = f"Losses must be scalars; not scalar: {not_scalar}"
            raise ValueError(error_msg)

        losses["loss"] = total
        return losses
