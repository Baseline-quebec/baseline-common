"""Tests for ``PassthroughBackbone``."""

from typing import Any

import pytest

from baseline_core.core.ml.data.batch import Batch
from baseline_core.core.ml.huggingface.passthrough_backbone import PassthroughBackbone


class TestPassthroughBackbone:
    """Tests for ``PassthroughBackbone``."""

    @pytest.fixture
    def backbone(self) -> PassthroughBackbone:
        """The backbone of a head owning its whole model."""
        return PassthroughBackbone()

    def test__call__should_return_the_batch_inputs(
        self, backbone: PassthroughBackbone, regression_batch: Batch[Any]
    ) -> None:
        """The head receives the inputs as they were collated."""
        assert backbone(regression_batch) is regression_batch.x
