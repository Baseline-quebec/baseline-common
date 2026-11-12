"""Fixtures for the vision-language pieces: a taxonomy, labels and a fake hub processor."""

import pytest

from baseline_core.core.ml.data.taxonomy import LabelDefinition, Taxonomy
from tests.core.ml.huggingface.vlm.dummy_classes import FakeHFProcessor


@pytest.fixture
def taxonomy() -> Taxonomy:
    """Two labels with descriptions."""
    return Taxonomy(
        labels=[
            LabelDefinition(name="box", description="A shut box.", status="retained"),
            LabelDefinition(name="bag", description="One bag.", status="retained"),
        ]
    )


@pytest.fixture
def fake_hf_processor() -> FakeHFProcessor:
    """A processor whose tokens are the characters of the text."""
    return FakeHFProcessor()
