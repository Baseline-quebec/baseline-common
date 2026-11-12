"""Fixtures of the indexing suite: the finder, the locator, offline reconcilers, built lines."""

from collections.abc import Callable
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from baseline_core.core.llm.document_extraction.indexing.detection.heading_finder import (
    HeadingFinder,
    HeadingFinderConfig,
)
from baseline_core.core.llm.document_extraction.indexing.detection.heading_locator import (
    HeadingLocator,
    HeadingLocatorConfig,
)
from baseline_core.core.llm.document_extraction.indexing.section_reconciler.agent import (
    SectionReconcilerAgent,
)
from baseline_core.core.llm.document_extraction.indexing.section_reconciler.config import (
    SectionReconcilerConfig,
)
from baseline_core.core.llm.document_extraction.layout.layout import (
    BBox,
    LayoutDocument,
    LayoutLine,
    LayoutSpan,
)
from baseline_core.core.llm.document_extraction.layout.layout_parser import LayoutParser


@pytest.fixture
def reconciler_config() -> SectionReconcilerConfig:
    """Return a reconciler configuration at the library defaults, for a maintenance project."""
    return SectionReconcilerConfig(
        relevance_criterion="The section concerns maintenance work on an identified machine."
    )


@pytest.fixture
def no_outline_layout(no_outline_bytes: bytes) -> LayoutDocument:
    """Return the outline-less sample document, parsed."""
    return LayoutParser().parse(no_outline_bytes)


@pytest.fixture
def with_outline_layout(with_outline_bytes: bytes) -> LayoutDocument:
    """Return the four-page sample document carrying an outline, parsed."""
    return LayoutParser().parse(with_outline_bytes)


@pytest.fixture
def finder() -> HeadingFinder:
    """Return a heading finder bound to the library detection thresholds."""
    return HeadingFinder(HeadingFinderConfig())


@pytest.fixture
def heading_locator() -> HeadingLocator:
    """Return a heading locator bound to the library match threshold."""
    return HeadingLocator(HeadingLocatorConfig())


@pytest.fixture
def make_reconciler(
    reconciler_config: SectionReconcilerConfig,
) -> Callable[[list[dict[str, Any]]], SectionReconcilerAgent]:
    """Return a factory building a reconciler whose model emits fixed sections."""

    def build(sections: list[dict[str, Any]]) -> SectionReconcilerAgent:
        def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name=info.output_tools[0].name, args={"sections": sections})
                ]
            )

        return SectionReconcilerAgent(model=FunctionModel(respond), config=reconciler_config)

    return build


@pytest.fixture
def make_line() -> Callable[..., LayoutLine]:
    """Return a factory building one parsed line over given runs, placed at ``y0``."""

    def build(spans: list[LayoutSpan], y0: float = 100.0) -> LayoutLine:
        return LayoutLine(
            text=" ".join(span.text for span in spans),
            bbox=BBox(x0=72.0, y0=y0, x1=400.0, y1=y0 + 12.0),
            spans=spans,
        )

    return build
