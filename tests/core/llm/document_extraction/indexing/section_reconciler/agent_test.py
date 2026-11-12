"""Unit tests for SectionReconcilerAgent.reconcile against an offline model."""

import asyncio
import re
from collections.abc import Callable
from typing import Any, ClassVar

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage

from baseline_core.core.llm.document_extraction.indexing.detection.candidates import (
    FormatCandidate,
    HeadingCandidates,
    Typography,
)
from baseline_core.core.llm.document_extraction.indexing.section_reconciler.agent import (
    SectionReconcilerAgent,
)
from baseline_core.core.llm.document_extraction.indexing.section_reconciler.config import (
    SectionReconcilerConfig,
)
from baseline_core.core.llm.document_extraction.layout.layout import BBox
from baseline_core.core.llm.document_extraction.section import Section, SectionRelevance


class PageEcho:
    """An offline model reporting one section per page block shown, remembering each prompt."""

    PAGE: ClassVar[re.Pattern[str]] = re.compile(r"^Page (\d+)", re.MULTILINE)
    """A page block's opening line."""

    CONVENTION: ClassVar[str] = "items hang one level under a CAPS rubric"
    """The one convention every answer settles."""

    def __init__(self) -> None:
        """Start with no prompt seen."""
        self.prompts: list[str] = []
        self.instructions: list[str | None] = []

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Answer the user prompt with a section on every page it names, remembering the prompt.

        Args:
            messages: The run so far; the last request carries the user prompt.
            info: The run's tools, naming the output tool to call.

        Returns:
            One section per page block, in prompt order, under the one convention.
        """
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        prompt = "".join(
            str(part.content) for part in request.parts if isinstance(part, UserPromptPart)
        )
        self.prompts.append(prompt)
        self.instructions.append(info.instructions)
        sections = [
            {
                "title": f"Heading {page}",
                "start_page": int(page),
                "level": 1,
                "relevance": "irrelevant",
            }
            for page in self.PAGE.findall(prompt)
        ]
        args = {"conventions": [self.CONVENTION], "sections": sections}
        return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=args)])


class TestReconcile:
    STYLE: ClassVar[Typography] = Typography(size=14.0, bold=True)

    @classmethod
    def spanning(cls, last_page: int) -> HeadingCandidates:
        """Build a document carrying one candidate on every page up to ``last_page``.

        Args:
            last_page: The page the last candidate sits on.

        Returns:
            The candidates of the whole document.
        """
        return HeadingCandidates(
            outline=[],
            format=[
                FormatCandidate(
                    text=f"Heading {page}",
                    typography=cls.STYLE,
                    page=page,
                    bbox=BBox(x0=72.0, y0=100.0, x1=400.0, y1=114.0),
                    position=0.13,
                )
                for page in range(1, last_page + 1)
            ],
        )

    @staticmethod
    def windowed(echo: PageEcho, config: SectionReconcilerConfig) -> SectionReconcilerAgent:
        """Build a reconciler cutting documents into windows of four pages.

        Args:
            echo: The offline model.
            config: The base configuration, its criterion kept.

        Returns:
            The reconciler.
        """
        return SectionReconcilerAgent(
            model=FunctionModel(echo.respond), config=config.model_copy(update={"window_pages": 4})
        )

    def test__a_long_document_is_reconciled_window_by_window_and_stitched_by_page(
        self, reconciler_config: SectionReconcilerConfig
    ) -> None:
        """Every page is reported exactly once, in page order, and every call is counted."""
        reconciler = self.windowed(PageEcho(), reconciler_config)
        usage = RunUsage()

        sections = asyncio.run(reconciler.reconcile(self.spanning(10), usage=usage))

        assert [s.start_page for s in sections] == list(range(1, 11))
        # Three windows (1-4, 5-8, 9-10), each one call, all of them counted in the one usage.
        assert usage.requests == 3

    def test__each_window_reads_what_the_previous_one_settled_and_left_open(
        self, reconciler_config: SectionReconcilerConfig
    ) -> None:
        """The windows run in order, and the second is handed the first's conventions and plan."""
        echo = PageEcho()

        asyncio.run(self.windowed(echo, reconciler_config).reconcile(self.spanning(6)))

        first, second = echo.prompts
        assert first.startswith("Window 1 of 2 of one document")
        assert "already reconciled" not in first
        assert second.startswith("Window 2 of 2 of one document")
        assert "Pages 1 to 4 are already reconciled." in second
        assert f"- {PageEcho.CONVENTION}" in second
        # A flat plan: nothing is open, but the last section is carried as the level anchor.
        assert "run on past" not in second
        assert '- L1 p4 "Heading 4"' in second
        assert second.index("already reconciled") < second.index("Page 5\n")

    def test__every_window_reads_the_same_system_prompt_with_the_criterion(
        self, reconciler_config: SectionReconcilerConfig
    ) -> None:
        """The cut never touches the instructions, which carry the project's criterion."""
        echo = PageEcho()

        asyncio.run(self.windowed(echo, reconciler_config).reconcile(self.spanning(6)))

        first, second = echo.instructions
        assert first == second
        assert first is not None
        assert reconciler_config.relevance_criterion in first

    def test__a_window_holding_no_candidate_spends_no_call(
        self, reconciler_config: SectionReconcilerConfig
    ) -> None:
        """A long document with an empty middle is two calls, not three."""
        reconciler = self.windowed(PageEcho(), reconciler_config)
        sparse = HeadingCandidates(
            outline=[], format=[c for c in self.spanning(12).format if c.page in {1, 12}]
        )
        usage = RunUsage()

        sections = asyncio.run(reconciler.reconcile(sparse, usage=usage))

        assert [s.start_page for s in sections] == [1, 12]
        assert usage.requests == 2

    def test__no_signals_short_circuits_without_the_model(
        self, make_reconciler: Callable[[list[dict[str, Any]]], SectionReconcilerAgent]
    ) -> None:
        """Return no sections, and never call the model, when nothing was detected."""
        reconciler = make_reconciler([
            {"title": "x", "start_page": 1, "level": 1, "relevance": "irrelevant"}
        ])
        usage = RunUsage()

        sections = asyncio.run(
            reconciler.reconcile(HeadingCandidates(outline=[], format=[]), usage=usage)
        )

        assert sections == []
        assert usage.requests == 0

    def test__maps_model_output_to_reconciled_sections(
        self, make_reconciler: Callable[[list[dict[str, Any]]], SectionReconcilerAgent]
    ) -> None:
        """Return the model's structured sections when a signal is present."""
        reconciler = make_reconciler([
            {"title": "Intro", "start_page": 3, "level": 1, "relevance": "irrelevant"}
        ])
        headings = self.spanning(3).between(3, 3)

        result = asyncio.run(reconciler.reconcile(headings))

        assert result == [
            Section(title="Intro", start_page=3, level=1, relevance=SectionRelevance.IRRELEVANT)
        ]
