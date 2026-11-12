"""Tests of SectionRelevanceAgent: the windows it cuts, and when it stops reading."""

import asyncio
from typing import ClassVar

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from baseline_core.core.llm.document_extraction.classification.section_relevance.agent import (
    SectionRelevanceAgent,
)
from baseline_core.core.llm.document_extraction.classification.section_relevance.config import (
    SectionRelevanceConfig,
)
from baseline_core.core.llm.document_extraction.section import LocatedSection, SectionRelevance


class RecordingModel:
    """Builds an offline model answering verdicts in order, recording each message it reads."""

    def __init__(self, verdicts: list[str]) -> None:
        """Bind the answers, first run first.

        Args:
            verdicts: The relevance each successive run returns.
        """
        self._verdicts = iter(verdicts)
        self.seen: list[str] = []

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Record the window the agent sent and answer the next verdict.

        Args:
            messages: The conversation so far.
            info: The agent's output tools.

        Returns:
            The next verdict, as the output tool's call.
        """
        request = next(item for item in reversed(messages) if isinstance(item, ModelRequest))
        prompt = next(part for part in request.parts if isinstance(part, UserPromptPart))
        self.seen.append(str(prompt.content))
        verdict = next(self._verdicts)
        return ModelResponse(
            parts=[ToolCallPart(tool_name=info.output_tools[0].name, args={"response": verdict})]
        )


class TestSectionRelevanceAgent:
    CONFIG: ClassVar[SectionRelevanceConfig] = SectionRelevanceConfig(
        relevance_criterion="A repair of equipment at an identified site.",
        window_chars=20,
        overlap_chars=5,
        max_windows=3,
    )
    """Small windows, so a two-line section already spans more than ``max_windows``."""

    BODY: ClassVar[str] = (
        "PUMP REPLACEMENT - 120 HARBOUR ROAD. "
        "The board approves the replacement of the cooling pump of line 2. "
        "All in accordance with the maintenance committee's report."
    )

    def agent(self, model: RecordingModel) -> SectionRelevanceAgent:
        """Build an agent on a recording model.

        Args:
            model: The offline model answering and recording.

        Returns:
            The agent.
        """
        return SectionRelevanceAgent(model=FunctionModel(model.respond), config=self.CONFIG)

    def located(self, text: str) -> LocatedSection:
        """Build an uncertain located section carrying ``text``.

        Args:
            text: The text printed under its heading.

        Returns:
            The section the fallback is asked to settle.
        """
        return LocatedSection(
            title="PUMP REPLACEMENT",
            section_number="4.2",
            start_page=4,
            level=2,
            relevance=SectionRelevance.UNCERTAIN,
            text=text,
        )

    def test__stops_at_the_first_window_that_settles_the_section(self) -> None:
        """A settled verdict is returned at once, leaving the remaining windows unread."""
        model = RecordingModel(["uncertain", "relevant"])

        verdict = asyncio.run(self.agent(model).settle(self.located(self.BODY)))

        assert verdict is SectionRelevance.RELEVANT
        assert len(model.seen) == 2

    def test__an_irrelevant_window_settles_the_section_too(self) -> None:
        """Irrelevant is a verdict, so it stops the reading as relevant does."""
        model = RecordingModel(["irrelevant"])

        verdict = asyncio.run(self.agent(model).settle(self.located(self.BODY)))

        assert verdict is SectionRelevance.IRRELEVANT
        assert len(model.seen) == 1

    def test__reads_no_more_than_max_windows_before_staying_uncertain(self) -> None:
        """The window stops sliding at ``max_windows``, and the section stays uncertain."""
        model = RecordingModel(["uncertain"] * self.CONFIG.max_windows)

        verdict = asyncio.run(self.agent(model).settle(self.located(self.BODY)))

        assert verdict is SectionRelevance.UNCERTAIN
        assert len(model.seen) == self.CONFIG.max_windows

    def test__each_window_carries_the_printed_heading_and_moves_down_the_text(self) -> None:
        """Successive windows show the heading, its opener restored, over advancing text."""
        model = RecordingModel(["uncertain"] * self.CONFIG.max_windows)

        asyncio.run(self.agent(model).settle(self.located(self.BODY)))

        assert all('"4.2 PUMP REPLACEMENT"' in prompt for prompt in model.seen)
        assert len(set(model.seen)) == len(model.seen)

    def test__a_section_with_no_text_is_never_sent_to_the_model(self) -> None:
        """An unplaced section stays uncertain without billing a call."""
        model = RecordingModel([])

        verdict = asyncio.run(self.agent(model).settle(self.located("")))

        assert verdict is SectionRelevance.UNCERTAIN
        assert model.seen == []

    def test__the_stub_settles_an_uncertain_section_as_relevant(
        self, stub_relevance: SectionRelevanceAgent
    ) -> None:
        """The stub other suites chain behind the reconciler answers relevant on any text."""
        verdict = asyncio.run(stub_relevance.settle(self.located(self.BODY)))

        assert verdict is SectionRelevance.RELEVANT


class TestWindows:
    TEXT: ClassVar[str] = "abcdefghij"
    """Ten characters, so every window boundary below is countable by hand."""

    def agent(self, size: int, overlap: int, limit: int) -> SectionRelevanceAgent:
        """Build an agent whose only job here is to cut windows.

        Args:
            size: Characters one window carries.
            overlap: Characters the next window re-reads.
            limit: Windows read at most.

        Returns:
            The agent, on an offline model no test below ever calls.
        """
        return SectionRelevanceAgent(
            model=TestModel(),
            config=SectionRelevanceConfig(
                relevance_criterion="Anything.",
                window_chars=size,
                overlap_chars=overlap,
                max_windows=limit,
            ),
        )

    def test__windows_overlap_by_the_asked_characters(self) -> None:
        """Each window re-reads the tail of the one before it, and the last one ends the text."""
        assert self.agent(4, 1, 4)._windows(self.TEXT) == ["abcd", "defg", "ghij"]

    def test__windows_stop_at_the_limit(self) -> None:
        """The limit caps how far the window slides, whatever the text is worth."""
        assert self.agent(4, 1, 2)._windows(self.TEXT) == ["abcd", "defg"]

    def test__a_text_that_fits_one_window_is_read_once(self) -> None:
        """No sliver of the tail: a second window would only re-read what window 1 covered."""
        assert self.agent(12, 3, 4)._windows(self.TEXT) == ["abcdefghij"]

    def test__the_last_window_reaches_the_end_without_repeating_it(self) -> None:
        """A window ending on the last character closes the section, however short it fell."""
        assert self.agent(8, 3, 4)._windows(self.TEXT) == ["abcdefgh", "fghij"]

    def test__a_section_without_text_has_no_window(self) -> None:
        """A section the locator left unplaced offers nothing to read."""
        assert self.agent(4, 1, 2)._windows("") == []
