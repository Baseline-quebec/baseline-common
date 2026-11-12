"""SectionReconcilerAgent — reconciles the outline and the format candidates into sections."""

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage

from baseline_core.core.llm.document_extraction.indexing.detection.candidates import (
    HeadingCandidates,
)
from baseline_core.core.llm.document_extraction.indexing.section_reconciler.config import (
    SectionReconcilerConfig,
)
from baseline_core.core.llm.document_extraction.indexing.section_reconciler.output import (
    TableOfContent,
)
from baseline_core.core.llm.document_extraction.indexing.section_reconciler.page_window import (
    PageWindow,
)
from baseline_core.core.llm.document_extraction.indexing.section_reconciler.prompt import (
    SectionReconcilerPrompt,
)
from baseline_core.core.llm.document_extraction.indexing.section_reconciler.window_carry import (
    WindowCarry,
)
from baseline_core.core.llm.document_extraction.section import Section


class SectionReconcilerAgent(Agent[None, TableOfContent]):
    """Reconciles per-page outline and format signals into sections, one page window at a time."""

    def __init__(self, model: Model, config: SectionReconcilerConfig) -> None:
        """Bind the model and the prompt the project's configuration renders.

        Args:
            model: The LLM model to use.
            config: The relevance criterion, the outline clipping, the window size.
        """
        prompt = SectionReconcilerPrompt(config)
        super().__init__(model=model, output_type=TableOfContent, instructions=prompt.system)
        self._config = config
        self._prompt = prompt

    async def reconcile(
        self, headings: HeadingCandidates, usage: RunUsage | None = None
    ) -> list[Section]:
        """Reconcile the detected candidates into sections, one call per window of pages.

        Args:
            headings: The outline entries and format candidates of one document.
            usage: Where every call of this reconciliation adds what it spent, when asked.

        Returns:
            The sections the model kept, window after window, each on a page its window owns.
        """
        windows = [
            window
            for window in PageWindow.tile(headings, self._config.window_pages)
            if not window.headings.empty
        ]
        kept: list[Section] = []
        carry: WindowCarry | None = None

        # In order, never in parallel: each window reads the plan the previous one left open.
        for window in windows:
            result = await self.run(self._prompt.user(window, carry), usage=usage)
            kept.extend(section for section in result.output.sections if window.owns(section))
            carry = WindowCarry.from_reconciled(window.last_page, result.output.conventions, kept)

        return kept
