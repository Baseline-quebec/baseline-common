"""SectionRepairAgent — settles a boundary the deterministic detector could not."""

import json

from pydantic_ai import Agent
from pydantic_ai.models import Model

from baseline_core.core.llm.document_extraction.classification.section_repair.config import (
    SectionRepairConfig,
)
from baseline_core.core.llm.document_extraction.classification.section_repair.merge_candidate_detector import (
    Evidence,
)
from baseline_core.core.llm.document_extraction.classification.section_repair.output import (
    BoundaryOpinion,
)
from baseline_core.core.llm.document_extraction.classification.section_repair.prompt import (
    SectionRepairPrompt,
)
from baseline_core.core.llm.document_extraction.section import ResolvedSection
from baseline_core.core.llm.document_extraction.section_entities import Mention, MentionRole


class SectionRepairAgent(Agent[None, BoundaryOpinion]):
    """Judges one boundary between two consecutive sections: merge or separate."""

    def __init__(self, model: Model, *, config: SectionRepairConfig) -> None:
        """Initialize the agent.

        Args:
            model: The LLM model to use.
            config: The identity kinds, and how wide the boundary window is.
        """
        super().__init__(
            model=model, output_type=BoundaryOpinion, instructions=SectionRepairPrompt.SYSTEM
        )
        self._config = config

    async def judge(
        self, previous: ResolvedSection, current: ResolvedSection, evidence: list[Evidence]
    ) -> BoundaryOpinion:
        """Judge one boundary.

        Args:
            previous: The section before the boundary, whole — the one that might absorb.
            current: The section after it, whole — whose independence is in question.
            evidence: Why the detector brought this boundary — reasons to look, not verdicts.

        Returns:
            The opinion.
        """
        result = await self.run(self._user_prompt(previous, current, evidence))
        return result.output

    def _user_prompt(
        self, previous: ResolvedSection, current: ResolvedSection, evidence: list[Evidence]
    ) -> str:
        """Render the boundary from the sections the caller handed over.

        Args:
            previous: The section before the boundary, whole.
            current: The section after it, whole.
            evidence: The detector's clues.

        Returns:
            The user prompt: the clues first, then each side.
        """
        clues = ", ".join(item.value for item in evidence)
        first = self._side("first_section", previous, self._tail(previous.text))
        second = self._side("second_section", current, self._head(current.body))
        return f"<clues>{clues}</clues>\n\n{first}\n\n{second}"

    def _tail(self, text: str) -> str:
        """Keep the end of the first section's text, where a split would show.

        Args:
            text: The whole text of the first section.

        Returns:
            Its last ``boundary_chars``, opened by an ellipsis when cut.
        """
        width = self._config.boundary_chars
        tail = text[-width:].strip()
        return f"…{tail}" if len(text) > width else tail

    def _head(self, body: str) -> str:
        """Keep the start of the second section's body, where a split would show.

        Args:
            body: The second section's text, its own heading already stripped.

        Returns:
            Its first ``boundary_chars``, closed by an ellipsis when cut.
        """
        width = self._config.boundary_chars
        head = body[:width].strip()
        return f"{head}…" if len(body) > width else head

    def _side(self, tag: str, section: ResolvedSection, text: str) -> str:
        """Render one side of the boundary, omitting what the extractor did not fill.

        Args:
            tag: The XML tag for this side.
            section: The resolved section — title, entities and text.
            text: The window cut from its text.

        Returns:
            The rendered block.
        """
        entities = section.entities
        mentions = entities.mentions

        # Everything the detector weighed is shown, or judging its clues is guesswork.
        stated = {
            "carries": self._printed([item for item in mentions if self._config.identifies(item)]),
            "points_at": self._printed([
                item for item in mentions if item.role is MentionRole.CITED
            ]),
            "names": self._printed([
                item
                for item in mentions
                if item.role is MentionRole.OWN and not self._config.identifies(item)
            ]),
            "dated": entities.dated.isoformat() if entities.dated else "",
        }

        # ``level`` stays out: the step that set it made the split, so it would vouch for itself.
        attributes = f'title="{self._quoted(section.title)}"'
        attributes += "".join(
            f' {name}="{self._quoted(value)}"' for name, value in stated.items() if value
        )

        facts = entities.model_dump(
            mode="json", exclude={"mentions", "dated"}, exclude_defaults=True
        )
        block = f"<facts>{json.dumps(facts, ensure_ascii=False)}</facts>\n" if facts else ""
        return f"<{tag} {attributes}>\n{block}{text}\n</{tag}>"

    @staticmethod
    def _printed(mentions: list[Mention]) -> str:
        """Kind and value of each mention, comma-separated.

        Args:
            mentions: The mentions of one attribute.

        Returns:
            The mentions as printed, empty when there are none.
        """
        return ", ".join(f"{item.kind} {item.value}" for item in mentions)

    @staticmethod
    def _quoted(value: str) -> str:
        """One attribute value safe to sit inside double quotes.

        Args:
            value: The text as the extractor produced it.

        Returns:
            The same text, its own double quotes turned to single ones.
        """
        return value.replace('"', "'")
