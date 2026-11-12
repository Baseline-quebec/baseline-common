"""ClassificationService: what each section is, relevant or not, a section or a piece of one."""

import asyncio

from loguru import logger
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from baseline_core.core.llm.document_extraction.classification.section_relevance.agent import (
    SectionRelevanceAgent,
)
from baseline_core.core.llm.document_extraction.classification.section_repair.agent import (
    SectionRepairAgent,
)
from baseline_core.core.llm.document_extraction.classification.section_repair.merge_candidate_detector import (
    MergeCandidateDetector,
)
from baseline_core.core.llm.document_extraction.classification.section_repair.output import (
    BoundaryDecision,
)
from baseline_core.core.llm.document_extraction.section import (
    LocatedSection,
    ResolvedSection,
    SectionRelevance,
)


class ClassificationService:
    """Settles the relevance the plan left uncertain, and folds back the sections split in two."""

    def __init__(
        self,
        relevance: SectionRelevanceAgent,
        repair: SectionRepairAgent,
        detector: MergeCandidateDetector,
    ) -> None:
        """Bind the relevance judge, the boundary judge and the gate in front of it.

        Args:
            relevance: The agent settling an uncertain section on the text under its heading.
            repair: The agent judging one boundary at a time.
            detector: The deterministic gate; most boundaries never reach the boundary agent.
        """
        self._relevance = relevance
        self._repair = repair
        self._detector = detector

    async def settle(self, sections: list[LocatedSection]) -> list[LocatedSection]:
        """Settle every section the plan left uncertain, before anything is read from it.

        Args:
            sections: The located sections of one document.

        Returns:
            The sections, each uncertain one with the relevance its text settled on.
        """
        return list(await asyncio.gather(*(self._settled(section) for section in sections)))

    async def repair(self, sections: list[ResolvedSection]) -> list[ResolvedSection]:
        """Fold every section the detector and the agent agree was split off the one above it.

        Args:
            sections: The resolved sections of one document, in printed order.

        Returns:
            The sections with each false split undone; the same sections when none was found.
        """
        if not sections:
            return []

        kept = [sections[0]]

        for trailing in sections[1:]:
            # ``kept[-1]`` may already be a merge, so « S, W1, W2 » chains through S's keys.
            absorbed = await self._absorbed(kept[-1], trailing)

            if absorbed is None:
                kept.append(trailing)

            else:
                kept[-1] = absorbed

        return kept

    async def _settled(self, section: LocatedSection) -> LocatedSection:
        """Settle a section the reconciler left uncertain, on the text under its heading.

        Args:
            section: One located section.

        Returns:
            The section with the relevance its text settled on; as is when already settled.
        """
        if section.relevance is not SectionRelevance.UNCERTAIN:
            return section

        try:
            settled = await self._relevance.settle(section)

        except (UnexpectedModelBehavior, ModelHTTPError) as error:
            logger.warning("relevance of {!r} unsettled: {}", section.title[:80], error)
            return section

        return section.model_copy(update={"relevance": settled})

    async def _absorbed(
        self, absorber: ResolvedSection, trailing: ResolvedSection
    ) -> ResolvedSection | None:
        """The two sections folded into one, or None when they stay two.

        Args:
            absorber: The section standing before the boundary, merges already folded in.
            trailing: The section printed just below it.

        Returns:
            The merged section, else None: no clue, or the agent kept them apart.
        """
        evidence = self._detector.evidence_for(absorber, trailing)

        if not evidence:
            return None

        try:
            opinion = await self._repair.judge(absorber, trailing, evidence)

        except (UnexpectedModelBehavior, ModelHTTPError) as error:
            logger.warning("boundary before {!r} unjudged: {}", trailing.title[:80], error)
            return None

        if opinion.decision is not BoundaryDecision.MERGE:
            return None

        return self._folded(absorber, trailing)

    @staticmethod
    def _folded(absorber: ResolvedSection, trailing: ResolvedSection) -> ResolvedSection:
        """Fold a trailing section into the one above it, the false split undone.

        Args:
            absorber: The section keeping its heading, place and depth.
            trailing: The section printed just below, judged to be part of it.

        Returns:
            A new section: both texts, every file either points at, everything either names.
        """
        # The false heading stays in the text, where the page printed it.
        text = f"{absorber.text.rstrip()}\n\n{trailing.printed_heading}\n{trailing.body}".strip()
        targets = {reference.target for reference in absorber.file_references}
        references = [
            *absorber.file_references,
            *[item for item in trailing.file_references if item.target not in targets],
        ]

        return absorber.model_copy(
            update={
                "text": text,
                "file_references": references,
                "entities": absorber.entities.merged_with(trailing.entities),
                "extraction_error": absorber.extraction_error or trailing.extraction_error,
            }
        )
