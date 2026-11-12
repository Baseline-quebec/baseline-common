"""ExtractionService — what the sections of a document name, in layers that each cost little."""

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import Annotated

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from baseline_core.core.llm.document_extraction.extraction.image_budget import ImageBudget
from baseline_core.core.llm.document_extraction.extraction.mention_harvester import (
    MentionHarvester,
)
from baseline_core.core.llm.document_extraction.extraction.page_flags import PageFlags
from baseline_core.core.llm.document_extraction.extraction.section_entities.agent import (
    SectionEntitiesAgent,
)
from baseline_core.core.llm.document_extraction.extraction.section_vision.agent import (
    SectionVisionAgent,
)
from baseline_core.core.llm.document_extraction.extraction.vision_router import VisionRouter
from baseline_core.core.llm.document_extraction.layout.layout import (
    LayoutDocument,
    LayoutPage,
    PageRegion,
)
from baseline_core.core.llm.document_extraction.layout.page_renderer import PageRenderer
from baseline_core.core.llm.document_extraction.section import (
    LocatedSection,
    ResolvedSection,
    SectionRelevance,
)
from baseline_core.core.llm.document_extraction.section_entities import SectionEntities


class ExtractionConfig(BaseModel):
    """How many sections are read at once, and which pixels the vision reading may spend."""

    model_config = ConfigDict(frozen=True)

    max_concurrent_sections: Annotated[
        int,
        Field(
            gt=0,
            description="Sections of one document read at once. Unbounded, a 100-section "
            "document fans out 100 calls at the same instant, and a refusal the HTTP retry "
            "window cannot absorb empties a section.",
        ),
    ] = 16
    vision_image_share: Annotated[
        float,
        Field(ge=0, le=1, description="Raster coverage above which a page is read as pixels."),
    ] = 0.30
    vision_page_cap: Annotated[
        int,
        Field(
            gt=0,
            description="Image-heavy pages one section may send to vision; the cap guards "
            "against a runaway attachment.",
        ),
    ] = 36
    vision_max_side: Annotated[
        int,
        Field(
            gt=0,
            description="Longest side, in pixels, an image is shrunk to: 1 600 keeps a letter "
            "page legible at about 2 500 tokens (a drawing sheet at 150 dpi ran past 5 000).",
        ),
    ] = 1600
    vision_batch_tokens: Annotated[
        int, Field(gt=0, description="Image tokens one vision call may carry.")
    ] = 20_000
    min_text: Annotated[
        int,
        Field(
            ge=0,
            description="Characters of text below which a section has no text layer to bound "
            "its vision reading by: a scan's few characters harvest nothing.",
        ),
    ] = 200


@dataclass
class ExtractionRun:
    """The state of one document's extraction: its pages, its bytes, and what was done."""

    pages: dict[int, LayoutPage]
    content: bytes
    cells: bool
    """Whether the document is a workbook, read as text alone."""
    gate: asyncio.Semaphore
    counts: Counter[str] = field(default_factory=Counter)


class ExtractionService[Out: SectionEntities]:
    """Reads sections in layers: code harvests, a model reads text then pixels, code decides."""

    def __init__(
        self,
        reader: SectionEntitiesAgent[Out],
        seer: SectionVisionAgent[Out],
        harvester: MentionHarvester,
        config: ExtractionConfig,
    ) -> None:
        """Bind the readers, the project's formats and the knobs.

        Args:
            reader: Agent reading a section off its text.
            seer: Agent reading a section off its rendered image-heavy pages.
            harvester: Finder of the values the project's formats recognise.
            config: Concurrency and the vision reading's pixel budget.
        """
        self._reader = reader
        self._seer = seer
        self._harvester = harvester
        self._config = config
        self._budget = ImageBudget(config.vision_max_side, config.vision_batch_tokens)

    async def extract(
        self, sections: list[LocatedSection], document: LayoutDocument, content: bytes
    ) -> list[ResolvedSection]:
        """Read what every section names.

        Args:
            sections: The indexed sections, each placed on the pages it runs across.
            document: The parsed tree, whose page geometry the bands are cut from.
            content: The document's bytes, rendered band by band where pixels are needed.

        Returns:
            The same sections in the same order, each with what it names; irrelevant ones empty.
        """
        routes = VisionRouter.route(sections, document)
        run = ExtractionRun(
            pages={page.number: page for page in document.pages},
            content=content,
            cells=document.origin == "cells",
            gate=asyncio.Semaphore(self._config.max_concurrent_sections),
        )
        resolved = list(
            await asyncio.gather(
                *(
                    self._resolve(section, regions, run)
                    for section, regions in zip(sections, routes, strict=True)
                )
            )
        )
        image_only = sum(1 for page in document.pages if page.images and not page.lines)

        # A section opening on such a page has no heading to place it, so it is never read.
        if image_only:
            run.counts["image-only pages"] = image_only

        if run.counts:
            logger.info("extraction: {}", dict(run.counts))

        return resolved

    async def _resolve(
        self, section: LocatedSection, regions: list[PageRegion], run: ExtractionRun
    ) -> ResolvedSection:
        """Read one section through every layer, or keep it unread when nothing earns a call.

        Args:
            section: The located section.
            regions: Its bands, one per page it runs across; empty when unplaced or irrelevant.
            run: The document's extraction state.

        Returns:
            The section and what it names, as one record.
        """
        placed = bool(section.body) if run.cells else bool(regions)

        if section.relevance is SectionRelevance.IRRELEVANT or not placed:
            return ResolvedSection.model_validate(section.model_dump())

        async with run.gate:
            return await self._read(section, regions, run)

    async def _read(
        self, section: LocatedSection, regions: list[PageRegion], run: ExtractionRun
    ) -> ResolvedSection:
        """Harvest, read the text, read the image-heavy pages, and merge the readings.

        Args:
            section: The located section, worth a call.
            regions: Its bands, one per page it runs across; empty for a sheet.
            run: The document's extraction state.

        Returns:
            The section and what it names, or kept empty with why its reading failed.
        """
        printed = f"{section.printed_heading}\n{section.text}"
        candidates = self._harvester.harvest(printed)

        try:
            reading = await self._reader.extract(section, candidates)
        except (UnexpectedModelBehavior, ModelHTTPError) as error:
            logger.warning("section {!r} unreadable: {}", section.title[:80], error)
            return ResolvedSection.model_validate({
                **section.model_dump(),
                "extraction_error": f"{type(error).__name__}: {error}",
            })

        restricted = self._harvester.restrict(reading, candidates)
        entities = self._harvester.drop_unprinted(restricted, printed)
        run.counts["mentions outside harvest"] += len(reading.mentions) - len(restricted.mentions)
        run.counts["mentions unprinted"] += len(restricted.mentions) - len(entities.mentions)

        # A scan's few characters harvest nothing, so they bound none of its vision codes.
        bounded = len(printed.strip()) >= self._config.min_text
        errors: list[str] = []

        for vision in await self._see(section, regions, run, errors):
            held = self._harvester.restrict(vision, candidates) if bounded else vision
            dropped = len(vision.mentions) - len(held.mentions)
            run.counts["vision mentions outside harvest"] += dropped
            entities = entities.merged_with(held)

        return ResolvedSection.model_validate({
            **section.model_dump(),
            "entities": entities,
            "extraction_error": "; ".join(errors) or None,
        })

    async def _see(
        self,
        section: LocatedSection,
        regions: list[PageRegion],
        run: ExtractionRun,
        errors: list[str],
    ) -> list[Out]:
        """Read the section's image-heavy bands, in calls of a token budget.

        Args:
            section: The located section.
            regions: Its bands; only the image-heavy ones are rendered.
            run: The document's extraction state.
            errors: Collects why each failed vision call failed.

        Returns:
            One reading per call that came back; none when no band is image-heavy.
        """
        share = self._config.vision_image_share
        heavy = [r for r in regions if PageFlags.is_image_heavy(run.pages[r.page], share)]
        cap = self._config.vision_page_cap

        if len(heavy) > cap:
            logger.warning(
                "vision: section {!r} has {} image-heavy pages; reading the first {}",
                section.title[:80],
                len(heavy),
                cap,
            )
            heavy = heavy[:cap]

        if not heavy:
            return []

        rendered = PageRenderer.render_regions(run.content, heavy, dpi=self._seer.dpi)
        readings: list[Out] = []

        for group in self._budget.batches(rendered):
            try:
                readings.append(await self._seer.extract(section.printed_heading, group))
            except (UnexpectedModelBehavior, ModelHTTPError) as error:
                logger.warning("vision of {!r} unreadable: {}", section.title[:80], error)
                errors.append(f"{type(error).__name__}: {error}")
                continue

            run.counts["vision pages"] += len(group)

        return readings
