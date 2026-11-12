"""HeadingFinder — the typed, positioned heading candidates the section reconciler reads."""

import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from itertools import pairwise
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz

from baseline_core.core.llm.document_extraction.indexing.detection.candidates import (
    FormatCandidate,
    HeadingCandidates,
    OutlineCandidate,
    SerialCode,
    SerialItem,
    StyledBlock,
    StyledText,
    Typography,
)
from baseline_core.core.llm.document_extraction.indexing.detection.config import (
    BodyProfileConfig,
    CandidateConfig,
    TextConfig,
)
from baseline_core.core.llm.document_extraction.indexing.detection.text_utils import TextUtils
from baseline_core.core.llm.document_extraction.layout.assembly.furniture_marker import (
    EdgeLine,
    HeaderFooterConfig,
    PageEdges,
)
from baseline_core.core.llm.document_extraction.layout.layout import (
    BBox,
    BodyProfile,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
    LayoutSpan,
    TextBlock,
)
from baseline_core.core.llm.document_extraction.text_search import TextSearch


class HeadingFinderConfig(BaseModel):
    """Every threshold the HeadingFinder reads with, grouped by what it governs."""

    model_config = ConfigDict(frozen=True)

    majority_ratio: Annotated[
        float, Field(description="Char-weighted share of a line/chunk that must share a trait.")
    ] = 0.6
    chunk_gap_factor: Annotated[
        float,
        Field(
            gt=0,
            description=(
                "Blank space over this many font sizes ends a chunk: a wrapped heading keeps "
                "about 1.2x, two headings sit about 2.3x apart."
            ),
        ),
    ] = 1.6
    body: Annotated[
        BodyProfileConfig, Field(description="Body-profile measurement thresholds.")
    ] = BodyProfileConfig()
    candidate: Annotated[CandidateConfig, Field(description="Candidate promotion thresholds.")] = (
        CandidateConfig()
    )
    header_footer: Annotated[
        HeaderFooterConfig, Field(description="Running header/footer thresholds.")
    ] = HeaderFooterConfig()
    text: Annotated[TextConfig, Field(description="All-caps heuristic thresholds.")] = TextConfig()


class HeadingFinder:
    """Heading detection over a parsed document's outline and styled runs, recall first."""

    SECTION_CODE_PREFIX: ClassVar[re.Pattern[str]] = re.compile(
        r"^\s*(?:"
        r"(?:chapter|section|article|part|appendix|annex)\s+[\w.\-]+"
        r"|[A-Za-z]{0,3}[.\-]?\d[\w.\-)]*"
        r"|[A-Za-z][.)]"
        r")\.?\s+\S",
        re.IGNORECASE,
    )
    """A leading section-code cue (1, 1.2, A., HG.123A, 'Chapter 3') - one soft heading signal."""

    TOC_MARKER: ClassVar[re.Pattern[str]] = re.compile(r"\.{4,}")
    """A run of dot leaders — a table-of-contents / index entry marker."""

    COMPOUND_CODE: ClassVar[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9]{1,4}\.[A-Za-z0-9]")
    """A compound-dotted section number (3.1, A.2, 10.2A) — a strong section signal."""

    NOISE: ClassVar[re.Pattern[str]] = re.compile(r"^[\d\s.,$%()/-]+$")
    """A line of only digits, currency, and separators — a folio or table amount, never a title."""

    IDENTITY_CODE: ClassVar[re.Pattern[str]] = re.compile(r"\d{5,}")
    """A five-plus digit run: with a word beside it, a file identifier no date or scale carries."""

    BOOKMARK_WORD: ClassVar[re.Pattern[str]] = re.compile(r"[a-z]{3,}")
    """A word of a simplified bookmark title, its codes and enumerators left aside."""

    def __init__(self, config: HeadingFinderConfig) -> None:
        """Bind the detection thresholds every document is read with.

        Args:
            config: Tunable typographic thresholds.
        """
        self._config = config

    def scan(self, document: LayoutDocument) -> HeadingCandidates:
        """Surface both heading sources of one parsed document.

        Args:
            document: The parsed document tree.

        Returns:
            The outline entries and the format-detected candidates, each positioned.
        """
        return HeadingCandidates(
            outline=self._outline_candidates(document),
            format=self._heading_candidates(self._drop_edge_furniture(document)),
        )

    def serial_items(self, candidates: list[FormatCandidate]) -> list[SerialItem]:
        """The candidates that open an item of a counting series, with the item's code.

        Args:
            candidates: The format candidates of one document, after every filter.

        Returns:
            One item for each code of a family that counts up, first printing only.
        """
        openers = [
            (candidate, code)
            for candidate in candidates
            if (code := SerialCode.opening(candidate.text)) is not None
        ]
        families = self._counting_families([code for _, code in openers])
        seen: set[str] = set()
        items = []

        for candidate, code in openers:
            if code.family in families and code.code not in seen:
                seen.add(code.code)
                items.append(SerialItem(code=code.code, candidate=candidate))

        return items

    def _edges(self, page: LayoutPage) -> PageEdges | None:
        """Reduce one page to its topmost and its bottommost line.

        Args:
            page: The parsed page.

        Returns:
            Its two edge lines, or None when the page carries no text.
        """
        lines = page.lines

        if not lines:
            return None

        # Ties broken on the text, so two pages sharing a baseline resolve to the same edge.
        ordered = sorted(lines, key=lambda line: (line.bbox.y0, line.text))
        bins = self._config.header_footer.bins
        return PageEdges(
            top=self._edge(ordered[0], page.height, bins),
            bottom=self._edge(ordered[-1], page.height, bins),
        )

    def _drop_edge_furniture(self, document: LayoutDocument) -> LayoutDocument:
        """Remove each page's recurring header/footer lines, baseline mates included.

        Args:
            document: The parsed document.

        Returns:
            The same document with the edge-furniture lines removed from every page.
        """
        # Unlike the parser's verdict on the edge line alone, every line in a furniture band
        # goes: a header printed as two runs on one baseline goes whole.
        config = self._config.header_footer
        edges = [self._edges(page) for page in document.pages]
        pages = []

        for index, page in enumerate(document.pages):
            here = edges[index]
            neighbours = self._neighbours(edges, index, config.window)

            if here is None or not neighbours:
                pages.append(page)
                continue

            floor = max(1.0, config.recurrence * len(neighbours))
            bands = {
                mine.band
                for mine, theirs in [
                    (here.top, [edge.top for edge in neighbours]),
                    (here.bottom, [edge.bottom for edge in neighbours]),
                ]
                if sum(mine.matches(other, config.tolerance) for other in theirs) >= floor
            }
            pages.append(self._page_sans_bands(page, bands, config.bins, config.tolerance))

        return document.model_copy(update={"pages": pages})

    @staticmethod
    def _page_sans_bands(
        page: LayoutPage, bands: set[int], bins: int, tolerance: int
    ) -> LayoutPage:
        """Strip one page of the lines sitting in the given furniture bands.

        Args:
            page: The parsed page.
            bands: The vertical bands proven to carry edge furniture.
            bins: Vertical quantization of the page.
            tolerance: Band slack allowed around a furniture band.

        Returns:
            The page without those lines; unchanged when no band is given.
        """
        if not bands:
            return page

        blocks = [
            TextBlock(lines=kept)
            for block in page.blocks
            if (
                kept := [
                    line
                    for line in block.lines
                    if all(
                        abs(int(line.bbox.y0 / page.height * bins) - band) > tolerance
                        for band in bands
                    )
                ]
            )
        ]
        return page.model_copy(update={"blocks": blocks})

    @staticmethod
    def _edge(line: LayoutLine, height: float, bins: int) -> EdgeLine:
        """Reduce one line to the text and band another page compares against.

        Args:
            line: The line at one edge of its page.
            height: Height of that page, the band is measured against it.
            bins: Vertical quantization of the page.

        Returns:
            The comparable edge line.
        """
        return EdgeLine(key=line.furniture_key, band=int(line.bbox.y0 / height * bins))

    @staticmethod
    def _neighbours(edges: list[PageEdges | None], index: int, window: int) -> list[PageEdges]:
        """Collect the edges of the pages around one page, that page itself excluded.

        Args:
            edges: The edges of every page, in page order; None on a page with no text.
            index: 0-based index of the page under test.
            window: Pages examined on each side of it.

        Returns:
            The edges of the surrounding pages that carry text.
        """
        before = edges[max(0, index - window) : index]
        after = edges[index + 1 : index + 1 + window]
        return [edge for edge in [*before, *after] if edge is not None]

    @staticmethod
    def _outline_candidates(document: LayoutDocument) -> list[OutlineCandidate]:
        """Read the embedded outline: one bookmark's title and the page it points at.

        Args:
            document: The parsed document.

        Returns:
            One entry per bookmark that lands on a page, in outline order.
        """
        return [
            OutlineCandidate(text=entry.title, page=entry.page)
            for entry in document.outline
            if entry.page is not None
        ]

    def _majority(self, spans: list[LayoutSpan], trait: Callable[[LayoutSpan], bool]) -> bool:
        """Return whether ``trait`` holds over a char-weighted majority of ``spans``.

        Args:
            spans: The runs of one line or chunk.
            trait: A per-run predicate over the style the document declares.

        Returns:
            True when the traited share clears ``config.majority_ratio``.
        """
        total = sum(len(span.text) for span in spans)

        if total == 0:
            return False

        hit = sum(len(span.text) for span in spans if trait(span))
        return hit / total >= self._config.majority_ratio

    @staticmethod
    def _is_duplicate(line: LayoutLine, seen: set[tuple[int, int, str]]) -> bool:
        """Tell whether this line repeats one already seen at the same position.

        Args:
            line: A parsed line.
            seen: Positions already emitted on this page; mutated in place.

        Returns:
            True when the line has been seen before (and should be skipped).
        """
        key = (round(line.bbox.x0), round(line.bbox.y0), line.text)

        if key in seen:
            return True

        seen.add(key)
        return False

    def _body_profile(self, document: LayoutDocument) -> BodyProfile:
        """Measure the document's body style, at this finder's thresholds.

        Args:
            document: The parsed document.

        Returns:
            The body size and whether the body is mostly bold.
        """
        return document.body_profile(
            sample_pages=self._config.body.sample_pages,
            bold_ratio=self._config.body.bold_ratio,
            default_size=self._config.body.default_size,
        )

    def _hard_format_deviates(self, style: StyledText, body: BodyProfile) -> bool:
        """Tell whether a run carries a bold, larger or underlined deviation from the body.

        Args:
            style: The run under test (line or chunk).
            body: The body profile to compare against.

        Returns:
            True when the run is bold, larger or underlined relative to the body.
        """
        typography = style.typography
        larger = typography.size > body.size + self._config.candidate.size_tolerance
        bold_signal = typography.bold and not body.bold
        return larger or bold_signal or typography.underlined

    def _format_deviates(self, style: StyledText, body: BodyProfile) -> bool:
        """Tell whether a run's FORMAT is unlike the body — the code cue excluded.

        Args:
            style: The run under test (line or chunk).
            body: The body profile to compare against.

        Returns:
            True on a hard format deviation or an all-caps run.
        """
        return self._hard_format_deviates(style, body) or self._is_caps(style.text)

    def _deviates(self, style: StyledText, body: BodyProfile) -> bool:
        """Tell whether a run is a candidate: format, a reference code, or a TOC marker.

        Args:
            style: The run under test (line or chunk).
            body: The body profile to compare against.

        Returns:
            True on a format deviation, a leading alphanumeric code, or a TOC marker.
        """
        return (
            self._format_deviates(style, body)
            or self._has_section_code(style.text)
            or bool(self.TOC_MARKER.search(style.text))
        )

    def _has_section_code(self, text: str) -> bool:
        """Whether the run opens with a distinctive section code.

        Args:
            text: The run's text.

        Returns:
            True when the run opens with a distinctive section code.
        """
        if not self.SECTION_CODE_PREFIX.match(text):
            return False

        token = next(iter(text.split()), "")
        alphanumeric = any(c.isalpha() for c in token) and any(c.isdigit() for c in token)
        return alphanumeric or bool(self.COMPOUND_CODE.match(token))

    def _styled_blocks(self, document: LayoutDocument) -> Iterator[StyledBlock]:
        """Yield paragraph chunks, merging contiguous lines into one located unit.

        Args:
            document: The parsed document.

        Yields:
            One located chunk per merged run of lines.
        """
        body = self._body_profile(document)
        bookmarks = self._bookmark_words_by_page(document)

        for page in document.pages:
            yield from self._merged_blocks(page, body, bookmarks.get(page.number, []))

    def _merged_blocks(
        self, page: LayoutPage, body: BodyProfile, named: list[set[str]]
    ) -> Iterator[StyledBlock]:
        """Merge the lines of one page's blocks into the located runs of text they form.

        Args:
            page: ONE page, its blocks in the document's own order.
            body: The body profile a style change is judged against.
            named: The words of every bookmark pointing at this page.

        Yields:
            One located chunk per merged run of lines; a block boundary always ends a run.
        """
        seen: set[tuple[int, int, str]] = set()

        for block in page.blocks:
            chunk: list[LayoutLine] = []

            for line in block.lines:
                # A turned line is a drawing's label or dimension, never a heading.
                if line.turned:
                    continue

                if self._is_duplicate(line, seen):
                    continue

                # A line a bookmark names word for word stands alone, never buried in a masthead.
                if chunk and (
                    self._gap_breaks(chunk[-1], line)
                    or self._breaks_chunk(
                        self._chunk_style(chunk), self._chunk_style([line]), body
                    )
                    or self._named_line(line, named)
                    or self._named_line(chunk[-1], named)
                ):
                    yield self._locate(chunk, page.number)
                    chunk = []

                chunk.append(line)

            if chunk:
                yield self._locate(chunk, page.number)

    @classmethod
    def _named_line(cls, line: LayoutLine, named: list[set[str]]) -> bool:
        """Tell whether a bookmark of the page names this very line.

        Args:
            line: A printed line.
            named: The words of every bookmark pointing at the line's page.

        Returns:
            True when the line's words are exactly some bookmark's words.
        """
        printed = set(cls.BOOKMARK_WORD.findall(TextSearch.simplified(line.text)))
        return bool(printed) and any(printed == words for words in named)

    def _chunk_style(self, lines: list[LayoutLine]) -> StyledText:
        """Read a run of contiguous lines as the one paragraph chunk they form.

        Args:
            lines: The lines of the chunk, in reading order; never empty.

        Returns:
            Their joined text, at their largest size, styled by a char-weighted majority.
        """
        spans = [span for line in lines for span in line.spans]
        text = " ".join(line.text for line in lines)
        return StyledText(
            text=text,
            typography=Typography(
                size=max(line.size for line in lines),
                bold=self._majority(spans, lambda span: span.bold),
                italic=self._majority(spans, lambda span: span.italic),
                underlined=self._majority(spans, lambda span: span.underlined),
                caps=self._is_caps(text),
            ),
        )

    def _locate(self, lines: list[LayoutLine], page: int) -> StyledBlock:
        """Place one chunk on its page, over the extent of its lines.

        Args:
            lines: The lines of the chunk, in reading order; never empty.
            page: 1-based physical page those lines sit on.

        Returns:
            The located chunk.
        """
        chunk = self._chunk_style(lines)
        return StyledBlock(
            text=chunk.text,
            typography=chunk.typography,
            page=page,
            bbox=BBox.union(line.bbox for line in lines) or lines[0].bbox,
        )

    def _gap_breaks(self, previous: LayoutLine, current: LayoutLine) -> bool:
        """Tell whether the blank space above ``current`` is too wide to continue ``previous``.

        Args:
            previous: The last line of the chunk accumulated so far.
            current: The incoming line.

        Returns:
            True when the two are set apart rather than wrapped.
        """
        gap = current.bbox.y0 - previous.bbox.y1
        return gap > self._config.chunk_gap_factor * current.size

    def _breaks_chunk(self, previous: StyledText, current: StyledText, body: BodyProfile) -> bool:
        """Tell whether ``current`` starts a new chunk rather than extending ``previous``.

        Args:
            previous: The chunk accumulated so far; never empty.
            current: The incoming line.
            body: The body profile.

        Returns:
            True on a size change, a style boundary, or bold switching on.
        """
        previous_deviates = self._format_deviates(previous, body)
        current_deviates = self._format_deviates(current, body)
        size_break = current.typography.size != previous.typography.size and not (
            previous_deviates and current_deviates
        )
        style_break = previous_deviates != current_deviates and not self._is_bare_code(
            previous.text
        )
        # Bold onset is one direction only, since legitimate wraps are uniformly bold.
        bold_onset = not previous.typography.bold and current.typography.bold
        return size_break or style_break or bold_onset

    def _is_caps(self, text: str) -> bool:
        """Tell whether a line is essentially all-caps.

        Args:
            text: Line text.

        Returns:
            True when the uppercase share clears ``TextConfig.caps_ratio``.
        """
        letters = [c for c in text if c.isalpha()]

        if len(letters) < self._config.text.min_caps_letters:
            return False

        return sum(c.isupper() for c in letters) / len(letters) >= self._config.text.caps_ratio

    @staticmethod
    def _is_bare_code(text: str) -> bool:
        """Whether ``text`` is a wordless section code awaiting its title.

        Args:
            text: The accumulated chunk text.

        Returns:
            True when the text holds a code/enumerator but no word.
        """
        return not TextUtils.has_word(text) and any(c.isdigit() for c in text)

    def _is_candidate(self, style: StyledText, body: BodyProfile) -> bool:
        """Tell whether a chunk qualifies as a format-detected heading.

        Args:
            style: The chunk under test.
            body: The body profile.

        Returns:
            True when the chunk holds a word or a section code and deviates.
        """
        text = style.text

        if len(text) < self._config.candidate.min_chars or self.NOISE.match(text):
            return False

        if not TextUtils.has_word(text) and not self.SECTION_CODE_PREFIX.match(text):
            return False

        if not self._deviates(style, body):
            return False

        return self._hard_format_deviates(style, body) or (
            len(text.split()) <= self._config.candidate.soft_max_words
        )

    def _heading_candidates(self, document: LayoutDocument) -> list[FormatCandidate]:
        """List every format-detected heading with its page, its extent and its position.

        Args:
            document: The parsed document, its edge furniture already dropped.

        Returns:
            The candidates in detection order, redundant repeats dropped.
        """
        body = self._body_profile(document)
        heights = {page.number: page.height for page in document.pages}
        found = [
            FormatCandidate.from_block(block, heights[block.page])
            for block in self._styled_blocks(document)
            if self._is_candidate(block, body)
        ]
        redundant = self._redundant_texts(c.text for c in found)
        bookmarks = self._bookmark_words_by_page(document)
        # A recurring text a bookmark of its own page names is an attachment's banner, kept.
        kept = [
            c
            for c in found
            if TextSearch.simplified(c.text) not in redundant or self._bookmarked(c, bookmarks)
        ]
        collapsed = self._collapse_near_duplicates(kept, bookmarks)
        # Added after both filters: an item code is identity, which neither may collapse.
        serial = self._serial_items(document, collapsed, heights)
        return self._drop_listing_prints(collapsed + serial)

    def _drop_listing_prints(self, candidates: list[FormatCandidate]) -> list[FormatCandidate]:
        """Drop a code family's listing: every first printing above every second one, in order.

        Args:
            candidates: Every candidate, typographic and serial.

        Returns:
            The candidates without the listing's printings.
        """
        prints: dict[str, dict[str, list[tuple[int, float, int]]]] = {}

        for index, candidate in enumerate(candidates):
            code = SerialCode.opening(candidate.text)

            if code is not None:
                printings = prints.setdefault(code.family, {}).setdefault(code.code, [])
                printings.append((candidate.page, candidate.position, index))

        dropped: set[int] = set()

        for codes in prints.values():
            repeated = [sorted(printed) for printed in codes.values() if len(printed) > 1]

            if len(repeated) < self._config.candidate.serial_min_items:
                continue

            last_first = max(printed[0][:2] for printed in repeated)
            first_second = min(printed[1][:2] for printed in repeated)

            if last_first < first_second:
                dropped.update(printed[0][2] for printed in repeated)

        return [candidate for index, candidate in enumerate(candidates) if index not in dropped]

    @classmethod
    def _bookmark_words_by_page(cls, document: LayoutDocument) -> dict[int, list[set[str]]]:
        """Read the words of every bookmark, keyed by the page it points at.

        Args:
            document: The parsed document.

        Returns:
            One word set per bookmark that lands on a page and carries a word, by 1-based page.
        """
        words: dict[int, list[set[str]]] = {}

        for entry in document.outline:
            named = set(cls.BOOKMARK_WORD.findall(TextSearch.simplified(entry.title)))

            if entry.page is not None and named:
                words.setdefault(entry.page, []).append(named)

        return words

    @classmethod
    def _bookmarked(cls, candidate: FormatCandidate, bookmarks: dict[int, list[set[str]]]) -> bool:
        """Tell whether a bookmark on the candidate's page names it.

        Args:
            candidate: The candidate under test.
            bookmarks: The words of every bookmark, by page.

        Returns:
            True when every word of some bookmark on that page is printed in the candidate.
        """
        printed = set(cls.BOOKMARK_WORD.findall(TextSearch.simplified(candidate.text)))
        return any(named <= printed for named in bookmarks.get(candidate.page, []))

    def _serial_items(
        self, document: LayoutDocument, found: list[FormatCandidate], heights: dict[int, float]
    ) -> list[FormatCandidate]:
        """Promote every LINE opening an item of a counting series, whatever its typography.

        Args:
            document: The parsed document, its edge furniture already dropped.
            found: The typographic candidates, so an item they already hold is not repeated.
            heights: Page heights, by page, for each candidate's relative position.

        Returns:
            The series lines no typographic candidate already opens with, in document order.
        """
        openers = [
            (page.number, line, code)
            for page in document.pages
            for line in page.lines
            if not line.turned and (code := SerialCode.opening(line.text)) is not None
        ]
        series = self._counting_families([code for _, _, code in openers])
        taken = {
            (candidate.page, code.code)
            for candidate in found
            if (code := SerialCode.opening(candidate.text)) is not None
        }
        promoted = []
        opened: set[str] = set()

        for number, line, code in openers:
            # An item opens once: a later line opening on its code cites it.
            if code.family not in series or (number, code.code) in taken or code.code in opened:
                continue

            opened.add(code.code)
            located = self._locate([line], number)
            promoted.append(FormatCandidate.from_block(located, heights[number]))

        return promoted

    def _counting_families(self, codes: list[SerialCode]) -> set[str]:
        """The code families whose item numbers count up, as a list's do.

        Args:
            codes: Every line-opening code.

        Returns:
            The families with enough distinct items, mostly one apart once sorted.
        """
        numbers: dict[str, set[int]] = {}

        for code in codes:
            numbers.setdefault(code.family, set()).add(code.number)

        config = self._config.candidate
        families = set()

        for family, values in numbers.items():
            ordered = sorted(values)

            if len(ordered) < config.serial_min_items:
                continue

            steps = sum(b - a == 1 for a, b in pairwise(ordered))

            if steps / (len(ordered) - 1) >= config.serial_step_share:
                families.add(family)

        return families

    def _collapse_near_duplicates(
        self, candidates: list[FormatCandidate], bookmarks: dict[int, list[set[str]]]
    ) -> list[FormatCandidate]:
        """Drop near-duplicate candidates, keeping the first of each cluster (EXPERIMENTAL).

        Args:
            candidates: The surviving candidates, in document order.
            bookmarks: The words of every bookmark, by page; a named candidate is never dropped.

        Returns:
            The same list with later near-duplicates of an earlier candidate removed.
        """
        prefix = self._config.candidate.near_duplicate_prefix
        ratio = self._config.candidate.near_duplicate_ratio
        buckets: dict[str, list[tuple[str, str]]] = {}
        kept: list[FormatCandidate] = []

        for candidate in candidates:
            norm = TextSearch.simplified(candidate.text)

            # A short code line and a bookmarked banner are identity: never collapsed here.
            if (len(norm) < prefix and any(ch.isdigit() for ch in norm)) or self._bookmarked(
                candidate, bookmarks
            ):
                kept.append(candidate)
                continue

            # Digits are identity, never noise: BT-2026-0146 is no reprint of BT-2026-0145.
            digits = "".join(ch for ch in norm if ch.isdigit())
            bucket = buckets.setdefault(norm[:prefix], [])

            if any(
                digits == seen_digits and fuzz.ratio(norm, seen) >= ratio
                for seen, seen_digits in bucket
            ):
                continue

            bucket.append((norm, digits))
            kept.append(candidate)

        return kept

    def _redundant_texts(self, texts: Iterable[str]) -> set[str]:
        """Collect the texts recurring at or above the redundancy threshold.

        Args:
            texts: Every candidate text of the document.

        Returns:
            The keys to drop as boilerplate.
        """
        frequency: Counter[str] = Counter(TextSearch.simplified(t) for t in texts)
        limit = self._config.candidate.redundancy_drop_count
        # A labelled reference line reprinted on every banner page is identity, not boilerplate.
        return {
            key
            for key, count in frequency.items()
            if count >= limit and not (self.IDENTITY_CODE.search(key) and TextUtils.has_word(key))
        }
