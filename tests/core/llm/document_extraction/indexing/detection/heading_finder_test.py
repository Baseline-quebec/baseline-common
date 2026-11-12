"""Unit tests for the heading finder: run predicates, chunking, redundancy, series, detection."""

from collections.abc import Callable
from typing import ClassVar

from baseline_core.core.llm.document_extraction.indexing.detection.candidates import (
    FormatCandidate,
    StyledText,
    Typography,
)
from baseline_core.core.llm.document_extraction.indexing.detection.config import TextConfig
from baseline_core.core.llm.document_extraction.indexing.detection.heading_finder import (
    HeadingFinder,
    HeadingFinderConfig,
)
from baseline_core.core.llm.document_extraction.layout.layout import (
    BBox,
    BodyProfile,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
    LayoutSpan,
    OutlineEntry,
    PageSize,
    TextBlock,
)


class TestHeadingFinder:
    """The finder: run traits, chunk boundaries, geometry, redundancy, detection."""

    BODY: ClassVar[BodyProfile] = BodyProfile(size=11.0, bold=False)
    LETTER: ClassVar[PageSize] = PageSize(width=612.0, height=792.0)
    """The shape every built page is printed at, so only the text under test varies."""

    EXPECTED_HEADINGS: ClassVar[list[str]] = ["Background", "Methodology", "Findings"]

    def test__majority_clears_ratio_by_char_weight(
        self, finder: HeadingFinder, make_span: Callable[..., LayoutSpan]
    ) -> None:
        """Report the trait when its char-weighted share reaches the ratio (0.6)."""
        spans = [make_span("AAAAAA", flags=16), make_span("BBBB")]

        assert finder._majority(spans, lambda span: span.bold) is True

    def test__majority_below_ratio_is_false(
        self, finder: HeadingFinder, make_span: Callable[..., LayoutSpan]
    ) -> None:
        """Report no majority when the traited share falls under the ratio."""
        spans = [make_span("AAAAA", flags=16), make_span("BBBBB")]

        assert finder._majority(spans, lambda span: span.bold) is False

    def test__empty_runs_are_not_a_majority(
        self, finder: HeadingFinder, make_span: Callable[..., LayoutSpan]
    ) -> None:
        """Report no majority when the runs carry no characters."""
        assert finder._majority([make_span("")], lambda span: span.bold) is False

    def test__chunk_merges_all_lines(
        self,
        finder: HeadingFinder,
        make_span: Callable[..., LayoutSpan],
        make_line: Callable[..., LayoutLine],
    ) -> None:
        """Join every line of the chunk, take its largest size and its char-weighted traits."""
        lines = [
            make_line([make_span("Alpha", flags=16)]),
            make_line([make_span("Beta", flags=16, size=12.0)]),
            make_line([make_span("Gamma")]),
        ]

        chunk = finder._chunk_style(lines)

        assert chunk.text == "Alpha Beta Gamma"
        assert chunk.typography.size == 12.0
        assert chunk.typography.bold is True  # 9 bold chars of 14 clears 0.6

    def test__chunk_keeps_the_whole_title_untruncated(
        self,
        finder: HeadingFinder,
        make_span: Callable[..., LayoutSpan],
        make_line: Callable[..., LayoutLine],
    ) -> None:
        """Never cap a wrapped title: a long heading keeps its full tail intact."""
        words = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
        lines = [make_line([make_span(word)]) for word in words]

        assert finder._chunk_style(lines).text == " ".join(words)

    def test__a_break_carries_nothing_into_the_next_chunk(
        self,
        finder: HeadingFinder,
        make_span: Callable[..., LayoutSpan],
        make_line: Callable[..., LayoutLine],
    ) -> None:
        """Each side of a chunk boundary holds its own lines only, text and extent alike."""
        block = TextBlock(
            lines=[
                make_line([make_span("Plain body line")]),
                make_line([make_span("BOLD HEADING", flags=16)], y0=120.0),
            ]
        )
        page = LayoutPage(number=1, size=self.LETTER, blocks=[block])

        emitted = list(finder._styled_blocks(LayoutDocument(pages=[page], outline=[])))

        assert [chunk.text for chunk in emitted] == ["Plain body line", "BOLD HEADING"]
        assert [chunk.bbox.y0 for chunk in emitted] == [100.0, 120.0]

    def test__bold_heading_breaks_from_the_numbered_body_line(self, finder: HeadingFinder) -> None:
        """A bold underlined heading must break off the '1. …' item beneath it."""
        heading = StyledText(
            text="SUBMISSION OF THE INSPECTION REPORTS",
            typography=Typography(size=11.0, bold=True, underlined=True),
        )
        numbered = StyledText(
            text="1. Report of the site inspection", typography=Typography(size=11.0)
        )

        assert finder._breaks_chunk(heading, numbered, self.BODY) is True

    def test__bold_after_non_bold_breaks(self, finder: HeadingFinder) -> None:
        """An underlined running header deviates like the bold heading it precedes."""
        # Underlined, not italic: italic is no axis of ``_format_deviates``, so an italic
        # header would break on the flip alone and the test would lock nothing.
        header = StyledText(
            text="January 13, 2020 Page 2", typography=Typography(size=11.0, underlined=True)
        )
        heading = StyledText(
            text="PRESENTATION OF THE DRAFT PROCEDURE", typography=Typography(size=11.0, bold=True)
        )

        assert finder._breaks_chunk(header, heading, self.BODY) is True

    def test__uniformly_bold_wrap_never_splits(self, finder: HeadingFinder) -> None:
        """Wrapped headings are uniformly bold: bold -> bold is untouched by construction."""
        first = StyledText(
            text="APPOINTMENT OF THE REVIEW BOARD MEMBERS",
            typography=Typography(size=11.0, bold=True),
        )
        second = StyledText(text="(SD-2025-2430)", typography=Typography(size=11.0, bold=True))

        assert finder._breaks_chunk(first, second, self.BODY) is False

    def test__bare_numeric_enumerator_does_not_promote(self, finder: HeadingFinder) -> None:
        """A plain '1. …' body line promotes on neither format nor its bare enumerator."""
        numbered = StyledText(
            text="1. Report of the site inspection", typography=Typography(size=11.0)
        )

        assert finder._format_deviates(numbered, self.BODY) is False
        assert finder._deviates(numbered, self.BODY) is False

    def test__alphanumeric_code_promotes_an_unformatted_line(self, finder: HeadingFinder) -> None:
        """A distinctive reference code (letter + digit) still raises a plain heading."""
        coded = StyledText(
            text="HG.123A Requirements for the thermal insulation",
            typography=Typography(size=11.0),
        )

        assert finder._format_deviates(coded, self.BODY) is False
        assert finder._deviates(coded, self.BODY) is True

    def test__dot_leader_toc_line_promotes(self, finder: HeadingFinder) -> None:
        """A numbered TOC entry with a dot-leader marker promotes though it is unformatted."""
        toc = StyledText(
            text="1. Introduction ....................... 5", typography=Typography(size=11.0)
        )

        assert finder._format_deviates(toc, self.BODY) is False
        assert finder._deviates(toc, self.BODY) is True

    def test__compound_number_promotes_but_bare_enumerator_does_not(
        self, finder: HeadingFinder
    ) -> None:
        """A multi-part number (3.1) is a section code; a bare enumerator (1.) is not."""
        compound = StyledText(text="3.1 Scope of delivery", typography=Typography(size=11.0))
        bare = StyledText(
            text="1. Report of the site inspection", typography=Typography(size=11.0)
        )

        assert finder._deviates(compound, self.BODY) is True
        assert finder._deviates(bare, self.BODY) is False

    def test__body_profile_is_not_bold(
        self, finder: HeadingFinder, no_outline_layout: LayoutDocument
    ) -> None:
        """Report a body that is not mostly bold for an ordinary document."""
        assert finder._body_profile(no_outline_layout).bold is False

    def test__heading_candidates_surface_the_headings_and_drop_body(
        self, finder: HeadingFinder, no_outline_layout: LayoutDocument
    ) -> None:
        """Surface the styled headings and drop the body lines."""
        found = finder._heading_candidates(no_outline_layout)

        assert [c.text for c in found] == self.EXPECTED_HEADINGS

    def test__toc_and_body_pair_survives(self, finder: HeadingFinder) -> None:
        """A title on the contents page AND as the body heading is a legitimate pair."""
        texts = ["Introduction", "Scope of the inspection", "Scope of the inspection"]

        assert finder._redundant_texts(texts) == set()

    def test__three_occurrences_are_boilerplate(self, finder: HeadingFinder) -> None:
        """A phrase recurring three times is boilerplate (approval stamps, running headers)."""
        texts = [
            "Introduction",
            "APPROVED AS SUBMITTED",
            "APPROVED AS SUBMITTED",
            "APPROVED AS SUBMITTED",
            "Conclusion",
        ]

        assert finder._redundant_texts(texts) == {"approved as submitted"}

    def test__a_recurring_reference_code_is_identity_not_boilerplate(
        self, finder: HeadingFinder
    ) -> None:
        """One file reprints its own code on every banner page; the code line survives."""
        assert finder._redundant_texts(["File # : 1267303003"] * 4) == set()

    def test__a_recurring_dateline_is_still_boilerplate(self, finder: HeadingFinder) -> None:
        """A date stamp bears digits but no identifier run; it drops like any other furniture."""
        assert finder._redundant_texts(["March 27, 2026"] * 4) == {"march 27, 2026"}

    def test__a_recurring_bare_code_stub_is_still_boilerplate(self, finder: HeadingFinder) -> None:
        """An unlabeled digit run (a contents-page stub) is not an identity line."""
        assert finder._redundant_texts(["251209 -...."] * 4) == {"251209 -...."}

    def test__a_recurring_banner_a_bookmark_names_survives_on_every_bookmarked_page(
        self,
        finder: HeadingFinder,
        make_span: Callable[..., LayoutSpan],
        make_line: Callable[..., LayoutLine],
    ) -> None:
        """The same masthead on five bookmarked pages is five banners, never boilerplate."""
        pages = [
            LayoutPage(
                number=number,
                size=self.LETTER,
                blocks=[
                    TextBlock(
                        lines=[
                            make_line([make_span("Records management system", flags=16)], y0=60.0),
                            make_line([make_span("SUMMARY SHEET", flags=16)], y0=74.0),
                        ]
                    ),
                    TextBlock(lines=[make_line([make_span("Body of the summary.")], y0=200.0)]),
                ],
            )
            for number in range(1, 6)
        ]
        # Bookmarked on four pages of five: the fifth masthead is the only one to drop.
        outline = [
            OutlineEntry(title=f"40.0{n}-123456000{n}-Summary sheet", page=n, depth=1)
            for n in range(1, 5)
        ]

        found = finder._heading_candidates(LayoutDocument(pages=pages, outline=outline))

        banners = [c for c in found if "SUMMARY" in c.text]
        assert [c.page for c in banners] == [1, 2, 3, 4]
        assert {c.text for c in banners} == {"SUMMARY SHEET"}

    def test__format_candidates_carry_their_full_extent(
        self, finder: HeadingFinder, with_outline_layout: LayoutDocument
    ) -> None:
        """Every format candidate spans a positive height and sits somewhere on its own page."""
        detected = finder.scan(with_outline_layout)

        assert detected.format
        assert all(c.bbox.y1 > c.bbox.y0 for c in detected.format)
        assert all(0.0 <= c.position <= 1.0 for c in detected.format)

    def test__reads_every_entry_with_its_physical_page(
        self, finder: HeadingFinder, with_outline_layout: LayoutDocument
    ) -> None:
        """Each bookmark becomes one candidate carrying the physical page it points at."""
        detected = finder.scan(with_outline_layout)

        assert [(c.text, c.page) for c in detected.outline] == [
            ("Introduction", 2),
            ("Methods", 3),
            ("Discussion", 4),
        ]

    def test__a_bookmark_resolving_nowhere_is_dropped(self, finder: HeadingFinder) -> None:
        """A dangling destination resolves to no page, so it can corroborate no candidate."""
        parsed = LayoutDocument(
            pages=[LayoutPage(number=1, size=self.LETTER, blocks=[])],
            outline=[
                OutlineEntry(title="REF-505_Schedule", page=None, depth=1),
                OutlineEntry(title="Introduction", page=1, depth=1),
            ],
        )

        assert [c.text for c in finder._outline_candidates(parsed)] == ["Introduction"]

    def test__no_outline_yields_no_outline_candidates(
        self, finder: HeadingFinder, no_outline_layout: LayoutDocument
    ) -> None:
        """A document with no embedded outline surfaces format candidates only."""
        detected = finder.scan(no_outline_layout)

        assert detected.outline == []
        assert detected.format

    def test__a_running_header_is_dropped_with_its_baseline_mates(
        self,
        finder: HeadingFinder,
        make_span: Callable[..., LayoutSpan],
        make_line: Callable[..., LayoutLine],
    ) -> None:
        """A bold header repeated at the top of every page never reaches the candidates."""
        pages = [
            LayoutPage(
                number=number,
                size=self.LETTER,
                blocks=[
                    TextBlock(lines=[make_line([make_span("ANNUAL REPORT", flags=16)], y0=20.0)]),
                    TextBlock(
                        lines=[make_line([make_span(f"PART {number}", flags=16)], y0=200.0)]
                    ),
                    TextBlock(lines=[make_line([make_span("Body text of the page.")], y0=400.0)]),
                ],
            )
            for number in range(1, 5)
        ]

        found = finder.scan(LayoutDocument(pages=pages, outline=[]))

        assert [c.text for c in found.format] == [f"PART {number}" for number in range(1, 5)]


class TestTextRules:
    """The line-level rules: noise and capitals."""

    def test__bare_number_and_amount_are_noise(self) -> None:
        """A folio and a lone table amount match the noise pattern."""
        assert HeadingFinder.NOISE.match("234")
        assert HeadingFinder.NOISE.match("38 800 $")

    def test__real_titles_are_not_noise(self) -> None:
        """A titled line, even a numbered one, is not noise."""
        assert not HeadingFinder.NOISE.match("1 Overview")
        assert not HeadingFinder.NOISE.match("Introduction")

    def test__all_caps_line_reads_as_caps(self, finder: HeadingFinder) -> None:
        """An all-caps line reads as caps under the library thresholds."""
        assert finder._is_caps("GENERAL PROVISIONS") is True

    def test__mixed_case_line_is_not_caps(self, finder: HeadingFinder) -> None:
        """A title-cased line is not caps."""
        assert finder._is_caps("Introduction To Things") is False

    def test__too_few_letters_is_not_caps(self, finder: HeadingFinder) -> None:
        """A line with fewer letters than the minimum is never judged caps."""
        assert finder._is_caps("AB") is False

    def test__the_thresholds_move_the_verdict(self) -> None:
        """Raising the letter minimum drops a short line; lowering the ratio lifts a long one."""
        strict = HeadingFinder(HeadingFinderConfig(text=TextConfig(min_caps_letters=8)))
        lenient = HeadingFinder(HeadingFinderConfig(text=TextConfig(caps_ratio=0.2)))

        assert strict._is_caps("CAT") is False
        assert strict._is_caps("GENERAL PROVISIONS") is True
        assert lenient._is_caps("Hello") is True


class TestListing:
    """An item listing printed again, item by item, below is never the items' anchor."""

    @staticmethod
    def candidate(text: str, page: int, y0: float = 100.0) -> FormatCandidate:
        """Build a candidate line on a letter page.

        Args:
            text: The line, as printed.
            page: Its 1-based page.
            y0: Its top edge, in points.

        Returns:
            The candidate.
        """
        return FormatCandidate(
            text=text,
            typography=Typography(size=10.0),
            page=page,
            bbox=BBox(x0=72.0, y0=y0, x1=400.0, y1=y0 + 12.0),
            position=y0 / 792.0,
        )

    def test__a_listing_repeated_below_in_order_is_dropped(self, finder: HeadingFinder) -> None:
        """Every item keeps only its body printing, never the listing line."""
        listing = [self.candidate(f"RPT 26-07.{i:02d}", 2, 100.0 + 20 * i) for i in range(1, 7)]
        bodies = [self.candidate(f"RPT 26-07.{i:02d}", 4 + i) for i in range(1, 7)]

        kept = finder._drop_listing_prints(listing + bodies)

        assert [c.page for c in kept] == [5, 6, 7, 8, 9, 10]

    def test__a_code_reprinted_inside_its_own_item_keeps_its_first_printing(
        self, finder: HeadingFinder
    ) -> None:
        """A code reprinted under its own item interleaves: no listing, nothing dropped."""
        interleaved = [
            printed
            for i in range(1, 7)
            for printed in [
                self.candidate(f"CV-2025-{i:02d}", i, 100.0),
                self.candidate(f"CV-2025-{i:02d}", i, 500.0),
            ]
        ]

        assert finder._drop_listing_prints(interleaved) == interleaved


class TestSerialItems:
    """Lines opening a counting code series are candidates whatever their typography."""

    LETTER: ClassVar[PageSize] = PageSize(width=612.0, height=792.0)

    def document(
        self,
        rows: list[str],
        make_span: Callable[..., LayoutSpan],
        make_line: Callable[..., LayoutLine],
    ) -> LayoutDocument:
        """One page of ``rows`` in plain 11pt body type, one block, as a table the parse merges.

        Args:
            rows: The printed lines.
            make_span: The span factory.
            make_line: The line factory.

        Returns:
            The one-page document.
        """
        lines = [make_line([make_span(text)], y0=80.0 + 14.0 * i) for i, text in enumerate(rows)]
        page = LayoutPage(number=1, size=self.LETTER, blocks=[TextBlock(lines=lines)])
        return LayoutDocument(pages=[page], outline=[])

    @staticmethod
    def openers(finder: HeadingFinder, document: LayoutDocument) -> list[str]:
        """Read the first word pair of every candidate the finder keeps.

        Args:
            finder: The finder under test.
            document: The document it reads.

        Returns:
            The two leading words of each candidate.
        """
        return [" ".join(c.text.split()[:2]) for c in finder._heading_candidates(document)]

    def test__a_work_order_list_in_body_type_yields_one_candidate_per_item(
        self,
        finder: HeadingFinder,
        make_span: Callable[..., LayoutSpan],
        make_line: Callable[..., LayoutLine],
    ) -> None:
        """Every item of a counting series opens a candidate, though printed as table rows."""
        rows = []

        for n in range(1, 6):
            rows += [
                # En dashes as printed: the rule must not depend on the separator.
                f"RPT 26-12.0{n} 2025-0008{n} \N{EN DASH} WO 2025-0302{n} \N{EN DASH} Windows",
                "The inspector recommends replacing the unit as described in the work order.",
            ]

        found = self.openers(finder, self.document(rows, make_span, make_line))

        assert [o for o in found if o.startswith("RPT")] == [
            f"RPT 26-12.0{n}" for n in range(1, 6)
        ]

    def test__budget_accounts_that_do_not_count_up_are_not_promoted(
        self,
        finder: HeadingFinder,
        make_span: Callable[..., LayoutSpan],
        make_line: Callable[..., LayoutLine],
    ) -> None:
        """A finance table's account codes recur at line openers but are no list of items."""
        rows = [
            "02-193-00-989 Professional services 12 500 $",
            "02-411-00-321 Network maintenance 8 200 $",
            "02-701-73-484 Training 4 750 $",
            "02-877-10-533 Equipment 1 300 $",
        ]

        assert self.openers(finder, self.document(rows, make_span, make_line)) == []

    def test__dates_opening_lines_are_never_a_series(
        self,
        finder: HeadingFinder,
        make_span: Callable[..., LayoutSpan],
        make_line: Callable[..., LayoutLine],
    ) -> None:
        """Consecutive dates count up by one, yet a dated line is not an item heading."""
        rows = [f"2025-12-0{d} site visit at the head office" for d in range(1, 5)]

        assert self.openers(finder, self.document(rows, make_span, make_line)) == []

    def test__an_item_cited_again_is_promoted_once(
        self,
        finder: HeadingFinder,
        make_span: Callable[..., LayoutSpan],
        make_line: Callable[..., LayoutLine],
    ) -> None:
        """A later line opening with an item's code cites it; only where it opens counts."""
        rows = [
            "CV-2025-1364 Adoption of the standard",
            "CV-2025-1365 Award of a contract",
            "CV-2025-1366 Site access variance",
            "CV-2025-1364 as amended hereby",
        ]

        found = self.openers(finder, self.document(rows, make_span, make_line))

        assert found.count("CV-2025-1364 Adoption") == 1
        assert "CV-2025-1364 as" not in found

    def test__two_items_are_not_yet_a_series(
        self,
        finder: HeadingFinder,
        make_span: Callable[..., LayoutSpan],
        make_line: Callable[..., LayoutLine],
    ) -> None:
        """Below ``serial_min_items`` distinct items, the series rule alone promotes nothing."""
        rows = ["SH-260413-2.1 Submission of the report", "SH-260413-2.2 Submission of a plan"]
        document = self.document(rows, make_span, make_line)

        assert finder._serial_items(document, [], {1: self.LETTER.height}) == []

    def test__serial_items_name_each_item_once_by_its_code(self, finder: HeadingFinder) -> None:
        """The items a counting family opens come back with their codes, first printing only."""
        candidates = [
            TestListing.candidate(f"RPT 26-07.{i:02d} Item {i}", page)
            for i, page in [(1, 1), (2, 2), (3, 3), (1, 4)]
        ]

        items = finder.serial_items(candidates)

        assert [(item.code, item.candidate.page) for item in items] == [
            ("RPT 26-07.01", 1),
            ("RPT 26-07.02", 2),
            ("RPT 26-07.03", 3),
        ]
