"""Tests of the merge-candidate detector: no suspect boundary missed, and no clue invented."""

from collections.abc import Callable
from typing import ClassVar

from baseline_core.core.llm.document_extraction.classification.section_repair.config import (
    SectionRepairConfig,
)
from baseline_core.core.llm.document_extraction.classification.section_repair.merge_candidate_detector import (
    Evidence,
    MergeCandidateDetector,
)
from baseline_core.core.llm.document_extraction.section import ResolvedSection, SectionRelevance
from baseline_core.core.llm.document_extraction.section_entities import Mention, MentionRole


class Codes:
    """The mentions the tests below name sections with."""

    FIRST: ClassVar[Mention] = Mention(kind="contract", value="C-2026-0307", role=MentionRole.OWN)
    SECOND: ClassVar[Mention] = Mention(kind="contract", value="C-2026-0412", role=MentionRole.OWN)
    CITED: ClassVar[Mention] = Mention(
        kind="contract", value="C-2026-0100", role=MentionRole.CITED
    )
    STANDARD: ClassVar[Mention] = Mention(
        kind="standard", value="ISO 9001", role=MentionRole.CITED
    )
    SITE: ClassVar[Mention] = Mention(kind="site", value="4120 Harbour Road", role=MentionRole.OWN)
    SUPPLIER: ClassVar[Mention] = Mention(
        kind="supplier", value="Northwind Tools", role=MentionRole.OWN
    )
    OTHER_SUPPLIER: ClassVar[Mention] = Mention(
        kind="supplier", value="Contoso Rigging", role=MentionRole.OWN
    )


class TestTheOnlyExclusions:
    """The two certainties, discarding a pair before any clue is gathered."""

    def test__a_long_first_section_does_not_disqualify_the_pair(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """One item's summary split in two can sit five pages apart."""
        first = make_section("SUMMARY REPORT", mentions=[Codes.FIRST])
        second = make_section("SUMMARY REPORT", mentions=[Codes.FIRST]).model_copy(
            update={"start_page": 6}
        )

        evidence = detector.evidence_for(first, second)

        assert evidence, "this case MUST be examined"

    def test__two_own_identities_are_not_a_candidate(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """Two sections each owning its own identity are two items."""
        first = make_section("Repair", mentions=[Codes.FIRST])
        second = make_section("Replacement", mentions=[Codes.SECOND])

        assert detector.evidence_for(first, second) == []

    def test__the_exclusion_holds_even_when_titles_continue(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """The veto reads the identities, never the typography of the titles."""
        first = make_section("Amendment to the maintenance plan of", mentions=[Codes.FIRST])
        second = make_section("the north assembly line", mentions=[Codes.SECOND])

        assert detector.evidence_for(first, second) == []

    def test__a_cited_code_is_not_an_own_identity(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """A CITED code does not make the section an item: the veto does not apply."""
        item = make_section("Item", mentions=[Codes.FIRST])
        citing = make_section("continued", mentions=[Codes.CITED])

        assert detector.evidence_for(item, citing) == [Evidence.NO_KEY_OF_ITS_OWN]

    def test__an_irrelevant_second_section_is_skipped(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """A closing section is not a boundary worth a call."""
        first = make_section("Repair", mentions=[Codes.FIRST])
        second = make_section("CLOSING", relevance=SectionRelevance.IRRELEVANT)

        assert detector.evidence_for(first, second) == []

    def test__an_irrelevant_first_section_is_skipped(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """A distribution list before a kept item raises no question."""
        first = make_section(
            "DISTRIBUTION LIST", mentions=[Codes.FIRST], relevance=SectionRelevance.IRRELEVANT
        )
        second = make_section("Repair")

        assert detector.evidence_for(first, second) == []

    def test__an_uncertain_keyless_fragment_still_reaches_the_agent(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """A broken-off fragment reads uncertain precisely because its key stayed above the cut."""
        first = make_section("Repair", mentions=[Codes.FIRST])
        second = make_section(
            "Considering the line stopped twice", relevance=SectionRelevance.UNCERTAIN
        )

        assert detector.evidence_for(first, second) == [Evidence.NO_KEY_OF_ITS_OWN]


class TestEvidenceGathering:
    """Every clue is found, and none is invented."""

    def test__the_same_identity_is_evidence_not_a_verdict(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """A shared identity sends the boundary to the agent — it never merges by itself."""
        first = make_section("Summary", mentions=[Codes.FIRST])
        second = make_section("Recommendation", mentions=[Codes.FIRST])

        assert detector.evidence_for(first, second) == [Evidence.SAME_KEY]

    def test__two_printings_of_one_code_are_the_same_identity(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """Case, spacing and marks do not make two codes."""
        reprinted = Mention(kind="contract", value="c 2026 0307", role=MentionRole.OWN)
        first = make_section("Summary", mentions=[Codes.FIRST])
        second = make_section("Recommendation", mentions=[reprinted])

        assert detector.evidence_for(first, second) == [Evidence.SAME_KEY]

    def test__a_section_without_its_own_identity_is_evidence(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """A keyless section beside a keyed one is often the latter's tail."""
        first = make_section("Item", mentions=[Codes.FIRST])
        second = make_section("Fragment")

        assert detector.evidence_for(first, second) == [Evidence.NO_KEY_OF_ITS_OWN]

    def test__two_keyless_sections_in_a_row_have_no_host(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """Two rubrics in a row: nothing to attach to."""
        first = make_section("FINANCE")
        second = make_section("MAINTENANCE")

        assert detector.evidence_for(first, second) == []

    def test__identical_titles_are_not_evidence(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """Within one report, an identical title is a category label, not a split."""
        first = make_section("Budget - Spending authorization", mentions=[Codes.FIRST])
        second = make_section("Budget - Spending authorization", mentions=[Codes.SECOND])

        assert detector.evidence_for(first, second) == []

    def test__a_kind_outside_the_identity_kinds_is_not_an_identity(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """A paragraph number identifies no item, so sharing one proves nothing."""
        point = Mention(kind="paragraph", value="10.01", role=MentionRole.OWN)
        first = make_section("A", mentions=[point])
        second = make_section("B", mentions=[point])

        assert detector.evidence_for(first, second) == []

    def test__every_declared_identity_kind_counts(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """The project's list decides what identifies, not the first kind it names."""
        order = Mention(kind="work_order", value="WO-2026-041", role=MentionRole.OWN)
        first = make_section("A", mentions=[order])
        second = make_section("B", mentions=[order])

        assert detector.evidence_for(first, second) == [Evidence.SAME_KEY]


class TestWeakSignalsOnlyCountTogether:
    """One weak signal proves nothing; several agreeing do."""

    def test__a_shared_site_alone_is_not_enough(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """Two items of one report can bear on the same site."""
        first = make_section("A", mentions=[Codes.SITE])
        second = make_section("B", mentions=[Codes.SITE])

        assert detector.evidence_for(first, second) == []

    def test__two_weak_signals_agreeing_are_a_clue(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """Same site AND same supplier, with no identity on either side."""
        first = make_section("A", mentions=[Codes.SITE, Codes.SUPPLIER])
        second = make_section("B", mentions=[Codes.SITE, Codes.SUPPLIER])

        assert detector.evidence_for(first, second) == [Evidence.WEAK_SIGNALS_AGREE]

    def test__one_weak_signal_agreeing_is_not_enough(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """The site matches but the supplier differs: still below the bar."""
        first = make_section("A", mentions=[Codes.SITE, Codes.SUPPLIER])
        second = make_section("B", mentions=[Codes.SITE, Codes.OTHER_SUPPLIER])

        assert detector.evidence_for(first, second) == []

    def test__a_printed_detail_keeps_two_sites_apart(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """Two bays of one building are two sites, so they do not agree."""
        bay_100 = Mention(kind="site", value="1500 Mill Road, bay 100", role=MentionRole.OWN)
        bay_250 = Mention(kind="site", value="1500 Mill Road, bay 250", role=MentionRole.OWN)
        first = make_section("A", mentions=[bay_100, Codes.SUPPLIER])
        second = make_section("B", mentions=[bay_250, Codes.SUPPLIER])

        assert detector.evidence_for(first, second) == []

    def test__an_identity_on_each_side_outranks_any_agreement(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """The veto holds even when every weak signal agrees."""
        first = make_section("A", mentions=[Codes.FIRST, Codes.SITE, Codes.SUPPLIER])
        second = make_section("B", mentions=[Codes.SECOND, Codes.SITE, Codes.SUPPLIER])

        assert detector.evidence_for(first, second) == []

    def test__the_threshold_is_the_projects(
        self, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """Lowered to one, a shared site alone earns a look."""
        detector = MergeCandidateDetector(
            SectionRepairConfig(identity_kinds=["contract"], min_shared_signals=1)
        )
        first = make_section("A", mentions=[Codes.SITE])
        second = make_section("B", mentions=[Codes.SITE])

        assert detector.evidence_for(first, second) == [Evidence.WEAK_SIGNALS_AGREE]


class TestTheRoleDecides:
    """An OWNED code identifies; the same code CITED is a shared framework."""

    def test__the_same_code_owned_or_cited_is_not_the_same_signal(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """Owned on both sides: an identity. Cited on both sides: one weak signal."""
        owned = detector.evidence_for(
            make_section("A", mentions=[Codes.FIRST]), make_section("B", mentions=[Codes.FIRST])
        )
        cited = detector.evidence_for(
            make_section("A", mentions=[Codes.CITED]), make_section("B", mentions=[Codes.CITED])
        )

        assert owned == [Evidence.SAME_KEY]
        assert cited == [], "a framework cited from both sides is not enough"

    def test__two_items_citing_one_framework_are_two_items(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """Two amendments side by side cite the same framework and own distinct codes."""
        first = make_section("Amendment A", mentions=[Codes.FIRST, Codes.STANDARD])
        second = make_section("Amendment B", mentions=[Codes.SECOND, Codes.STANDARD])

        assert detector.evidence_for(first, second) == []

    def test__a_cited_code_counts_toward_a_weak_agreement(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """Shared framework AND shared site: two weak signals agreeing."""
        first = make_section("A", mentions=[Codes.STANDARD, Codes.SITE])
        second = make_section("B", mentions=[Codes.STANDARD, Codes.SITE])

        assert detector.evidence_for(first, second) == [Evidence.WEAK_SIGNALS_AGREE]

    def test__a_cited_code_never_satisfies_the_veto(
        self, detector: MergeCandidateDetector, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """Two sections citing distinct codes stay examinable."""
        other = Mention(kind="contract", value="C-2026-0200", role=MentionRole.CITED)
        first = make_section("A", mentions=[Codes.FIRST, Codes.CITED])
        second = make_section("B", mentions=[other])

        assert detector.evidence_for(first, second) == [Evidence.NO_KEY_OF_ITS_OWN]
