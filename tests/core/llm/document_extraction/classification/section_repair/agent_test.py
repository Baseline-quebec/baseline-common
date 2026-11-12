"""Tests of the boundary agent's prompt logic: it alone decides what the model sees."""

from collections.abc import Callable
from datetime import date
from typing import ClassVar

from baseline_core.core.llm.document_extraction.classification.section_repair.agent import (
    SectionRepairAgent,
)
from baseline_core.core.llm.document_extraction.classification.section_repair.merge_candidate_detector import (
    Evidence,
)
from baseline_core.core.llm.document_extraction.classification.section_repair.output import (
    BoundaryDecision,
)
from baseline_core.core.llm.document_extraction.classification.section_repair.prompt import (
    SectionRepairPrompt,
)
from baseline_core.core.llm.document_extraction.section import ResolvedSection
from baseline_core.core.llm.document_extraction.section_entities import (
    Mention,
    MentionRole,
    SectionEntities,
)
from tests.core.llm.document_extraction.classification.section_repair.report_entities import (
    ReportEntities,
)


class TestBoundaryWindow:
    """The window shows the junction, and says where text is missing."""

    def test__both_cut_edges_carry_an_ellipsis(self, boundary_agent: SectionRepairAgent) -> None:
        """A cut edge must say text is missing, not let it read as absent."""
        tail, head = boundary_agent._tail("x" * 900), boundary_agent._head("y" * 900)

        assert tail.startswith("…")
        assert head.endswith("…")

    def test__a_short_section_is_not_marked_as_cut(
        self, boundary_agent: SectionRepairAgent
    ) -> None:
        """A text fitting whole in the window carries no ellipsis."""
        tail, head = boundary_agent._tail("short"), boundary_agent._head("brief")

        assert tail == "short"
        assert head == "brief"

    def test__the_window_keeps_the_tail_and_the_head(
        self, boundary_agent: SectionRepairAgent
    ) -> None:
        """The evidence of a split lives where the two sections meet."""
        tail = boundary_agent._tail("start " + "x" * 900 + " END-OF-FIRST")
        head = boundary_agent._head("START-OF-SECOND " + "y" * 900)

        assert tail.endswith("END-OF-FIRST")
        assert head.startswith("START-OF-SECOND")

    def test__the_second_side_is_the_body_with_its_heading_already_stripped(
        self, boundary_agent: SectionRepairAgent, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """The title is an attribute of the rendered side; ``body`` keeps it out of the text."""
        second = make_section("PUMP REPLACEMENT", text="PUMP REPLACEMENT\nConsidering the pump")

        rendered = boundary_agent._user_prompt(
            make_section("A", text="end"), second, [Evidence.SAME_KEY]
        )

        assert "\nConsidering the pump\n" in rendered
        assert rendered.count("PUMP REPLACEMENT") == 1, "the heading must appear only as title"


class TestUserPrompt:
    """What the model receives, exactly."""

    OWNED: ClassVar[Mention] = Mention(kind="contract", value="SD-2025-2171", role=MentionRole.OWN)
    CITED: ClassVar[Mention] = Mention(
        kind="contract", value="C-2026-0313", role=MentionRole.CITED
    )
    SITE: ClassVar[Mention] = Mention(
        kind="site", value="1500 Mill Road, bay 250", role=MentionRole.OWN
    )

    def test__optional_attributes_are_omitted_when_absent(
        self, boundary_agent: SectionRepairAgent, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """An empty attribute would read as an extractor verdict; absence stays absent."""
        rendered = boundary_agent._user_prompt(
            make_section("A"), make_section("B"), [Evidence.SAME_KEY]
        )

        for attribute in ["carries=", "points_at=", "names=", "dated=", "<facts>"]:
            assert attribute not in rendered

    def test__a_cited_code_is_rendered_as_pointed_at(
        self, boundary_agent: SectionRepairAgent, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """What a side only refers to reaches the model as ``points_at``, never ``carries``."""
        rendered = boundary_agent._user_prompt(
            make_section("A"), make_section("B", mentions=[self.CITED]), [Evidence.SAME_KEY]
        )

        assert 'points_at="contract C-2026-0313"' in rendered
        assert "carries=" not in rendered

    def test__an_owned_identity_is_rendered_as_carried(
        self, boundary_agent: SectionRepairAgent, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """Judging ``same_key`` without seeing the keys was guesswork."""
        rendered = boundary_agent._user_prompt(
            make_section("A", mentions=[self.OWNED]),
            make_section("B", mentions=[self.OWNED]),
            [Evidence.SAME_KEY],
        )

        assert rendered.count('carries="contract SD-2025-2171"') == 2
        assert "points_at=" not in rendered

    def test__an_absent_carries_is_what_no_key_of_its_own_looks_like(
        self, boundary_agent: SectionRepairAgent, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """The clue and the rendering agree: nothing carried, nothing shown."""
        rendered = boundary_agent._user_prompt(
            make_section("A", mentions=[self.OWNED]),
            make_section("B"),
            [Evidence.NO_KEY_OF_ITS_OWN],
        )

        assert rendered.count("carries=") == 1

    def test__an_owned_mention_of_another_kind_is_named_not_carried(
        self, boundary_agent: SectionRepairAgent, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """``weak_signals_agree`` weighs what a side is about without being identified by it."""
        rendered = boundary_agent._user_prompt(
            make_section("A"), make_section("B", mentions=[self.SITE]), [Evidence.SAME_KEY]
        )

        assert 'names="site 1500 Mill Road, bay 250"' in rendered
        assert "carries=" not in rendered

    def test__both_dates_are_shown(
        self, boundary_agent: SectionRepairAgent, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """Two sides stating two dates of their own weigh for separate, so the model sees both."""
        first = make_section("A").model_copy(
            update={"entities": SectionEntities(dated=date(2026, 4, 7))}
        )
        second = make_section("B").model_copy(
            update={"entities": SectionEntities(dated=date(2026, 5, 12))}
        )

        rendered = boundary_agent._user_prompt(first, second, [Evidence.SAME_KEY])

        assert 'dated="2026-04-07"' in rendered
        assert 'dated="2026-05-12"' in rendered

    def test__the_projects_own_fields_reach_the_model_as_facts(
        self, boundary_agent: SectionRepairAgent, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """A field the generic code never reads still informs the judge, as data."""
        summarised = make_section("A").model_copy(
            update={"entities": ReportEntities(summary="Approves a pump replacement.")}
        )

        rendered = boundary_agent._user_prompt(summarised, make_section("B"), [Evidence.SAME_KEY])

        assert '<facts>{"summary": "Approves a pump replacement."}</facts>' in rendered
        assert rendered.count("<facts>") == 1

    def test__a_title_quoting_something_does_not_close_its_own_attribute(
        self, boundary_agent: SectionRepairAgent, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """A double quote inside a value would end the attribute early and garble the rest."""
        rendered = boundary_agent._user_prompt(
            make_section('The "North" site'), make_section("B"), [Evidence.SAME_KEY]
        )

        assert "title=\"The 'North' site\"" in rendered

    def test__the_clues_come_first(
        self, boundary_agent: SectionRepairAgent, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """The model reads why it was called before it reads the boundary."""
        rendered = boundary_agent._user_prompt(
            make_section("A"), make_section("B"), [Evidence.NO_KEY_OF_ITS_OWN]
        )

        assert rendered.startswith("<clues>no_key_of_its_own</clues>")

    def test__the_prompt_cuts_the_window_itself(
        self, boundary_agent: SectionRepairAgent, make_section: Callable[..., ResolvedSection]
    ) -> None:
        """The caller hands whole sections; the agent decides what the model sees."""
        rendered = boundary_agent._user_prompt(
            make_section("A", text="start " + "x" * 900 + " END-OF-FIRST"),
            make_section("B", text="START-OF-SECOND " + "y" * 900),
            [Evidence.SAME_KEY],
        )

        assert "…" in rendered
        assert "END-OF-FIRST" in rendered
        assert "START-OF-SECOND" in rendered
        assert "start x" not in rendered


class TestOnlyTwoDecisions:
    """The boundary has two outcomes: one section, or two."""

    def test__the_decision_space_is_binary(self) -> None:
        """A tie between two sections is the linking step's business, not a third outcome."""
        assert {decision.value for decision in BoundaryDecision} == {"merge", "separate"}

    def test__the_prompt_offers_no_third_way(self) -> None:
        """No removed outcome resurfaces in the instructions."""
        prompt = SectionRepairPrompt.SYSTEM.lower()

        assert "subsection" not in prompt
        assert "drop:" not in prompt
