"""Tests of TextSearch: one comparable form, and the three searches that read through it."""

from baseline_core.core.llm.document_extraction.text_search import TextSearch


class TestSimplified:
    """The comparable form: accents gone, case gone, whitespace collapsed."""

    def test__two_printings_of_one_line_compare_equal(self) -> None:
        """Accents, case and spacing are rendering, not identity."""
        assert TextSearch.simplified("RÉVISION  générale") == TextSearch.simplified(
            "revision generale"
        )

    def test__whitespace_collapses_across_line_breaks(self) -> None:
        """A wrapped printing reads as its one-line form."""
        assert TextSearch.simplified("Dépôt du\nrapport d'essai") == "depot du rapport d'essai"


class TestExactLines:
    """Every line reading exactly as the needle, by index."""

    def test__the_matching_lines_are_found_in_printed_order(self) -> None:
        """An opener printed alone is found however it was accented or cased."""
        lines = ["Revue de conception", "VL-260120-2.1", "Dépôt du rapport", "vl-260120-2.1"]
        assert TextSearch.exact_lines("VL-260120-2.1", lines) == [1, 3]

    def test__a_line_merely_containing_the_needle_is_not_exact(self) -> None:
        """A citation inside a sentence never counts as the opener's own line."""
        lines = ["selon la révision VL-260120-2.1 approuvée"]
        assert TextSearch.exact_lines("VL-260120-2.1", lines) == []

    def test__an_empty_needle_matches_nothing(self) -> None:
        """A needle reducing to nothing would match every blank line."""
        assert TextSearch.exact_lines("  ", ["", "x"]) == []


class TestFuzzyWindow:
    """A window of consecutive lines sized to the needle, scored whole."""

    def test__a_needle_spanning_two_lines_scores_as_one(self) -> None:
        """A wrapped heading reads whole, wherever the line break fell."""
        lines = ["8.7 Validation du protocole", "d'essai 2021-126", "Considering that…"]
        score, stop = TextSearch.fuzzy_window(
            "8.7 Validation du protocole d'essai 2021-126", lines, 0
        )
        assert score == 100.0
        assert stop == 2

    def test__the_window_never_reads_past_the_corpus(self) -> None:
        """A needle longer than what remains stops at the end, scored on what was read."""
        score, stop = TextSearch.fuzzy_window("un très long titre jamais imprimé", ["court"], 0)
        assert stop == 1
        assert score < 50.0

    def test__a_wrong_start_scores_below_the_right_one(self) -> None:
        """Ranking start lines is the caller's whole use of the score."""
        lines = ["Considering that…", "8.7 Validation du protocole"]
        wrong, _ = TextSearch.fuzzy_window("8.7 Validation du protocole", lines, 0)
        right, _ = TextSearch.fuzzy_window("8.7 Validation du protocole", lines, 1)
        assert right > wrong


class TestLeading:
    """Where the needle ends when the corpus opens on it."""

    def test__the_boundary_lands_just_past_the_needle(self) -> None:
        """The needle may render with its own spacing; the boundary is found by form."""
        text = "Révision   générale — Recommendation"
        boundary = TextSearch.leading("Révision générale", text)
        assert boundary is not None
        assert text[boundary:].lstrip(" —") == "Recommendation"

    def test__a_corpus_not_opening_on_the_needle_has_no_boundary(self) -> None:
        """A needle appearing later, or not at all, is not a leading match."""
        assert TextSearch.leading("Recommendation", "Révision Recommendation") is None

    def test__an_empty_needle_has_no_boundary(self) -> None:
        """Nothing to strip: the caller keeps the corpus whole."""
        assert TextSearch.leading("", "texte") is None
