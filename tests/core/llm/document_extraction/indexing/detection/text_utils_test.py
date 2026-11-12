"""Unit tests for TextUtils."""

from baseline_core.core.llm.document_extraction.indexing.detection.text_utils import TextUtils


class TestHasWord:
    def test__three_plus_letter_word_is_a_word(self) -> None:
        """A title naming anything carries at least one word of three letters."""
        assert TextUtils.has_word("of contents") is True

    def test__numbering_and_short_tokens_name_nothing(self) -> None:
        """A wordless bookmark ("7.2", "Ab 4x") is numbering, not a title."""
        assert TextUtils.has_word("7.2") is False
        assert TextUtils.has_word("Ab 123 4x") is False
