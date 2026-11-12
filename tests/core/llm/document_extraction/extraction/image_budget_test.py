"""Tests of the image budget: pages shrunk, then grouped under a token budget."""

from io import BytesIO

from PIL import Image

from baseline_core.core.llm.document_extraction.extraction.image_budget import ImageBudget


class TestImageBudget:
    """Each call carries at most the budget, and at least one image."""

    def png(self, width: int, height: int) -> bytes:
        """A blank PNG of the given size."""
        out = BytesIO()
        Image.new("RGB", (width, height), "white").save(out, format="PNG")
        return out.getvalue()

    def test__a_page_is_shrunk_to_the_longest_side(self) -> None:
        """A 2 000 x 1 000 page comes back 1 000 x 500 under a 1 000 px side."""
        [[shrunk]] = ImageBudget(max_side=1000, batch_tokens=100_000).batches([
            self.png(2000, 1000)
        ])

        assert Image.open(BytesIO(shrunk)).size == (1000, 500)

    def test__pages_are_grouped_under_the_budget(self) -> None:
        """Three pages of ~1 300 tokens each: two fit under 3 000, the third opens a call."""
        pages = [self.png(1000, 1000)] * 3

        groups = ImageBudget(max_side=1000, batch_tokens=3000).batches(pages)

        assert [len(group) for group in groups] == [2, 1]

    def test__a_page_over_budget_is_sent_alone(self) -> None:
        """A budget never drops a page: one too large goes in a call of its own."""
        groups = ImageBudget(max_side=1000, batch_tokens=10).batches([self.png(500, 500)])

        assert [len(group) for group in groups] == [1]
