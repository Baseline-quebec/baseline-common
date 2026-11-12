"""ImageBudget — rendered pages shrunk and grouped into vision calls under a token budget."""

from io import BytesIO
from typing import ClassVar

from PIL import Image


class ImageBudget:
    """Shrinks rendered pages to a legible side and groups them into calls of a token budget."""

    TOKEN_PIXELS: ClassVar[int] = 784
    """A vision model reading 28x28-pixel tiles merged 2x2 spends about one token per 784 px."""

    IMAGE_TOKENS: ClassVar[int] = 16
    """Tokens an image costs beyond its pixels: its delimiters."""

    def __init__(self, max_side: int, batch_tokens: int) -> None:
        """Bind the limits.

        Args:
            max_side: Longest side, in pixels, an image is shrunk to.
            batch_tokens: Image tokens one call may carry.
        """
        self._max_side = max_side
        self._batch_tokens = batch_tokens

    def batches(self, pngs: list[bytes]) -> list[list[bytes]]:
        """Shrink each page, then group them in order into calls of at most the budget.

        Args:
            pngs: The rendered pages, in reading order.

        Returns:
            The shrunk PNGs grouped per call, in order; an image over budget alone in its call.
        """
        groups: list[list[bytes]] = []
        used = self._batch_tokens + 1

        for png in pngs:
            image = Image.open(BytesIO(png))
            image.thumbnail((self._max_side, self._max_side))
            cost = image.width * image.height // self.TOKEN_PIXELS + self.IMAGE_TOKENS

            if used + cost > self._batch_tokens:
                groups.append([])
                used = 0

            shrunk = BytesIO()
            image.save(shrunk, format="PNG")
            groups[-1].append(shrunk.getvalue())
            used += cost

        return groups
