"""A router for the tests: one route validating its parameters, one raising a given exception."""

from fastapi import APIRouter


class ProbeRouter(APIRouter):
    """Echoes two validated query parameters, and raises the exception it is built with."""

    def __init__(self, failure: Exception) -> None:
        """Register the two routes.

        Args:
            failure: What ``/fail`` raises.
        """
        super().__init__()
        self.failure = failure
        self.add_api_route("/echo", self.echo, methods=["GET"])
        self.add_api_route("/fail", self.fail, methods=["GET"])

    async def echo(self, count: int, ratio: float) -> dict[str, float]:
        """Echo the parameters FastAPI validated.

        Args:
            count: An integer.
            ratio: A decimal number.

        Returns:
            Both values.
        """
        return {"count": count, "ratio": ratio}

    async def fail(self) -> None:
        """Raise the configured exception."""
        raise self.failure
