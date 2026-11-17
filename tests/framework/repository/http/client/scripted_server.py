"""A server that answers scripted responses."""

import httpx


class ScriptedServer:
    """A server answering from a script, and remembering every request it was sent."""

    def __init__(self, *answers: httpx.Response | Exception) -> None:
        """Queue the answers, the last one repeating once the script runs out.

        Args:
            *answers: A response to return, or a transport error to raise, per request.
        """
        self._answers = list(answers)
        self.requests: list[httpx.Request] = []
        """The requests received, in order."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Answer one request from the script.

        Args:
            request: The request the client sent.

        Returns:
            The next scripted response.

        Raises:
            Exception: The next scripted transport error.
        """  # noqa: DOC502 - the scripted error is raised, whatever its type
        self.requests.append(request)
        answer = self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]

        if isinstance(answer, Exception):
            raise answer

        return answer

    @property
    def transport(self) -> httpx.MockTransport:
        """The transport a client reaches this server through."""
        return httpx.MockTransport(self)

    def paths(self) -> list[str]:
        """Every path the server was asked for, in order."""
        return [request.url.path for request in self.requests]
