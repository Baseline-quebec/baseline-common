"""Contract of a submitted job, whatever backend runs it."""

from abc import ABC, abstractmethod

from baseline_core.core.ml.job.job_spec import JobStatus


class AbstractJobHandle(ABC):
    """A submitted job, to follow and control."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Identifier of the job on its backend."""

    @abstractmethod
    def status(self) -> JobStatus:
        """Current status, fetched from the backend."""

    @abstractmethod
    def wait(self, *, follow: bool = True) -> JobStatus:
        """Block until the job ends.

        Args:
            follow: Print the job's output to the terminal while waiting.

        Returns:
            The final status, ``SUCCEEDED`` or ``CANCELED``.

        Raises:
            JobFailedError: If the job ends in failure.
        """

    @abstractmethod
    def logs(self) -> list[str]:
        """Lines of output produced so far."""

    @abstractmethod
    def cancel(self) -> None:
        """Ask the backend to stop the job."""
