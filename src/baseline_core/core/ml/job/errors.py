"""Errors raised by job launchers; the SDK errors of a backend never leak past its package."""

from collections.abc import Sequence


class JobError(Exception):
    """Base of every job error."""


class JobSubmissionError(JobError):
    """The backend refused the job: unknown compute or environment, quota, authentication."""


class JobFailedError(JobError):
    """The job ran and ended in failure."""

    def __init__(self, job_id: str, last_lines: Sequence[str]) -> None:
        """Build the message from the job id and the tail of its log.

        Args:
            job_id: Identifier of the failed job.
            last_lines: Last lines of the job's output, for the message.
        """
        tail = "\n".join(last_lines)
        super().__init__(f"Job {job_id} failed. Last output:\n{tail}")
        self.job_id = job_id
        self.last_lines = list(last_lines)


class JobNotFoundError(JobError):
    """No job has this id on the backend."""

    def __init__(self, job_id: str) -> None:
        """Name the missing job.

        Args:
            job_id: Identifier that was looked up.
        """
        super().__init__(f"Job {job_id} not found.")
        self.job_id = job_id
