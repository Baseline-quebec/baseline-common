"""Contract of a job backend: a GPU cluster, a local subprocess."""

from abc import ABC, abstractmethod

from baseline_core.core.ml.job.abstract_job_handle import AbstractJobHandle
from baseline_core.core.ml.job.job_spec import JobSpec


class AbstractJobLauncher(ABC):
    """Submits jobs to one backend."""

    @abstractmethod
    def submit(self, spec: JobSpec) -> AbstractJobHandle:
        """Start a job.

        Args:
            spec: What to run.

        Returns:
            A handle on the started job.

        Raises:
            JobSubmissionError: If the backend refuses the job.
        """
