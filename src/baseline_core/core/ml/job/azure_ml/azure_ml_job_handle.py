"""Follow and control a job submitted to Azure ML."""

import tempfile
from enum import Enum
from pathlib import Path
from time import sleep
from typing import ClassVar

from azure.ai.ml import MLClient
from azure.core.exceptions import AzureError, ResourceNotFoundError
from loguru import logger

from baseline_core.core.ml.job.abstract_job_handle import AbstractJobHandle
from baseline_core.core.ml.job.errors import JobError, JobFailedError, JobNotFoundError
from baseline_core.core.ml.job.job_spec import JobStatus


class AzureMLJobHandle(AbstractJobHandle):
    """One job on an Azure ML workspace.

    Attributes:
        AZURE_STATUS: Azure ML job statuses mapped onto the common vocabulary.
        STD_LOG: Path of the job's standard output inside its downloaded artifacts.
        POLL_SECONDS: Delay between two status checks in ``wait`` without ``follow``.
        LOG_TAIL_LINES: Lines of output quoted in a ``JobFailedError``.
    """

    AZURE_STATUS: ClassVar[dict[str, JobStatus]] = {
        "NotStarted": JobStatus.QUEUED,
        "Queued": JobStatus.QUEUED,
        "Starting": JobStatus.QUEUED,
        "Preparing": JobStatus.QUEUED,
        "Provisioning": JobStatus.QUEUED,
        "Running": JobStatus.RUNNING,
        "Finalizing": JobStatus.RUNNING,
        "Completed": JobStatus.SUCCEEDED,
        "Failed": JobStatus.FAILED,
        "Canceled": JobStatus.CANCELED,
        "CancelRequested": JobStatus.CANCELED,
    }
    STD_LOG: ClassVar[str] = "artifacts/user_logs/std_log.txt"
    POLL_SECONDS: ClassVar[float] = 15.0
    LOG_TAIL_LINES: ClassVar[int] = 20

    def __init__(self, client: MLClient, job_name: str) -> None:
        """Wrap an existing job.

        Args:
            client: Workspace client.
            job_name: Name of the job on the workspace.
        """
        self._client = client
        self._name = job_name

    @property
    def id(self) -> str:
        """Name of the job on the workspace."""
        return self._name

    def status(self) -> JobStatus:
        """Fetch the job and map its status.

        Raises:
            JobNotFoundError: If the workspace has no job with this name.
            JobError: If Azure cannot be reached or reports a status this code does not know.
        """
        try:
            job = self._client.jobs.get(self._name)
        except ResourceNotFoundError as error:
            raise JobNotFoundError(self._name) from error
        except AzureError as error:
            error_msg = f"Could not fetch job {self._name}: {error}"
            raise JobError(error_msg) from error

        job_status = job.status
        azure_status = job_status.value if isinstance(job_status, Enum) else str(job_status)

        if azure_status not in self.AZURE_STATUS:
            error_msg = f"Job {self._name} has unknown Azure ML status '{azure_status}'."
            raise JobError(error_msg)

        return self.AZURE_STATUS[azure_status]

    def wait(self, *, follow: bool = True) -> JobStatus:
        """Block until the job ends.

        Args:
            follow: Stream the job output; otherwise poll the status every ``POLL_SECONDS``.

        Returns:
            The final status.

        Raises:
            JobFailedError: If the job ended in failure; the message holds the log tail.
        """
        if follow:
            try:
                self._client.jobs.stream(self._name)
            except Exception as error:  # noqa: BLE001
                logger.warning("Streaming of job {} stopped: {}", self._name, error)

        status = self.status()

        while not status.is_final:
            sleep(self.POLL_SECONDS)
            status = self.status()

        if status is JobStatus.FAILED:
            raise JobFailedError(self._name, self.logs()[-self.LOG_TAIL_LINES :])

        return status

    def logs(self) -> list[str]:
        """Download the outputs of the finished job and return the lines of its standard log.

        Returns:
            The log lines, or an empty list when the job wrote no log.

        Raises:
            JobError: If the job has not ended yet or the download fails.
        """
        with tempfile.TemporaryDirectory() as download_dir:
            try:
                self._client.jobs.download(self._name, download_path=download_dir)
            except AzureError as error:
                error_msg = f"Could not download the log of job {self._name}: {error}"
                raise JobError(error_msg) from error

            log = Path(download_dir) / self.STD_LOG

            if not log.exists():
                return []

            return log.read_text().splitlines()

    def cancel(self) -> None:
        """Request cancellation and wait for Azure to acknowledge it.

        Raises:
            JobError: If Azure refuses or cannot be reached.
        """
        try:
            self._client.jobs.begin_cancel(self._name).result()
        except AzureError as error:
            error_msg = f"Could not cancel job {self._name}: {error}"
            raise JobError(error_msg) from error
