"""Tests for the Azure ML job handle against a mocked client."""

from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from azure.ai.ml._restclient.arm_ml_service.models import JobStatus as SdkJobStatus  # noqa: PLC2701
from azure.ai.ml.exceptions import JobException
from azure.core.exceptions import ResourceNotFoundError, ServiceRequestError
from pytest_mock import MockerFixture

from baseline_core.core.ml.job.azure_ml.azure_ml_job_handle import AzureMLJobHandle
from baseline_core.core.ml.job.errors import JobError, JobFailedError, JobNotFoundError
from baseline_core.core.ml.job.job_spec import JobStatus


class TestAzureMLJobHandle:
    """Tests for ``AzureMLJobHandle``."""

    POLLED_STATUSES: ClassVar[list[str]] = ["Queued", "Running", "Completed"]

    @pytest.fixture
    def handle(self, client: MagicMock, azure_job: MagicMock) -> AzureMLJobHandle:
        """Handle on ``job-123``, which the client reports as completed."""
        client.jobs.get.return_value = azure_job
        return AzureMLJobHandle(client, "job-123")

    @pytest.fixture
    def write_log(self, client: MagicMock) -> None:
        """Make ``client.jobs.download`` write a two-line standard log in the download dir."""

        def download(_name: str, download_path: str, **_kwargs: bool) -> None:
            """Write the standard log under ``download_path``."""
            log = Path(download_path) / "artifacts" / "user_logs" / "std_log.txt"
            log.parent.mkdir(parents=True)
            log.write_text("epoch 1\nepoch 2\n")

        client.jobs.download.side_effect = download

    @pytest.mark.parametrize(
        ("azure_status", "expected"),
        [
            ("NotStarted", JobStatus.QUEUED),
            ("Queued", JobStatus.QUEUED),
            ("Starting", JobStatus.QUEUED),
            ("Preparing", JobStatus.QUEUED),
            ("Provisioning", JobStatus.QUEUED),
            ("Running", JobStatus.RUNNING),
            ("Finalizing", JobStatus.RUNNING),
            ("Completed", JobStatus.SUCCEEDED),
            ("Failed", JobStatus.FAILED),
            ("Canceled", JobStatus.CANCELED),
            ("CancelRequested", JobStatus.CANCELED),
        ],
    )
    def test__status__should_map_azure_statuses(
        self,
        handle: AzureMLJobHandle,
        azure_job: MagicMock,
        azure_status: str,
        expected: JobStatus,
    ) -> None:
        """Every Azure status maps onto the five common ones."""
        azure_job.status = azure_status

        assert handle.status() is expected

    @pytest.mark.parametrize(
        "azure_status",
        [
            status
            for status in SdkJobStatus
            if status.value not in {"NotResponding", "Paused", "Unknown"}
        ],
    )
    def test__status__should_read_the_sdk_enum_by_its_value(
        self, handle: AzureMLJobHandle, azure_job: MagicMock, azure_status: SdkJobStatus
    ) -> None:
        """``jobs.get`` returns the SDK's ``JobStatus`` enum, whose ``str`` is not its value."""
        azure_job.status = azure_status

        assert handle.status() is handle.AZURE_STATUS[azure_status.value]

    def test__status__when_unknown_azure_status__should_raise_job_error(
        self, handle: AzureMLJobHandle, azure_job: MagicMock
    ) -> None:
        """A status this code does not know is reported, not guessed."""
        azure_job.status = "Paused"

        with pytest.raises(JobError, match="Paused"):
            handle.status()

    def test__status__when_job_missing__should_raise_job_not_found(
        self, client: MagicMock, handle: AzureMLJobHandle
    ) -> None:
        """Azure's not-found error becomes ours."""
        client.jobs.get.side_effect = ResourceNotFoundError("gone")

        with pytest.raises(JobNotFoundError, match="job-123"):
            handle.status()

    def test__status__when_azure_fails__should_raise_job_error(
        self, client: MagicMock, handle: AzureMLJobHandle
    ) -> None:
        """Other Azure errors (network, authentication) become ``JobError``."""
        client.jobs.get.side_effect = ServiceRequestError("network down")

        with pytest.raises(JobError, match="network down"):
            handle.status()

    def test__logs__when_job_not_finished__should_raise_job_error(
        self, client: MagicMock, handle: AzureMLJobHandle
    ) -> None:
        """Azure ML only serves the log of a finished job."""
        client.jobs.download.side_effect = JobException(
            "Job is not in a terminal state.", no_personal_data_message="not terminal"
        )

        with pytest.raises(JobError, match="terminal state"):
            handle.logs()

    def test__cancel__when_azure_fails__should_raise_job_error(
        self, client: MagicMock, handle: AzureMLJobHandle
    ) -> None:
        """Cancellation errors do not leak SDK types."""
        client.jobs.begin_cancel.side_effect = ServiceRequestError("network down")

        with pytest.raises(JobError, match="network down"):
            handle.cancel()

    @pytest.mark.usefixtures("write_log")
    def test__logs__should_return_lines_of_standard_log(
        self, client: MagicMock, handle: AzureMLJobHandle
    ) -> None:
        """The standard log is downloaded and split into lines."""
        lines = handle.logs()

        assert lines == ["epoch 1", "epoch 2"]
        client.jobs.download.assert_called_once()

    def test__logs__when_no_log_yet__should_return_empty_list(
        self, handle: AzureMLJobHandle
    ) -> None:
        """A job that has not started has no log file; that is not an error."""
        assert handle.logs() == []

    def test__wait__when_follow__should_stream_and_return_final_status(
        self, client: MagicMock, handle: AzureMLJobHandle
    ) -> None:
        """Streaming delegates to the SDK, then the status is read once more."""
        status = handle.wait(follow=True)

        client.jobs.stream.assert_called_once_with("job-123")
        assert status is JobStatus.SUCCEEDED

    @pytest.mark.usefixtures("write_log")
    def test__wait__when_job_failed__should_raise_with_log_tail(
        self, client: MagicMock, handle: AzureMLJobHandle, azure_job: MagicMock
    ) -> None:
        """Failure raises even if the SDK's stream raised first."""
        azure_job.status = "Failed"
        client.jobs.stream.side_effect = RuntimeError("stream aborted")

        with pytest.raises(JobFailedError, match="epoch 2"):
            handle.wait(follow=True)

    def test__wait__when_not_following__should_poll_until_final(
        self,
        client: MagicMock,
        handle: AzureMLJobHandle,
        azure_job: MagicMock,
        mocker: MockerFixture,
    ) -> None:
        """Without streaming, the status is polled until it is final."""
        statuses = iter(self.POLLED_STATUSES)
        type(azure_job).status = mocker.PropertyMock(side_effect=lambda: next(statuses))
        sleep = mocker.patch("baseline_core.core.ml.job.azure_ml.azure_ml_job_handle.sleep")

        status = handle.wait(follow=False)

        assert status is JobStatus.SUCCEEDED
        assert sleep.call_count == len(self.POLLED_STATUSES) - 1
        client.jobs.stream.assert_not_called()

    def test__cancel__should_request_cancellation(
        self, client: MagicMock, handle: AzureMLJobHandle
    ) -> None:
        """Cancellation goes through the SDK's long-running operation."""
        handle.cancel()

        client.jobs.begin_cancel.assert_called_once_with("job-123")
        client.jobs.begin_cancel.return_value.result.assert_called_once()
