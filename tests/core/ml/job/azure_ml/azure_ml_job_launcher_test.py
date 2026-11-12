"""Tests for the Azure ML launcher and its settings."""

import os
import re
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import Environment
from azure.core.exceptions import HttpResponseError, ServiceRequestError

from baseline_core.core.ml.job.azure_ml.azure_ml_job_launcher import (
    AzureMLConfig,
    AzureMLJobLauncher,
)
from baseline_core.core.ml.job.errors import JobError, JobSubmissionError
from baseline_core.core.ml.job.job_spec import JobSpec, JobStatus


class TestAzureMLJobLauncher:
    """Tests for ``AzureMLJobLauncher`` against a mocked client."""

    PROJECT_FILES: ClassVar[tuple[str, ...]] = ("pyproject.toml", "uv.lock", "README.md")

    @pytest.fixture
    def config(self, tmp_path: Path) -> AzureMLConfig:
        """Settings pointing at a fake workspace, code in a temporary directory."""
        return AzureMLConfig(
            subscription_id="sub-1",
            resource_group="rg-ml-dev",
            workspace_name="mlw-ml-dev",
            code_dir=tmp_path,
        )

    @pytest.fixture
    def launcher(
        self, config: AzureMLConfig, client: MagicMock, azure_job: MagicMock
    ) -> AzureMLJobLauncher:
        """Launcher using the mocked client, which accepts every job."""
        client.jobs.create_or_update.return_value = azure_job
        return AzureMLJobLauncher(config, client=client)

    @pytest.fixture
    def dockerfile(self, tmp_path: Path) -> Path:
        """A Dockerfile at its repository path, next to the project files it installs from."""
        dockerfile = tmp_path / "docker" / "azure-ml" / "Dockerfile"
        dockerfile.parent.mkdir(parents=True)
        dockerfile.write_text("FROM python:3.12-slim\n")

        for name in self.PROJECT_FILES:
            (tmp_path / name).write_text(f"# {name}\n")

        return dockerfile

    def test__submit__should_build_command_job_from_spec(
        self, launcher: AzureMLJobLauncher, client: MagicMock, spec: JobSpec
    ) -> None:
        """Every field of the spec lands on the Azure ML command job."""
        handle = launcher.submit(spec)

        job = client.jobs.create_or_update.call_args.args[0]
        assert job.command == spec.launch_command(spec.command, python="python")
        assert job.inputs["dataset"].type == AssetTypes.URI_FOLDER
        assert job.inputs["dataset"].path == spec.inputs["dataset"]
        assert job.inputs["dataset"].mode == InputOutputModes.RO_MOUNT
        assert job.environment_variables == {
            **AzureMLJobLauncher.DEFAULT_ENVIRONMENT_VARIABLES,
            **spec.environment_variables,
        }
        assert job.display_name == spec.display_name
        assert job.experiment_name == spec.experiment_name
        assert handle.id == "job-123"

    def test__submit__should_mount_outputs_read_write(
        self, launcher: AzureMLJobLauncher, client: MagicMock
    ) -> None:
        """An output folder is mounted so the job writes straight to blob storage."""
        spec = JobSpec(
            command="python teach.py --dataset ${{outputs.dataset}}",
            outputs={"dataset": "azureml://datastores/datasets/paths/frames_v1"},
        )

        launcher.submit(spec)

        job = client.jobs.create_or_update.call_args.args[0]
        assert job.outputs["dataset"].type == AssetTypes.URI_FOLDER
        assert job.outputs["dataset"].path == spec.outputs["dataset"]
        assert job.outputs["dataset"].mode == InputOutputModes.RW_MOUNT

    def test__submit__should_name_job_after_experiment_and_start_time(
        self, launcher: AzureMLJobLauncher, client: MagicMock, spec: JobSpec
    ) -> None:
        """A readable job name is what a later job passes to restart from this one."""
        launcher.submit(spec)

        job = client.jobs.create_or_update.call_args.args[0]
        assert re.fullmatch(rf"{spec.experiment_name}_\d{{8}}_\d{{6}}", job.name)

    def test__submit__should_give_dataloader_workers_shared_memory(
        self, launcher: AzureMLJobLauncher, client: MagicMock, config: AzureMLConfig, spec: JobSpec
    ) -> None:
        """DataLoader workers pass batches through /dev/shm, which is tiny by default."""
        launcher.submit(spec)

        job = client.jobs.create_or_update.call_args.args[0]
        assert job.resources.shm_size == config.shm_size

    def test__submit__when_input_mode_download__should_download_every_input(
        self, config: AzureMLConfig, client: MagicMock, azure_job: MagicMock, spec: JobSpec
    ) -> None:
        """Inputs are copied to the node's disk before the command, when asked."""
        client.jobs.create_or_update.return_value = azure_job
        launcher = AzureMLJobLauncher(
            config.model_copy(update={"input_mode": "download"}), client=client
        )

        launcher.submit(spec)

        job = client.jobs.create_or_update.call_args.args[0]
        assert job.inputs["dataset"].mode == InputOutputModes.DOWNLOAD

    def test__submit__when_monitor_disabled__should_run_command_as_is(
        self, launcher: AzureMLJobLauncher, client: MagicMock
    ) -> None:
        """Without a monitor interval the command is not wrapped."""
        spec = JobSpec(command="echo hi", monitor_interval_s=None)

        launcher.submit(spec)

        assert client.jobs.create_or_update.call_args.args[0].command == "echo hi"

    def test__submit__when_compute_and_environment_unset__should_use_config_defaults(
        self, launcher: AzureMLJobLauncher, client: MagicMock, spec: JobSpec
    ) -> None:
        """The cluster and environment default to the settings."""
        launcher.submit(spec)

        job = client.jobs.create_or_update.call_args.args[0]
        assert job.compute == "gpu"
        assert job.environment == "training@latest"

    def test__submit__when_spec_overrides__should_use_them(
        self, launcher: AzureMLJobLauncher, client: MagicMock
    ) -> None:
        """A spec may target another cluster or environment."""
        spec = JobSpec(command="echo hi", compute="cpu-cluster", environment="other")

        launcher.submit(spec)

        job = client.jobs.create_or_update.call_args.args[0]
        assert job.compute == "cpu-cluster"
        assert job.environment == "other@latest"

    def test__submit__when_azure_refuses__should_raise_job_submission_error(
        self, launcher: AzureMLJobLauncher, client: MagicMock, spec: JobSpec
    ) -> None:
        """Azure errors are wrapped so callers never depend on the SDK."""
        client.jobs.create_or_update.side_effect = HttpResponseError("quota exceeded")

        with pytest.raises(JobSubmissionError, match="quota exceeded"):
            launcher.submit(spec)

    def test__submit__when_network_fails__should_raise_job_submission_error(
        self, launcher: AzureMLJobLauncher, client: MagicMock, spec: JobSpec
    ) -> None:
        """Errors raised before any HTTP response are wrapped too."""
        client.jobs.create_or_update.side_effect = ServiceRequestError("network down")

        with pytest.raises(JobSubmissionError, match="network down"):
            launcher.submit(spec)

    @pytest.mark.usefixtures("dockerfile")
    def test__register_environment__when_azure_fails__should_raise_job_error(
        self, launcher: AzureMLJobLauncher, client: MagicMock, dockerfile: Path
    ) -> None:
        """A failed registration does not leak SDK types."""
        client.environments.create_or_update.side_effect = HttpResponseError("forbidden")

        with pytest.raises(JobError, match="forbidden"):
            launcher.register_environment(dockerfile=dockerfile)

    def test__register_environment__should_build_from_dockerfile_and_project_files(
        self, launcher: AzureMLJobLauncher, client: MagicMock, dockerfile: Path
    ) -> None:
        """The build context holds the Dockerfile plus the files it installs from."""
        contexts: list[set[str]] = []

        def register(environment: Environment) -> MagicMock:
            """Record the build context of ``environment`` and return it registered."""
            assert environment.build is not None
            assert isinstance(environment.build.path, str)
            contexts.append({path.name for path in Path(environment.build.path).iterdir()})
            registered = MagicMock()
            registered.name = environment.name
            registered.version = "3"
            return registered

        client.environments.create_or_update.side_effect = register

        reference = launcher.register_environment(dockerfile=dockerfile)

        assert reference == "training:3"
        assert contexts == [{"Dockerfile", *self.PROJECT_FILES}]


@pytest.mark.usefixtures("environment")
class TestAzureMLConfig:
    """Tests for ``AzureMLConfig``."""

    @pytest.fixture
    def environment(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Required settings in the environment, and no ``.env`` file in reach."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AZURE_ML_SUBSCRIPTION_ID", "sub-1")
        monkeypatch.setenv("AZURE_ML_RESOURCE_GROUP", "rg-ml-dev")
        monkeypatch.setenv("AZURE_ML_WORKSPACE_NAME", "mlw-ml-dev")

    def test__init__should_read_prefixed_environment_variables(self) -> None:
        """Settings come from ``AZURE_ML_*`` variables."""
        config = AzureMLConfig()

        assert config.subscription_id == "sub-1"
        assert config.resource_group == "rg-ml-dev"
        assert config.workspace_name == "mlw-ml-dev"

    def test__init__should_default_compute_and_environment(self) -> None:
        """The Terraform cluster and environment names are the defaults."""
        config = AzureMLConfig()

        assert config.compute == "gpu"
        assert config.environment == "training"
        assert config.code_dir == Path()
        assert config.input_mode == InputOutputModes.RO_MOUNT

    def test__init__when_workspace_missing__should_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing required setting fails at construction."""
        monkeypatch.delenv("AZURE_ML_WORKSPACE_NAME")

        with pytest.raises(ValueError, match="workspace_name"):
            AzureMLConfig()


@pytest.mark.azure
@pytest.mark.skipif(
    "AZURE_ML_WORKSPACE_NAME" not in os.environ, reason="AZURE_ML_* not configured"
)
class TestAzureMLRoundTrip:
    """One real job, end to end, on a real workspace; run by hand with ``pytest -m azure``."""

    def test__submit__when_echo_job__should_succeed_and_expose_logs(self) -> None:
        """A trivial job runs on the cluster and its output comes back."""
        launcher = AzureMLJobLauncher(AzureMLConfig())
        spec = JobSpec(command="echo launcher-ok", display_name="launcher-smoke-test")

        handle = launcher.submit(spec)
        status = handle.wait(follow=True)

        assert status is JobStatus.SUCCEEDED
        assert any("launcher-ok" in line for line in handle.logs())
