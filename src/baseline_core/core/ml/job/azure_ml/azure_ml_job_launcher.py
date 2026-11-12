"""Submit jobs to an Azure ML workspace and build their environment image."""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Literal

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import BuildContext, Environment
from azure.core.exceptions import AzureError
from azure.identity import DefaultAzureCredential
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

from baseline_core.core.ml.job.abstract_job_launcher import AbstractJobLauncher
from baseline_core.core.ml.job.azure_ml.azure_ml_job_handle import AzureMLJobHandle
from baseline_core.core.ml.job.errors import JobError, JobSubmissionError
from baseline_core.core.ml.job.job_spec import JobSpec


class AzureMLConfig(BaseSettings):
    """Azure ML workspace and job defaults, read from ``AZURE_ML_<FIELD>`` or ``.env``.

    Attributes:
        subscription_id: Azure subscription of the workspace.
        resource_group: Resource group of the workspace.
        workspace_name: Azure ML workspace.
        compute: Default compute cluster for jobs.
        environment: Default registered environment for jobs.
        code_dir: Directory uploaded as the job code: the root of the project that runs it.
        shm_size: Shared memory of the job container; DataLoader workers pass batches through it.
        input_mode: ``ro_mount`` streams inputs from blob storage; ``download`` copies them first.
    """

    model_config = SettingsConfigDict(env_prefix="AZURE_ML_", env_file=".env", extra="ignore")

    subscription_id: str
    resource_group: str
    workspace_name: str
    compute: str = "gpu"
    environment: str = "training"
    code_dir: Path = Path()
    shm_size: str = "8g"
    input_mode: Literal["ro_mount", "download"] = "ro_mount"


class AzureMLJobLauncher(AbstractJobLauncher):
    """Runs jobs on an Azure ML compute cluster, with ``config.code_dir`` as the job code.

    Attributes:
        DOCKERFILE: Dockerfile of the job environment, relative to ``config.code_dir``.
        BUILD_CONTEXT_FILES: Project files copied next to the Dockerfile for ``uv sync``.
        DEFAULT_ENVIRONMENT_VARIABLES: Set in every job, under the spec's own (system metrics).
        JOB_PYTHON: Python interpreter of the job environment.
        config: Workspace coordinates and defaults.
    """

    DOCKERFILE: ClassVar[Path] = Path("docker/azure-ml/Dockerfile")
    BUILD_CONTEXT_FILES: ClassVar[tuple[str, ...]] = ("pyproject.toml", "uv.lock", "README.md")
    DEFAULT_ENVIRONMENT_VARIABLES: ClassVar[dict[str, str]] = {
        "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "true"
    }
    JOB_PYTHON: ClassVar[str] = "python"

    def __init__(self, config: AzureMLConfig, client: MLClient | None = None) -> None:
        """Connect to the workspace.

        Args:
            config: Workspace coordinates and defaults.
            client: Workspace client, or ``None`` to build one with ``DefaultAzureCredential``.
        """
        self.config = config
        self._client = client or MLClient(
            DefaultAzureCredential(),
            config.subscription_id,
            config.resource_group,
            config.workspace_name,
        )

    def submit(self, spec: JobSpec) -> AzureMLJobHandle:
        """Upload the code and start a command job named ``<experiment>_<YYYYmmdd_HHMMSS>``.

        Args:
            spec: What to run; ``compute`` and ``environment`` fall back to the config.

        Returns:
            A handle whose ``id`` is the job name, also the MLflow run id to restart from.

        Raises:
            JobSubmissionError: If Azure refuses the job.
        """
        environment = spec.environment or self.config.environment
        start_time = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        job = command(
            name=f"{spec.experiment_name or 'job'}_{start_time}",
            code=str(self.config.code_dir),
            command=spec.launch_command(spec.command, python=self.JOB_PYTHON),
            inputs={
                name: Input(type=AssetTypes.URI_FOLDER, path=uri, mode=self.config.input_mode)
                for name, uri in spec.inputs.items()
            },
            outputs={
                name: Output(type=AssetTypes.URI_FOLDER, path=uri, mode=InputOutputModes.RW_MOUNT)
                for name, uri in spec.outputs.items()
            },
            environment=f"{environment}@latest",
            compute=spec.compute or self.config.compute,
            environment_variables={
                **self.DEFAULT_ENVIRONMENT_VARIABLES,
                **spec.environment_variables,
            },
            display_name=spec.display_name,
            experiment_name=spec.experiment_name,
            shm_size=self.config.shm_size,
        )

        try:
            created = self._client.jobs.create_or_update(job)
        except AzureError as error:
            error_msg = f"Azure ML refused the job: {error}"
            raise JobSubmissionError(error_msg) from error

        logger.info("Submitted job {} ({})", created.name, created.studio_url)
        return AzureMLJobHandle(self._client, str(created.name))

    def register_environment(self, name: str | None = None, dockerfile: Path | None = None) -> str:
        """Build the job image from the Dockerfile and register it on the workspace.

        Args:
            name: Environment name, or ``None`` for the config default.
            dockerfile: Dockerfile under ``config.code_dir``, or ``None`` for ``DOCKERFILE``.

        Returns:
            The registered ``name:version``.

        Raises:
            JobError: If Azure refuses the environment or cannot be reached.
        """
        environment_name = name or self.config.environment
        dockerfile_path = self.config.code_dir / (dockerfile or self.DOCKERFILE)

        with tempfile.TemporaryDirectory() as context_dir:
            context = Path(context_dir)
            shutil.copy(dockerfile_path, context / "Dockerfile")

            for file_name in self.BUILD_CONTEXT_FILES:
                shutil.copy(self.config.code_dir / file_name, context / file_name)

            environment = Environment(
                name=environment_name,
                build=BuildContext(path=str(context)),
                description="The project with CUDA torch and its training dependencies.",
            )

            try:
                registered = self._client.environments.create_or_update(environment)
            except AzureError as error:
                error_msg = f"Azure ML refused environment {environment_name}: {error}"
                raise JobError(error_msg) from error

        reference = f"{registered.name}:{registered.version}"
        logger.info("Registered environment {}", reference)
        return reference
