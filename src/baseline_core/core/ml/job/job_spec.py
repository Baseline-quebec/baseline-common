"""What a job is: a command, its inputs and where it runs."""

import re
import shlex
from collections.abc import Mapping
from enum import StrEnum
from typing import ClassVar, Self

from pydantic import BaseModel, Field, model_validator


class JobStatus(StrEnum):
    """Lifecycle of a job, common to every launcher."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"

    @property
    def is_final(self) -> bool:
        """Whether the job is over, whatever the outcome."""
        return self in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELED}


class JobSpec(BaseModel):
    """Everything a launcher needs to run one job.

    Attributes:
        command: Shell command run from the repository root; ``${{inputs.x}}`` becomes a path.
        inputs: Read-only folder URI of each input, by name (``azureml://`` or a local path).
        outputs: Read-write folder URI of each output, by name, in the same forms.
        environment_variables: Set in the job's process.
        display_name: Human-readable name shown by the launcher.
        experiment_name: Group of jobs the launcher files this one under.
        compute: Compute target, or ``None`` for the launcher's default.
        environment: Software environment, or ``None`` for the launcher's default.
        monitor_interval_s: Seconds between two resource lines, or ``None`` for no monitor.
    """

    PLACEHOLDER: ClassVar[re.Pattern[str]] = re.compile(r"\$\{\{\s*(inputs|outputs)\.(\w+)\s*\}\}")
    """``${{inputs.<name>}}`` or ``${{outputs.<name>}}`` in a command, the Azure ML syntax."""

    MONITOR_MODULE: ClassVar[str] = f"{__package__}.monitor"
    """Module run with ``python -m`` to wrap a command; follows the package once it is copied."""

    command: str = Field(min_length=1)
    inputs: dict[str, str] = {}
    outputs: dict[str, str] = {}
    environment_variables: dict[str, str] = {}
    display_name: str | None = None
    experiment_name: str | None = None
    compute: str | None = None
    environment: str | None = None
    monitor_interval_s: float | None = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def validate_placeholders(self) -> Self:
        """Every ``${{inputs.x}}`` must name an input and every ``${{outputs.x}}`` an output."""
        declared = {"inputs": self.inputs, "outputs": self.outputs}

        for kind, names in declared.items():
            unknown = sorted({
                name
                for found_kind, name in self.PLACEHOLDER.findall(self.command)
                if found_kind == kind and name not in names
            })

            if unknown:
                error_msg = f"Command references undefined {kind}: {', '.join(unknown)}."
                raise ValueError(error_msg)

        return self

    def render_command(
        self, input_paths: Mapping[str, str], output_paths: Mapping[str, str] | None = None
    ) -> str:
        """Replace each placeholder with the path of that input or output.

        Args:
            input_paths: Path of each input, by name. Must cover every input placeholder.
            output_paths: Path of each output, by name. Must cover every output placeholder.

        Returns:
            The command with placeholders replaced.
        """
        paths = {"inputs": input_paths, "outputs": output_paths or {}}
        return self.PLACEHOLDER.sub(
            lambda match: paths[match.group(1)][match.group(2)], self.command
        )

    def launch_command(self, command: str, python: str) -> str:
        """Wrap ``command`` with the resource monitor, unless the monitor is disabled.

        Args:
            command: Command to run, with or without its placeholders replaced.
            python: Python interpreter of the job's environment.

        Returns:
            The command the launcher starts.
        """
        if self.monitor_interval_s is None:
            return command

        return shlex.join([
            python,
            "-m",
            self.MONITOR_MODULE,
            "--interval",
            str(self.monitor_interval_s),
            command,
        ])
