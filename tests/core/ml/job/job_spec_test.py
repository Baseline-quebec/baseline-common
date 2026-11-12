"""Tests for the job vocabulary."""

import shlex

import pytest

from baseline_core.core.ml.job.job_spec import JobSpec, JobStatus


class TestJobStatus:
    """Tests for ``JobStatus``."""

    @pytest.mark.parametrize("status", [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELED])
    def test__is_final__when_terminal__should_be_true(self, status: JobStatus) -> None:
        """Terminal statuses end a ``wait``."""
        assert status.is_final

    @pytest.mark.parametrize("status", [JobStatus.QUEUED, JobStatus.RUNNING])
    def test__is_final__when_in_progress__should_be_false(self, status: JobStatus) -> None:
        """A queued or running job is not over."""
        assert not status.is_final


class TestJobSpec:
    """Tests for ``JobSpec`` validation and rendering."""

    def test__init__when_placeholder_has_no_input__should_raise_value_error(self) -> None:
        """A placeholder naming an undefined input is a configuration error."""
        with pytest.raises(ValueError, match="undefined inputs: dataset"):
            JobSpec(command="python train.py --data ${{inputs.dataset}}")

    def test__init__when_command_empty__should_raise_value_error(self) -> None:
        """A job needs a command."""
        with pytest.raises(ValueError, match="command"):
            JobSpec(command="")

    def test__render_command__should_replace_placeholders_with_paths(self, spec: JobSpec) -> None:
        """Placeholders become the paths given, for launchers that mount inputs themselves."""
        rendered = spec.render_command({"dataset": "/mnt/data/frames_v1"})

        assert rendered == "python train.py --data /mnt/data/frames_v1"

    def test__launch_command__should_run_command_under_resource_monitor(
        self, spec: JobSpec
    ) -> None:
        """By default the command runs under the monitor, as one quoted shell argument."""
        launched = spec.launch_command(spec.command, python="python3")

        assert shlex.split(launched) == [
            "python3",
            "-m",
            JobSpec.MONITOR_MODULE,
            "--interval",
            str(spec.monitor_interval_s),
            spec.command,
        ]

    def test__launch_command__when_monitor_disabled__should_return_command(self) -> None:
        """Without a monitor interval the command runs as is."""
        spec = JobSpec(command="echo hello", monitor_interval_s=None)

        assert spec.launch_command(spec.command, python="python3") == "echo hello"

    def test__init__when_monitor_interval_not_positive__should_raise_value_error(self) -> None:
        """A zero interval would make the monitor spin."""
        with pytest.raises(ValueError, match="monitor_interval_s"):
            JobSpec(command="echo hello", monitor_interval_s=0)

    def test__init__when_placeholder_has_no_output__should_raise_value_error(self) -> None:
        """A placeholder naming an undefined output is a configuration error."""
        with pytest.raises(ValueError, match="undefined outputs: dataset"):
            JobSpec(command="python teach.py --dataset ${{outputs.dataset}}")

    def test__render_command__should_replace_output_placeholders_too(self) -> None:
        """Inputs and outputs render from their own paths."""
        spec = JobSpec(
            command="python teach.py --images ${{inputs.images}} --out ${{outputs.dataset}}",
            inputs={"images": "azureml://datastores/datasets/paths/raw"},
            outputs={"dataset": "azureml://datastores/datasets/paths/frames_v1"},
        )

        rendered = spec.render_command({"images": "/mnt/in"}, {"dataset": "/mnt/out"})

        assert rendered == "python teach.py --images /mnt/in --out /mnt/out"

    def test__render_command__when_no_placeholder__should_return_command(self) -> None:
        """A command without inputs is returned unchanged."""
        spec = JobSpec(command="echo hello")

        assert spec.render_command({}) == "echo hello"
