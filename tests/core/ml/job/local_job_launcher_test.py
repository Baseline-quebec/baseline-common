"""Tests for the subprocess launcher."""

import sys
from pathlib import Path
from typing import ClassVar

import pytest

from baseline_core.core.ml.job.errors import JobFailedError, JobSubmissionError
from baseline_core.core.ml.job.job_spec import JobSpec, JobStatus
from baseline_core.core.ml.job.local_job_launcher import LocalJobLauncher


class TestLocalJobLauncher:
    """Tests for ``LocalJobLauncher`` and the handles it returns."""

    LINE_COUNT: ClassVar[int] = 5000

    @pytest.fixture
    def launcher(self, tmp_path: Path) -> LocalJobLauncher:
        """Launcher running commands from a temporary directory."""
        return LocalJobLauncher(working_dir=tmp_path)

    def test__submit__when_command_succeeds__should_end_succeeded(
        self, launcher: LocalJobLauncher
    ) -> None:
        """A zero exit code is a success."""
        spec = JobSpec(command=f"{sys.executable} -c \"print('hello')\"")

        handle = launcher.submit(spec)
        status = handle.wait(follow=False)

        assert status is JobStatus.SUCCEEDED
        assert handle.status() is JobStatus.SUCCEEDED
        assert handle.logs() == ["hello"]

    def test__submit__when_executable_missing__should_raise_job_submission_error(
        self, launcher: LocalJobLauncher
    ) -> None:
        """A command that cannot start is refused like a backend would refuse it."""
        spec = JobSpec(command="no-such-executable --flag", monitor_interval_s=None)

        with pytest.raises(JobSubmissionError, match="no-such-executable"):
            launcher.submit(spec)

    def test__wait__when_command_fails__should_raise_job_failed_error(
        self, launcher: LocalJobLauncher
    ) -> None:
        """A non-zero exit code raises, with the output in the message."""
        spec = JobSpec(command=f"{sys.executable} -c \"print('boom'); raise SystemExit(3)\"")

        handle = launcher.submit(spec)

        with pytest.raises(JobFailedError, match="boom"):
            handle.wait(follow=False)

        assert handle.status() is JobStatus.FAILED

    def test__submit__should_replace_input_placeholders_with_local_paths(
        self, launcher: LocalJobLauncher, tmp_path: Path
    ) -> None:
        """Inputs are plain local folders for this launcher."""
        data = tmp_path / "frames"
        data.mkdir()
        spec = JobSpec(
            command=f'{sys.executable} -c "import sys; print(sys.argv[1])" ${{{{inputs.data}}}}',
            inputs={"data": str(data)},
        )

        handle = launcher.submit(spec)
        handle.wait(follow=False)

        assert handle.logs() == [str(data)]

    def test__submit__should_replace_output_placeholders_with_local_paths(
        self, launcher: LocalJobLauncher, tmp_path: Path
    ) -> None:
        """Outputs are plain local folders for this launcher."""
        spec = JobSpec(
            command=f'{sys.executable} -c "import sys; print(sys.argv[1])" ${{{{outputs.out}}}}',
            outputs={"out": str(tmp_path)},
        )

        handle = launcher.submit(spec)
        handle.wait(follow=False)

        assert handle.logs() == [str(tmp_path)]

    def test__submit__should_pass_environment_variables(self, launcher: LocalJobLauncher) -> None:
        """Environment variables of the spec reach the process."""
        spec = JobSpec(
            command=f"{sys.executable} -c \"import os; print(os.environ['JOB_FLAG'])\"",
            environment_variables={"JOB_FLAG": "on"},
        )

        handle = launcher.submit(spec)
        handle.wait(follow=False)

        assert handle.logs() == ["on"]

    def test__logs__when_output_is_large__should_not_block(
        self, launcher: LocalJobLauncher
    ) -> None:
        """Output is drained continuously, so a chatty job never fills the pipe."""
        spec = JobSpec(
            command=f"{sys.executable} -c \"for i in range({self.LINE_COUNT}): print('line', i)\""
        )

        handle = launcher.submit(spec)
        status = handle.wait(follow=False)

        assert status is JobStatus.SUCCEEDED
        assert len(handle.logs()) == self.LINE_COUNT

    def test__cancel__should_end_canceled(self, launcher: LocalJobLauncher) -> None:
        """A canceled job reports ``CANCELED``, not ``FAILED``."""
        spec = JobSpec(command=f'{sys.executable} -c "import time; time.sleep(30)"')
        handle = launcher.submit(spec)

        handle.cancel()
        status = handle.wait(follow=False)

        assert status is JobStatus.CANCELED

    def test__submit__should_log_resource_usage_while_command_runs(
        self, launcher: LocalJobLauncher
    ) -> None:
        """The command runs under the resource monitor, whose lines join the job output."""
        spec = JobSpec(
            command=f'{sys.executable} -c "import time; time.sleep(1)"', monitor_interval_s=0.2
        )

        handle = launcher.submit(spec)
        handle.wait(follow=False)

        assert any("[monitor " in line for line in handle.logs())

    def test__wait__when_follow__should_print_output(
        self, launcher: LocalJobLauncher, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """With ``follow`` the output goes to the terminal as it arrives."""
        spec = JobSpec(command=f"{sys.executable} -c \"print('visible')\"")

        launcher.submit(spec).wait(follow=True)

        assert "visible" in capsys.readouterr().out
