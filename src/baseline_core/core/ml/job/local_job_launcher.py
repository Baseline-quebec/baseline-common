"""Run a job as a subprocess: for tests, dry runs and machines with their own GPU."""

import os
import shlex
import subprocess  # noqa: S404
import sys
import threading
import time
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

from loguru import logger

from baseline_core.core.ml.job.abstract_job_handle import AbstractJobHandle
from baseline_core.core.ml.job.abstract_job_launcher import AbstractJobLauncher
from baseline_core.core.ml.job.errors import JobFailedError, JobSubmissionError
from baseline_core.core.ml.job.job_spec import JobSpec, JobStatus


class LocalJobHandle(AbstractJobHandle):
    """A running subprocess, with its output captured line by line.

    Attributes:
        POLL_SECONDS: Delay between two checks of the process in ``wait``.
        LOG_TAIL_LINES: Lines of output quoted in a ``JobFailedError``.
    """

    POLL_SECONDS: ClassVar[float] = 0.05
    LOG_TAIL_LINES: ClassVar[int] = 20

    def __init__(self, process: subprocess.Popen[str], job_id: str) -> None:
        """Start draining the process output.

        Args:
            process: Started process with ``stdout`` piped (stderr merged into it).
            job_id: Identifier reported by ``id``.
        """
        self._process = process
        self._id = job_id
        self._lines: list[str] = []
        self._canceled = False
        # Drained from the start so the process never blocks on a full pipe, and ``logs()``
        # works before and after ``wait()``.
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    @property
    def id(self) -> str:
        """Identifier of the job."""
        return self._id

    def status(self) -> JobStatus:
        """Status derived from the process state and exit code."""
        return_code = self._process.poll()

        if return_code is None:
            return JobStatus.RUNNING

        if self._canceled:
            return JobStatus.CANCELED

        return JobStatus.SUCCEEDED if return_code == 0 else JobStatus.FAILED

    def wait(self, *, follow: bool = True) -> JobStatus:
        """Block until the process exits.

        Args:
            follow: Write the output to stdout as it arrives.

        Returns:
            The final status.

        Raises:
            JobFailedError: If the exit code is not zero and the job was not canceled.
        """
        printed = 0

        while self._process.poll() is None:
            if follow:
                printed = self._print_from(printed)

            time.sleep(self.POLL_SECONDS)

        self._reader.join()

        if follow:
            self._print_from(printed)

        status = self.status()

        if status is JobStatus.FAILED:
            raise JobFailedError(self._id, self._lines[-self.LOG_TAIL_LINES :])

        return status

    def logs(self) -> list[str]:
        """Output lines captured so far."""
        return list(self._lines)

    def cancel(self) -> None:
        """Terminate the process."""
        if self._process.poll() is None:
            self._canceled = True
            self._process.terminate()

    def _drain(self) -> None:
        """Copy the process output into ``_lines`` until EOF, in the reader thread."""
        if self._process.stdout is None:
            return

        for line in self._process.stdout:
            self._lines.append(line.rstrip("\n"))

    def _print_from(self, index: int) -> int:
        """Write the lines from ``index`` on to stdout.

        Args:
            index: First line not printed yet.

        Returns:
            The index of the next line to print.
        """
        lines = self._lines[index:]

        for line in lines:
            sys.stdout.write(line + "\n")

        sys.stdout.flush()
        return index + len(lines)


class LocalJobLauncher(AbstractJobLauncher):
    """Runs jobs as subprocesses of the current machine.

    Attributes:
        working_dir: Directory the command runs from.
    """

    def __init__(self, working_dir: Path | None = None) -> None:
        """Choose where commands run.

        Args:
            working_dir: Directory the command runs from, or ``None`` for the current one.
        """
        self.working_dir = working_dir or Path.cwd()

    def submit(self, spec: JobSpec) -> LocalJobHandle:
        """Start the command in a subprocess, with placeholders replaced by local paths.

        Args:
            spec: What to run; its input and output URIs are local paths here.

        Returns:
            A handle on the process.

        Raises:
            JobSubmissionError: If the process cannot start (missing executable, bad directory).
        """
        command = spec.launch_command(
            spec.render_command(spec.inputs, spec.outputs), python=sys.executable
        )
        job_id = f"local-{uuid4().hex[:8]}"
        logger.info("Starting job {}: {}", job_id, command)

        try:
            process = subprocess.Popen(  # noqa: S603
                shlex.split(command),
                cwd=self.working_dir,
                env={**os.environ, **spec.environment_variables},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as error:
            error_msg = f"Could not start job {job_id}: {error}"
            raise JobSubmissionError(error_msg) from error

        return LocalJobHandle(process, job_id)
