"""SpawnedProcessPool — run each job in a spawned process, never in the caller's."""

import asyncio
import multiprocessing
import os
import threading
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing.connection import wait

from loguru import logger


class SpawnedProcessDiedError(RuntimeError):
    """The process running a job died before answering."""


class SpawnedProcessPool[J, R]:
    """Hand each job to a spawned process, replaced after it, while the caller only waits."""

    def __init__(self, processes: int, worker: Callable[[J], R]) -> None:
        """Keep the settings and the work; the first process starts with the first job.

        Args:
            processes: How many jobs run at once, each in a process of its own.
            worker: What a process runs for a job: a function it imports by its name.
        """
        self.processes = processes
        """How many jobs run at once, each in a process of its own."""
        self.worker = worker
        """What a process runs for a job: a function it imports by its name."""
        self._executor = self.new_executor()

    def new_executor(self) -> ProcessPoolExecutor:
        """Spawned processes, one per job: a fork would copy the caller's threads and connections.

        Returns:
            The pool; a process's replacement starts as soon as it ends, and imports while idle.
        """
        return ProcessPoolExecutor(
            max_workers=self.processes,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=SpawnedProcessPool.watch_parent,
            max_tasks_per_child=1,
        )

    async def run(self, job: J) -> R:
        """Run the job in a process of the pool, the caller's event loop free meanwhile.

        Args:
            job: What the worker receives.

        Returns:
            What the worker returned.

        Raises:
            SpawnedProcessDiedError: When the process died; the next job gets a new pool.
        """
        loop = asyncio.get_running_loop()

        try:
            return await loop.run_in_executor(self._executor, self.worker, job)
        except BrokenProcessPool as error:
            self._executor = self.new_executor()
            msg = "A pool process died before answering; the next job gets a new one"
            raise SpawnedProcessDiedError(msg) from error

    def close(self) -> None:
        """Stop the processes, a job running included: its caller gets a died error."""
        # Python 3.12 has no public way to stop a busy worker; 3.14 adds terminate_workers().
        for process in list(self._executor._processes.values()):  # noqa: SLF001
            process.terminate()

        self._executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def watch_parent() -> None:
        """In a pool process, end it with the process that spawned it: no job outlives its caller."""
        parent = multiprocessing.parent_process()

        if parent is not None:
            watch = threading.Thread(
                target=SpawnedProcessPool.end_with, args=[parent.sentinel], daemon=True
            )
            watch.start()

    @staticmethod
    def end_with(sentinel: int) -> None:
        """Wait for the parent process to end, then end this one at once.

        Args:
            sentinel: Ready when the parent process has ended.
        """
        wait([sentinel])
        logger.warning("The parent process ended: its pool process stops.")
        os._exit(1)
