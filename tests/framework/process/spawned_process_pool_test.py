"""Each job runs in a spawned process; a dead one is replaced, and none outlives its parent."""

import asyncio
import multiprocessing
import os
import threading
import time
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path

import pytest

from baseline_common.framework.process.spawned_process_pool import (
    SpawnedProcessDiedError,
    SpawnedProcessPool,
)
from tests.framework.process.workers import answer_or_die, hold, host


@pytest.fixture
def pool() -> Iterator[SpawnedProcessPool]:
    """A pool of one process, whose worker answers with its process's id or dies.

    Yields:
        The pool, closed after the test.
    """
    pool = SpawnedProcessPool(1, answer_or_die)
    yield pool
    pool.close()


@pytest.fixture
def started(tmp_path: Path) -> Path:
    """The file a held job writes its process's id in, once it runs.

    Returns:
        Its path, not written yet.
    """
    return tmp_path / "started"


class TestSpawnedProcessPool:
    @staticmethod
    def until(condition: Callable[[], bool]) -> None:
        """Wait until a spawned process makes the condition true.

        Args:
            condition: What the process makes true.

        Raises:
            TimeoutError: When it is still false after ten seconds.
        """
        deadline = time.monotonic() + 10

        while not condition():
            if time.monotonic() > deadline:
                msg = "A spawned process did not act within ten seconds"
                raise TimeoutError(msg)

            time.sleep(0.01)

    def written(self, path: Path) -> str:
        """Wait until a spawned process has written the file, then read it.

        Args:
            path: The file, written whole at once.

        Returns:
            What it holds.
        """
        self.until(path.exists)
        return path.read_text(encoding="utf-8")

    @staticmethod
    def has_ended(process_id: int) -> bool:
        """Whether no process of this id exists any more.

        Args:
            process_id: The process's id.

        Returns:
            True once it has ended and been reaped.
        """
        try:
            os.kill(process_id, 0)
        except ProcessLookupError:
            return True

        return False

    async def test__a_job__runs_in_a_spawned_process_while_the_event_loop_runs(
        self, pool: SpawnedProcessPool
    ) -> None:
        """The worker runs away from the test's process, and the loop ticks meanwhile."""
        running = asyncio.ensure_future(pool.run(False))  # noqa: FBT003 - the job is a flag

        await asyncio.sleep(0.05)
        waiting = not running.done()
        process_id = await running

        assert waiting
        assert process_id != os.getpid()

    async def test__two_jobs__run_in_two_fresh_processes(self, pool: SpawnedProcessPool) -> None:
        """A process runs one job only: the next one starts in its replacement."""
        first = await pool.run(False)  # noqa: FBT003 - the job is a flag

        second = await pool.run(False)  # noqa: FBT003 - the job is a flag

        assert first != second

    async def test__a_dead_process__raises_and_the_next_job_gets_a_new_pool(
        self, pool: SpawnedProcessPool
    ) -> None:
        """A process killed mid-job breaks its job only: the next one runs in a fresh pool."""
        with pytest.raises(SpawnedProcessDiedError, match="died before answering"):
            await pool.run(True)  # noqa: FBT003 - the job is a flag

        assert await pool.run(False) != os.getpid()  # noqa: FBT003 - the job is a flag

    async def test__closing__terminates_a_busy_process(self, started: Path) -> None:
        """A job still running when the pool closes is stopped, and its caller told it died."""
        pool = SpawnedProcessPool(1, hold)
        running = asyncio.ensure_future(pool.run(started))
        process_id = int(await asyncio.to_thread(self.written, started))

        pool.close()

        with pytest.raises(SpawnedProcessDiedError):
            await running

        # The pool's own thread reaps the process: the test can only see that it is gone.
        await asyncio.to_thread(self.until, partial(self.has_ended, process_id))

    async def test__closing__returns_once_the_pool_s_thread_has_ended(self) -> None:
        """Its thread still replacing the answered process would find the pool already gone."""
        before = set(threading.enumerate())
        pool = SpawnedProcessPool(1, answer_or_die)
        await pool.run(False)  # noqa: FBT003 - the job is a flag

        pool.close()

        assert set(threading.enumerate()) <= before

    def test__the_parent_s_death__ends_its_pool_process(self, started: Path) -> None:
        """A pool process watches the process that spawned it, and ends when it dies."""
        parent = multiprocessing.get_context("spawn").Process(target=host, args=[started])
        parent.start()
        process_id = int(self.written(started))

        parent.kill()
        parent.join()

        self.until(partial(self.has_ended, process_id))

    def test__no_process__is_refused(self) -> None:
        """A pool needs at least one process to run a job in."""
        with pytest.raises(ValueError, match="max_workers must be greater than 0"):
            SpawnedProcessPool(0, answer_or_die)
