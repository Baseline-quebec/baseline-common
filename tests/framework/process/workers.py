"""Workers sent by name to a pool's spawned processes: answer, die, or hold their process."""

import asyncio
import os
import time
from pathlib import Path

from baseline_common.framework.process.spawned_process_pool import SpawnedProcessPool


def answer_or_die(dies: bool) -> int:  # noqa: FBT001 - the pool sends the job alone
    """Answer with the process's id, or end the process at once, as a crash would.

    Args:
        dies: Whether the process ends instead of answering.

    Returns:
        The id of the process that ran the job.
    """
    if dies:
        os._exit(1)

    return os.getpid()


def hold(started: Path) -> None:
    """Write the process's id in a file, then keep the process busy until something stops it.

    Args:
        started: The file written once the job runs, whole as soon as it exists.
    """
    draft = started.with_suffix(".draft")
    draft.write_text(str(os.getpid()), encoding="utf-8")
    draft.replace(started)
    time.sleep(60)


def host(started: Path) -> None:
    """In a spawned process, run a pool whose own process holds its job.

    Args:
        started: The file the pool's process writes once its job runs.
    """
    pool = SpawnedProcessPool(1, hold)
    asyncio.run(pool.run(started))
