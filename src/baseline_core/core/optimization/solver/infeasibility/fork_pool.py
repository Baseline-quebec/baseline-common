"""Independent checks run in forked process copies, each sharing the kernel already bound."""

import os
import pickle  # noqa: S403  results travel between a process and the copies it forked itself
import traceback
from collections.abc import Callable, Sequence


class ForkPool:
    """Maps a function over items in ``workers`` forked copies of the process (Linux), in order."""

    def __init__(self, workers: int) -> None:
        """Keep how many copies to fork.

        Args:
            workers: How many processes run the items; 1 runs them here, in order.
        """
        self.workers = max(workers, 1)

    @staticmethod
    def available_cores() -> int:
        """The cores this process may run on: its CPU affinity (a container's pinning), else all.

        Returns:
            How many cores.
        """
        affinity = getattr(os, "sched_getaffinity", None)
        return len(affinity(0)) if affinity is not None else os.cpu_count() or 1

    def map[Item, Result](
        self, function: Callable[[Item], Result], items: Sequence[Item]
    ) -> list[Result]:
        """``[function(item) for item in items]``, the items shared out between the copies.

        Args:
            function: What to compute per item; it may change process state (only the copy's).
            items: The items.

        Returns:
            One result per item, in the items' order.

        Raises:
            RuntimeError: A copy failed; its traceback is in the message.
        """
        workers = min(self.workers, len(items))

        if workers <= 1:
            return [function(item) for item in items]

        children = [self._fork(function, items, worker, workers) for worker in range(workers)]
        results: list[Result | None] = [None] * len(items)

        for worker, (pid, reader) in enumerate(children):
            with os.fdopen(reader, "rb") as stream:
                status, payload = pickle.loads(stream.read())  # noqa: S301  written by our fork

            os.waitpid(pid, 0)

            if status != "ok":
                msg = f"a forked check failed:\n{payload}"
                raise RuntimeError(msg)

            for position, result in zip(range(worker, len(items), workers), payload, strict=True):
                results[position] = result

        return results  # type: ignore[return-value]  # each position filled by exactly one copy

    @staticmethod
    def _fork[Item, Result](
        function: Callable[[Item], Result], items: Sequence[Item], worker: int, workers: int
    ) -> list[int]:
        """Fork a copy computing every ``workers``-th item from ``worker``, writing results back.

        Args:
            function: What to compute per item.
            items: The items.
            worker: Which share this copy takes.
            workers: How many shares there are.

        Returns:
            ``[process id, pipe]``: the copy, and where to read its results from.
        """
        reader, writer = os.pipe()
        pid = os.fork()

        if pid:
            os.close(writer)
            return [pid, reader]

        os.close(reader)

        try:
            payload = pickle.dumps(["ok", [function(item) for item in items[worker::workers]]])
        except BaseException:  # noqa: BLE001  any failure goes back to the parent, which raises it
            payload = pickle.dumps(["error", traceback.format_exc()])

        with os.fdopen(writer, "wb") as stream:
            stream.write(payload)

        os._exit(0)
