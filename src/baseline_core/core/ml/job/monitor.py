"""Run a job command and log GPU, CPU, memory and disk usage at a fixed interval."""

import argparse
import os
import signal
import subprocess  # noqa: S404
import sys
import threading
import time
from collections.abc import Callable, Sequence
from types import FrameType, ModuleType, TracebackType
from typing import ClassVar, Self

import psutil
import pynvml
from loguru import logger
from pydantic import BaseModel


class GpuSample(BaseModel):
    """Usage of one GPU at one instant.

    Attributes:
        utilization_percent: Share of time a kernel was running.
        memory_used_gb: Allocated GPU memory.
        memory_total_gb: GPU memory size.
        throttle_reasons: Why the clock is slowed down, e.g. ``hw_thermal``; empty when it is not.
    """

    utilization_percent: float
    memory_used_gb: float
    memory_total_gb: float
    throttle_reasons: list[str]

    def format(self, index: int) -> str:
        """One-line summary of GPU ``index``."""
        throttle = ",".join(self.throttle_reasons) or "none"
        return (
            f"gpu{index} {self.utilization_percent:.0f}% "
            f"vram {self.memory_used_gb:.1f}/{self.memory_total_gb:.1f} GB throttle={throttle}"
        )


class ResourceSample(BaseModel):
    """Usage of the machine at one instant.

    Attributes:
        elapsed_s: Seconds since the monitor started.
        cpu_percent: CPU usage since the previous sample, all cores together.
        ram_percent: Memory in use.
        disk_read_mb_s: Disk read rate since the previous sample, or ``None`` without counters.
        gpus: One sample per NVIDIA GPU.
    """

    elapsed_s: float
    cpu_percent: float
    ram_percent: float
    disk_read_mb_s: float | None
    gpus: list[GpuSample]

    def format(self) -> str:
        """One line with every metric, prefixed with the elapsed time."""
        hours, remainder = divmod(int(self.elapsed_s), 3600)
        minutes, seconds = divmod(remainder, 60)
        gpus = " | ".join(gpu.format(index) for index, gpu in enumerate(self.gpus))
        disk = "n/a" if self.disk_read_mb_s is None else f"{self.disk_read_mb_s:.1f} MB/s"
        return (
            f"[monitor {hours:02}:{minutes:02}:{seconds:02}] {gpus or 'gpu none'} "
            f"| cpu {self.cpu_percent:.0f}% | ram {self.ram_percent:.0f}% | disk read {disk}"
        )


class GpuReader:
    """Reads GPU usage through NVIDIA's management library; no GPU when the driver is missing.

    Attributes:
        THROTTLE_REASONS: Clock event bits that mean the GPU is slowed down, by name.
    """

    THROTTLE_REASONS: ClassVar[dict[int, str]] = {
        pynvml.nvmlClocksEventReasonSwPowerCap: "sw_power_cap",
        pynvml.nvmlClocksEventReasonHwSlowdown: "hw_slowdown",
        pynvml.nvmlClocksEventReasonSwThermalSlowdown: "sw_thermal",
        pynvml.nvmlClocksEventReasonHwThermalSlowdown: "hw_thermal",
        pynvml.nvmlClocksEventReasonHwPowerBrakeSlowdown: "power_brake",
    }
    BYTES_PER_GB: ClassVar[float] = 1e9

    def __init__(self, nvml: ModuleType | None = None) -> None:
        """Find the GPUs.

        Args:
            nvml: NVML bindings, or ``None`` for ``pynvml``.
        """
        self._nvml = nvml or pynvml

        try:
            self._nvml.nvmlInit()
            count = self._nvml.nvmlDeviceGetCount()
            self._handles = [
                self._nvml.nvmlDeviceGetHandleByIndex(index) for index in range(count)
            ]
        except self._nvml.NVMLError:
            self._handles = []

    def read(self) -> list[GpuSample]:
        """Sample every GPU."""
        return [self._read_one(handle) for handle in self._handles]

    def close(self) -> None:
        """Release NVML."""
        if self._handles:
            self._nvml.nvmlShutdown()

    def _read_one(self, handle: object) -> GpuSample:
        """Sample the GPU behind the NVML ``handle``."""
        memory = self._nvml.nvmlDeviceGetMemoryInfo(handle)
        reasons = self._nvml.nvmlDeviceGetCurrentClocksEventReasons(handle)
        return GpuSample(
            utilization_percent=self._nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            memory_used_gb=memory.used / self.BYTES_PER_GB,
            memory_total_gb=memory.total / self.BYTES_PER_GB,
            throttle_reasons=[
                name for bit, name in self.THROTTLE_REASONS.items() if reasons & bit
            ],
        )


class ResourceSampler:
    """Samples the machine; CPU and disk rates cover the time since the previous sample."""

    BYTES_PER_MB: ClassVar[float] = 1e6

    def __init__(self, gpu_reader: GpuReader | None = None) -> None:
        """Start the CPU and disk counters.

        Args:
            gpu_reader: GPU source, or ``None`` for NVML.
        """
        self._gpu_reader = gpu_reader or GpuReader()
        psutil.cpu_percent()
        self._last_disk_bytes = self._disk_read_bytes()
        self._last_time = time.monotonic()

    def sample(self, elapsed_s: float) -> ResourceSample:
        """Measure the machine now.

        Args:
            elapsed_s: Seconds since the monitor started.

        Returns:
            The usage of the machine.
        """
        now, disk_bytes = time.monotonic(), self._disk_read_bytes()
        disk_read_mb_s = None

        if disk_bytes is not None and self._last_disk_bytes is not None:
            seconds = max(now - self._last_time, 1e-9)
            disk_read_mb_s = (disk_bytes - self._last_disk_bytes) / self.BYTES_PER_MB / seconds

        self._last_disk_bytes, self._last_time = disk_bytes, now
        return ResourceSample(
            elapsed_s=elapsed_s,
            cpu_percent=psutil.cpu_percent(),
            ram_percent=psutil.virtual_memory().percent,
            disk_read_mb_s=disk_read_mb_s,
            gpus=self._gpu_reader.read(),
        )

    def close(self) -> None:
        """Release the GPU reader."""
        self._gpu_reader.close()

    @staticmethod
    def _disk_read_bytes() -> int | None:
        """Bytes read from disk since boot, or ``None`` when the platform has no counters."""
        counters = psutil.disk_io_counters()
        return None if counters is None else int(counters.read_bytes)


class ResourceMonitor:
    """Emits one sample line per interval from a background thread, the first after one wait."""

    def __init__(
        self,
        sampler: ResourceSampler,
        interval_s: float,
        emit: Callable[[str], None] = logger.info,
    ) -> None:
        """Prepare the monitor; ``with`` runs it.

        Args:
            sampler: Source of the samples.
            interval_s: Seconds between two lines.
            emit: Where lines go.
        """
        self._sampler = sampler
        self._interval_s = interval_s
        self._emit = emit
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> Self:
        """Start emitting."""
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop emitting and wait for the thread."""
        self._stopped.set()
        self._thread.join()

    def _run(self) -> None:
        """Emit one sample per interval until stopped."""
        start = time.monotonic()

        while not self._stopped.wait(self._interval_s):
            try:
                self._emit(self._sampler.sample(time.monotonic() - start).format())
            except Exception as error:  # noqa: BLE001
                logger.warning("Resource sampling failed: {}", error)


class MonitoredCommand:
    """A shell command in its own process group, stopped with its children on SIGTERM or SIGINT.

    Attributes:
        SHELL: Shell running the command, as on Azure ML.
        FORWARDED_SIGNALS: Signals passed on to the command's process group.
    """

    SHELL: ClassVar[str] = "/bin/bash"
    FORWARDED_SIGNALS: ClassVar[list[signal.Signals]] = [signal.SIGTERM, signal.SIGINT]

    def __init__(self, command: str) -> None:
        """Prepare the command.

        Args:
            command: Shell command.
        """
        self._command = command
        self._process: subprocess.Popen[bytes] | None = None
        self._stopped = False

    def run(self, monitor: ResourceMonitor) -> int:
        """Run the command while ``monitor`` emits.

        Args:
            monitor: Resource monitor running for the duration of the command.

        Returns:
            The command's exit code, ``128 + signal`` when a signal ended it.
        """
        previous = {sig: signal.signal(sig, self._forward) for sig in self.FORWARDED_SIGNALS}

        try:
            if self._stopped:
                return 128 + signal.SIGTERM

            self._process = subprocess.Popen(  # noqa: S602
                self._command, shell=True, executable=self.SHELL, start_new_session=True
            )

            with monitor:
                return_code = self._process.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

        return 128 - return_code if return_code < 0 else return_code

    def _forward(self, signum: int, frame: FrameType | None) -> None:
        """Pass ``signum`` on to the command's process group, or skip a command not started."""
        del frame
        self._stopped = True

        if self._process is not None and self._process.poll() is None:
            os.killpg(self._process.pid, signum)

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        """Run a command under the resource monitor and return its exit code.

        Args:
            argv: Command-line arguments, or ``None`` for ``sys.argv``.

        Returns:
            The command's exit code.
        """
        parser = argparse.ArgumentParser(
            description="Run a shell command, logging resource usage."
        )
        parser.add_argument("--interval", type=float, required=True, help="Seconds between lines.")
        parser.add_argument("command", help="Shell command to run, as one argument.")
        arguments = parser.parse_args(argv)
        sampler = ResourceSampler()

        try:
            return cls(arguments.command).run(
                ResourceMonitor(sampler, interval_s=arguments.interval)
            )

        finally:
            sampler.close()


if __name__ == "__main__":
    sys.exit(MonitoredCommand.main())
