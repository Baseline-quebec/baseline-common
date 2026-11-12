"""Tests for the resource monitor that wraps job commands."""

import threading
from typing import ClassVar
from unittest.mock import MagicMock

import pynvml
import pytest

from baseline_core.core.ml.job.monitor import (
    GpuReader,
    GpuSample,
    MonitoredCommand,
    ResourceMonitor,
    ResourceSample,
    ResourceSampler,
)


class TestResourceSample:
    """Tests for ``ResourceSample.format``."""

    def test__format__should_fit_every_metric_on_one_line(
        self, resource_sample: ResourceSample
    ) -> None:
        """One line per sample keeps the job output readable."""
        line = resource_sample.format()

        assert line == (
            "[monitor 00:12:30] gpu0 87% vram 11.2/16.1 GB throttle=sw_power_cap,hw_thermal "
            "| cpu 45% | ram 43% | disk read 85.0 MB/s"
        )

    def test__format__when_no_gpu_and_no_disk_counters__should_say_so(
        self, resource_sample: ResourceSample
    ) -> None:
        """A CPU node, or a container without disk counters, still gets a line."""
        sample = resource_sample.model_copy(update={"gpus": [], "disk_read_mb_s": None})

        line = sample.format()

        assert "gpu none" in line
        assert "disk read n/a" in line


class TestGpuReader:
    """Tests for ``GpuReader`` against a fake NVML library."""

    @pytest.fixture
    def nvml(self) -> MagicMock:
        """NVML reporting the GPU of ``gpu_sample``."""
        nvml = MagicMock()
        nvml.NVMLError = pynvml.NVMLError
        nvml.nvmlDeviceGetCount.return_value = 1
        nvml.nvmlDeviceGetUtilizationRates.return_value = MagicMock(gpu=87)
        nvml.nvmlDeviceGetMemoryInfo.return_value = MagicMock(used=11.2e9, total=16.1e9)
        nvml.nvmlDeviceGetCurrentClocksEventReasons.return_value = (
            pynvml.nvmlClocksEventReasonSwPowerCap
            | pynvml.nvmlClocksEventReasonHwThermalSlowdown
            | pynvml.nvmlClocksEventReasonGpuIdle
        )
        return nvml

    def test__read__should_convert_nvml_units(
        self, nvml: MagicMock, gpu_sample: GpuSample
    ) -> None:
        """Bytes become GB, the throttle bitmask becomes names, an idle GPU is not throttled."""
        reader = GpuReader(nvml)

        samples = reader.read()

        assert samples == [gpu_sample]

    def test__read__when_no_driver__should_return_no_gpu(self, nvml: MagicMock) -> None:
        """A machine without an NVIDIA driver is a CPU machine, not an error."""
        nvml.nvmlInit.side_effect = pynvml.NVMLError(pynvml.NVML_ERROR_DRIVER_NOT_LOADED)

        reader = GpuReader(nvml)

        assert reader.read() == []


class TestResourceSampler:
    """Tests for ``ResourceSampler`` on the machine running the tests."""

    def test__sample__should_measure_this_machine_and_its_gpus(
        self, gpu_sample: GpuSample
    ) -> None:
        """CPU and memory come from the OS, GPUs from the reader."""
        gpu_reader = MagicMock(spec=GpuReader)
        gpu_reader.read.return_value = [gpu_sample]
        sampler = ResourceSampler(gpu_reader)

        sample = sampler.sample(elapsed_s=1)

        assert 0 < sample.ram_percent < 100
        assert sample.gpus == [gpu_sample]


class TestResourceMonitor:
    """Tests for ``ResourceMonitor``."""

    LINES_TO_WAIT_FOR: ClassVar[int] = 2

    def test__start__should_emit_one_line_per_interval(
        self, resource_sample: ResourceSample
    ) -> None:
        """The monitor keeps emitting until stopped."""
        sampler = MagicMock(spec=ResourceSampler)
        sampler.sample.return_value = resource_sample
        lines: list[str] = []
        enough = threading.Event()

        def emit(line: str) -> None:
            """Collect ``line``, signaling once enough lines arrived."""
            lines.append(line)

            if len(lines) >= self.LINES_TO_WAIT_FOR:
                enough.set()

        with ResourceMonitor(sampler, interval_s=0.01, emit=emit):
            assert enough.wait(timeout=5)

        assert lines[0] == resource_sample.format()

    def test__stop__should_end_emission_before_first_interval(
        self, resource_sample: ResourceSample
    ) -> None:
        """Nothing is emitted for a job shorter than one interval."""
        sampler = MagicMock(spec=ResourceSampler)
        sampler.sample.return_value = resource_sample
        lines: list[str] = []

        with ResourceMonitor(sampler, interval_s=60, emit=lines.append):
            pass

        assert lines == []


class TestMain:
    """Tests for the ``python -m baseline_core.core.ml.job.monitor`` entry point."""

    def test__main__should_return_exit_code_of_command(self) -> None:
        """The wrapper is transparent to the launcher: the job fails when the command fails."""
        exit_code = MonitoredCommand.main(["--interval", "60", "exit 3"])

        assert exit_code == 3

    def test__main__should_run_command_through_shell(
        self, capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Variables and operators work as on Azure ML, which runs commands with bash."""
        monkeypatch.setenv("JOB_FLAG", "on")

        exit_code = MonitoredCommand.main(["--interval", "60", "echo $JOB_FLAG && echo done"])

        assert exit_code == 0
        assert capfd.readouterr().out.splitlines() == ["on", "done"]
