"""Fixtures shared by the job tests: a job spec and resource samples."""

import pytest

from baseline_core.core.ml.job.job_spec import JobSpec
from baseline_core.core.ml.job.monitor import GpuSample, ResourceSample


@pytest.fixture
def spec() -> JobSpec:
    """A job reading one input folder."""
    return JobSpec(
        command="python train.py --data ${{inputs.dataset}}",
        inputs={"dataset": "azureml://datastores/datasets/paths/frames_v1"},
        environment_variables={"HF_HUB_OFFLINE": "1"},
        display_name="train-frames",
        experiment_name="box_detector",
    )


@pytest.fixture
def gpu_sample() -> GpuSample:
    """A T4 under load, slowed down by temperature and its power cap."""
    return GpuSample(
        utilization_percent=87,
        memory_used_gb=11.2,
        memory_total_gb=16.1,
        throttle_reasons=["sw_power_cap", "hw_thermal"],
    )


@pytest.fixture
def resource_sample(gpu_sample: GpuSample) -> ResourceSample:
    """Twelve and a half minutes into a job."""
    return ResourceSample(
        elapsed_s=750, cpu_percent=45, ram_percent=43, disk_read_mb_s=85, gpus=[gpu_sample]
    )
