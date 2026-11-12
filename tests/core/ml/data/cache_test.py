"""Tests for ``DiskTransformCache``."""

from pathlib import Path

import pytest
import torch

from baseline_core.core.ml.data.cache import DiskTransformCache


class TestDiskTransformCache:
    """Tests for ``DiskTransformCache`` in a temporary folder."""

    @pytest.fixture
    def cache_dir(self, tmp_path: Path) -> Path:
        """A folder that does not exist yet."""
        return tmp_path / "cache"

    @pytest.fixture
    def cache(self, cache_dir: Path) -> DiskTransformCache:
        """Cache writing into ``cache_dir``."""
        return DiskTransformCache(cache_dir)

    @pytest.fixture
    def value(self) -> dict[str, torch.Tensor]:
        """A value as a cached feature group returns it."""
        return {"image": torch.arange(6).reshape(2, 3)}

    def test__load__when_never_saved__should_return_none(self, cache: DiskTransformCache) -> None:
        """A miss is ``None``, not an error."""
        assert cache.load("missing") is None

    def test__save__should_be_read_back_by_load(
        self, cache: DiskTransformCache, value: dict[str, torch.Tensor]
    ) -> None:
        """What is saved under a key is loaded back under that key."""
        cache.save("sample", value)

        loaded = cache.load("sample")

        assert loaded is not None
        assert torch.equal(loaded["image"], value["image"])

    def test__save__should_leave_one_file_and_no_temporary_one(
        self, cache: DiskTransformCache, cache_dir: Path, value: dict[str, torch.Tensor]
    ) -> None:
        """The folder is created and the temporary file is renamed into place."""
        cache.save("sample", value)

        assert [path.name for path in cache_dir.iterdir()] == ["sample.pt"]
