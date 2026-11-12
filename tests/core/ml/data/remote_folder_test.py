"""Tests for ``RemoteFolder`` on local folders and on fsspec's in-memory file system."""

from collections.abc import Iterator
from pathlib import Path

import fsspec
import pytest

from baseline_core.core.ml.data.remote_folder import RemoteFolder, RemoteFolderConfig


class TestRemoteFolder:
    """Tests for ``RemoteFolder``."""

    @pytest.fixture
    def file_contents(self) -> dict[str, str]:
        """Relative path to content of the files of a dataset of images."""
        return {"annotations.json": "{}", "images/a.jpg": "a", "images/b.PNG": "b"}

    @pytest.fixture
    def memory_uri(self, tmp_path: Path, file_contents: dict[str, str]) -> Iterator[str]:
        """The dataset on the in-memory file system, removed after the test.

        Yields:
            The URI of the dataset's root.
        """
        memory_fs = fsspec.filesystem("memory")
        root = f"memory://{tmp_path.name}/dataset"

        for relative_path, content in file_contents.items():
            with memory_fs.open(f"{root}/{relative_path}", "w") as handle:
                handle.write(content)

        yield root
        memory_fs.rm(root, recursive=True)

    @pytest.fixture
    def remote_folder(self, memory_uri: str) -> RemoteFolder:
        """The in-memory dataset, as a remote folder."""
        return RemoteFolder(RemoteFolderConfig(uri=memory_uri))

    @pytest.fixture
    def local_folder(self, tmp_path: Path, file_contents: dict[str, str]) -> RemoteFolder:
        """The same dataset written to a local folder."""
        root = tmp_path / "dataset"

        for relative_path, content in file_contents.items():
            (root / relative_path).parent.mkdir(parents=True, exist_ok=True)
            (root / relative_path).write_text(content)

        return RemoteFolder(RemoteFolderConfig(uri=str(root)))

    def test__init__when_remote_uri__should_pass_the_options_to_its_file_system(self) -> None:
        """Options such as an ``account_name`` reach the file system serving the URI."""
        config = RemoteFolderConfig(uri="memory://datasets/v1", options={"account_name": "lab"})

        folder = RemoteFolder(config)

        assert folder.fs.storage_options == {"account_name": "lab"}
        assert folder.root == "/datasets/v1"

    def test__init__when_local_path__should_ignore_the_remote_options(
        self, tmp_path: Path
    ) -> None:
        """Options name a remote store; a local folder works without them."""
        config = RemoteFolderConfig(uri=str(tmp_path), options={"account_name": "datasets"})

        folder = RemoteFolder(config)

        assert folder.root == str(tmp_path)

    def test__sync__when_local_path__should_return_it_unchanged(
        self, local_folder: RemoteFolder, tmp_path: Path
    ) -> None:
        """Local datasets are used in place, never copied."""
        local_dir = local_folder.sync(tmp_path / "cache")

        assert local_dir == tmp_path / "dataset"
        assert not (tmp_path / "cache").exists()

    @pytest.mark.parametrize("scheme", ["", "memory://"])
    def test__sync__when_nothing_at_uri__should_raise_file_not_found(
        self, tmp_path: Path, scheme: str
    ) -> None:
        """A wrong dataset URI fails before training, local or remote."""
        folder = RemoteFolder(RemoteFolderConfig(uri=f"{scheme}{tmp_path / 'missing'}"))

        with pytest.raises(FileNotFoundError, match="Nothing found"):
            folder.sync(tmp_path / "cache")

    def test__sync__when_remote__should_copy_the_tree_under_its_name(
        self, remote_folder: RemoteFolder, tmp_path: Path, file_contents: dict[str, str]
    ) -> None:
        """Remote trees land in ``<cache_dir>/<folder name>`` with the same layout."""
        local_dir = remote_folder.sync(tmp_path / "cache")

        assert local_dir == tmp_path / "cache" / "dataset"
        assert {
            str(path.relative_to(local_dir)) for path in local_dir.rglob("*") if path.is_file()
        } == set(file_contents)

        for relative_path, content in file_contents.items():
            assert (local_dir / relative_path).read_text() == content

    def test__sync__when_already_synced__should_not_copy_unchanged_files(
        self, remote_folder: RemoteFolder, tmp_path: Path
    ) -> None:
        """A second sync keeps the local files untouched."""
        local_dir = remote_folder.sync(tmp_path / "cache")
        modified_times = {path: path.stat().st_mtime_ns for path in local_dir.rglob("*.*")}

        remote_folder.sync(tmp_path / "cache")

        assert {path: path.stat().st_mtime_ns for path in modified_times} == modified_times

    def test__sync__when_remote_file_changed__should_copy_it_again(
        self, remote_folder: RemoteFolder, memory_uri: str, tmp_path: Path
    ) -> None:
        """A file whose size changed is copied again; the others are kept."""
        local_dir = remote_folder.sync(tmp_path / "cache")

        with fsspec.open(f"{memory_uri}/images/a.jpg", "w") as handle:
            handle.write("a, relabeled")

        remote_folder.sync(tmp_path / "cache")

        assert (local_dir / "images" / "a.jpg").read_text() == "a, relabeled"

    def test__list__should_give_every_file_relative_and_sorted(
        self, remote_folder: RemoteFolder, file_contents: dict[str, str]
    ) -> None:
        """Every file of the tree, at any depth, relative to the folder."""
        paths = remote_folder.list()

        assert paths == sorted(file_contents)

    @pytest.mark.parametrize("folder_name", ["remote_folder", "local_folder"])
    def test__list__when_suffixes__should_keep_matching_files_case_insensitively(
        self, request: pytest.FixtureRequest, folder_name: str
    ) -> None:
        """Image files only, whatever the case of their suffix, local or remote."""
        folder: RemoteFolder = request.getfixturevalue(folder_name)

        paths = folder.list(suffixes={".jpg", ".png"})

        assert paths == ["images/a.jpg", "images/b.PNG"]

    def test__open_and_exists__should_read_remote_files(self, remote_folder: RemoteFolder) -> None:
        """Files can be checked and read without syncing."""
        with remote_folder.open("annotations.json", "r") as handle:
            content = handle.read()

        assert content == "{}"
        assert remote_folder.exists("annotations.json")
        assert remote_folder.exists("images")
        assert not remote_folder.exists("missing.json")
