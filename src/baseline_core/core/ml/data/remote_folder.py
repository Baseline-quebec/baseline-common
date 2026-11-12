"""A folder of files addressed by URI, local or remote, read in place or copied to a cache."""

from pathlib import Path, PurePosixPath
from typing import IO, Any

import fsspec
from fsspec.implementations.local import LocalFileSystem
from loguru import logger
from pydantic import BaseModel, Field


class RemoteFolderConfig(BaseModel):
    """Where the folder lives and how to reach it.

    Attributes:
        uri: A local path, or a URL fsspec serves (``az://container/datasets/v1``, ``s3://``).
        options: fsspec options of a remote URI (``{"account_name": ...}``); never a secret.
    """

    uri: str
    options: dict[str, Any] = Field(default_factory=dict)


class RemoteFolder:
    """A file tree on any fsspec file system, e.g. a dataset of images on blob storage.

    Attributes:
        config: Folder URI and file system options.
        fs: File system serving the folder.
        root: Path of the folder within ``fs``, without a trailing slash.
    """

    def __init__(self, config: RemoteFolderConfig) -> None:
        """Resolve the file system serving ``config.uri``.

        Args:
            config: Folder URI and file system options.
        """
        self.config = config
        # A local file system rejects the options of a remote one.
        options = config.options if "://" in config.uri else {}
        fs, root = fsspec.core.url_to_fs(config.uri, **options)
        self.fs: fsspec.AbstractFileSystem = fs
        self.root: str = root.rstrip("/")

    def sync(self, cache_dir: Path) -> Path:
        """Copy the tree to ``cache_dir/<folder name>``, skipping files already there unchanged.

        Args:
            cache_dir: Local directory receiving remote trees; unused for a local folder.

        Returns:
            The local folder holding the tree: the folder itself when it is local.

        Raises:
            FileNotFoundError: If the folder does not exist.
        """
        if not self.fs.exists(self.root):
            error_msg = f"Nothing found at {self.config.uri}."
            raise FileNotFoundError(error_msg)

        if isinstance(self.fs, LocalFileSystem):
            return Path(self.root)

        target = cache_dir / PurePosixPath(self.root).name
        copied = 0

        for remote in self.fs.find(self.root):
            local = target / PurePosixPath(remote).relative_to(self.root)

            # Same size counts as unchanged: remote stores do not keep local modification times.
            if local.exists() and local.stat().st_size == self.fs.size(remote):
                continue

            local.parent.mkdir(parents=True, exist_ok=True)
            self.fs.get_file(remote, str(local))
            copied += 1

        logger.info("Synced {} to {} ({} files copied)", self.config.uri, target, copied)
        return target

    def list(self, suffixes: set[str] | None = None) -> list[str]:
        """Paths of the files of the tree, recursively, relative to the folder and sorted.

        Args:
            suffixes: Lower-case suffixes to keep, matched in any case, or ``None`` for all.

        Returns:
            The POSIX paths, relative to the folder.
        """
        paths = [
            str(PurePosixPath(found).relative_to(self.root)) for found in self.fs.find(self.root)
        ]

        if suffixes is None:
            return sorted(paths)

        return sorted(path for path in paths if PurePosixPath(path).suffix.lower() in suffixes)

    def open(self, path: str, mode: str = "rb") -> IO[Any]:
        """Open a file of the tree without copying the tree.

        Args:
            path: POSIX path relative to the folder, as ``list`` gives it.
            mode: File mode, ``"rb"`` or ``"r"`` to read.

        Returns:
            The open file, to use as a context manager.
        """
        handle: IO[Any] = self.fs.open(f"{self.root}/{path}", mode)
        return handle

    def exists(self, path: str) -> bool:
        """Whether a file or folder exists in the tree.

        Args:
            path: POSIX path relative to the folder.

        Returns:
            ``True`` if something is at ``path``.
        """
        return bool(self.fs.exists(f"{self.root}/{path}"))
