"""Capture what is needed to reproduce a run: host, git state, installed packages."""

import os
import socket
from importlib import metadata

from git import Repo
from git.exc import GitError
from loguru import logger


class SystemContext:
    """Describes the machine, code and packages a run executes with."""

    @staticmethod
    def collect() -> dict[str, str]:
        """Hostname, platform and git commit, branch and dirty flag.

        Returns:
            Flat ``key -> value`` mapping; git fields are ``"unknown"`` outside a repository.
        """
        context = {"system.hostname": socket.gethostname(), "system.platform": os.name}

        try:
            repo = Repo(search_parent_directories=True)
        except GitError:
            logger.warning("Not inside a git repository; commit will be logged as unknown.")
            context["git.commit"] = "unknown"
            return context

        context["git.commit"] = repo.head.commit.hexsha
        context["git.branch"] = "detached" if repo.head.is_detached else repo.active_branch.name

        if repo.is_dirty():
            logger.warning("Git repository has uncommitted changes.")
            context["git.dirty"] = "true"

        return context

    @staticmethod
    def freeze_dependencies() -> str:
        """``name==version`` of every installed distribution, one per line, sorted.

        Returns:
            A ``requirements.txt``-style listing.
        """
        lines = sorted(
            f"{dist.metadata['Name']}=={dist.version}" for dist in metadata.distributions()
        )
        return "\n".join(lines) + "\n"
