"""Builds a generated kernel module once per source, cached on disk, safe across processes."""

import fcntl
import hashlib
import importlib.util
import os
import subprocess  # noqa: S404  builds the Cython source this package generated
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import numpy as np


class KernelBuilder:
    """Compiles pure-Python Cython sources to extension modules, or imports them as Python."""

    ENV_DIR: ClassVar[str] = "MODELING_KERNEL_CACHE"
    """The environment variable naming the cache directory."""
    SETUP: ClassVar[str] = """import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup
ext = [Extension("{name}", ["{name}.py"], include_dirs=["{include}"], extra_compile_args=["-O3"],
                 define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])]
setup(name="{name}",
      ext_modules=cythonize(ext, compiler_directives={{"language_level": 3}}, quiet=True))
"""
    """The ``setup.py`` that builds one module."""
    # an extension module can be imported only once per process: later loads must reuse it
    _loaded: ClassVar[dict[tuple[str, bool], ModuleType]] = {}

    def __init__(self, *, cythonize: bool = True) -> None:
        """Choose how kernels run, cached in ``$MODELING_KERNEL_CACHE`` or ``~/.cache/modeling``.

        Args:
            cythonize: Whether to compile with Cython; ``False`` imports it as plain Python.
        """
        self.cythonize = cythonize
        self.cache_dir = Path(os.environ.get(self.ENV_DIR, Path.home() / ".cache" / "modeling"))

    def load(self, source: str) -> ModuleType:
        """The module of ``source``: compiled on first use, or imported as plain Python.

        Args:
            source: The pure-Python-mode Cython source.

        Returns:
            The imported module.
        """
        name = "kernel_" + hashlib.sha256(source.encode()).hexdigest()[:16]

        if (name, self.cythonize) in self._loaded:
            return self._loaded[name, self.cythonize]

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        with (self.cache_dir / f"{name}.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)

            if not self.cythonize:
                built = self.cache_dir / f"{name}.py"

                if not built.exists():
                    partial = self.cache_dir / f"{name}.py.part"
                    partial.write_text(source)
                    partial.replace(built)
            else:
                built = self._find(name) or self._compile(name, source)

        spec = importlib.util.spec_from_file_location(name, built)

        if spec is None or spec.loader is None:
            msg = f"cannot import the built kernel {built}"
            raise ImportError(msg)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._loaded[name, self.cythonize] = module
        return module

    def _find(self, name: str) -> Path | None:
        """A build of the module already in the cache.

        Args:
            name: The module name.

        Returns:
            Its path, or ``None``.
        """
        found = sorted(self.cache_dir.glob(f"{name}.*.so")) + sorted(
            self.cache_dir.glob(f"{name}.so")
        )
        return found[0] if found else None

    def _compile(self, name: str, source: str) -> Path:
        """Build the module into the cache.

        Args:
            name: The module name.
            source: The Cython source.

        Returns:
            The path of the built module.

        Raises:
            RuntimeError: The build failed.
        """
        with tempfile.TemporaryDirectory(dir=self.cache_dir) as temporary:
            work = Path(temporary)
            (work / f"{name}.py").write_text(source)
            (work / "setup.py").write_text(self.SETUP.format(name=name, include=np.get_include()))

            out = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=work,
                capture_output=True,
                text=True,
                check=False,
            )

            if out.returncode != 0:
                msg = (
                    "generated kernel failed to compile:\n"
                    f"{out.stdout[-4000:]}\n{out.stderr[-4000:]}"
                )
                raise RuntimeError(msg)

            built = next(work.glob(f"{name}*.so"))
            target = self.cache_dir / built.name
            built.replace(target)  # atomic: readers never see a half-written module
            return target
