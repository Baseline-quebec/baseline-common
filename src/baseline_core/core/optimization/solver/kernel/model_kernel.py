"""The template of a model's generated kernel: ``KernelSource`` fills its slots from the model."""

# Cython's typed shadow cannot express memoryviews (`cython.long[::1]`), so `cython` stays
# untyped and its decorators (`cfunc`, `exceptval`, ...) with it.
# mypy: disable-error-code="untyped-decorator"
# pyright: reportCallIssue=false, reportReturnType=false

from typing import Any

import cython
import numpy as np
from numpy.typing import NDArray

from baseline_core.core.optimization.solver.kernel.kernel import Kernel, KernelLayout


@cython.cclass
class ModelKernel(Kernel):
    """Generated from a model: do not edit, regenerate from the model."""

    # keep: each model array is declared here, or the compiled __init__ cannot bind it
    # generated fields

    def __init__(
        self,
        arrays: dict[str, NDArray[np.int64]],  # noqa: ARG002  the generated binds read it
        layout: KernelLayout,
    ) -> None:
        """Bind the model's arrays, then count the instances that read no decision.

        Args:
            arrays: Every array the generated blocks read, by name.
            layout: The elements, sets and constraints counted, and the names reported.
        """
        Kernel.__init__(self, layout)
        # generated binds
        self._count_constants()

    @cython.cfunc
    @cython.exceptval(check=False)
    def _around(self, j: cython.long, r_other: cython.long) -> cython.void:
        """Add to ``cnt`` the ``act`` instances element ``j`` can change.

        Args:
            j: The element.
            r_other: Another set whose decided instances it can change, or -1.
        """
        # generated around

    @cython.cfunc
    @cython.exceptval(check=False)
    def _full(self) -> cython.void:  # noqa: PLR6301  the generated block reads self
        """Add to ``cnt`` every instance of the ``act`` constraints."""
        j: cython.long = -1  # noqa: F841  the generated block reads it
        # generated full

    @cython.cfunc
    @cython.exceptval(check=False)
    def _around_set(  # noqa: PLR6301  the generated block reads self
        self,
        n_elements: cython.long,  # noqa: ARG002  the generated block reads it
    ) -> cython.void:
        """Like ``_around``, over the first ``n_elements`` of ``elements`` and ``others``, each once.

        Args:
            n_elements: How many of ``elements`` are in the set.
        """
        j: cython.long  # noqa: F842  the generated block assigns it
        jj_: cython.long  # noqa: F842  the generated block assigns it
        # generated around_set

    @cython.cfunc
    @cython.exceptval(check=False)
    def _constants(self) -> cython.void:  # noqa: PLR6301  the generated block reads self
        """Add to ``cnt`` every instance that reads no decision."""
        j: cython.long = -1  # noqa: F841  the generated block reads it
        # generated constants

    def _violations(self) -> list[list[Any]]:  # noqa: PLR6301  the generated block reads self
        """``[template, instance, members, constraint index]`` per violation.

        Returns:
            The violations.
        """
        found: list[list[Any]] = []
        j: cython.long = -1  # noqa: F841  the generated block reads it
        # generated violations
        return found
