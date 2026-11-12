"""What a search reads off a bound model besides its kernel: placements, what moves together."""

from dataclasses import dataclass

import numpy as np

from baseline_core.core.optimization.problem_formulation.model import Tie
from baseline_core.core.optimization.solver.binding.model_compiler import BoundModel
from baseline_core.core.optimization.solver.infeasibility.precheck_run import PrecheckRun
from baseline_core.core.optimization.solver.kernel.kernel import Domains
from baseline_core.core.optimization.solver.model_solution import VariableAssignment


@dataclass(frozen=True, slots=True)
class SearchIndex:
    """Each element's placements, and the elements a destroy or a repair moves together."""

    domains: Domains
    """Per element, the starts and the sets no hard constraint on it alone forbids."""
    families: list[list[int]]
    """The elements released together: the ties of the ``Tie.RELEASED`` constraint."""
    family_of: list[int]
    """Per element, the first family it belongs to; -1 in none (released only one by one)."""
    partner: list[int]
    """Per element, the one it takes a value with: its ``Tie.PAIRED`` pair, or -1."""
    groups: list[list[int]]
    """The elements released together whatever the destroy: the ``Tie.GROUPED`` ties."""
    span: list[int]
    """Per element, its span (0 without one)."""
    period: int
    """Values per period: a band stays inside one, a pair is aligned on one."""
    fixed: list[bool]
    """Per element, whether ``fixed`` names it: the search never leaves it out."""

    @classmethod
    def of(cls, bound: BoundModel, fixed: list[VariableAssignment] | None = None) -> "SearchIndex":
        """Read the lone domains, cut to what ``fixed`` gives, the model's ties and its period.

        Args:
            bound: The model bound to the data; its kernel is left as it was.
            fixed: The assignments the search may not change.

        Returns:
            The index.
        """
        model = bound.model
        run = PrecheckRun.read(bound, [])

        tied: dict[str, list[list[int]]] = {}

        for link in run.links:
            tied.setdefault(run.names[link.constraint], []).append(link.elements)

        ties = {how: tied.get(name, []) for how, name in model.ties.items()}

        families = ties.get(Tie.RELEASED, [])
        family_of = [-1] * run.kernel.n

        for position, family in enumerate(families):
            for element in family:
                if family_of[element] < 0:
                    family_of[element] = position

        partner = [-1] * run.kernel.n

        for pair in ties.get(Tie.PAIRED, []):
            if len(pair) == 2:  # noqa: PLR2004  a pair, by name
                partner[pair[0]], partner[pair[1]] = pair[1], pair[0]

        domain_starts = [list(row) for row in run.starts]
        domain_sets = [list(row) for row in run.sets]
        is_fixed = cls._fix(bound, fixed or [], domain_starts, domain_sets, partner)

        values: list[int] = []
        offsets = [0]

        for starts, sets in zip(domain_starts, domain_sets, strict=True):
            values += starts
            offsets.append(len(values))
            values += sets
            offsets.append(len(values))

        # without a declared period, the whole horizon is one period
        decision = model.integer_decision
        horizon = max((start for row in domain_starts for start in row), default=0) + 1
        period = horizon if decision is None or decision.period is None else decision.period

        return cls(
            domains=Domains(np.asarray(values, dtype=np.int64), np.asarray(offsets, np.int64)),
            families=families,
            family_of=family_of,
            partner=partner,
            groups=ties.get(Tie.GROUPED, []),
            span=run.span,
            period=period,
            fixed=is_fixed,
        )

    def held(self, element: int) -> bool:
        """Whether a fix leaves an element one start in one set: no destroy releases it.

        Args:
            element: The element.

        Returns:
            Whether it is held where it is.
        """
        offsets = self.domains.offsets
        n_starts = offsets[2 * element + 1] - offsets[2 * element]
        n_sets = offsets[2 * element + 2] - offsets[2 * element + 1]
        return self.fixed[element] and n_starts == 1 and n_sets == 1

    @staticmethod
    def _fix(
        bound: BoundModel,
        fixed: list[VariableAssignment],
        starts: list[list[int]],
        sets: list[list[int]],
        partner: list[int],
    ) -> list[bool]:
        """Cut each fixed element's domain to what it is given, a fixed value to its pair's too.

        Args:
            bound: The model bound to the data, which reads the keys.
            fixed: The assignments the search may not change.
            starts: Per element, its starts; cut in place.
            sets: Per element, its sets; cut in place.
            partner: Per element, the one it takes a value with, or -1.

        Returns:
            Per element, whether ``fixed`` names it.

        Raises:
            ValueError: An unreadable or twice-fixed assignment, or one its hard rules forbid.
        """
        is_fixed = [False] * len(starts)

        for assignment in fixed:
            if (reason := bound.unreadable_reason(assignment)) is not None:
                raise ValueError(reason)

            element = bound.element_positions[assignment.element]

            if is_fixed[element]:
                msg = f"{assignment.element!r} is fixed twice"
                raise ValueError(msg)

            is_fixed[element] = True

            if assignment.value is not None:
                held = f"{assignment.element!r} at {assignment.value}"

                for k in [element, partner[element]]:
                    if k >= 0:
                        SearchIndex._narrow(starts, k, assignment.value, held)

            if assignment.target is not None:
                held = f"{assignment.element!r} on {assignment.target!r}"
                SearchIndex._narrow(sets, element, bound.target_sets[assignment.target], held)

        return is_fixed

    @staticmethod
    def _narrow(domain: list[list[int]], element: int, kept: int, held: str) -> None:
        """Cut one element's domain to one member.

        Args:
            domain: Per element, its starts or its sets; cut in place.
            element: The element.
            kept: The one member it keeps.
            held: The fix, as the error names it.

        Raises:
            ValueError: Its hard rules alone, or its pair's fix, already forbid that member.
        """
        if kept not in domain[element]:
            msg = f"{held}: its hard rules alone, or its pair's fix, forbid it"
            raise ValueError(msg)

        domain[element] = [kept]

    def starts(self, element: int) -> list[int]:
        """The starts an element may take.

        Args:
            element: The element.

        Returns:
            Its starts.
        """
        offsets = self.domains.offsets
        low, high = offsets[2 * element], offsets[2 * element + 1]
        return [int(start) for start in self.domains.values[low:high]]

    def sets(self, element: int) -> list[int]:
        """The sets an element may take.

        Args:
            element: The element.

        Returns:
            Its sets.
        """
        offsets = self.domains.offsets
        low, high = offsets[2 * element + 1], offsets[2 * element + 2]
        return [int(target) for target in self.domains.values[low:high]]
