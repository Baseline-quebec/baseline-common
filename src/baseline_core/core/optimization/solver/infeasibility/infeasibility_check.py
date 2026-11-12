"""Pre-check: what the hard constraints of a model make impossible, proven before any search."""

import heapq
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import IntEnum
from functools import partial
from typing import Any

import numpy as np
from pydantic import BaseModel

from baseline_core.core.optimization.problem_formulation.declaration import (
    Assignment,
    IndexSet,
    Parameter,
    SubsetFamily,
)
from baseline_core.core.optimization.problem_formulation.model import BreakRule, Resource
from baseline_core.core.optimization.solver.binding.model_compiler import BoundModel
from baseline_core.core.optimization.solver.infeasibility.fork_pool import ForkPool
from baseline_core.core.optimization.solver.infeasibility.placement_search import (
    ExhaustiveSearch,
    HeuristicSearch,
)
from baseline_core.core.optimization.solver.infeasibility.precheck_issue import (
    IssueKind,
    LoadGrade,
    PrecheckIssue,
)
from baseline_core.core.optimization.solver.infeasibility.precheck_run import PrecheckRun
from baseline_core.core.optimization.solver.kernel.kernel import (
    LoneDomain,
    SearchOutcome,
    SearchPlan,
)
from baseline_core.core.optimization.solver.stop_condition import (
    AnyOf,
    MaxIterations,
    MaxTime,
    StopCondition,
)


@dataclass(frozen=True, slots=True)
class Load:
    """A load to grade: the resource, the entity it is reported on, its members, their demand."""

    resource: Resource
    """The resource the members share."""
    kind: IssueKind
    """The kind of issue the load is reported as."""
    entity: str
    """The name it is reported under."""
    members: list[int]
    """The elements that must use the capacities."""
    demand: list[int]
    """Per element, what it uses of a target."""
    capacities: list[int]
    """Per target, what it offers at once; for a fixed group, its one capacity."""
    pool: set[int] | None
    """The targets the members can only go to; ``None`` when no pool bounds them."""

    @property
    def shared(self) -> list[int]:
        """The capacities the members share: those of the pool's targets, else all of them."""
        if self.pool is None:
            return self.capacities

        return [self.capacities[target] for target in sorted(self.pool)]


@dataclass(frozen=True, slots=True)
class LoadBound:
    """What the members of a load demand, and the most the capacities can give them."""

    used: int
    """The demand."""
    offered: int
    """The offer."""


@dataclass(frozen=True, slots=True)
class GroupProof:
    """A full search of one group."""

    verdict: bool | None
    """``True`` placeable, ``False`` proven not, ``None`` undecided."""
    constraints: list[int]
    """When ``False``, the constraints to blame, as name positions."""


class PrecheckLevel(IntEnum):
    """How far a pre-check goes: each level runs every level before it, cheapest first.

    Attributes:
        ELEMENT: Each element alone (its domain), and each resource's load.
        GROUP: Then each small group of elements one hard constraint ties (``group_limit``).
        COMPONENT: Then every set of elements the hard constraints link, of any size.
    """

    ELEMENT = 1
    GROUP = 2
    COMPONENT = 3


class PrecheckConfig(BaseModel):
    """The limits of a pre-check: how far it goes, and what each search may spend."""

    up_to: PrecheckLevel = PrecheckLevel.COMPONENT
    """The last level a run reaches."""

    group_limit: int = 8
    """Largest group of elements one hard constraint ties that level 2 searches."""
    group_budget: int = 20_000_000
    """Most kernel evaluations spent proving one group; beyond, it proves nothing."""
    witness_budget: int = 5_000
    """Most evaluations per element spent first on a placement of a group, which needs no proof."""
    nearly_full_ratio: float = 0.9
    """A load at least this share of its offer is reported nearly full (a warning, no proof)."""
    underused_ratio: float = 0.5
    """A pool of targets loaded below this share of its offer is reported underused (a warning)."""
    workers: int = max(ForkPool.available_cores() - 2, 1)
    """Forked copies searching the undecided groups: every core this process may use but two."""
    seed: int = 12345
    """Where the heuristic search's random draws start: the same seed, the same issues."""
    time_budget_seconds: float | None = None
    """Seconds a run takes at most; ``None`` waits for every search to reach its cap."""


class InfeasibilityCheck:
    """An element alone, each resource's load, a group one rule ties, every element linked."""

    def __init__(self, config: PrecheckConfig) -> None:
        """Keep the limits every run reads.

        Args:
            config: The limits.
        """
        self.config = config

    def run(self, bound: BoundModel) -> list[PrecheckIssue]:
        """Every impossibility proven over the bound data within the configured time budget.

        Args:
            bound: The model bound to the data; its kernel ends as it was given.

        Returns:
            One issue per proof, naming the constraints that suffice for it.
        """
        budget = self.config.time_budget_seconds
        stops: list[StopCondition] = [] if budget is None else [MaxTime(budget)]

        # a clock latches on its first look: the run's time counts from here
        for stop in stops:
            stop.should_stop(0)

        run = PrecheckRun.read(bound, stops)

        with PrecheckRun.isolated(bound.kernel):
            return self._issues(run)

    def _issues(self, run: PrecheckRun) -> list[PrecheckIssue]:
        """The issues of each level up to the configured one, in order, until the run's stop.

        Args:
            run: The run.

        Returns:
            The issues.
        """
        issues = self._alone_issues(run)

        loads: list[PrecheckIssue] = []

        for resource in run.model.resources:
            if run.late:
                break

            loads += self._resource_issues(run, resource)

        issues += self._distinct(loads)

        if self.config.up_to < PrecheckLevel.GROUP:
            return issues

        proven = [set(issue.elements) for issue in issues if issue.proven]
        groups = run.linked_groups(self.config.group_limit)
        issues += self._group_issues(run, groups, IssueKind.GROUP, proven)

        if self.config.up_to < PrecheckLevel.COMPONENT:
            return issues

        searched = {tuple(group) for group in groups}
        components = [part for part in run.components() if tuple(part) not in searched]
        issues += self._group_issues(run, components, IssueKind.COMPONENT, proven)

        return issues

    def _alone_issues(self, run: PrecheckRun) -> list[PrecheckIssue]:
        """Elements without a placement on their own, each with the constraints that leave it none.

        Args:
            run: The run.

        Returns:
            One issue per such element.
        """
        placeable = run.placeable
        stuck = [element for element in range(run.known) if not placeable[element]]
        reasons = [self._alone_constraints(run, element) for element in stuck]

        return [
            PrecheckIssue(
                kind=IssueKind.NO_PLACEMENT,
                elements=[element],
                constraints=[run.names[index] for index in constraints],
                entity=self._label(run, element),
            )
            for element, constraints in zip(stuck, reasons, strict=True)
        ]

    @staticmethod
    def _distinct(loads: list[PrecheckIssue]) -> list[PrecheckIssue]:
        """One load per kind and entity, the tightest, kept when graded.

        Args:
            loads: The loads of every resource, graded or not, and the elements no target holds.

        Returns:
            The graded ones in their first order: two resources measuring one pool give one.
        """
        kept: dict[tuple[str, str], PrecheckIssue] = {}

        for issue in loads:
            key = (issue.kind, issue.entity)
            known = kept.get(key)

            if known is None or (issue.demand or 0) * (known.offer or 1) > (known.demand or 0) * (
                issue.offer or 1
            ):
                kept[key] = issue

        return [
            issue
            for issue in kept.values()
            if issue.grade is not None or issue.kind == IssueKind.NO_PLACEMENT
        ]

    def _group_issues(
        self, run: PrecheckRun, groups: list[list[int]], kind: IssueKind, proven: list[set[int]]
    ) -> list[PrecheckIssue]:
        """Groups that cannot be placed together, each with the constraints to blame.

        Args:
            run: The run.
            groups: The groups, searched in this order until the run's stop.
            kind: The kind of their issues.
            proven: Elements already proven impossible, extended with each group proven here.

        Returns:
            One issue per group proven, then one ``UNDECIDED`` per searched group left open.
        """
        blamed_by_group: dict[int, list[int]] = {}
        undecided: list[int] = []

        for position, group in enumerate(groups):
            if run.late:
                break

            if group[-1] >= run.known or self._covered(group, proven):
                continue

            cap = MaxIterations(self.config.witness_budget * len(group))
            witness = HeuristicSearch(AnyOf([cap, *run.stops]), self.config.seed)
            searched = self._search(run, group, run.hard, witness)

            if searched.verdict is False:
                blamed_by_group[position] = self._group_constraints(
                    run, group, searched.blamed, searched.work
                )
            elif searched.verdict is None:
                undecided.append(position)

        proofs = ForkPool(self.config.workers).map(
            partial(self._group_proof, run), [groups[position] for position in undecided]
        )
        open_groups = [
            groups[p] for p, proof in zip(undecided, proofs, strict=True) if proof.verdict is None
        ]
        blamed_by_group.update({
            position: proof.constraints
            for position, proof in zip(undecided, proofs, strict=True)
            if proof.verdict is False
        })

        issues = []

        for position in sorted(blamed_by_group):
            group = groups[position]

            if self._covered(group, proven):
                continue

            issues.append(
                PrecheckIssue(
                    kind=kind,
                    elements=group,
                    constraints=[run.names[index] for index in blamed_by_group[position]],
                    entity=", ".join(self._label(run, element) for element in group),
                )
            )
            proven.append(set(group))

        issues += [
            PrecheckIssue(
                kind=IssueKind.UNDECIDED,
                elements=group,
                constraints=[],
                entity=", ".join(self._label(run, element) for element in group),
            )
            for group in open_groups
            if not self._covered(group, proven)
        ]
        return issues

    @staticmethod
    def _covered(group: list[int], proven: list[set[int]]) -> bool:
        """Whether a proven group holds the group, or lies within it.

        Args:
            group: The elements.
            proven: The groups already proven impossible.

        Returns:
            Whether one of them covers the group.
        """
        members = set(group)
        return any(elements <= members or members <= elements for elements in proven)

    def _group_proof(self, run: PrecheckRun, group: list[int]) -> GroupProof:
        """A full search of the group, and the constraints to blame when it proves it unplaceable.

        Args:
            run: The run.
            group: The elements, undecided by the witness pass.

        Returns:
            The verdict (``None`` undecided); the constraints, as name positions, when ``False``.
        """
        cap = MaxIterations(self.config.group_budget)
        proof = ExhaustiveSearch(AnyOf([cap, *run.stops]))
        searched = self._search(run, group, run.hard, proof)

        if searched.verdict is not False:
            return GroupProof(searched.verdict, [])

        return GroupProof(
            searched.verdict, self._group_constraints(run, group, searched.blamed, searched.work)
        )

    @staticmethod
    def _explain(
        fails: Callable[[list[int]], bool],
        constraint_indices: list[int],
        stops: list[StopCondition],
    ) -> list[int]:
        """The ``constraint_indices`` the proof needs: each dropped in turn, kept if needed.

        Args:
            fails: Whether the impossibility is still proven with only the given constraints.
            constraint_indices: The constraints the proof used, as name positions.
            stops: What ends the run: once one says to give up, the rest is kept untried.

        Returns:
            The constraints the impossibility holds under.
        """
        needed = list(constraint_indices)

        for constraint_index in constraint_indices:
            if any(stop.should_stop(0) for stop in stops):
                break

            trial = [other for other in needed if other != constraint_index]

            if fails(trial):
                needed = trial

        return needed

    def _alone_constraints(self, run: PrecheckRun, element: int) -> list[int]:
        """The constraints that leave an element without a placement on their own.

        Args:
            run: The run.
            element: The element.

        Returns:
            The constraints, as name positions.
        """
        used = np.flatnonzero(run.filtered[element]).tolist()
        return self._explain(partial(self._alone_fails, run, element), used, run.stops)

    @staticmethod
    def _alone_fails(run: PrecheckRun, element: int, constraint_indices: list[int]) -> bool:
        """Whether the element has no placement under ``constraint_indices`` alone.

        Args:
            run: The run.
            element: The element.
            constraint_indices: The constraints that filter, as name positions.

        Returns:
            No start or no set is left.
        """
        domain = run.domain(element, constraint_indices)
        return not domain.starts or not domain.sets

    def _group_constraints(
        self, run: PrecheckRun, group: list[int], blamed: set[int], work: int
    ) -> list[int]:
        """The constraints that make a group unplaceable on their own.

        Args:
            run: The run.
            group: The elements, all with placements alone.
            blamed: The constraints that rejected something in the proof: the others never acted.
            work: The evaluations the proof took: a trial gets as many, else keeps its rule.

        Returns:
            The constraints, as name positions.
        """
        cap = MaxIterations(max(work, self.config.witness_budget))
        fails = partial(self._group_fails, run, group, ExhaustiveSearch(AnyOf([cap, *run.stops])))
        return self._explain(fails, sorted(blamed), run.stops)

    def _group_fails(
        self,
        run: PrecheckRun,
        group: list[int],
        exhaustive: ExhaustiveSearch,
        constraint_indices: list[int],
    ) -> bool:
        """Whether the group is proven unplaceable under ``constraint_indices`` alone.

        Args:
            run: The run.
            group: The elements.
            exhaustive: The search that may prove it, with its budget.
            constraint_indices: The constraints that apply, as name positions.

        Returns:
            The search proves no placement exists.
        """
        return self._search(run, group, constraint_indices, exhaustive).verdict is False

    @staticmethod
    def _order(run: PrecheckRun, group: list[int], sizes: dict[int, int]) -> list[int]:
        """The search order: next, the element most linked to those placed, fewest candidates.

        Args:
            run: The run.
            group: The elements.
            sizes: Per element, its candidates.

        Returns:
            The elements, in the order they are placed.
        """
        members = set(group)
        linked = dict.fromkeys(group, 0)
        heap = [(0, sizes[element], element) for element in group]
        heapq.heapify(heap)
        order: list[int] = []
        placed: set[int] = set()

        while heap:
            links, _, element = heapq.heappop(heap)

            if element in placed or -links != linked[element]:
                continue

            order.append(element)
            placed.add(element)

            for neighbor in run.neighbors[element] & members - placed:
                linked[neighbor] += 1
                heapq.heappush(heap, (-linked[neighbor], sizes[neighbor], neighbor))

        return order

    def _plan(
        self,
        run: PrecheckRun,
        domain: dict[int, LoneDomain],
        constraint_indices: list[int],
        search: ExhaustiveSearch | HeuristicSearch,
    ) -> SearchPlan:
        """The search order, each element's candidates, and what each placement reads.

        Args:
            run: The run.
            domain: Per element of the group, in its order, its starts and its sets.
            constraint_indices: The constraints that apply, as name positions.
            search: The search, which says what is read once placed and what forward.

        Returns:
            The plan, as the kernel reads it.
        """
        group = list(domain)
        members = set(group)

        # a constraint reading elements outside the group cannot be read with them absent; one
        # read per decided set is counted on every set, so only when the group holds all takers
        closed = [
            index
            for index in constraint_indices
            if all(run.ready(element, index, members) for element in group)
            and (index not in run.set_level or run.takers <= members)
        ]

        order = self._order(
            run,
            group,
            {element: len(lone.starts) * len(lone.sets) for element, lone in domain.items()},
        )

        placed: set[int] = set()
        checks: list[list[int]] = []

        for element in order:
            placed.add(element)
            checks.append(search.reads(closed, partial(run.ready, element, present=placed)))

        for element in order:
            alone = partial(run.ready, element, present={element})
            checks.append(search.ahead(closed, alone))

        whole = [
            index for index in closed if any(index in run.needs[element] for element in group)
        ]
        values = [
            getattr(domain[element], half) for element in order for half in ["starts", "sets"]
        ]

        return SearchPlan(
            order=np.asarray(order, dtype=np.int64),
            domains=np.asarray([value for part in values for value in part], dtype=np.int64),
            domain_offsets=np.cumsum([0, *map(len, values)]).astype(np.int64),
            checks=np.asarray([index for part in checks for index in part], dtype=np.int64),
            check_offsets=np.cumsum([0, *map(len, checks)]).astype(np.int64),
            whole=np.asarray(whole, dtype=np.int64),
            closed=run.array(closed),
        )

    def _search(
        self,
        run: PrecheckRun,
        group: list[int],
        constraint_indices: list[int],
        search: ExhaustiveSearch | HeuristicSearch,
    ) -> SearchOutcome:
        """Whether the group can be placed alone, searched in the kernel, other elements absent.

        Args:
            run: The run.
            group: The elements.
            constraint_indices: The constraints that apply, as name positions.
            search: Exhaustive, to prove there is no placement, or heuristic, to find one.

        Returns:
            What it found, the evaluations spent, and what rejected candidates on the way.
        """
        blame = np.zeros(len(run.names), dtype=np.int64)

        if constraint_indices == run.hard:
            domain = {element: run.lone(element) for element in group}
        else:
            domain = {element: run.domain(element, constraint_indices) for element in group}

        for lone in domain.values():
            blame += lone.blame

        if not all(lone.starts and lone.sets for lone in domain.values()):
            return SearchOutcome(found=0, work=0, blame=blame)

        plan = self._plan(run, domain, constraint_indices, search)
        searched = search.search(run.kernel, plan)
        return replace(searched, blame=searched.blame + blame)

    def _resource_issues(self, run: PrecheckRun, resource: Resource) -> list[PrecheckIssue]:
        """The loads a resource cannot carry: per element, per fixed group, per pool of targets.

        Args:
            run: The run.
            resource: A hard resource the model declares.

        Returns:
            One issue per element, group or pool whose demand exceeds what it is offered.
        """
        kernel = run.kernel
        demand = self._per_element(run, resource.demand, range(kernel.n))

        if resource.mask is not None:
            taking = run.data.values[resource.mask]
            demand = [need if taking[element] else 0 for element, need in enumerate(demand)]

        if resource.interval is not None:
            demand = [need if run.span[element] else 0 for element, need in enumerate(demand)]

        if isinstance(resource.groups, Assignment):
            capacity = self._per_element(run, resource.capacity, range(kernel.n_sets))
            return self._pool_issues(run, resource, demand, capacity)

        issues = []

        for load in self._group_loads(run, resource, demand):
            if run.late:
                break

            members = [element for element in load.members if element < run.known]
            issues.append(self._load_issue(run, replace(load, members=members)))

        return issues

    @staticmethod
    def _per_element(
        run: PrecheckRun, value: int | Parameter[Any], elements: Iterable[int]
    ) -> list[int]:
        """A number per given element (or target): the parameter's value there, or the one number.

        Args:
            run: The run.
            value: A number, or a parameter over the elements (the targets).
            elements: The elements (targets), in order.

        Returns:
            The numbers.
        """
        if isinstance(value, Parameter):
            data = run.data.values[value]
            return [int(data[element]) for element in elements]

        return [int(value) for _ in elements]

    @staticmethod
    def _group_loads(run: PrecheckRun, resource: Resource, demand: list[int]) -> list[Load]:
        """The fixed groups of elements a resource is shared by, each a load to grade.

        Args:
            run: The run.
            resource: A resource shared by a family of element subsets, or by every element.
            demand: Per element, what it uses of the capacity.

        Returns:
            One load per group; none for any other collection (not checked).
        """
        capacities = [int(resource.capacity) if isinstance(resource.capacity, int) else 0]
        elements, groups = run.elements, resource.groups

        if isinstance(groups, SubsetFamily) and groups.of is elements:
            rows = run.data.subsets[groups]
            keys = run.data.keys.get(groups) or [
                f"{groups.name} #{position}" for position in range(len(rows))
            ]
            labels, groups_of = [str(key) for key in keys], [sorted(set(row)) for row in rows]
        elif groups is elements:
            labels, groups_of = [elements.name], [list(range(run.kernel.n))]
        else:
            return []

        return [
            Load(
                resource=resource,
                kind=IssueKind.GROUP_LOAD,
                entity=label,
                members=members,
                demand=demand,
                capacities=capacities,
                pool=None,
            )
            for label, members in zip(labels, groups_of, strict=True)
        ]

    def _pool_issues(
        self, run: PrecheckRun, resource: Resource, demand: list[int], capacity: list[int]
    ) -> list[PrecheckIssue]:
        """Per element, a demand above every target it may take; per pool, a Hall condition.

        Args:
            run: The run.
            resource: A resource shared per target of the assignment.
            demand: Per element, what it uses of a target.
            capacity: Per target, what it offers at once.

        Returns:
            The issues: an element no target can hold, or a pool its elements overload.
        """
        issues = []
        names = run.names
        placeable = run.placeable
        by_pool: dict[tuple[int, ...], list[int]] = {}

        for element in range(run.known):
            if not placeable[element] or not demand[element]:
                continue

            largest = max(capacity[target] for target in run.sets[element])

            if demand[element] <= largest:
                by_pool.setdefault(tuple(run.sets[element]), []).append(element)
                continue

            alone = Load(
                resource=resource,
                kind=IssueKind.NO_PLACEMENT,
                entity=self._label(run, element),
                members=[element],
                demand=demand,
                capacities=capacity,
                pool=None,
            )
            fails = partial(self._too_large_fails, run, alone)
            filters = np.flatnonzero(run.filtered[element]).tolist()
            explained = self._explain(fails, filters, run.stops)

            issues.append(
                PrecheckIssue(
                    kind=IssueKind.NO_PLACEMENT,
                    elements=[element],
                    constraints=sorted(
                        [resource.name, *(names[index] for index in explained)], key=names.index
                    ),
                    entity=self._label(run, element),
                    demand=demand[element],
                    offer=largest,
                )
            )

        for pool in sorted(by_pool):
            if run.late:
                break

            members = sorted(
                element
                for other, elements in by_pool.items()
                if set(other) <= set(pool)
                for element in elements
            )

            label = ", ".join(self._target_label(run, target) for target in pool)
            load = Load(
                resource=resource,
                kind=IssueKind.POOL_LOAD,
                entity=label,
                members=members,
                demand=demand,
                capacities=capacity,
                pool=set(pool),
            )
            issues.append(self._load_issue(run, load))

        if resource.mask is None and run.known == run.kernel.n:
            taken = {target for element in range(run.kernel.n) for target in run.sets[element]}
            issues += [
                PrecheckIssue(
                    kind=IssueKind.POOL_LOAD,
                    elements=[],
                    constraints=[resource.name],
                    entity=self._target_label(run, target),
                    demand=0,
                    grade=LoadGrade.UNDERUSED,
                )
                for target in range(run.kernel.n_sets)
                if target not in taken
            ]

        return issues

    def _load_issue(self, run: PrecheckRun, load: Load) -> PrecheckIssue:
        """One issue when the members' demand overloads, nearly fills, or underuses the offer.

        Args:
            run: The run.
            load: The load to grade.

        Returns:
            The load, graded or not (an overload names the level-1 rules its bound relies on).
        """
        names, placeable = run.names, run.placeable
        members = [e for e in load.members if placeable[e] and load.demand[e]]
        load = replace(load, members=members)

        domain = {element: run.lone(element) for element in members}

        breaks = self._breaks(run, load, domain)
        bound = self._load(run, load, domain, breaks)
        grade = self._grade(bound, underused=load.kind == IssueKind.POOL_LOAD)

        # every element pinned and the offer exactly what they use: the load is the data itself
        pinned = all(len(domain[e].starts) == len(domain[e].sets) == 1 for e in members)

        if grade is LoadGrade.NEARLY_FULL and pinned and bound.used == bound.offered:
            grade = None

        constraints = [load.resource.name, *([breaks.name] if breaks is not None else [])]

        if grade is LoadGrade.OVERLOADED:
            filters = np.flatnonzero(run.filtered[members].sum(axis=0)).tolist()
            fails = partial(self._load_fails, run, load)
            constraints += [names[index] for index in self._explain(fails, filters, run.stops)]

        return PrecheckIssue(
            kind=load.kind,
            elements=members,
            constraints=sorted(constraints, key=names.index),
            entity=load.entity,
            demand=bound.used,
            offer=bound.offered,
            grade=grade,
        )

    def _grade(self, bound: LoadBound, *, underused: bool) -> LoadGrade | None:
        """How a load compares with its offer, ``None`` in between.

        Args:
            bound: The demand and the offer.
            underused: Whether a low load is reported.

        Returns:
            Overloaded past the offer, nearly full or underused by the class's shares.
        """
        used, offered = bound.used, bound.offered

        if used > offered:
            return LoadGrade.OVERLOADED

        if used > 0 and used >= self.config.nearly_full_ratio * offered:
            return LoadGrade.NEARLY_FULL

        if underused and used < self.config.underused_ratio * offered:
            return LoadGrade.UNDERUSED

        return None

    @staticmethod
    def _too_large_fails(run: PrecheckRun, alone: Load, constraint_indices: list[int]) -> bool:
        """Whether no target the element may take holds it, its domain filtered by these alone.

        Args:
            run: The run.
            alone: The load of the one element, against every target's capacity.
            constraint_indices: The level-1 filters kept, as name positions.

        Returns:
            Its demand exceeds every target its domain keeps.
        """
        element = alone.members[0]
        sets = run.domain(element, constraint_indices).sets
        return all(alone.demand[element] > alone.capacities[target] for target in sets)

    def _load_fails(self, run: PrecheckRun, load: Load, constraint_indices: list[int]) -> bool:
        """Whether the members still overload their capacities, domains filtered by these alone.

        Args:
            run: The run.
            load: The load, its members already kept.
            constraint_indices: The level-1 filters kept, as name positions.

        Returns:
            Every member still can only use the pool, and demand still exceeds offer.
        """
        domain = {e: run.domain(e, constraint_indices) for e in load.members}
        pool = load.pool

        if pool is not None and any(not set(lone.sets) <= pool for lone in domain.values()):
            return False

        bound = self._load(run, load, domain, self._breaks(run, load, domain))
        return bound.used > bound.offered

    @staticmethod
    def _breaks(run: PrecheckRun, load: Load, domain: dict[int, LoneDomain]) -> BreakRule | None:
        """The breaks posted on the resource's groups, when no member may reach the next period.

        Args:
            run: The run.
            load: The load: its resource and its members.
            domain: Per member, its starts and sets.

        Returns:
            The breaks the bound may take off each period; ``None`` falls back to no breaks.
        """
        resource = load.resource

        # paired with a no-overlap on the same groups; a break frees every member it counts, so
        # those the no-overlap counts must all be among them
        breaks = next(
            (
                rule
                for rule in run.model.break_rules
                if rule.groups is resource.groups
                and rule.interval is resource.interval
                and rule.mask in {None, resource.mask}
                and resource.demand == resource.capacity == 1
            ),
            None,
        )

        if breaks is None:
            return None

        size = breaks.period
        # a member reaching into the next period does not take that period's break
        crossing = any(
            start // size != (start + run.span[element] - 1) // size
            for element in load.members
            for start in domain[element].starts
        )
        return None if crossing else breaks

    def _load(
        self, run: PrecheckRun, load: Load, domain: dict[int, LoneDomain], breaks: BreakRule | None
    ) -> LoadBound:
        """What the members demand, and the most the capacities can give them.

        Args:
            run: The run.
            load: The load: its resource, members, demand and capacities.
            domain: Per member, its starts and sets.
            breaks: The breaks each period keeps, if any.

        Returns:
            The demand, and the offer it is measured against.
        """
        # with time, the offer is the slots members can cover, less breaks; a capacity of 1
        # holds whole members end to end, or each member past one per run adds the least wait
        members, demand, capacities = load.members, load.demand, load.shared

        if load.resource.interval is None:
            return LoadBound(sum(demand[element] for element in members), sum(capacities))

        length = run.span
        used = sum(demand[element] * length[element] for element in members)

        covered = 0

        for element in members:
            for start in domain[element].starts:
                covered |= ((1 << length[element]) - 1) << start

        open_slots = self._slots(run, covered, None, breaks)

        if any(capacity != 1 for capacity in capacities):
            return LoadBound(used, sum(capacities) * open_slots)

        lengths = frozenset(length[element] for element in members)
        fillable = self._slots(run, covered, lengths, breaks)

        # a break may also serve as a wait: with breaks, no wait is added
        gaps = 0 if breaks is not None else self._gaps(run, load, covered)

        if used * open_slots > (used + gaps) * fillable:
            return LoadBound(used, len(capacities) * fillable)

        return LoadBound(used + gaps, len(capacities) * open_slots)

    def _gaps(self, run: PrecheckRun, load: Load, covered: int) -> int:
        """The least transition time the members spend between two of them on one target.

        Args:
            run: The run.
            load: The load: its resource, with or without a transition, its members and targets.
            covered: The slots some member can cover, as bits.

        Returns:
            One least wait per member past one per run of ``covered`` and target.
        """
        transition, members, n_targets = load.resource.transition, load.members, len(load.shared)

        if transition is None or not members:
            return 0

        same = self._per_element(run, transition.same, members)
        other = self._per_element(run, transition.other, members)
        least = min(max(0, min(pair)) for pair in zip(same, other, strict=True))
        runs = len([part for part in f"{covered:b}".split("0") if part])
        return max(0, len(members) - n_targets * runs) * least

    def _slots(
        self,
        run: PrecheckRun,
        covered: int,
        lengths: frozenset[int] | None,
        breaks: BreakRule | None,
    ) -> int:
        """The slots a capacity of 1 can fill inside ``covered``, keeping each period's break.

        Args:
            run: The run.
            covered: The slots some member can cover, as bits.
            lengths: The members' spans, laid whole end to end; ``None``: any hour counts.
            breaks: The break each period keeps, if any.

        Returns:
            The slots, summed over the runs; with breaks, the best place of each period's break.
        """
        if breaks is None:
            return self._filled(run, covered, lengths)

        size, (first, last) = breaks.period, breaks.window
        interval = run.model.integer_decision
        total = 0

        for index in range(interval.high // size if interval else 0):
            period = (covered >> (index * size)) & ((1 << size) - 1)
            total += max(
                (
                    self._filled(run, period & ~(((1 << breaks.length) - 1) << start), lengths)
                    for start in range(first, last - breaks.length + 1)
                ),
                default=self._filled(run, period, lengths),
            )

        return total

    def _filled(self, run: PrecheckRun, covered: int, lengths: frozenset[int] | None) -> int:
        """The slots whole pieces of ``lengths`` fill in the runs of ``covered``.

        Args:
            run: The run.
            covered: The slots, as bits.
            lengths: The pieces' lengths; ``None``: every slot counts.

        Returns:
            The slots.
        """
        runs = [len(part) for part in f"{covered:b}".split("0") if part]

        if lengths is None:
            return sum(runs)

        return sum(self._fillable(run, part, lengths) for part in runs)

    @staticmethod
    def _fillable(run: PrecheckRun, size: int, lengths: frozenset[int]) -> int:
        """The most a run of slots holds of pieces of ``lengths`` laid end to end, any number each.

        Args:
            run: The pre-check run, which keeps each answer once worked out.
            size: The run of slots' length.
            lengths: The pieces' lengths.

        Returns:
            The largest sum of those lengths not past ``size``.
        """
        key = (size, lengths)

        if key not in run.fill:
            reachable = [True] + [False] * size

            for total in range(1, size + 1):
                reachable[total] = any(
                    0 < piece <= total and reachable[total - piece] for piece in lengths
                )

            run.fill[key] = max(total for total in range(size + 1) if reachable[total])

        return run.fill[key]

    def _label(self, run: PrecheckRun, element: int) -> str:
        """An element's name: its key in the schema, else its position.

        Args:
            run: The run.
            element: The element.

        Returns:
            The name.
        """
        return self._name(run.kernel.objects[element], run.elements, element)

    def _target_label(self, run: PrecheckRun, target: int) -> str:
        """A target's name: its key in the schema, else its position.

        Args:
            run: The run.
            target: The target.

        Returns:
            The name.
        """
        assignment = run.model.assignment

        if assignment is None:
            return str(target)

        return self._name(run.data.targets[target], assignment.to, target)

    @staticmethod
    def _name(item: object, index_set: IndexSet[Any], position: int) -> str:
        """An item's name: the value of its set's key attribute, else its position.

        Args:
            item: The schema object.
            index_set: Its set.
            position: Its position in the set.

        Returns:
            The name.
        """
        return str(getattr(item, index_set.key)) if index_set.key else str(position)
