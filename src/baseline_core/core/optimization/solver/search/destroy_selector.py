"""Draw one destroy arm per iteration by adaptive weight, and emit the move repairing it."""

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, assert_never

from numpy.random import Generator, RandomState
from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.optimization.solver.search.destroy_repair import DestroyRepair, Release
from baseline_core.core.optimization.solver.search.kernel_state import KernelState
from baseline_core.core.optimization.solver.search.search_index import SearchIndex


class AlnsConfig(BaseModel):
    """How the destroy arms are weighted, and how much a band destroy releases."""

    model_config = ConfigDict(frozen=True)

    roulette_scores: list[float] = Field(default=[25.0, 5.0, 1.0, 0.5], min_length=4)
    """Adaptive scores for a new best, a better, an accepted and a rejected outcome."""
    roulette_decay: float = Field(default=0.8, ge=0.0, le=1.0)
    """Exponential decay of the adaptive weights."""
    band: int = Field(default=4, gt=0)
    """The width of the band of values a band destroy releases."""


class DestroyKind(StrEnum):
    """How an iteration picks what it releases; all but two release whole families.

    Attributes:
        VIOLATING: The families of the elements in a hard violation, plus a few at random.
        RANDOM: Families drawn uniformly at random.
        BAND: The families with an element starting inside one random band of one period.
        SET: The families using one set, the least used one half of the time.
        SET_POOL: Every element that may take a set of one violating element; capped in elements.
        TAIL: The members of a set after a random cut, in order of value; the first one stays.
    """

    VIOLATING = "violating"
    RANDOM = "random"
    BAND = "band"
    SET = "set"
    SET_POOL = "set_pool"
    TAIL = "tail"


@dataclass(frozen=True, slots=True)
class DestroyArm:
    """One roulette arm: how it picks, how many it releases, whether they leave first."""

    kind: DestroyKind
    """How the arm picks what it releases."""
    cap: int
    """Families the arm releases at most (elements for ``SET_POOL``, sets for ``TAIL``)."""
    remove_first: bool
    """Whether the released elements leave the placement before the repair."""
    in_order: bool = False
    """Whether the repair re-places them in the order of their values, not tightest first."""


class DestroySelector:
    """Pick a destroy arm by adaptive weight and emit the move that releases and repairs."""

    ARMS: ClassVar[list[DestroyArm]] = [
        DestroyArm(DestroyKind.VIOLATING, 40, remove_first=False),
        DestroyArm(DestroyKind.RANDOM, 6, remove_first=False),
        DestroyArm(DestroyKind.BAND, 12, remove_first=False),
        DestroyArm(DestroyKind.SET, 40, remove_first=False),
        DestroyArm(DestroyKind.VIOLATING, 40, remove_first=True),
        DestroyArm(DestroyKind.RANDOM, 6, remove_first=True),
        DestroyArm(DestroyKind.BAND, 12, remove_first=True),
        DestroyArm(DestroyKind.SET, 40, remove_first=True),
        DestroyArm(DestroyKind.SET_POOL, 60, remove_first=True),
        DestroyArm(DestroyKind.TAIL, 1, remove_first=True, in_order=True),
        DestroyArm(DestroyKind.TAIL, 2, remove_first=True, in_order=True),
    ]
    """The default arms: in place, then removal, which reaches moves of several at once."""
    RANDOM_PADDING: ClassVar[int] = 4
    """Families ``VIOLATING`` adds at random, so its draw never collapses."""
    LEAST_USED_ODDS: ClassVar[float] = 0.5
    """Share of ``SET`` draws taking the least used set."""

    def __init__(self, index: SearchIndex, config: AlnsConfig, rng: Generator, seed: int) -> None:
        """Keep what moves together, weight the arms, and keep the draws.

        Args:
            index: The elements' placements and what moves together.
            config: The arms' adaptive weighting and the band's width.
            rng: Draws the released elements and the repair's ties.
            seed: Seeds the draw of the arms.
        """
        self.index = index
        self.band = config.band
        self.scores = config.roulette_scores
        self.decay = config.roulette_decay
        self.weights = [1.0] * len(self.ARMS)
        """Each arm's adaptive weight; an arm is drawn in proportion to its weight."""
        self.rng = rng
        self.draw = RandomState(seed)
        self._arm = 0

    def select(self, state: KernelState) -> DestroyRepair | None:
        """Draw an arm and emit the move releasing what it picked.

        Args:
            state: The kernel, read to pick what is released.

        Returns:
            The move, ``None`` when nothing is released.
        """
        n_arms, total = len(self.weights), sum(self.weights)
        odds = [weight / total for weight in self.weights] if total > 0 else [1 / n_arms] * n_arms
        self._arm = int(self.draw.choice(n_arms, p=odds))
        arm = self.ARMS[self._arm]
        indices = self._closed(self._pick(state, arm))

        if not indices:
            return None

        return DestroyRepair(
            Release(indices, arm.remove_first, arm.in_order), self.index, self.rng
        )

    def feedback(self, *, improved_best: bool, improved_current: bool, accepted: bool) -> None:
        """Reward the arm last drawn by what its iteration did.

        Args:
            improved_best: Whether its move beat the best.
            improved_current: Whether its move beat the current placement.
            accepted: Whether its move was kept.
        """
        best, better, kept, rejected = self.scores[:4]

        if improved_best:
            score = best
        elif improved_current:
            score = better
        elif accepted:
            score = kept
        else:
            score = rejected

        self.weights[self._arm] = self.decay * self.weights[self._arm] + (1 - self.decay) * score

    def _pick(self, state: KernelState, arm: DestroyArm) -> list[int]:
        """The elements one arm releases.

        Args:
            state: The kernel, read only.
            arm: The arm drawn.

        Returns:
            The released elements, possibly none.
        """
        match arm.kind:
            case DestroyKind.VIOLATING:
                return self._violating(state, arm.cap)
            case DestroyKind.RANDOM:
                return self._random(arm.cap)
            case DestroyKind.BAND:
                return self._band(state, arm.cap)
            case DestroyKind.SET:
                return self._set(state, arm.cap)
            case DestroyKind.SET_POOL:
                return self._set_pool(state, arm.cap)
            case DestroyKind.TAIL:
                return self._tail(state, arm.cap)
            case _:
                assert_never(arm.kind)

    def _violating(self, state: KernelState, cap: int) -> list[int]:
        """The families of the elements in a hard violation, plus a few at random.

        Args:
            state: The kernel, read only.
            cap: Families released at most.

        Returns:
            The released elements, none when there is no family.
        """
        n_families = len(self.index.families)

        if n_families == 0:
            return []

        violating = self._families_of(state.kernel.violating)
        padding = [int(f) for f in self.rng.integers(0, n_families, size=self.RANDOM_PADDING)]
        return self._elements_of(list(dict.fromkeys([*violating, *padding]))[:cap])

    def _random(self, cap: int) -> list[int]:
        """Families drawn uniformly at random.

        Args:
            cap: Families drawn.

        Returns:
            The released elements, none when there is no family.
        """
        n_families = len(self.index.families)

        if n_families == 0:
            return []

        return self._elements_of([int(f) for f in self.rng.integers(0, n_families, cap)])

    def _band(self, state: KernelState, cap: int) -> list[int]:
        """The families with an element starting inside one random band of one period.

        Args:
            state: The kernel, read only.
            cap: Families released at most.

        Returns:
            The released elements.
        """
        index, rng = self.index, self.rng
        starts, _ = state.kernel.placement
        absent = state.kernel.absent_mask
        period = index.period
        n_periods = max(1, max(starts, default=0) // period + 1)
        low = int(rng.integers(n_periods)) * period + int(rng.integers(period))

        inside = [
            j for j, start in enumerate(starts) if not absent[j] and low <= start < low + self.band
        ]

        families = self._families_of(inside)
        rng.shuffle(families)
        return self._elements_of(families[:cap])

    def _set(self, state: KernelState, cap: int) -> list[int]:
        """The families using one set, the least used one half of the time.

        Args:
            state: The kernel, read only.
            cap: Families released at most.

        Returns:
            The released elements.
        """
        _, targets = state.kernel.placement
        absent = state.kernel.absent_mask

        by_set: dict[int, list[int]] = {}

        for j, target in enumerate(targets):
            if not absent[j]:
                by_set.setdefault(target, []).append(j)

        if not by_set:
            return []

        used = sorted(by_set)

        # emptying a thinly used set gathers the placement onto fewer
        if self.rng.random() < self.LEAST_USED_ODDS:
            fewest = min(len(by_set[target]) for target in used)
            chosen = next(target for target in used if len(by_set[target]) == fewest)
        else:
            chosen = used[int(self.rng.integers(len(used)))]

        return self._elements_of(self._families_of(by_set[chosen])[:cap])

    def _set_pool(self, state: KernelState, cap: int) -> list[int]:
        """Every element that may take a set of one violating element, at most ``cap``.

        Args:
            state: The kernel, read only.
            cap: Elements released at most.

        Returns:
            The released elements, the drawn one always among them.
        """
        index, rng = self.index, self.rng
        n = state.kernel.n

        if n == 0:
            return []

        violating = state.kernel.violating
        drawn = int(rng.choice(violating)) if violating else int(rng.integers(n))
        sets = set(index.sets(drawn))
        pool = [k for k in range(n) if k != drawn and sets.intersection(index.sets(k))]

        if len(pool) >= cap:
            pool = [int(k) for k in rng.choice(pool, size=max(cap - 1, 0), replace=False)]

        return sorted([drawn, *pool])

    def _tail(self, state: KernelState, n_sets: int) -> list[int]:
        """The members after a random cut of up to ``n_sets`` sets, by value; each first one stays.

        Args:
            state: The kernel, read only.
            n_sets: Sets cut at most.

        Returns:
            The released elements.
        """
        rng = self.rng
        starts, targets = state.kernel.placement
        absent = state.kernel.absent_mask
        by_set: dict[int, list[list[int]]] = {}

        for j, target in enumerate(targets):
            if not absent[j]:
                by_set.setdefault(target, []).append([starts[j], j])

        cuttable = sorted(target for target, members in by_set.items() if len(members) > 1)

        if not cuttable:
            return []

        chosen = rng.choice(cuttable, size=min(n_sets, len(cuttable)), replace=False)
        released: list[int] = []

        for target in chosen:
            members = [j for _, j in sorted(by_set[int(target)])]
            released += members[int(rng.integers(1, len(members))) :]

        return sorted(released)

    def _families_of(self, elements: list[int]) -> list[int]:
        """The distinct families of the elements, ascending, an element in none skipped.

        Args:
            elements: The elements.

        Returns:
            Their families.
        """
        family_of = self.index.family_of
        return sorted({family_of[j] for j in elements if family_of[j] >= 0})

    def _elements_of(self, families: list[int]) -> list[int]:
        """Every element of the families, ascending.

        Args:
            families: The families.

        Returns:
            Their elements.
        """
        released: set[int] = set()

        for family in families:
            released.update(self.index.families[family])

        return sorted(released)

    def _closed(self, picked: list[int]) -> list[int]:
        """The picked elements no fix holds, each partner's whole family and each group touched.

        Args:
            picked: The elements an arm released.

        Returns:
            Those, and what must move with them, ascending.
        """
        index = self.index
        released = {j for j in picked if not index.held(j)}
        changed = True

        while changed:
            changed = False

            for j in list(released):
                partner = index.partner[j]

                if partner < 0 or partner in released:
                    continue

                released.add(partner)
                released.update(self._elements_of(self._families_of([partner])))
                changed = True

            # re-placing one part of a group alone would drift the score off the true one
            for group in index.groups:
                if released.intersection(group) and not released.issuperset(group):
                    released.update(group)
                    changed = True

        return sorted(released)
