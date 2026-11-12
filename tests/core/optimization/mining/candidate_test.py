import math
from collections.abc import Callable

import pytest

from baseline_core.core.optimization.mining.candidate import (
    Candidate,
    CandidateFeature,
    ConstrainedDefinition,
)
from baseline_core.core.optimization.mining.plan import Plan
from baseline_core.core.optimization.solver.binding.model_compiler import BoundModel, ModelCompiler
from tests.core.optimization.mining.shop import Shop, ShopDefinition


class TestCandidate:
    @pytest.fixture
    def candidates(self) -> dict[str, Candidate]:
        """One candidate per feature, on operator ``a``, machine ``B`` or everything.

        Returns:
            The candidates, by a short name.
        """
        a_earliest = Candidate(
            role="op", key="a", feature=CandidateFeature.EARLIEST_START, value=3, kept=1, plans=1
        )
        updates = {
            "a latest 5": {"feature": CandidateFeature.LATEST_END, "value": 5},
            "a not day 1": {"feature": CandidateFeature.UNUSED_DAY, "value": 1},
            "a one per day": {"feature": CandidateFeature.MAX_PER_DAY, "value": 1},
            "B latest 4": {
                "role": "machine",
                "key": "B",
                "feature": CandidateFeature.LATEST_END,
                "value": 4,
            },
            "A one per day": {
                "role": "machine",
                "key": "A",
                "feature": CandidateFeature.MAX_PER_DAY,
                "value": 1,
            },
            "closed at 4": {
                "role": "*",
                "key": "*",
                "feature": CandidateFeature.CLOSED_HOURS,
                "value": 0,
                "hours": [4],
            },
            "closed 3 to 4": {
                "role": "*",
                "key": "*",
                "feature": CandidateFeature.CLOSED_HOURS,
                "value": 0,
                "hours": [3, 4],
            },
            "closed 3 and 6": {
                "role": "*",
                "key": "*",
                "feature": CandidateFeature.CLOSED_HOURS,
                "value": 0,
                "hours": [3, 6],
            },
        }
        candidates = {"a earliest 3": a_earliest}

        for name, update in updates.items():
            candidates[name] = a_earliest.model_copy(update=update)

        return candidates

    @pytest.fixture
    def bind_with(self, candidates: dict[str, Candidate]) -> Callable[[str], BoundModel]:
        """Bind the shop of two ``a`` jobs and one ``b`` job, one candidate posted last.

        Args:
            candidates: The candidates, by short name.

        Returns:
            The function that binds it, given the candidate's short name.
        """
        shop = Shop.of(["a", "a", "b"])

        def bind(name: str) -> BoundModel:
            definition = ConstrainedDefinition(ShopDefinition(), shop.view, [candidates[name]])
            return ModelCompiler(definition, cythonize=False).bind(shop)

        return bind

    def test__the_chance__is_the_binomial_tail_of_the_planner_s_count(
        self, candidates: dict[str, Candidate]
    ) -> None:
        """Never broken: certain; broken half the time: 12 respects out of 12 is 1 in 4 096."""
        candidate = candidates["a not day 1"].model_copy(update={"kept": 12, "plans": 12})

        assert math.isclose(candidate.chance(0.0), 1.0)
        assert math.isclose(candidate.chance(0.5), 0.5**12)

    @pytest.mark.parametrize(
        ("name", "starts", "broken"),
        [
            ("a earliest 3", [2, 11, 0], True),
            ("a earliest 3", [3, 11, 0], False),
            ("a latest 5", [4, 8, 0], True),
            ("a latest 5", [3, 8, 0], False),
            ("a not day 1", [0, 8, 0], True),
            ("a not day 1", [0, 3, 0], False),
            ("a one per day", [0, 3, 0], True),
            ("a one per day", [0, 8, 0], False),
            ("B latest 4", [0, 8, 3], True),
            ("B latest 4", [0, 8, 2], False),
            ("A one per day", [0, 3, 0], True),
            ("A one per day", [0, 8, 0], False),
            ("closed at 4", [0, 8, 3], True),
            ("closed at 4", [0, 8, 0], False),
            ("closed 3 to 4", [0, 8, 2], True),
            ("closed 3 to 4", [0, 8, 5], False),
            ("closed 3 and 6", [0, 8, 5], True),
            ("closed 3 and 6", [0, 8, 4], False),
        ],
    )
    def test__the_posted_rule__counts_a_violation_exactly_when_the_plan_breaks_it(
        self,
        bind_with: Callable[[str], BoundModel],
        name: str,
        starts: list[int],
        broken: bool,  # noqa: FBT001
    ) -> None:
        """Both ``a`` jobs on machine A, the ``b`` job on machine B: only real breaks count."""
        bound = bind_with(name)

        Plan(starts=starts, targets=[0, 0, 1]).load(bound.kernel)

        assert bool(bound.kernel.full_counts[-1] > 0) is broken

    def test__an_absent_element__breaks_nothing(
        self, bind_with: Callable[[str], BoundModel]
    ) -> None:
        """An ``a`` job left out on day 1 does not break « a never on day 1 »."""
        bound = bind_with("a not day 1")

        Plan(starts=[0, 8, 0], targets=[0, 0, 1], absent=[1]).load(bound.kernel)

        assert bound.kernel.full_counts[-1] == 0

    @pytest.mark.parametrize(
        ("update", "message"),
        [({"role": "Atelier"}, "no role 'Atelier'"), ({"key": "z"}, "carries the key 'z'")],
    )
    def test__a_role_or_a_key_the_view_lacks__is_refused_when_posted(
        self, candidates: dict[str, Candidate], update: dict[str, str], message: str
    ) -> None:
        """A candidate naming a role or a key the instance lacks is refused, not ignored."""
        shop = Shop.of(["a", "b"])
        candidate = candidates["a earliest 3"].model_copy(update=update)

        with pytest.raises(ValueError, match=message):
            ModelCompiler(ConstrainedDefinition(ShopDefinition(), shop.view, [candidate]))
