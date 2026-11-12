from baseline_core.core.optimization.solver.infeasibility.placement_search import (
    ExhaustiveSearch,
    HeuristicSearch,
)
from baseline_core.core.optimization.solver.stop_condition import MaxIterations


class TestPlacementSearch:
    def test__an_exhaustive_search__reads_only_what_its_placed_elements_allow(self) -> None:
        """It reads a constraint once it is ready, and prunes ahead with those read alone."""
        exhaustive = ExhaustiveSearch(MaxIterations(10))

        assert exhaustive.reads([3, 1, 2], ready={1, 2}.__contains__) == [1, 2]
        assert exhaustive.ahead([3, 1, 2], alone={3}.__contains__) == [3]

    def test__a_heuristic_search__reads_everything_at_once_and_never_prunes(self) -> None:
        """It reads every closed constraint as soon as an element is placed, and none ahead."""
        heuristic = HeuristicSearch(MaxIterations(10), seed=12345)

        assert heuristic.reads([3, 1, 2], ready=set().__contains__) == [3, 1, 2]
        assert heuristic.ahead([3, 1, 2], alone={1, 2, 3}.__contains__) == []
