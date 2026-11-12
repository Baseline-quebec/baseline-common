from baseline_core.core.optimization.solver.binding.model_compiler import BoundModel
from baseline_core.core.optimization.solver.search.search_index import SearchIndex


class TestSearchIndex:
    def test__a_crew_s_tasks__are_one_family_with_their_monday_starts(
        self, bound: BoundModel
    ) -> None:
        """The crew's three tasks form one family; each may start from 8 h to 16 h."""
        index = SearchIndex.of(bound)

        assert index.families == [[0, 1, 2]]
        assert index.starts(0) == list(range(8, 17))
        assert index.sets(0) == [0, 1]

    def test__the_period__is_the_one_the_start_decision_declares(self, bound: BoundModel) -> None:
        """A day of the crew model is its slots per day."""
        decision = bound.model.integer_decision

        assert decision is not None
        assert SearchIndex.of(bound).period == decision.period
