import pytest

from baseline_core.core.optimization.solver.binding.model_compiler import BoundModel
from baseline_core.core.optimization.solver.search.search_index import SearchIndex
from tests.core.optimization.job_shop_model import EarlyStartModelDefinition


class TestRelocate:
    def test__a_present_element__reports_the_change_a_full_recount_gives(
        self, bound: BoundModel
    ) -> None:
        """Moved to its best placement, the change it reports is the whole score's change."""
        index = SearchIndex.of(bound)
        kernel = bound.kernel
        kernel.soft_mode = True
        before = [kernel.full_hard, kernel.full_soft]
        change = kernel.relocate(0, -1, index.domains, 0.5)

        assert change == [kernel.full_hard - before[0], kernel.full_soft - before[1]]
        assert change < [0, 0]

    def test__an_absent_element__is_placed_and_reports_its_change_from_where_it_stood(
        self, bound: BoundModel
    ) -> None:
        """Taken out first, it comes back present, its change counted from its old spot."""
        index = SearchIndex.of(bound)
        kernel = bound.kernel
        kernel.soft_mode = True
        before = [kernel.full_hard, kernel.full_soft]
        kernel.remove(1)
        change = kernel.relocate(1, -1, index.domains, 0.5)

        assert kernel.absent_mask[1] == 0
        assert change == [kernel.full_hard - before[0], kernel.full_soft - before[1]]

    def test__an_element_in_a_hard_conflict__is_the_one_named_violating(
        self, bound: BoundModel
    ) -> None:
        """At their defaults the tasks start before 8 h: each breaks its window."""
        kernel = bound.kernel

        assert kernel.violating == [0, 1, 2]
        assert all(kernel.element_hard(element) > 0 for element in range(3))
        kernel.remove(2)
        assert kernel.element_hard(2) == 0

    def test__a_soft_saving__never_buys_a_hard_violation(self, late: BoundModel) -> None:
        """Two hours later cost two million, still less than overlapping: the overlap goes."""
        kernel = late.kernel
        kernel.soft_mode = True

        change = kernel.relocate(1, -1, SearchIndex.of(late).domains, 0.5)

        assert kernel.full_hard == 0
        assert change == [-1, 2 * EarlyStartModelDefinition.HOUR_PRICE]

    def test__a_placement_s_change__is_hard_then_soft(self, late: BoundModel) -> None:
        """Leaving the overlap is an improvement, however much later the operation starts."""
        late.kernel.soft_mode = True

        change = late.kernel.delta_place(1, 2, 0)

        assert change == [-1, 2 * EarlyStartModelDefinition.HOUR_PRICE]
        assert change < [0, 0]


class TestAroundSetCount:
    def test__an_element_s_other_set__is_read_with_its_own(self, overloaded: BoundModel) -> None:
        """Moved to truck B, a parcel counted with truck A too sees every parcel's rule."""
        kernel = overloaded.kernel
        kernel.insert(8, 0, 1)
        whole = kernel.full_count(0)

        assert kernel.around_set_count(0, [8], [0]) == whole
        assert kernel.around_set_count(0, [8], [-1]) < whole

    def test__an_element_left_out__counts_in_none_of_its_own_evaluations(
        self, shop: BoundModel
    ) -> None:
        """Left out on its machine, an operation's overlaps and job order go with it."""
        kernel = shop.kernel
        indexes = range(len(shop.model.names))
        around = [kernel.around_set_count(index, [0], [-1]) for index in indexes]
        full = kernel.full_counts

        kernel.remove(0)
        counted = [kernel.around_set_count(index, [0], [-1]) - around[index] for index in indexes]

        assert counted == [new - old for new, old in zip(kernel.full_counts, full, strict=True)]

    def test__others_of_another_length__are_refused(self, overloaded: BoundModel) -> None:
        """Exactly one other set is given per element."""
        with pytest.raises(ValueError, match="one set per element"):
            overloaded.kernel.around_set_count(0, [7, 8], [0])
