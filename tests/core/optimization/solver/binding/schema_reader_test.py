import pytest
from pydantic import BaseModel

from baseline_core.core.optimization.problem_formulation.model import Model
from baseline_core.core.optimization.solver.binding.schema_reader import SchemaReader


class Item(BaseModel):
    ref: str
    zone: str


class Store(BaseModel):
    items: list[Item]


class TestKeys:
    def test__a_grouping__names_each_subset_by_its_key_in_order(self) -> None:
        """A grouping's subsets are named by their keys, in the keys' order."""
        model = Model(Store)
        items = model.set(lambda s: s.items, key="ref")
        model.var("at", items, high=5)
        zones = model.group_by(lambda item: item.zone, items)
        store = Store(items=[Item(ref=f"i{k}", zone=f"z{k % 3}") for k in range(4)])

        assert SchemaReader(store).keys(zones) == ["z0", "z1", "z2"]

    def test__a_family_read_as_is__has_no_keys(self) -> None:
        """A family read from the schema has no keys to name its subsets by."""
        model = Model(Store)
        items = model.set(lambda s: s.items, key="ref")
        model.var("at", items, high=5)
        rows = model.subsets(lambda item: [item.zone], items)
        store = Store(items=[Item(ref="i0", zone="z")])

        assert SchemaReader(store).keys(rows) is None


class TestSchemaReader:
    @pytest.mark.parametrize("distinct_first", [True, False])
    def test__two_sets_on_one_list__each_count_their_own_elements(
        self, *, distinct_first: bool
    ) -> None:
        """A plain set and a distinct one on the same list count 3 and 2, whichever reads first."""
        model = Model(Store)
        every = model.set(lambda s: s.items)
        unique = model.set(lambda s: s.items, distinct=True)
        twin = Item(ref="a", zone="z")
        store = Store(items=[twin, twin.model_copy(), Item(ref="b", zone="z")])
        reader = SchemaReader(store)
        order = [unique, every] if distinct_first else [every, unique]

        counts = {id(index_set): reader.read(index_set) for index_set in order}

        assert [counts[id(every)], counts[id(unique)]] == [3, 2]
