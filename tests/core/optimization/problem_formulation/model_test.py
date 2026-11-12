import pytest

from baseline_core.core.optimization.problem_formulation.expression import (
    Comparison,
    Constant,
    Expr,
)
from baseline_core.core.optimization.problem_formulation.model import Model, Tie
from baseline_core.core.optimization.solver.binding.model_compiler import ModelCompiler
from tests.core.optimization.counter import Counter
from tests.core.optimization.crew_model import CrewModelConfig, Workshop
from tests.core.optimization.models import SharedStationDefinition, WorkshopFactory


class TestGuards:
    def test__a_fractional_lookup_value__is_refused(self) -> None:
        """The kernel counts in integers: 2.5 is refused, never truncated to 2."""
        with pytest.raises(ValueError, match="must be integers"):
            Model.lookup_table([1, 2.5])  # type: ignore[list-item]  # deliberate wrong value

    def test__whole_lookup_values__are_kept(self) -> None:
        """A whole number written as a float is the same integer."""
        table = Model.lookup_table([1, 2.0])  # type: ignore[list-item]  # deliberate float

        assert table.values == [1, 2]

    @staticmethod
    def _start() -> Expr:
        """The start of the first task of a workshop model.

        Returns:
            The expression.
        """
        model = Model(Workshop)
        return model.var("start", model.set(lambda s: s.tasks, key="id_"), high=120)[0]

    def test__an_expression__refuses_a_truth_value(self) -> None:
        """An ``if`` on an expression fails where written instead of deciding once for all."""
        with pytest.raises(TypeError, match="no truth value"):
            bool(self._start() <= 1)

    def test__an_expression__refuses_membership(self) -> None:
        """``in`` on expressions fails instead of answering true for any list."""
        with pytest.raises(TypeError, match="no truth value"):
            _ = self._start() in [self._start()]  # noqa: FURB171, PLR6201  what went wrong

    def test__a_second_integer_decision__is_refused(self) -> None:
        """The kernel holds one integer per element: a second ``m.var`` is refused, not ignored."""
        model = Model(Workshop)
        tasks = model.set(lambda s: s.tasks, key="id_")
        model.var("start", tasks, high=120)

        with pytest.raises(ValueError, match=r"already has an m\.var"):
            model.var("finish", tasks, high=120)

    def test__an_integer_decision_without_high__is_refused_where_written(self) -> None:
        """The search and the pre-check list every value: an unbounded ``m.var`` fails at once."""
        model = Model(Workshop)
        tasks = model.set(lambda s: s.tasks, key="id_")

        with pytest.raises(TypeError, match="'high'"):
            # the missing bound is the point: the engine, not the type checker, must refuse it
            model.var("start", tasks)  # type: ignore[call-arg]

    def test__a_second_assignment__is_refused(self) -> None:
        """The kernel holds one set per element: a second ``m.assign`` is refused, not ignored."""
        model = Model(Workshop)
        tasks = model.set(lambda s: s.tasks, key="id_")
        stations = model.set(lambda s: s.stations, key="id_")
        model.assign("station", tasks, stations)

        with pytest.raises(ValueError, match=r"already has an m\.assign"):
            model.assign("backup_station", tasks, stations)

    def test__a_name_posted_hard_then_soft__is_refused(self) -> None:
        """A name is one kind: a constraint, then a term to minimize, fails where it is written."""
        model = Model(Workshop)
        start = model.var("start", model.set(lambda s: s.tasks, key="id_"), high=120)
        model.constraint(start[0] >= 0, name="rule")

        with pytest.raises(ValueError, match="one name, one kind"):
            model.minimize(start[0], name="rule", weight=1)

    def test__a_soft_term_posted_with_two_weights__is_refused(self) -> None:
        """A term to minimize keeps one weight: a second weight fails where it is written."""
        model = Model(Workshop)
        start = model.var("start", model.set(lambda s: s.tasks, key="id_"), high=120)
        model.minimize(start[0], name="late", weight=2)

        with pytest.raises(ValueError, match="one name, one kind"):
            model.minimize(start[1], name="late", weight=3)

    def test__a_tie_on_a_constraint_never_posted__is_refused(self) -> None:
        """A tie on a missing name would silently tie nothing: it fails where it is written."""
        model = Model(Workshop)
        model.var("start", model.set(lambda s: s.tasks, key="id_"), high=120)

        with pytest.raises(ValueError, match="'no_such_rule' ties nothing"):
            model.tie("no_such_rule", Tie.RELEASED)

    def test__two_constraints_tied_the_same_way__are_refused(self) -> None:
        """A search reads one constraint per tie: a second one fails instead of replacing it."""
        model = Model(Workshop)
        start = model.var("start", model.set(lambda s: s.tasks, key="id_"), high=120)
        model.constraint(start[0] >= 0, name="first")
        model.constraint(start[1] >= 0, name="second")
        model.tie("first", Tie.PAIRED)

        with pytest.raises(ValueError, match="one constraint per tie"):
            model.tie("second", Tie.PAIRED)


class TestSelectors:
    def test__a_wrong_field__is_refused_naming_the_fields(self) -> None:
        """A selector reading a field the schema lacks fails at definition, listing the fields."""
        model = Model(Workshop)
        tasks = model.set(lambda s: s.tasks, key="id_")

        with pytest.raises(ValueError, match=r"no field 'crew'.*'crew_id'"):
            # the wrong field is the point: the engine, not the type checker, must refuse it
            model.param(lambda task: task.crew, tasks)  # type: ignore[attr-defined]

    def test__a_wrong_key__is_refused_naming_the_fields(self) -> None:
        """A set keyed by a field its elements lack fails at definition, listing the fields."""
        model = Model(Workshop)

        with pytest.raises(ValueError, match=r"keyed by 'code'.*'id_'"):
            model.set(lambda s: s.routes, key="code")

    def test__a_reference__is_followed_by_the_key_of_its_set(self) -> None:
        """``routes[task.route_id]`` reads the route's field, joined on the route's key."""
        model = Model(Workshop)
        tasks = model.set(lambda s: s.tasks, key="id_")
        routes = model.set(lambda s: s.routes, key="id_")
        length = model.param(lambda task: routes[task.route_id].options[0].duration, tasks)

        assert length.through is not None
        base, index = length.through.base, length.through.index
        assert (base.name, base.over, index.name, index.of) == (
            "routes.options.0.duration",
            routes,
            "tasks.route_id",
            routes,
        )

    def test__a_set_without_key__cannot_be_followed(self) -> None:
        """Following a reference into a set with no key fails at definition."""
        model = Model(Workshop)
        tasks = model.set(lambda s: s.tasks, key="id_")
        routes = model.set(lambda s: s.routes)

        with pytest.raises(ValueError, match="no key"):
            model.param(lambda task: routes[task.route_id].name, tasks)


class TestOperators:
    def test__greater_than__is_less_than_with_the_operands_swapped(self) -> None:
        """``a > 2`` is written ``2 < a``: one operator per comparison, no mirror twin."""
        start = TestGuards._start()
        node = start > 2

        assert isinstance(node, Comparison)
        assert isinstance(node.left, Constant)
        assert (node.operator, node.left.value, node.right is start) == ("<", 2.0, True)


class TestNoOverlap:
    def test__two_tasks_of_one_crew_at_once__count_once(
        self, counter: Counter, factory: WorkshopFactory
    ) -> None:
        """Overlapping tasks of one crew at two stations: a crew overlap, no station overlap."""
        workshop = factory.workshop(
            [factory.task("a", crew_id="C"), factory.task("b", crew_id="C")],
            [["S1"], ["S2"]],
            [2, 2],
        )

        counts = counter.of(workshop, starts=[8, 9], targets=[0, 1])

        assert (counts["no_crew_overlap"], counts["no_station_overlap"]) == (1, 0)

    def test__two_tasks_at_one_station_at_once__count_once(
        self, counter: Counter, factory: WorkshopFactory
    ) -> None:
        """Overlapping tasks of two crews at one station: a station overlap, no crew overlap."""
        workshop = factory.workshop(
            [factory.task("a"), factory.task("b")], [["S1"], ["S1"]], [2, 2]
        )
        counts = counter.of(workshop, starts=[8, 9], targets=[0, 0])

        assert (counts["no_crew_overlap"], counts["no_station_overlap"]) == (0, 1)

    def test__back_to_back_tasks__do_not_overlap(
        self, counter: Counter, factory: WorkshopFactory
    ) -> None:
        """A task starting where the other ends overlaps neither by crew nor by station."""
        workshop = factory.workshop(
            [factory.task("a", crew_id="C"), factory.task("b", crew_id="C")],
            [["S1"], ["S1"]],
            [2, 2],
        )

        counts = counter.of(workshop, starts=[8, 10], targets=[0, 0])

        assert (counts["no_crew_overlap"], counts["no_station_overlap"]) == (0, 0)

    def test__a_decision_without_length__is_refused(self) -> None:
        """A primitive over a decision with no span is refused where it is written."""
        model = Model(Workshop)
        tasks = model.set(lambda s: s.tasks, key="id_")
        start = model.var("start", tasks, high=120)

        with pytest.raises(ValueError, match="has no span"):
            model.no_overlap(tasks, start, name="no_overlap")


class TestCumulative:
    def test__three_tasks_at_once_at_a_station_for_two__overload_it_once(
        self, config: CrewModelConfig, factory: WorkshopFactory
    ) -> None:
        """Three overlapping tasks at a station for two: one overload, at the third one's start."""
        workshop = factory.workshop(
            [factory.task("a"), factory.task("b"), factory.task("c")],
            [["S1"]] * 3,
            [4] * 3,
            capacities={"S1": 2},
        )

        counter = Counter(compiled=ModelCompiler(SharedStationDefinition(config), cythonize=False))

        assert counter.of(workshop, starts=[8, 9, 10], targets=[0, 0, 0])["station_capacity"] == 1

    def test__tasks_within_the_capacity__break_nothing(
        self, config: CrewModelConfig, factory: WorkshopFactory
    ) -> None:
        """Two overlapping tasks fit a station for two; the third starts when the first ends."""
        workshop = factory.workshop(
            [factory.task("a"), factory.task("b"), factory.task("c")],
            [["S1"]] * 3,
            [4] * 3,
            capacities={"S1": 2},
        )

        counter = Counter(compiled=ModelCompiler(SharedStationDefinition(config), cythonize=False))

        assert counter.of(workshop, starts=[8, 9, 12], targets=[0, 0, 0])["station_capacity"] == 0

    def test__each_station__has_its_own_capacity(
        self, config: CrewModelConfig, factory: WorkshopFactory
    ) -> None:
        """The same three tasks overload a station for two, not a station for three."""
        tasks = [factory.task("a"), factory.task("b"), factory.task("c")]
        workshop = factory.workshop(
            tasks, [["S1", "S2"]] * 3, [4] * 3, capacities={"S1": 2, "S2": 3}
        )
        counter = Counter(compiled=ModelCompiler(SharedStationDefinition(config), cythonize=False))
        in_small = counter.of(workshop, starts=[8, 9, 10], targets=[0, 0, 0])["station_capacity"]
        in_large = counter.of(workshop, starts=[8, 9, 10], targets=[1, 1, 1])["station_capacity"]

        assert (in_small, in_large) == (1, 0)

    def test__a_capacity_per_target_without_an_assignment__is_refused(self) -> None:
        """A capacity per station needs groups that are the stations of an assignment."""
        model = Model(Workshop)
        tasks = model.set(lambda s: s.tasks, key="id_")
        stations = model.set(lambda s: s.stations, key="id_")
        start = model.var("start", tasks, span=model.param(lambda task: task.run, tasks), high=120)

        with pytest.raises(ValueError, match="capacity per target"):
            model.cumulative(
                tasks,
                start,
                demand=1,
                capacity=model.param(lambda station: station.capacity, stations),
                name="station_capacity",
            )
