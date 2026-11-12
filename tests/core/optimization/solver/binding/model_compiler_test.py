import gc
import itertools

import numpy as np
import pytest
from numpy.random import default_rng
from pydantic import BaseModel, ValidationError

from baseline_core.core.optimization.problem_formulation.abstract_model_definition import (
    AbstractModelDefinition,
)
from baseline_core.core.optimization.problem_formulation.declaration import (
    IndexSet,
    IntegerDecision,
)
from baseline_core.core.optimization.problem_formulation.model import Model
from baseline_core.core.optimization.solver.binding.model_compiler import ModelCompiler
from baseline_core.core.optimization.solver.model_solution import ModelSolution, VariableAssignment
from tests.core.optimization.crew_model import CrewModelConfig, Spacing, Workshop
from tests.core.optimization.large_crew_problem import LargeCrewProblem
from tests.core.optimization.models import WorkshopFactory


class TestPausedCollector:
    def test__the_collector__is_paused_inside_and_restored_after(self) -> None:
        """Inside the block the collector is off; after it, it is on again."""
        with ModelCompiler._paused_collector():
            inside = gc.isenabled()

        assert (inside, gc.isenabled()) == (False, True)

    def test__an_error__still_restores_the_collector(self) -> None:
        """A bind that fails leaves the collector as it found it."""
        with pytest.raises(ValueError, match="bind failed"), ModelCompiler._paused_collector():
            self._fail()

        assert gc.isenabled()

    @staticmethod
    def _fail() -> None:
        """Stand for a bind that raises.

        Raises:
            ValueError: Always.
        """
        msg = "bind failed"
        raise ValueError(msg)

    def test__a_collector_already_off__stays_off(self) -> None:
        """A caller who turned the collector off keeps it off."""
        gc.disable()

        try:
            with ModelCompiler._paused_collector():
                pass

            assert not gc.isenabled()
        finally:
            gc.enable()


class TestVariableAssignment:
    def test__an_assignment_giving_nothing__is_refused(self) -> None:
        """An assignment with neither a value nor a target names nothing to hold or start from."""
        with pytest.raises(ValidationError, match="gives neither a value nor a target"):
            VariableAssignment(element="a")


class TestKernel:
    def test__the_compiled_kernel__counts_as_its_pure_python_run(
        self,
        config: CrewModelConfig,
        cythonized: ModelCompiler[Workshop],
        compiled: ModelCompiler[Workshop],
        factory: WorkshopFactory,
    ) -> None:
        """On random placements, the compiled kernel counts every rule as its Python run."""
        spacings = list(Spacing)
        tasks = [
            factory.task(
                f"t{index}",
                crew_id=f"C{index % 3}",
                order_key=f"O{index % 4}",
                run=index % 3 + 1,
                spacing=spacings[index % len(spacings)],
                days=range(config.n_days),
            )
            for index in range(12)
        ]

        workshop = factory.workshop(tasks, [["S1", "S2"]] * len(tasks), [2] * len(tasks))
        kernels = [model.bind(workshop).kernel for model in (cythonized, compiled)]

        assert [kernel.COMPILED for kernel in kernels] == [True, False]

        generator, horizon = default_rng(7), config.n_days * config.slots_per_day

        for _ in range(30):
            starts = generator.integers(0, horizon - 2, len(tasks)).astype(np.int64)
            stations = generator.integers(0, 2, len(tasks)).astype(np.int64)

            for kernel in kernels:
                kernel.load(start=starts, station=stations)

            assert kernels[0].full_counts == kernels[1].full_counts

    def test__tasks_left_out__are_counted_alike_compiled_or_not(
        self, cythonized: ModelCompiler[Workshop], compiled: ModelCompiler[Workshop]
    ) -> None:
        """With clean tasks, co-starts and random tasks left out, both kernels count alike."""
        workshop = LargeCrewProblem(6).build(seed=0)
        kernels = [model.bind(workshop).kernel for model in [cythonized, compiled]]
        n_tasks, n_stations = len(workshop.tasks.root), len(workshop.stations)
        everyone = np.zeros(n_tasks, dtype=np.int64)
        generator, changed = default_rng(0), 0

        for _ in range(10):
            starts = generator.integers(0, 117, n_tasks)
            stations = generator.integers(0, n_stations, n_tasks)
            absent = (generator.random(n_tasks) < 0.3).astype(np.int64)

            for kernel in kernels:
                kernel.set_absent(everyone)
                kernel.load(start=starts, station=stations)

            present = kernels[1].full_counts

            for kernel in kernels:
                kernel.set_absent(absent)

            counts = [kernel.full_counts for kernel in kernels]
            changed += counts[1] != present

            assert counts[0] == counts[1]

        assert changed > 0

    def test__a_placement_put_in__comes_back_out_with_its_counts(
        self, compiled: ModelCompiler[Workshop], factory: WorkshopFactory
    ) -> None:
        """Two tasks placed by key come back at their slots and station, counted by the model."""
        tasks = [factory.task("t1", crew_id="C"), factory.task("t2", crew_id="C")]
        workshop = factory.workshop(tasks, [["S1", "S2"], ["S1", "S2"]], [2, 3])
        bound = compiled.bind(workshop)
        bound.place([
            VariableAssignment(element="t2", value=9, target="S2"),
            VariableAssignment(element="t1", value=9, target="S2"),
        ])
        placed = bound.solution

        assert [[a.element, a.value, a.target] for a in placed.assignments] == [
            ["t1", 9, "S2"],
            ["t2", 9, "S2"],
        ]

        assert placed.summary.constraints["no_station_overlap"] == 1
        assert placed.summary.constraints["no_crew_overlap"] == 1
        assert placed.summary.score.hard == bound.kernel.full_hard
        assert placed.summary.terms["station_utilization"].cost == 20
        assert placed.unassigned == []

    def test__a_violation__is_named_by_its_constraint_and_its_members(
        self, compiled: ModelCompiler[Workshop], factory: WorkshopFactory
    ) -> None:
        """A crew's two tasks at once: one ``no_crew_overlap`` violation naming both."""
        workshop = factory.workshop(
            [factory.task("a", crew_id="C"), factory.task("b", crew_id="C")],
            [["S1"], ["S2"]],
            [2, 2],
        )

        kernel = compiled.bind(workshop).kernel
        kernel.load(start=np.array([8, 9]), station=np.array([0, 1]))
        overlaps = [members for name, _, members in kernel.violations if name == "no_crew_overlap"]

        assert overlaps == [[0, 1]]

    def test__a_move__changes_the_score_by_its_delta_compiled_or_not(
        self,
        config: CrewModelConfig,
        cythonized: ModelCompiler[Workshop],
        compiled: ModelCompiler[Workshop],
        factory: WorkshopFactory,
    ) -> None:
        """Moving a task as ``delta_place`` was asked changes the recount by that delta."""
        tasks = [
            factory.task(f"t{index}", crew_id=f"C{index % 2}", days=range(config.n_days))
            for index in range(6)
        ]

        workshop = factory.workshop(tasks, [["S1", "S2"]] * len(tasks), [2] * len(tasks))
        generator, horizon = default_rng(3), config.n_days * config.slots_per_day

        for model in (cythonized, compiled):
            kernel = model.bind(workshop).kernel
            kernel.soft_mode = True

            for _ in range(20):
                starts = generator.integers(0, horizon - 2, len(tasks))
                stations = generator.integers(0, 2, len(tasks))
                kernel.load(start=starts, station=stations)
                before = [kernel.full_hard, kernel.full_soft]

                element = int(generator.integers(0, len(tasks)))
                value, target = (
                    int(generator.integers(0, horizon - 2)),
                    int(generator.integers(0, 2)),
                )
                delta = kernel.delta_place(element, value, target)

                starts[element], stations[element] = value, target
                kernel.load(start=starts, station=stations)

                assert [kernel.full_hard - before[0], kernel.full_soft - before[1]] == delta
                assert [list(column) for column in kernel.placement] == [
                    list(starts),
                    list(stations),
                ]


class Item(BaseModel):
    ref: str
    zone: str


class Shelf(BaseModel):
    code: str


class Store(BaseModel):
    items: list[Item]
    shelves: list[Shelf]


class ReadAroundDefinition(AbstractModelDefinition[Store]):
    """Terms whose evaluations also read other elements: a zone, a shelf, a fixed item."""

    schema = Store

    def define(self, model: Model[Store]) -> None:
        """Post the rules.

        Args:
            model: The model to post on.
        """
        items = model.set(lambda s: s.items, key="ref")
        shelves = model.set(lambda s: s.shelves, key="code")

        at, shelf = model.var("at", items, high=6), model.assign("shelf", items, shelves)
        zones = model.group_by(lambda item: item.zone, items)

        for zone in zones:
            item, other = items.member, zone.member
            model.constraint(model.count(other, at[other] == at[item]) <= 1, name="zone")

        for group in shelf.groups:
            item, other = items.member, group.member
            model.constraint(model.count(other, at[other] < at[item]) <= 2, name="shelf")
            held, zoned = group.member, zones[0].member
            model.constraint(model.count(zoned, at[zoned] < at[held]) <= 1, name="shelf_zone")

        first, second = items.pair
        model.constraint((at[first] != at[second]) | (at[first] == at[0]), name="pair_ref")


class TestReadAround:
    def test__a_move__changes_the_count_by_its_delta_when_others_read_the_element(self) -> None:
        """An element read by another element's evaluation still moves the count by its delta."""
        store = Store(
            items=[Item(ref=f"i{index}", zone=f"z{index % 3}") for index in range(9)],
            shelves=[Shelf(code=f"s{index}") for index in range(3)],
        )

        generator = default_rng(5)

        for cythonize in (True, False):
            kernel = ModelCompiler(ReadAroundDefinition(), cythonize=cythonize).bind(store).kernel

            for _ in range(200):
                starts, shelves = generator.integers(0, 6, 9), generator.integers(0, 3, 9)
                kernel.load(at=starts, shelf=shelves)
                before = kernel.full_hard

                element = int(generator.integers(0, 9))
                value, target = int(generator.integers(0, 6)), int(generator.integers(0, 3))
                delta = kernel.delta_place(element, value, target)

                starts[element], shelves[element] = value, target
                kernel.load(at=starts, shelf=shelves)

                assert [kernel.full_hard - before, 0] == delta


class Partner(BaseModel):
    ref: str
    partner: str


class Partners(BaseModel):
    items: list[Partner]


class PartnerDefinition(AbstractModelDefinition[Partners]):
    """A start read at the element another element's data names (``at[partner[item]]``)."""

    schema = Partners

    def __init__(self, *, aggregated: bool) -> None:
        """Choose where the read element appears.

        Args:
            aggregated: Whether it is read inside an aggregate rather than by a member term.
        """
        self.aggregated = aggregated

    def define(self, model: Model[Partners]) -> None:
        """Post the rule.

        Args:
            model: The model to post on.
        """
        items = model.set(lambda s: s.items, key="ref")
        partner = model.param(lambda item: item.partner, items, of=items)
        at = model.var("at", items, high=5)
        item = items.member

        if self.aggregated:
            model.constraint(model.count(item, at[partner[item]] == 0) <= 0, name="late")
        else:
            model.constraint(at[partner[item]] != at[item] + 1, name="after")


class TestDataNamedElement:
    @pytest.mark.parametrize("aggregated", [False, True])
    def test__an_element_named_by_data__is_refused_at_bind(self, *, aggregated: bool) -> None:
        """A decision read at an element that data names is refused when bound, not miscounted."""
        partners = Partners(
            items=[Partner(ref=f"i{index}", partner=f"i{(index + 1) % 3}") for index in range(3)]
        )
        compiled = ModelCompiler(PartnerDefinition(aggregated=aggregated), cythonize=False)

        with pytest.raises(ValueError, match="named by data"):
            compiled.bind(partners)


class NoElementDefinition(AbstractModelDefinition[Store]):
    """A rule that reads no element: the number of shelves."""

    schema = Store

    def define(self, model: Model[Store]) -> None:
        """Post the rule.

        Args:
            model: The model to post on.
        """
        items = model.set(lambda s: s.items, key="ref")
        shelves = model.set(lambda s: s.shelves, key="code")
        model.var("at", items, high=5)
        model.constraint(shelves.size >= 2, name="shelves")


class TestNoElement:
    def test__a_rule_reading_no_element__binds_and_counts(self) -> None:
        """A rule no element can change compiles, counts once, and moves no delta."""
        store = Store(items=[Item(ref="a", zone="z")], shelves=[Shelf(code="s")])

        for cythonize in (True, False):
            kernel = ModelCompiler(NoElementDefinition(), cythonize=cythonize).bind(store).kernel
            kernel.load(at=np.array([0]))
            assert [kernel.full_hard, kernel.delta_place(0, 3, 0)] == [1, [0, 0]]


class Job(BaseModel):
    num: int


class Plant(BaseModel):
    jobs: list[Job]


class IntKeyDefinition(AbstractModelDefinition[Plant]):
    """Jobs keyed by an int field, each at a start."""

    schema = Plant

    def define(self, model: Model[Plant]) -> None:
        """Post the rule.

        Args:
            model: The model to post on.
        """
        jobs = model.set(lambda s: s.jobs, key="num")
        at = model.var("at", jobs, high=5)
        job = jobs.member
        model.constraint(at[job] <= 3, name="early")


class TestBoundModel:
    def test__an_int_key__binds_as_its_digits(self) -> None:
        """Jobs keyed by an int bind; the keys a solution names are their digits."""
        compiled = ModelCompiler(IntKeyDefinition(), cythonize=False)
        bound = compiled.bind(Plant(jobs=[Job(num=1), Job(num=20)]))

        assert bound.element_positions == {"1": 0, "20": 1}

    def test__a_solution_run_as_pure_python__survives_json(self) -> None:
        """A solution counted by the pure-Python kernel holds plain ints, so JSON gives it back."""
        bound = ModelCompiler(IntKeyDefinition(), cythonize=False).bind(Plant(jobs=[Job(num=1)]))
        bound.place([VariableAssignment(element="1", value=4)])
        solution = bound.solution

        assert ModelSolution.model_validate_json(solution.model_dump_json()) == solution
        assert solution.summary.constraints == {"early": 1}

    def test__an_element_out_of_range__is_refused_by_remove(self) -> None:
        """Removing an element the kernel does not hold raises, rather than writing past it."""
        kernel = ModelCompiler(IntKeyDefinition(), cythonize=False).bind(Plant(jobs=[])).kernel

        with pytest.raises(ValueError, match="out of range"):
            kernel.remove(3)


class Depot(BaseModel):
    code: str
    opens: int


class Truck(BaseModel):
    code: str


class Slot(BaseModel):
    start: int
    end: int


class Client(BaseModel):
    ref: str
    depot_code: str
    zone: str
    extra_zones: list[str]
    blocked: bool
    slots: list[Slot]

    @property
    def zones(self) -> list[str]:
        """The zones it is delivered in: none while blocked."""
        return [] if self.blocked else [self.zone, *self.extra_zones]


class Network(BaseModel):
    depots: list[Depot]
    trucks: list[Truck]
    clients: list[Client]


class DeliveryDefinition(AbstractModelDefinition[Network]):
    """Deliveries: a depot named by text, zones as lists of keys, slots copied between clients."""

    schema = Network

    def define(self, model: Model[Network]) -> None:
        """Post the rules.

        Args:
            model: The model to post on.
        """
        clients = model.set(lambda s: s.clients, key="ref")
        depots = model.set(lambda s: s.depots, key="code")
        depot = model.param(lambda client: client.depot_code, clients, of=depots)
        opens = model.param(lambda depot: depot.opens, depots)
        zone = model.param(lambda client: client.zone, clients)
        zones = model.group_by(lambda client: client.zones, clients)
        at = model.var("at", clients, high=24)
        truck = model.assign("truck", clients, model.set(lambda s: s.trucks))

        client = clients.member
        model.constraint(at[client] >= opens[depot[client]], name="after_opening")

        for group in zones:
            first, second = group.pair
            model.constraint(at[first] != at[second], name="zone_apart")

        for load in truck.groups:
            first, second = load.pair
            model.constraint(
                (zone[first] == zone[second]) | (at[first] != at[second]),
                name="one_zone_per_truck",
            )

        self._post_in_a_slot(model, clients, at)

    @staticmethod
    def _post_in_a_slot(
        model: Model[Network], clients: IndexSet[Client], at: IntegerDecision
    ) -> None:
        """Each client delivered inside one of its slots, the copies of a slot read once.

        Args:
            model: The model to post on.
            clients: The clients.
            at: Per client, its delivery time.
        """
        slots = model.set(lambda s: [slot for c in s.clients for slot in c.slots], distinct=True)
        low = model.param(lambda slot: slot.start, slots)
        high = model.param(lambda slot: slot.end, slots)
        slots_of = model.subsets(lambda client: client.slots, clients, of=slots)

        client = clients.member
        slot = slots_of[client].member
        model.constraint(
            model.count(slot, (low[slot] <= at[client]) & (at[client] < high[slot])) > 0,
            name="in_a_slot",
        )


class TestDeliveries:
    @pytest.fixture
    def network(self) -> Network:
        """Three depots, fourteen clients, some blocked, the slots each holds a copy of.

        Returns:
            The network.
        """
        generator = default_rng(0)
        depots = [Depot(code=f"D{k}", opens=int(generator.integers(6, 10))) for k in range(3)]
        zones = ["north", "south", "east", "west"]
        shared = [Slot(start=8, end=9), Slot(start=10, end=14)]
        clients = [
            Client(
                ref=f"C{index}",
                depot_code=f"D{generator.integers(3)}",
                zone=zones[generator.integers(4)],
                extra_zones=list(generator.permutation(zones)[: generator.integers(3)]),
                blocked=bool(generator.random() < 1 / 3),
                slots=[shared[k].model_copy() for k in generator.permutation(2)[: index % 2 + 1]],
            )
            for index in range(14)
        ]
        trucks = [Truck(code=f"T{index}") for index in range(3)]
        return Network(depots=depots, trucks=trucks, clients=clients)

    def test__a_schema_unrelated_to_the_workshop__counts_each_rule_as_its_objects_say(
        self, network: Network
    ) -> None:
        """A depot named by text, clients in many zones or none, copied slots: counted as read."""
        compiled = ModelCompiler(DeliveryDefinition(), cythonize=False)
        kernel = compiled.bind(network).kernel
        generator = default_rng(3)

        for _ in range(30):
            at = generator.integers(5, 11, len(network.clients))
            truck = generator.integers(0, len(network.trucks), len(network.clients))
            expected = self._direct(network, at.tolist(), truck.tolist())

            kernel.load(at=at, truck=truck)
            counted = {
                name: kernel.full_count(compiled.model.names.index(name)) for name in expected
            }

            assert counted == expected

    @staticmethod
    def _direct(network: Network, at: list[int], truck: list[int]) -> dict[str, int]:
        """Each rule's violations, counted straight on the schema objects.

        Args:
            network: The network.
            at: Per client, its delivery time.
            truck: Per client, its truck.

        Returns:
            The count per rule name.
        """
        opens = {depot.code: depot.opens for depot in network.depots}
        members: dict[str, set[int]] = {}

        for index, client in enumerate(network.clients):
            for zone in client.zones:
                members.setdefault(zone, set()).add(index)

        clients = network.clients
        pairs = itertools.combinations(range(len(clients)), 2)
        return {
            "after_opening": sum(
                at[index] < opens[client.depot_code] for index, client in enumerate(clients)
            ),
            "zone_apart": sum(
                at[first] == at[second]
                for group in members.values()
                for first, second in itertools.combinations(sorted(group), 2)
            ),
            "one_zone_per_truck": sum(
                truck[first] == truck[second]
                and clients[first].zone != clients[second].zone
                and at[first] == at[second]
                for first, second in pairs
            ),
            "in_a_slot": sum(
                not any(slot.start <= at[index] < slot.end for slot in client.slots)
                for index, client in enumerate(clients)
            ),
        }
