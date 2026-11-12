import pytest

from baseline_core.core.optimization.solver.binding.model_compiler import ModelCompiler
from baseline_core.core.optimization.solver.model_solution import ModelSolution, VariableAssignment
from baseline_core.core.optimization.solver.search.kernel_solver import (
    KernelSolver,
    KernelSolverConfig,
)
from tests.core.optimization.bin_packing_model import (
    BinPackingModelDefinition,
    Fleet,
    Parcel,
    Truck,
)
from tests.core.optimization.crew_model import Workshop
from tests.core.optimization.job_shop_model import Shop
from tests.core.optimization.models import WorkshopFactory


@pytest.fixture
def search() -> KernelSolverConfig:
    """Return a short search.

    Returns:
        The settings.
    """
    return KernelSolverConfig(n_iters=60, patience=30)


@pytest.fixture(scope="module")
def bin_packing() -> ModelCompiler[Fleet]:
    """Return the bin packing model, compiled once.

    Returns:
        The compiled model.
    """
    return ModelCompiler(BinPackingModelDefinition(), cythonize=False)


class TestKernelSolver:
    def test__one_crew_s_tasks__end_apart_in_their_windows_with_nothing_left_out(
        self,
        compiled: ModelCompiler[Workshop],
        factory: WorkshopFactory,
        search: KernelSolverConfig,
    ) -> None:
        """Three two-hour tasks of one crew on Monday are placed, none overlapping."""
        tasks = [factory.task(name, crew_id="C") for name in ["a", "b", "c"]]
        workshop = factory.workshop(tasks, [["S1", "S2"]] * 3, [2, 2, 2])
        solution = KernelSolver(search).solve(compiled.bind(workshop))

        assert solution.summary.score.hard == 0
        assert solution.unassigned == []
        starts = sorted(a.value for a in solution.assignments if a.value is not None)
        assert all(starts[k] + 2 <= starts[k + 1] for k in range(len(starts) - 1))
        assert all(8 <= start <= 16 for start in starts)

    def test__two_tasks_one_window_holds_only_once__leave_one_out(
        self,
        compiled: ModelCompiler[Workshop],
        factory: WorkshopFactory,
        search: KernelSolverConfig,
    ) -> None:
        """Two hours fit one of two two-hour tasks of one crew: the other is left out."""
        tasks = [factory.task(name, crew_id="C", hours=[8, 10]) for name in ["a", "b"]]
        workshop = factory.workshop(tasks, [["S1"]] * 2, [2, 2])
        solution = KernelSolver(search).solve(compiled.bind(workshop))

        assert solution.summary.score.hard == 0
        assert len(solution.assignments) == 1
        assert len(solution.unassigned) == 1

    def test__the_same_seed__reaches_the_same_placement(
        self,
        compiled: ModelCompiler[Workshop],
        factory: WorkshopFactory,
        search: KernelSolverConfig,
    ) -> None:
        """Two runs with one seed place every task at the same start at the same station."""
        tasks = [factory.task(name, crew_id="C") for name in ["a", "b", "c", "d"]]
        workshop = factory.workshop(tasks, [["S1", "S2"]] * 4, [2, 2, 2, 2])
        runs = [KernelSolver(search).solve(compiled.bind(workshop)) for _ in range(2)]

        assert runs[0].assignments == runs[1].assignments


class TestFixed:
    @staticmethod
    def _by_element(solution: ModelSolution) -> dict[str, list[int | str | None]]:
        """Each assigned element's value and target, by its key.

        Args:
            solution: The solution.

        Returns:
            ``[value, target]`` per element key.
        """
        return {a.element: [a.value, a.target] for a in solution.assignments}

    def test__a_fixed_value_and_target__hold_while_the_others_move_around_them(
        self,
        compiled: ModelCompiler[Workshop],
        factory: WorkshopFactory,
        search: KernelSolverConfig,
    ) -> None:
        """Task b fixed at 12 at S2 ends there; its crew's other tasks fit around it."""
        tasks = [factory.task(name, crew_id="C") for name in ["a", "b", "c"]]
        workshop = factory.workshop(tasks, [["S1", "S2"]] * 3, [2, 2, 2])
        fixed = [VariableAssignment(element="b", value=12, target="S2")]
        solution = KernelSolver(search).solve(compiled.bind(workshop), fixed=fixed)

        assert self._by_element(solution)["b"] == [12, "S2"]
        assert solution.summary.score.hard == 0
        assert solution.unassigned == []

    def test__a_fixed_target_alone__keeps_the_station_and_lets_the_time_move(
        self,
        compiled: ModelCompiler[Workshop],
        factory: WorkshopFactory,
        search: KernelSolverConfig,
    ) -> None:
        """Task a fixed at S2 with no time ends at S2, at a time the search chose."""
        tasks = [factory.task(name, crew_id="C") for name in ["a", "b"]]
        workshop = factory.workshop(tasks, [["S1", "S2"]] * 2, [2, 2])
        fixed = [VariableAssignment(element="a", target="S2")]
        solution = KernelSolver(search).solve(compiled.bind(workshop), fixed=fixed)

        assert self._by_element(solution)["a"][1] == "S2"
        assert solution.summary.score.hard == 0

    def test__a_fixed_element__is_never_the_one_left_out(
        self,
        compiled: ModelCompiler[Workshop],
        factory: WorkshopFactory,
        search: KernelSolverConfig,
    ) -> None:
        """Two tasks share a window that holds one: the fixed b stays, the free a goes."""
        tasks = [factory.task(name, crew_id="C", hours=[8, 10]) for name in ["a", "b"]]
        workshop = factory.workshop(tasks, [["S1"]] * 2, [2, 2])
        fixed = [VariableAssignment(element="b", value=8)]
        solution = KernelSolver(search).solve(compiled.bind(workshop), fixed=fixed)

        assert self._by_element(solution) == {"b": [8, "S1"]}
        assert solution.unassigned == ["a"]

    def test__a_fixed_value__holds_its_paired_element_to_the_same_value(
        self,
        compiled: ModelCompiler[Workshop],
        factory: WorkshopFactory,
        search: KernelSolverConfig,
    ) -> None:
        """Tasks a and b start together; a fixed at 10 brings b to 10 as well."""
        tasks = [factory.task(name) for name in ["a", "b"]]
        workshop = factory.workshop(tasks, [["S1", "S2"]] * 2, [2, 2])
        paired = workshop.model_copy(update={"co_start_groups": [["a", "b"]]})
        fixed = [VariableAssignment(element="a", value=10)]
        solution = KernelSolver(search).solve(compiled.bind(paired), fixed=fixed)

        assert [values[0] for values in self._by_element(solution).values()] == [10, 10]

    def test__an_unknown_element__is_refused(
        self,
        compiled: ModelCompiler[Workshop],
        factory: WorkshopFactory,
        search: KernelSolverConfig,
    ) -> None:
        """Fixing a key no element has stops the solve before it starts."""
        workshop = factory.workshop([factory.task("a")], [["S1"]], [2])
        fixed = [VariableAssignment(element="ghost", value=9)]

        with pytest.raises(ValueError, match="'ghost' is no element of the model"):
            KernelSolver(search).solve(compiled.bind(workshop), fixed=fixed)

    def test__a_value_its_hard_rules_forbid__is_refused(
        self,
        compiled: ModelCompiler[Workshop],
        factory: WorkshopFactory,
        search: KernelSolverConfig,
    ) -> None:
        """A two-hour task fixed at 17 would end past its 18 h window: the fix is refused."""
        workshop = factory.workshop([factory.task("a")], [["S1"]], [2])
        fixed = [VariableAssignment(element="a", value=17)]

        with pytest.raises(ValueError, match="'a' at 17: its hard rules alone"):
            KernelSolver(search).solve(compiled.bind(workshop), fixed=fixed)


class TestWarmStart:
    def test__a_solution_passed_back__starts_a_search_no_worse_than_it(
        self,
        compiled: ModelCompiler[Workshop],
        factory: WorkshopFactory,
        search: KernelSolverConfig,
    ) -> None:
        """A second solve warm-started from the first one's assignments scores no worse."""
        tasks = [factory.task(name, crew_id="C") for name in ["a", "b", "c", "d"]]
        workshop = factory.workshop(tasks, [["S1", "S2"]] * 4, [2, 2, 2, 2])
        first = KernelSolver(search).solve(compiled.bind(workshop))
        second = KernelSolver(search).solve(compiled.bind(workshop), warm_start=first.assignments)

        assert second.summary.score <= first.summary.score

    def test__assignments_the_model_cannot_read__are_skipped(
        self,
        compiled: ModelCompiler[Workshop],
        factory: WorkshopFactory,
        search: KernelSolverConfig,
    ) -> None:
        """An unknown element and an unknown station in the warm start are passed over."""
        tasks = [factory.task(name, crew_id="C") for name in ["a", "b"]]
        workshop = factory.workshop(tasks, [["S1", "S2"]] * 2, [2, 2])
        warm_start = [
            VariableAssignment(element="ghost", value=9),
            VariableAssignment(element="a", value=9, target="nowhere"),
        ]
        solution = KernelSolver(search).solve(compiled.bind(workshop), warm_start=warm_start)

        assert solution.summary.score.hard == 0
        assert solution.unassigned == []


class TestOtherModels:
    def test__a_job_shop_operation__fixed_on_a_machine_at_a_time_stays_there(
        self, job_shop: ModelCompiler[Shop], search: KernelSolverConfig
    ) -> None:
        """In a job shop, operation o3 fixed on M2 at 5 ends there, nothing else broken."""
        shop = Shop.of([
            ["o1", "J1", 0, 3, ["M1", "M2"], 0, 40],
            ["o2", "J1", 1, 2, ["M1"], 0, 40],
            ["o3", "J2", 0, 4, ["M1", "M2"], 0, 40],
        ])
        fixed = [VariableAssignment(element="o3", value=5, target="M2")]
        solution = KernelSolver(search).solve(job_shop.bind(shop), fixed=fixed)

        assert TestFixed._by_element(solution)["o3"] == [5, "M2"]
        assert solution.summary.score.hard == 0
        assert solution.unassigned == []

    def test__parcels_with_no_time__are_fixed_and_warm_started_by_truck_alone(
        self, bin_packing: ModelCompiler[Fleet], search: KernelSolverConfig
    ) -> None:
        """Bin packing has no value: p1 fixed on T1 stays there, and every value is None."""
        fleet = Fleet(
            trucks=[Truck(code="T1", capacity=10), Truck(code="T2", capacity=10)],
            parcels=[
                Parcel(ref="p1", weight=6, trucks=["T1", "T2"]),
                Parcel(ref="p2", weight=6, trucks=["T1", "T2"]),
                Parcel(ref="p3", weight=3, trucks=["T1"]),
            ],
        )
        fixed = [VariableAssignment(element="p1", target="T1")]
        warm_start = [VariableAssignment(element="p2", target="T1")]
        solution = KernelSolver(search).solve(
            bin_packing.bind(fleet), warm_start=warm_start, fixed=fixed
        )

        assert TestFixed._by_element(solution) == {
            "p1": [None, "T1"],
            "p2": [None, "T2"],
            "p3": [None, "T1"],
        }
        assert solution.summary.score.hard == 0

    def test__a_value_on_a_model_with_no_time__is_refused(
        self, bin_packing: ModelCompiler[Fleet], search: KernelSolverConfig
    ) -> None:
        """Fixing a value where the model declares no integer variable is refused."""
        fleet = Fleet(
            trucks=[Truck(code="T1", capacity=10)],
            parcels=[Parcel(ref="p1", weight=6, trucks=["T1"])],
        )
        fixed = [VariableAssignment(element="p1", value=3)]

        with pytest.raises(ValueError, match="declares no integer variable"):
            KernelSolver(search).solve(bin_packing.bind(fleet), fixed=fixed)
