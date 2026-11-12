import numpy as np
import pytest

from baseline_core.core.optimization.problem_formulation.model import Model
from baseline_core.core.optimization.solver.binding.model_compiler import ModelCompiler
from baseline_core.core.optimization.solver.infeasibility.infeasibility_check import (
    InfeasibilityCheck,
    PrecheckConfig,
    PrecheckLevel,
)
from baseline_core.core.optimization.solver.infeasibility.precheck_issue import (
    LoadGrade,
    PrecheckIssue,
)
from baseline_core.core.optimization.solver.kernel.kernel import SearchPlan
from baseline_core.core.optimization.solver.stop_condition import MaxIterations
from tests.core.optimization.bin_packing_model import (
    BinPackingModelDefinition,
    Fleet,
    Parcel,
    Truck,
)
from tests.core.optimization.crew_model import CrewModelConfig, Spacing, TaskKind, Workshop
from tests.core.optimization.job_shop_model import JobShopModelDefinition, Shop
from tests.core.optimization.models import DistinctStartDefinition, WorkshopFactory
from tests.core.optimization.resource_model import (
    Judge,
    Machine,
    Op,
    ResourceModelDefinition,
    TeamBreak,
)
from tests.core.optimization.resource_model import Shop as ResourceShop


class TestPrecheck:
    @staticmethod
    def _spaced_runs(
        factory: WorkshopFactory, spacing: Spacing, count: int, open_days: int
    ) -> Workshop:
        """One order of ``count`` runs spaced as asked, each its own crew and station.

        Args:
            factory: The test factory.
            spacing: How the runs are spaced.
            count: How many runs.
            open_days: How many days, from Monday, the runs may take.

        Returns:
            The workshop: two-hour tasks free from 8 h to 18 h on the open days.
        """
        tasks = [
            factory.task(
                f"t{index}", order_key="O", run=index + 1, spacing=spacing, days=range(open_days)
            )
            for index in range(count)
        ]
        return factory.workshop(tasks, [[f"S{index}"] for index in range(count)], [2] * count)

    def test__tasks_that_only_may_not_share_a_start__prove_nothing(
        self, config: CrewModelConfig, factory: WorkshopFactory
    ) -> None:
        """Three two-hour tasks that may not share a start fit in four hours: nothing proven."""
        tasks = [factory.task(f"t{index}", crew_id="C", hours=[8, 12]) for index in range(3)]
        workshop = factory.workshop(tasks, [["S1"]] * 3, [2] * 3)
        bound = ModelCompiler(DistinctStartDefinition(config), cythonize=False).bind(workshop)
        bound.kernel.load(start=np.array([8, 9, 10], dtype=np.int64))

        assert bound.kernel.full_hard == 0
        assert InfeasibilityCheck(PrecheckConfig()).run(bound) == []

    def test__a_run__gives_the_kernel_back_as_it_found_it(
        self, compiled: ModelCompiler[Workshop], factory: WorkshopFactory
    ) -> None:
        """The pre-check searches the bound kernel, then restores its placement and its counts."""
        workshop = factory.workshop(
            [factory.task("a", crew_id="C"), factory.task("b", crew_id="C")],
            [["S1"], ["S2"]],
            [2, 2],
        )

        bound = compiled.bind(workshop)
        bound.kernel.load(start=np.array([8, 9]), station=np.array([0, 1]))
        before = (bound.kernel.full_counts, [list(column) for column in bound.kernel.placement])
        InfeasibilityCheck(PrecheckConfig()).run(bound)
        after = (bound.kernel.full_counts, [list(column) for column in bound.kernel.placement])

        assert after == before

    def test__a_run__leaves_absent_elements_absent_where_they_were(
        self, compiled: ModelCompiler[Workshop], factory: WorkshopFactory
    ) -> None:
        """Elements absent before a run are absent after it, with the same placement and counts."""
        workshop = factory.workshop(
            [factory.task(name, crew_id="C") for name in "abc"],
            [["S1"], ["S2"], ["S3"]],
            [2, 2, 2],
        )

        bound = compiled.bind(workshop)
        kernel = bound.kernel
        kernel.load(start=np.array([8, 9, 10]), station=np.array([0, 1, 2]))
        kernel.set_absent(np.array([1, 0, 1]))

        before = (
            list(kernel.absent_mask),
            [list(column) for column in kernel.placement],
            kernel.full_counts,
        )

        InfeasibilityCheck(PrecheckConfig()).run(bound)
        after = (
            list(kernel.absent_mask),
            [list(column) for column in kernel.placement],
            kernel.full_counts,
        )

        assert after == before
        assert before[0] == [1, 0, 1]

    def test__three_runs_on_consecutive_days_open_on_two__are_a_group_issue(
        self, compiled: ModelCompiler[Workshop], factory: WorkshopFactory
    ) -> None:
        """Three runs on back-to-back days need three open days, not two."""
        workshop = self._spaced_runs(factory, Spacing.CONSECUTIVE_DAYS, 3, 2)
        issues = InfeasibilityCheck(PrecheckConfig()).run(compiled.bind(workshop))

        assert [
            (issue.kind, issue.elements, issue.entity) for issue in issues if issue.proven
        ] == [("group", [0, 1, 2], "t0, t1, t2")]

    @pytest.mark.parametrize(
        ("up_to", "kinds"), [(PrecheckLevel.ELEMENT, []), (PrecheckLevel.GROUP, ["group"])]
    )
    def test__a_run__stops_after_the_level_it_is_configured_up_to(
        self,
        compiled: ModelCompiler[Workshop],
        factory: WorkshopFactory,
        up_to: PrecheckLevel,
        kinds: list[str],
    ) -> None:
        """A group no run alone shows is found from level 2 on, and not by level 1 alone."""
        workshop = self._spaced_runs(factory, Spacing.CONSECUTIVE_DAYS, 3, 2)
        issues = InfeasibilityCheck(PrecheckConfig(up_to=up_to)).run(compiled.bind(workshop))

        assert [issue.kind for issue in issues if issue.proven] == kinds

    def test__two_runs_a_day_apart_open_on_two_days__are_a_group_issue(
        self, compiled: ModelCompiler[Workshop], factory: WorkshopFactory
    ) -> None:
        """Two runs a day apart find no free day between them on two open days."""
        workshop = self._spaced_runs(factory, Spacing.DAY_APART, 2, 2)
        issues = InfeasibilityCheck(PrecheckConfig()).run(compiled.bind(workshop))

        assert [
            (issue.kind, issue.elements, issue.entity) for issue in issues if issue.proven
        ] == [("group", [0, 1], "t0, t1")]

    def test__a_group_its_search_cannot_decide__is_reported_undecided(
        self, compiled: ModelCompiler[Workshop], factory: WorkshopFactory
    ) -> None:
        """A group its search would prove, given no budget, is undecided and proves nothing."""
        workshop = self._spaced_runs(factory, Spacing.CONSECUTIVE_DAYS, 3, 2)
        precheck = InfeasibilityCheck(PrecheckConfig(witness_budget=1, group_budget=1))
        issues = precheck.run(compiled.bind(workshop))

        assert [
            (issue.kind, issue.elements, issue.proven) for issue in issues if issue.grade is None
        ] == [("undecided", [0, 1, 2], False)]

    def test__a_load_its_pinned_elements_fill_exactly__is_not_nearly_full(
        self, compiled: ModelCompiler[Workshop], factory: WorkshopFactory
    ) -> None:
        """Two tasks pinned to their own hours fill them exactly: no choice left to warn about."""
        tasks = [
            factory.task("a", crew_id="C", hours=[8, 10]),
            factory.task("b", crew_id="C", hours=[10, 12]),
        ]

        workshop = factory.workshop(tasks, [["S1"], ["S2"]], [2, 2])
        issues = InfeasibilityCheck(PrecheckConfig()).run(compiled.bind(workshop))

        assert [(issue.kind, issue.entity, issue.grade) for issue in issues] == []

    def test__a_window_shorter_than_the_task__names_the_window_alone(
        self, compiled: ModelCompiler[Workshop], factory: WorkshopFactory
    ) -> None:
        """A three-hour task in a two-hour window has no placement; only the window is named."""
        workshop = factory.workshop([factory.task("a", hours=[8, 10])], [["S1"]], [3])
        issues = InfeasibilityCheck(PrecheckConfig()).run(compiled.bind(workshop))

        assert [(issue.kind, issue.constraints) for issue in issues] == [
            ("no_placement", ["window_coverage"])
        ]

    def test__a_crew_and_a_co_start__together__leave_three_tasks_no_placement(
        self, compiled: ModelCompiler[Workshop], factory: WorkshopFactory
    ) -> None:
        """Each rule alone is satisfiable; the component they link is proven, by these rules."""
        tasks = [
            factory.task("a", crew_id="C", hours=[8, 10]),
            factory.task("b", crew_id="C", hours=[8, 12]),
            factory.task("c", hours=[8, 11]),
        ]

        workshop = factory.workshop(tasks, [["S1"], ["S2"], ["S3"]], [2] * 3)
        workshop.co_start_groups = [["b", "c"]]
        issues = InfeasibilityCheck(PrecheckConfig()).run(compiled.bind(workshop))

        assert [
            (issue.kind, issue.elements, issue.constraints) for issue in issues if issue.proven
        ] == [("component", [0, 1, 2], ["window_coverage", "no_crew_overlap", "co_start"])]

    @pytest.mark.parametrize("cythonize", [True, False])
    def test__a_job_s_chain_too_long_for_its_windows__is_proven_by_the_same_search(
        self, *, cythonize: bool
    ) -> None:
        """Three operations in order cannot end by their due times, compiled or in pure Python."""
        compiled = ModelCompiler(JobShopModelDefinition(), cythonize=cythonize)
        problem = Shop.of([
            ["A1", "A", 1, 4, ["M1"], 0, 8],
            ["A2", "A", 2, 4, ["M2"], 0, 8],
            ["A3", "A", 3, 3, ["M1"], 0, 10],
        ])

        issues = InfeasibilityCheck(PrecheckConfig()).run(compiled.bind(problem))

        assert [(issue.kind, issue.elements, issue.constraints) for issue in issues] == [
            ("group", [0, 1, 2], ["time_window", "job_order"])
        ]

    def test__a_feasible_job_shop__proves_nothing(self) -> None:
        """Two jobs that fit on their machines leave nothing to prove."""
        compiled = ModelCompiler(JobShopModelDefinition(), cythonize=False)
        problem = Shop.of([
            ["A1", "A", 1, 4, ["M1"], 0, 20],
            ["A2", "A", 2, 3, ["M2"], 0, 20],
            ["B1", "B", 1, 5, ["M2"], 0, 20],
            ["B2", "B", 2, 2, ["M1"], 0, 20],
        ])

        issues = InfeasibilityCheck(PrecheckConfig()).run(compiled.bind(problem))

        assert [issue for issue in issues if issue.proven] == []

    @pytest.mark.parametrize("cythonize", [True, False])
    def test__local_search__finds_a_placement_of_a_feasible_shop(self, *, cythonize: bool) -> None:
        """From every start and machine, the witness local search reaches no hard violation."""
        compiled = ModelCompiler(JobShopModelDefinition(), cythonize=cythonize)
        problem = Shop.of([
            ["A1", "A", 1, 4, ["M1"], 0, 20],
            ["A2", "A", 2, 3, ["M2"], 0, 20],
            ["B1", "B", 1, 5, ["M2"], 0, 20],
            ["B2", "B", 2, 2, ["M1"], 0, 20],
        ])

        kernel = compiled.bind(problem).kernel
        kernel.set_absent(np.ones(4, dtype=np.int64))

        domains = [
            part for duration in (4, 3, 5, 2) for part in (list(range(41 - duration)), [0, 1])
        ]
        plan = SearchPlan(
            order=np.arange(4, dtype=np.int64),
            domains=np.asarray([value for part in domains for value in part], dtype=np.int64),
            domain_offsets=np.cumsum([0, *map(len, domains)]).astype(np.int64),
            checks=np.zeros(0, dtype=np.int64),
            check_offsets=np.zeros(9, dtype=np.int64),
            whole=np.zeros(0, dtype=np.int64),
            closed=np.arange(len(compiled.model.names), dtype=np.int64),
        )

        assert kernel.min_conflicts(plan, MaxIterations(100_000), seed=12345) == 1

    def test__a_crew_with_more_hours_than_its_tasks_can_take__is_named_with_numbers(
        self, compiled: ModelCompiler[Workshop], factory: WorkshopFactory
    ) -> None:
        """Four 3 h tasks in a 10 h day hold 9 h at most: the crew's group is overloaded."""
        tasks = [factory.task(f"t{index}", crew_id="C") for index in range(4)]
        workshop = factory.workshop(tasks, [[f"S{index}"] for index in range(4)], [3] * 4)
        issues = InfeasibilityCheck(PrecheckConfig()).run(compiled.bind(workshop))

        assert [
            (issue.kind, issue.entity, issue.demand, issue.offer, issue.constraints)
            for issue in issues
            if issue.proven
        ] == [("group_load", "C", 12, 9, ["window_coverage", "no_crew_overlap", "crew_break"])]

    def test__a_pool_of_stations_its_tasks_overload__is_named_with_numbers(
        self, compiled: ModelCompiler[Workshop], factory: WorkshopFactory
    ) -> None:
        """Seven 3 h tasks that may only take S1 or S2, in a 10 h day: 21 h for 18 h."""
        tasks = [factory.task(f"t{index}") for index in range(7)]
        workshop = factory.workshop(tasks, [["S1", "S2"]] * 7, [3] * 7)
        issues = InfeasibilityCheck(PrecheckConfig()).run(compiled.bind(workshop))

        assert [
            (issue.kind, issue.entity, issue.demand, issue.offer, issue.constraints)
            for issue in issues
            if issue.proven
        ] == [("pool_load", "S1, S2", 21, 18, ["window_coverage", "no_station_overlap"])]

    def test__a_machine_its_operations_overload__is_named_with_numbers(self) -> None:
        """Three 5 h operations of three jobs, all on M1 by 12: whole operations fill 10 h."""
        compiled = ModelCompiler(JobShopModelDefinition(), cythonize=False)
        problem = Shop.of([[f"{job}1", job, 1, 5, ["M1"], 0, 12] for job in "ABC"])
        issues = InfeasibilityCheck(PrecheckConfig()).run(compiled.bind(problem))

        assert [
            (issue.kind, issue.entity, issue.demand, issue.offer, issue.constraints)
            for issue in issues
            if issue.proven
        ] == [("pool_load", "M1", 15, 10, ["time_window", "machine_option", "no_machine_overlap"])]

    def test__without_time__a_parcel_and_a_pool_of_trucks_are_proven_by_weight(self) -> None:
        """A parcel heavier than its trucks has no place; parcels outweighing their trucks too."""
        compiled = ModelCompiler(BinPackingModelDefinition(), cythonize=False)
        fleet = Fleet(
            trucks=[Truck(code="A", capacity=10), Truck(code="B", capacity=10)],
            parcels=[
                Parcel(ref="p1", weight=12, trucks=["A", "B"]),
                Parcel(ref="p2", weight=8, trucks=["A", "B"]),
                Parcel(ref="p3", weight=8, trucks=["A", "B"]),
                Parcel(ref="p4", weight=8, trucks=["A"]),
            ],
        )

        issues = InfeasibilityCheck(PrecheckConfig()).run(compiled.bind(fleet))

        assert [
            (issue.kind, issue.entity, issue.demand, issue.offer)
            for issue in issues
            if issue.proven
        ] == [("no_placement", "p1", 12, 10), ("pool_load", "A, B", 24, 20)]

    def test__without_time__parcels_that_fit_prove_nothing(self) -> None:
        """Parcels within their trucks' capacities leave nothing proven."""
        compiled = ModelCompiler(BinPackingModelDefinition(), cythonize=False)
        fleet = Fleet(
            trucks=[Truck(code="A", capacity=10), Truck(code="B", capacity=10)],
            parcels=[Parcel(ref=f"p{index}", weight=6, trucks=["A", "B"]) for index in range(3)],
        )

        issues = InfeasibilityCheck(PrecheckConfig()).run(compiled.bind(fleet))

        assert [issue for issue in issues if issue.proven] == []

    def test__clean_tasks_waiting_an_hour_between_them__overload_their_station(
        self, compiled: ModelCompiler[Workshop], factory: WorkshopFactory
    ) -> None:
        """Five 2 h clean tasks fill a 10 h station only with the four hours of cleaning: 14/10."""
        tasks = [
            factory.task(
                f"t{index}",
                kind=TaskKind.CLEAN,
                clean_gap_same_product=1,
                clean_gap_other_product=1,
            )
            for index in range(5)
        ]

        workshop = factory.workshop(tasks, [["S1"]] * 5, [2] * 5)
        issues = InfeasibilityCheck(PrecheckConfig()).run(compiled.bind(workshop))

        assert [
            (issue.kind, issue.entity, issue.demand, issue.offer, issue.constraints)
            for issue in issues
            if issue.proven
        ] == [("pool_load", "S1", 14, 10, ["window_coverage", "clean_task_spacing"])]

    def test__a_crew_s_daily_break__is_taken_off_what_its_tasks_can_fill(
        self, compiled: ModelCompiler[Workshop], factory: WorkshopFactory
    ) -> None:
        """Five 2 h tasks fill a 10 h day, but not around a break hour: 10 h for 8 h."""
        tasks = [factory.task(f"t{index}", crew_id="C") for index in range(5)]
        workshop = factory.workshop(tasks, [[f"S{index}"] for index in range(5)], [2] * 5)
        issues = InfeasibilityCheck(PrecheckConfig()).run(compiled.bind(workshop))

        assert [
            (issue.kind, issue.entity, issue.demand, issue.offer, issue.constraints)
            for issue in issues
            if issue.proven
        ] == [("group_load", "C", 10, 8, ["window_coverage", "no_crew_overlap", "crew_break"])]

    def test__a_target_no_element_may_take__is_reported_underused(self) -> None:
        """A truck no parcel may go in is underused, with no demand and nothing to measure."""
        compiled = ModelCompiler(BinPackingModelDefinition(), cythonize=False)
        fleet = Fleet(
            trucks=[Truck(code="A", capacity=10), Truck(code="C", capacity=10)],
            parcels=[Parcel(ref="p1", weight=9, trucks=["A"])],
        )

        issues = InfeasibilityCheck(PrecheckConfig()).run(compiled.bind(fleet))

        assert [
            (issue.kind, issue.entity, issue.demand, issue.offer, issue.grade) for issue in issues
        ] == [("pool_load", "A", 9, 10, "nearly_full"), ("pool_load", "C", 0, None, "underused")]

    def test__an_explanation__keeps_only_the_rules_the_proof_needs(self) -> None:
        """Rules dropped one at a time leave those without which the proof fails."""
        assert InfeasibilityCheck._explain({2, 4}.issubset, [0, 1, 2, 3, 4], []) == [2, 4]


class TestLoadBounds:
    @pytest.fixture(params=[True, False], ids=["compiled", "python"])
    def cythonize(self, request: pytest.FixtureRequest) -> bool:
        """Every test runs on the compiled kernel and on the same kernel in pure Python.

        Args:
            request: The parameter.

        Returns:
            Whether the kernel is compiled.
        """
        return bool(request.param)

    @staticmethod
    def run(
        rules: set[str],
        high: int,
        shop: ResourceShop,
        *,
        cythonize: bool,
        breaks: TeamBreak | None = None,
    ) -> list[list[PrecheckIssue]]:
        """The pre-check's issues on a shop, and those a brute-force search proves wrong.

        Args:
            rules: The rules of the resource model.
            high: Where the horizon ends; 0: no time.
            shop: The shop.
            cythonize: Whether the kernel is compiled.
            breaks: The break of the teams, if any.

        Returns:
            The issues, and the proven ones whose named constraints some placement satisfies.
        """
        compiled = ModelCompiler(
            ResourceModelDefinition(frozenset(rules), high, breaks), cythonize=cythonize
        )
        issues = InfeasibilityCheck(PrecheckConfig()).run(compiled.bind(shop))
        return [issues, Judge(compiled, high).wrong(shop, issues)]

    def test__breaks_on_a_horizon_not_cut_in_whole_periods__are_refused(
        self, *, cythonize: bool
    ) -> None:
        """Six slots are no whole number of four-slot periods: the model is refused."""
        with pytest.raises(ValueError, match="period"):
            ModelCompiler(
                ResourceModelDefinition(
                    frozenset({"team_overlap"}), 6, TeamBreak(period=4, window=[1, 3], length=1)
                ),
                cythonize=cythonize,
            )

    def test__a_break_window_past_its_period__is_refused(self) -> None:
        """A window ending after its period is refused when the break is posted."""
        with pytest.raises(ValueError, match="window"):
            ResourceModelDefinition(
                frozenset({"team_overlap"}), 8, TeamBreak(period=4, window=[2, 5], length=1)
            ).define(Model(ResourceShop))

    def test__blocks_that_may_cross_into_the_next_period__prove_nothing_false(
        self, *, cythonize: bool
    ) -> None:
        """Blocks of 3, 2 and 2 slots fit eight slots with a break each period: nothing proven."""
        shop = ResourceShop(
            machines=[Machine(code="M0", cap=3)],
            operations=[
                Op(ref=f"o{index}", duration=duration, machines=["M0"])
                for index, duration in enumerate([3, 2, 2])
            ],
        )

        issues, wrong = self.run(
            {"team_overlap"},
            8,
            shop,
            cythonize=cythonize,
            breaks=TeamBreak(period=4, window=[1, 3], length=1),
        )

        assert [issue for issue in issues if issue.proven] == []
        assert wrong == []

    def test__a_rule_on_every_machine__is_not_read_on_machines_a_group_cannot_take(
        self, *, cythonize: bool
    ) -> None:
        """Team X takes M1 only; M2 waits for team Y's operation: nothing is proven."""
        shop = ResourceShop(
            machines=[Machine(code="M1", need=1, cap=2), Machine(code="M2", need=1, cap=2)],
            operations=[
                Op(ref="a1", machines=["M1"]),
                Op(ref="a2", machines=["M1"]),
                Op(ref="b1", team="Y", machines=["M2"]),
            ],
        )

        compiled = ModelCompiler(
            ResourceModelDefinition(frozenset({"machine_min", "team_overlap"}), 4),
            cythonize=cythonize,
        )

        precheck = InfeasibilityCheck(PrecheckConfig(witness_budget=1))
        issues = precheck.run(compiled.bind(shop))

        assert [issue for issue in issues if issue.proven] == []
        assert Judge(compiled, 4).wrong(shop, issues) == []

    def test__a_zero_length_operation__uses_no_capacity(self, *, cythonize: bool) -> None:
        """An operation of length 0 demanding 2 on a machine of capacity 1 is not proven."""
        shop = ResourceShop(
            machines=[Machine(code="M0", cap=1)],
            operations=[Op(ref="z", duration=0, demand=2, machines=["M0"])],
        )

        issues, wrong = self.run({"cumulative"}, 3, shop, cythonize=cythonize)

        assert [issue for issue in issues if issue.proven] == []
        assert wrong == []

    def test__bin_packing_in_a_timed_model__is_bounded_by_capacity(
        self, *, cythonize: bool
    ) -> None:
        """Three units on a machine holding two, hours aside: the pool is overloaded, 3 for 2."""
        shop = ResourceShop(
            machines=[Machine(code="M0", cap=2)],
            operations=[Op(ref=f"p{index}", duration=2, machines=["M0"]) for index in range(3)],
        )

        issues, wrong = self.run({"bin"}, 4, shop, cythonize=cythonize)

        assert [
            (issue.kind, issue.entity, issue.demand, issue.offer)
            for issue in issues
            if issue.proven
        ] == [("pool_load", "M0", 3, 2)]

        assert wrong == []

    def test__an_element_too_large_for_its_pool__names_the_filter_that_shrank_it(
        self, *, cythonize: bool
    ) -> None:
        """The big machine refuses p's kind and the small one holds 2: p's 3 needs both rules."""
        shop = ResourceShop(
            machines=[Machine(code="big", cap=5, banned_kind=1), Machine(code="small", cap=2)],
            operations=[
                Op(ref="p", demand=3, kind=1, machines=["big", "small"]),
                Op(ref="q", machines=["small"]),
            ],
        )

        issues, wrong = self.run({"bin", "ban"}, 0, shop, cythonize=cythonize)

        assert [
            (issue.kind, issue.entity, issue.constraints) for issue in issues if issue.proven
        ] == [("no_placement", "p", ["ban", "bin"])]

        assert wrong == []

    def test__breaks_count_only_the_members_that_take_part(self, *, cythonize: bool) -> None:
        """A member left out by ``where=`` covering the window does not take the team's break."""
        shop = ResourceShop(
            machines=[Machine(code="M0", cap=2)],
            operations=[
                Op(ref="out", duration=4, takes=0, machines=["M0"]),
                Op(ref="in", duration=1, machines=["M0"]),
            ],
        )

        compiled = ModelCompiler(
            ResourceModelDefinition(
                frozenset({"team_overlap", "where"}),
                4,
                TeamBreak(period=4, window=[1, 3], length=2),
            ),
            cythonize=cythonize,
        )

        kernel = compiled.bind(shop).kernel
        kernel.load(
            start=np.array([0, 0], dtype=np.int64), machine=np.array([0, 0], dtype=np.int64)
        )

        assert kernel.full_count(compiled.model.names.index("brk")) == 0

    def test__a_pool_two_resources_overload__is_reported_once_the_tighter(
        self, *, cythonize: bool
    ) -> None:
        """No overlap with waits and a capacity on M0 both bound it: one issue, 8 for 6."""
        shop = ResourceShop(
            machines=[Machine(code="M0", cap=1)],
            operations=[
                Op(ref=f"o{index}", duration=2, same=1, other=1, machines=["M0"])
                for index in range(3)
            ],
        )

        issues, wrong = self.run(
            {"machine_overlap", "transition", "cumulative"}, 6, shop, cythonize=cythonize
        )

        assert [
            (issue.entity, issue.demand, issue.offer, issue.grade)
            for issue in issues
            if issue.entity == "M0"
        ] == [("M0", 8, 6, LoadGrade.OVERLOADED)]

        assert wrong == []
