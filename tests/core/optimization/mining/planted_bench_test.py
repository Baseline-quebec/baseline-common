import pytest
from pydantic import ValidationError

from baseline_core.core.optimization.mining.candidate import CandidateFeature
from baseline_core.core.optimization.mining.plan import Plan
from tests.core.optimization.mining.planted_bench import (
    BenchInstanceConfig,
    BenchTables,
    BenchTruth,
    BlockedHours,
    GenericRules,
    PlantedBench,
    Problem,
    Spacing,
)


class TestBenchInstanceConfig:
    def test__a_bench_too_small_for_its_superpositions__is_refused(self) -> None:
        """Operator t12 carries a superposition: twelve operators cannot hold it."""
        with pytest.raises(ValidationError, match="operator t12"):
            BenchInstanceConfig(operators=12)

    def test__a_rule_on_a_machine_the_bench_lacks__is_refused(self) -> None:
        """The default rules name C5: a bench of four standard machines has none."""
        with pytest.raises(ValidationError, match="machine C5"):
            BenchInstanceConfig(standard=4)


class TestPlantedBench:
    @pytest.fixture
    def bench(self) -> PlantedBench:
        """A default bench, with a blocked hour and a spacing on a superposition's second task.

        Returns:
            The bench.
        """
        generic = GenericRules(
            blocked=[BlockedHours(field="operator", key="t1", day=0, hours=[7, 9])],
            spacing=[Spacing(product="k1", days=2)],
        )
        return PlantedBench(BenchInstanceConfig(rules=[], generic=generic))

    @pytest.fixture
    def instances(self, bench: PlantedBench) -> list[Problem]:
        """The bench's analyst and true instances.

        Args:
            bench: The bench.

        Returns:
            ``[analyst, truth]``.
        """
        return bench.build()

    def test__the_true_instance__merges_each_superposition_into_its_first_task(
        self, instances: list[Problem]
    ) -> None:
        """t0b2 carries t1's blocked hours and spaced lot; t1b2 is gone."""
        analyst, truth = instances
        merged = next(task for task in truth.tasks if task.name == "t0b2")

        assert len(truth.tasks) == len(analyst.tasks) - 2
        assert "t1b2" not in {task.name for task in truth.tasks}
        assert merged.operators == ["t0", "t1"]
        assert merged.blocked == [1, 2]
        assert merged.spaced_lots == ["k1|t0", "k1|t1"]

    def test__a_true_plan__reads_in_the_analyst_s_tasks_with_followers_on_their_leaders(
        self, bench: PlantedBench, instances: list[Problem]
    ) -> None:
        """t1b2 takes t0b2's slot and machine; an absent leader leaves its follower absent too."""
        analyst, truth = instances
        names = [task.name for task in truth.tasks]
        starts = list(range(len(truth.tasks)))
        leader = names.index("t0b2")

        plan = bench.as_analyst(
            analyst, truth, Plan(starts=starts, targets=[0] * len(starts), absent=[leader])
        )

        analyst_names = [task.name for task in analyst.tasks]
        follower = analyst_names.index("t1b2")
        assert plan.starts[follower] == leader
        assert analyst_names.index("t0b2") in plan.absent
        assert follower in plan.absent

    def test__drawn_rules__never_name_a_merged_operator_or_product(
        self, bench: PlantedBench
    ) -> None:
        """t1, t12 and k20 live only on merged tasks: no draw names them."""
        followers = set(bench.followers)

        rules = bench.draw_rules(40, seed=3)
        generic = bench.draw_generic({"operator_blocked": 8, "spacing": 6}, seed=3)

        assert followers == {"t1", "t12", "k20"}
        assert not followers & {rule.name for rule in rules}
        assert not followers & {rule.key for rule in generic.blocked}
        assert not followers & {rule.product for rule in generic.spacing}


class TestBenchTables:
    def test__the_rows__leave_absent_tasks_out_and_write_each_role(self) -> None:
        """One row per present task, its operator, product, lot, workshop, machine and hours."""
        analyst, _ = PlantedBench(BenchInstanceConfig()).build()
        n = len(analyst.tasks)

        rows = BenchTables.rows(analyst, [Plan(starts=[12] * n, targets=[0] * n, absent=[0])])

        assert len(rows) == n - 1
        assert rows[0] | {"Fin": "-"} == {
            "solution": 0,
            "Operateur": "op0",
            "Produit": "P0",
            "Lot": "lot-0",
            "Atelier": "A0",
            "Machine": "M-C0",
            "Jour": "J2",
            "Debut": "08:00",
            "Fin": "-",
        }


class TestBenchTruth:
    @pytest.fixture
    def truth(self) -> BenchTruth:
        """The default planted rules, in the tables' words.

        Returns:
            The truth.
        """
        return BenchTruth(BenchInstanceConfig())

    def test__the_planted_rules__end_with_the_wednesday_closure(self, truth: BenchTruth) -> None:
        """Ten planted rules, then Wednesday 10 h-12 h closed for everyone."""
        closure = truth.planted[-1]

        assert len(truth.planted) == 11
        assert [closure.feature, closure.value, closure.hours] == [
            CandidateFeature.CLOSED_HOURS,
            2,
            [10, 11],
        ]

    def test__a_candidate__is_labelled_true_effect_or_false(self, truth: BenchTruth) -> None:
        """A planted rule is true; another feature on its key is an effect; anything else false."""
        planted = truth.planted[0]
        effect = planted.model_copy(update={"feature": CandidateFeature.LATEST_END})
        unrelated = planted.model_copy(update={"key": "op13"})

        labels = [truth.label(candidate) for candidate in [planted, effect, unrelated]]

        assert labels == ["vraie", "effet", "fausse"]

    def test__the_grades__say_how_each_rule_was_found_and_whether_it_was_encoded(
        self, truth: BenchTruth
    ) -> None:
        """A rule found and encoded, a question left alone, the rest missed."""
        first, second = truth.planted[:2]

        grades = truth.grade([first], [second], [first])

        assert [[grade.found, grade.encoded] for grade in grades[:3]] == [
            ["rule", True],
            ["question", False],
            ["missed", False],
        ]
