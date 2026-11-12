import pytest

from tests.core.optimization.counter import Counter
from tests.core.optimization.crew_model import CrewModelConfig, Spacing
from tests.core.optimization.models import WorkshopFactory


class TestCrewBreak:
    def test__tasks_over_every_break_hour__break_the_break(
        self, counter: Counter, factory: WorkshopFactory
    ) -> None:
        """Tasks from 10 h to 12 h and 12 h to 14 h leave no free hour among 11, 12 and 13."""
        workshop = factory.workshop(
            [factory.task("a", crew_id="C"), factory.task("b", crew_id="C")],
            [["S1"], ["S1"]],
            [2, 2],
        )

        assert counter.of(workshop, starts=[10, 12], targets=[0, 0])["crew_break"] == 1

    def test__a_free_hour_at_noon__keeps_the_break(
        self, counter: Counter, factory: WorkshopFactory
    ) -> None:
        """Tasks from 10 h to 12 h and 13 h to 15 h leave noon free: the break holds."""
        workshop = factory.workshop(
            [factory.task("a", crew_id="C"), factory.task("b", crew_id="C")],
            [["S1"], ["S1"]],
            [2, 2],
        )

        assert counter.of(workshop, starts=[10, 13], targets=[0, 0])["crew_break"] == 0

    def test__a_break__is_counted_per_day(
        self, config: CrewModelConfig, counter: Counter, factory: WorkshopFactory
    ) -> None:
        """A crew that loses its break on the second day only counts one broken day."""
        tasks = [factory.task("a", crew_id="C"), factory.task("b", crew_id="C", days=[1])]
        workshop = factory.workshop(tasks, [["S1"], ["S1"]], [4, 4])
        tuesday = config.slots_per_day

        assert counter.of(workshop, starts=[8, tuesday + 10], targets=[0, 0])["crew_break"] == 1


class TestRunDaySpacing:
    @pytest.mark.parametrize(
        ("spacing", "days", "bad_steps"),
        [
            (Spacing.CONSECUTIVE_DAYS, [0, 1], 0),
            (Spacing.CONSECUTIVE_DAYS, [0, 0], 1),
            (Spacing.CONSECUTIVE_DAYS, [0, 2], 1),
            (Spacing.DAY_APART, [0, 1], 1),
            (Spacing.DAY_APART, [0, 2], 0),
            (Spacing.NONE, [0, 0], 0),
        ],
    )
    def test__two_runs_of_an_order__keep_the_spacing_it_asks(
        self,
        *,
        config: CrewModelConfig,
        counter: Counter,
        factory: WorkshopFactory,
        spacing: Spacing,
        days: list[int],
        bad_steps: int,
    ) -> None:
        """The step between the two runs is bad exactly when the order's spacing forbids it."""
        tasks = [
            factory.task("a", order_key="O", run=1, spacing=spacing, days=[days[0]]),
            factory.task("b", order_key="O", run=2, spacing=spacing, days=[days[1]]),
        ]

        workshop = factory.workshop(tasks, [["S1"], ["S2"]], [2, 2])
        starts = [day * config.slots_per_day + 8 for day in days]

        if days[0] == days[1]:
            starts[1] += 4  # same day: far enough apart for the same-day gap

        assert counter.of(workshop, starts=starts, targets=[0, 1])["run_day_spacing"] == bad_steps

    def test__tasks_of_one_run__count_as_one_run(
        self, config: CrewModelConfig, counter: Counter, factory: WorkshopFactory
    ) -> None:
        """A run of two tasks on Monday, the next run on Tuesday: one step, a day long."""
        spacing = Spacing.CONSECUTIVE_DAYS
        tasks = [
            factory.task("a", order_key="O", run=1, spacing=spacing),
            factory.task("b", order_key="O", run=1, spacing=spacing),
            factory.task("c", order_key="O", run=2, spacing=spacing, days=[1]),
        ]

        workshop = factory.workshop(tasks, [["S1"], ["S2"], ["S3"]], [2, 2, 2])
        starts = [8, 12, config.slots_per_day + 8]

        assert counter.of(workshop, starts=starts, targets=[0, 1, 2])["run_day_spacing"] == 0
