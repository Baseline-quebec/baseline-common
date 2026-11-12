from baseline_core.core.optimization.solver.binding.model_compiler import ModelCompiler
from baseline_core.core.optimization.solver.model_solution import VariableAssignment
from baseline_core.core.optimization.solver.output_mapper import OutputMapper
from tests.core.optimization.crew_model import Reservation, TaskKind, Workshop
from tests.core.optimization.crew_schedule import CrewSchedule
from tests.core.optimization.models import WorkshopFactory


class TestCrewSchedule:
    def test__the_output_schema__reads_its_tasks_off_the_solution(
        self, compiled: ModelCompiler[Workshop], factory: WorkshopFactory
    ) -> None:
        """Two tasks come out with their id, order, kind, station and slots, as declared."""
        reservation = Reservation(day_index=0, start_slot=9, station_id="S2")
        tasks = [
            factory.task("t1", crew_id="C", reservation=reservation),
            factory.task("t2", crew_id="C"),
        ]

        workshop = factory.workshop(tasks, [["S1", "S2"], ["S1", "S2"]], [2, 3])
        bound = compiled.bind(workshop)
        bound.place([
            VariableAssignment(element="t1", value=9, target="S2"),
            VariableAssignment(element="t2", value=9, target="S2"),
        ])

        solution = bound.solution
        schedule = OutputMapper(CrewSchedule).map(bound, solution)
        placed = [
            [t.task_id, t.order_key, t.kind, t.station_id, t.start_slot, t.end_slot]
            for t in schedule.tasks
        ]

        assert placed == [
            ["t1", "O_t1", TaskKind.STANDARD, "S2", 9, 11],
            ["t2", "O_t2", TaskKind.STANDARD, "S2", 9, 12],
        ]

        assert solution.summary.constraints["no_station_overlap"] == 1
        assert solution.summary.terms["station_utilization"].cost == 20
        assert schedule.reservations == {"total": 1, "kept": 1}
        assert solution.unassigned == []
