from baseline_core.core.optimization.mining.candidate import CandidateFeature
from baseline_core.core.optimization.mining.findings import Findings


class TestFindings:
    def test__t17_rows_and_t1_cells__become_rules_and_questions(self) -> None:
        """T17 rows keep their counts; T1 cells become one closed-hours rule per day."""
        row = {
            "role": "op",
            "cle": "a",
            "feature": "unused_day",
            "bound": 4,
            "plans_qui_respectent": "11/12",
        }
        findings = Findings.model_validate({
            "T17_profils": {"regles_probables": [row], "regle_ou_habitude": [row]},
            "T1_recurrent_empty_cells": {
                "recurrent_empty_cells": [["J3", 11], ["J3", 10]],
                "plans_exiges": "12/12",
            },
        })

        first, closed = findings.rules(["J1", "J2", "J3", "J4", "J5"])
        [question] = findings.questions

        assert [first.feature, first.key, first.kept, first.plans] == [
            CandidateFeature.UNUSED_DAY,
            "a",
            11,
            12,
        ]
        assert [closed.feature, closed.value, closed.hours, closed.kept] == [
            CandidateFeature.CLOSED_HOURS,
            2,
            [10, 11],
            12,
        ]
        assert question == first
