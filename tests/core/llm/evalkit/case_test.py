"""Tests of the dataset: narrowing it to named cases."""

import pytest

from baseline_core.core.llm.evalkit.case import EvalCase, EvalDataset


class TestEvalDatasetSelect:
    """A dataset narrowed to the cases named."""

    @pytest.fixture
    def dataset(self) -> EvalDataset[int, int]:
        """Three cases, in order."""
        return EvalDataset(
            name="sample",
            cases=[EvalCase(name=name, input=0, expected=0) for name in ("a", "b", "c")],
        )

    def test_keeps_the_named_cases_in_order(self, dataset: EvalDataset[int, int]) -> None:
        """The subset follows the dataset's order, whatever the order named."""
        subset = dataset.select(["c", "a"])
        assert [case.name for case in subset.cases] == ["a", "c"]
        assert subset.name == "sample (2 of 3)"

    def test_naming_every_case_returns_the_dataset(self, dataset: EvalDataset[int, int]) -> None:
        """A full selection is the dataset itself, its name kept."""
        assert dataset.select(["a", "b", "c"]) is dataset

    def test_an_unknown_name_is_refused(self, dataset: EvalDataset[int, int]) -> None:
        """A typo must not silently run fewer cases."""
        with pytest.raises(ValueError, match="no case named d"):
            dataset.select(["a", "d"])
