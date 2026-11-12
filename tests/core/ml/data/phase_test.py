"""Tests for ``Phase``."""

from baseline_core.core.ml.data.phase import Phase


class TestPhase:
    """Tests for ``Phase``."""

    def test__values__should_match_lightning_metric_prefixes(self) -> None:
        """Values are the prefixes used in logged metric names."""
        assert [phase.value for phase in Phase] == ["train", "val", "test"]

    def test__phase__should_behave_as_string(self) -> None:
        """``StrEnum`` members format as their value."""
        assert f"{Phase.TRAIN}_loss" == "train_loss"
