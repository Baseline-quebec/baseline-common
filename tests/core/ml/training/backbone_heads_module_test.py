"""Tests for ``BackboneHeadsModule``."""

from unittest.mock import MagicMock

import pytest
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanSquaredError, MetricCollection

from baseline_core.core.ml.data.batch import Batch, TensorTargets
from baseline_core.core.ml.data.phase import Phase
from baseline_core.core.ml.model.head_container import HeadContainer
from baseline_core.core.ml.training.backbone_heads_module import BackboneHeadsModule
from baseline_core.core.ml.training.optimizer import OptimizerFactory
from tests.core.ml.dummy_classes import DummyBackbone


class TestBackboneHeadsModule:
    """Tests for ``BackboneHeadsModule``."""

    @pytest.fixture
    def ghost_metrics(self) -> dict[str, MetricCollection]:
        """Metrics of a target no head predicts."""
        return {"ghost": MetricCollection([MeanSquaredError()])}

    def test__init__should_clone_metrics_per_phase_and_target(
        self, module: BackboneHeadsModule
    ) -> None:
        """One metric collection per phase and target, prefixed accordingly."""
        for phase in Phase:
            assert set(module.metrics_for(phase)) == {"y_user", "y_flight"}

        assert module.metrics_for(Phase.VAL)["y_user"].prefix == "val_y_user_"

    def test__init__when_metric_target_has_no_head__should_raise_value_error(
        self,
        backbone: DummyBackbone,
        heads: HeadContainer,
        optimizer_factory: OptimizerFactory,
        ghost_metrics: dict[str, MetricCollection],
    ) -> None:
        """Metrics must belong to a head."""
        with pytest.raises(ValueError, match=r"targets without a head: \['ghost'\]"):
            BackboneHeadsModule(backbone, heads, optimizer_factory, ghost_metrics)

    def test__init__should_save_the_head_configs_as_hyperparameters(
        self, module: BackboneHeadsModule
    ) -> None:
        """The head configs are recorded for checkpoints and loggers."""
        assert set(module.hparams["heads"]) == {"y_user", "y_flight"}
        assert module.hparams["heads"]["y_user"]["weight"] == pytest.approx(0.5)

    def test__forward__should_return_predictions_for_every_head(
        self, module: BackboneHeadsModule, batch: Batch[TensorTargets]
    ) -> None:
        """Backbone then heads."""
        predictions = module(batch)

        assert set(predictions) == {"y_user", "y_flight"}

    def test__training_step__should_return_the_loss_only(
        self, module: BackboneHeadsModule, batch: Batch[TensorTargets]
    ) -> None:
        """Lightning back-propagates the scalar; the predictions are not kept."""
        loss = module.training_step(batch, 0)

        assert loss.requires_grad
        assert loss.ndim == 0

    def test__validation_step__should_hand_the_predictions_to_callbacks(
        self, module: BackboneHeadsModule, batch: Batch[TensorTargets]
    ) -> None:
        """The step output carries the loss, its components and the predictions."""
        output = module.validation_step(batch, 0)

        assert set(output["losses"]) == {"loss", "y_user_loss", "y_flight_loss"}
        assert set(output["outputs"]) == {"y_user", "y_flight"}

    def test__training_step__should_log_losses_per_step_and_per_epoch(
        self, module: BackboneHeadsModule, batch: Batch[TensorTargets], log: MagicMock
    ) -> None:
        """Every loss is logged as ``<phase>_<name>``, the total on the progress bar."""
        module.training_step(batch, 0)

        logged = {call.args[0]: call.kwargs for call in log.call_args_list}
        assert set(logged) == {"train_loss", "train_y_user_loss", "train_y_flight_loss"}
        assert logged["train_loss"]["prog_bar"] is True
        assert logged["train_y_user_loss"]["prog_bar"] is False
        assert all(kwargs["on_epoch"] for kwargs in logged.values())

    @pytest.mark.parametrize(
        ("method", "phase"),
        [
            ("training_step", Phase.TRAIN),
            ("validation_step", Phase.VAL),
            ("test_step", Phase.TEST),
        ],
    )
    def test__steps__should_update_metrics_of_their_phase(
        self,
        module: BackboneHeadsModule,
        batch: Batch[TensorTargets],
        log_dict: MagicMock,
        method: str,
        phase: Phase,
    ) -> None:
        """Each step updates the metric collections of its own phase only; nothing is logged."""
        getattr(module, method)(batch, 0)

        updated_phases = {
            metrics_phase
            for metrics_phase in Phase
            for collection in module.metrics_for(metrics_phase).values()
            if all(metric.update_called for metric in collection.values())
        }
        assert updated_phases == {phase}
        log_dict.assert_not_called()

    def test__on_validation_epoch_end__should_log_the_epoch_metrics_then_reset(
        self, module: BackboneHeadsModule, batch: Batch[TensorTargets], log_dict: MagicMock
    ) -> None:
        """The epoch's metrics are computed once, logged, and the next epoch starts clean."""
        module.validation_step(batch, 0)
        collections = list(module.metrics_for(Phase.VAL).values())
        expected_names = {name for collection in collections for name in collection.compute()}

        module.on_validation_epoch_end()

        logged_names = {name for call in log_dict.call_args_list for name in call.args[0]}
        assert logged_names == expected_names
        assert not any(
            metric.update_called for collection in collections for metric in collection.values()
        )

    def test__on_train_epoch_end__when_metrics_never_updated__should_not_log_them(
        self, module: BackboneHeadsModule, log_dict: MagicMock
    ) -> None:
        """A phase whose metrics saw no batch logs no metric."""
        module.on_train_epoch_end()

        log_dict.assert_not_called()

    def test__configure_optimizers__should_return_factory_output(
        self, module: BackboneHeadsModule
    ) -> None:
        """Optimizer and scheduler come from the factory, built on the module's parameters."""
        result = module.configure_optimizers()

        assert isinstance(result["optimizer"], Adam)
        assert isinstance(result["lr_scheduler"]["scheduler"], ReduceLROnPlateau)
        assert result["lr_scheduler"]["monitor"] == "val_loss"
