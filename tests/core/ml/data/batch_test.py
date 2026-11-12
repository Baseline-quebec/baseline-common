"""Tests for the batch schemas."""

from collections.abc import Callable

import pytest
import torch
from lightning_fabric.utilities.apply_func import move_data_to_device

from baseline_core.core.ml.data.batch import AbstractBatch, Batch, TensorTargets


class TestAbstractBatch:
    """Tests for the tensor traversal every batch shares."""

    @pytest.fixture
    def to_double(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Convert a tensor to ``float64``."""
        return lambda tensor: tensor.to(torch.float64)

    def test__map_tensors__when_nested_dict__should_convert_every_tensor(
        self, to_double: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        """Tensors nested in dicts are converted in place of the originals."""
        data = {"a": torch.tensor([1.0]), "b": {"c": torch.tensor([2.0])}}

        result = AbstractBatch.map_tensors(data, to_double)

        assert result["a"].dtype == torch.float64
        assert result["b"]["c"].dtype == torch.float64

    @pytest.mark.parametrize("container", [list, tuple])
    def test__map_tensors__when_sequence__should_keep_its_type(
        self,
        to_double: Callable[[torch.Tensor], torch.Tensor],
        container: type[list[torch.Tensor]] | type[tuple[torch.Tensor, ...]],
    ) -> None:
        """Lists stay lists and tuples stay tuples, not generators."""
        data = container([torch.tensor([1.0]), torch.tensor([2.0])])

        result = AbstractBatch.map_tensors(data, to_double)

        assert isinstance(result, container)
        assert all(tensor.dtype == torch.float64 for tensor in result)

    def test__map_tensors__when_not_a_tensor__should_return_it_unchanged(
        self, to_double: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        """Strings, numbers and other leaves pass through."""
        data = {"name": "a.png", "count": 3}

        assert AbstractBatch.map_tensors(data, to_double) == data

    def test__map_fields__should_convert_every_field_and_extra(
        self, to_double: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        """Declared fields and extra fields alike; the batch keeps its class."""
        batch = Batch[TensorTargets](
            x={"image": torch.zeros(2)}, targets={"label": torch.zeros(2)}, mask=torch.ones(2)
        )

        converted = batch.map_fields(to_double)

        assert isinstance(converted, Batch)
        assert converted.x["image"].dtype == torch.float64
        assert converted.targets is not None
        assert converted.targets["label"].dtype == torch.float64
        assert converted.model_extra is not None
        assert converted.model_extra["mask"].dtype == torch.float64


class TestBatch:
    """Tests for ``Batch``."""

    def test__init__when_extra_field__should_keep_it(self) -> None:
        """Extra fields are allowed so projects can carry task-specific tensors."""
        mask = torch.tensor([True, False])
        batch = Batch[TensorTargets](x={"image": torch.zeros(2, 3)}, mask=mask)

        assert batch.model_extra is not None
        assert torch.equal(batch.model_extra["mask"], mask)

    def test__to__when_lightning_moves_the_batch__should_move_every_nested_tensor(self) -> None:
        """Lightning finds ``to`` and calls it; tensors in lists, dicts and extras all move."""
        batch = Batch[list[dict[str, torch.Tensor]]](
            x={"pixel_values": torch.zeros(1)},
            targets=[{"boxes": torch.zeros(1, 4)}],
            metadata={"file_names": ["a.png"]},
            mask=torch.ones(1),
        )

        moved = move_data_to_device(batch, torch.device("meta"))

        assert isinstance(moved, Batch)
        assert moved.x["pixel_values"].is_meta
        assert moved.targets is not None
        assert moved.targets[0]["boxes"].is_meta
        assert moved.model_extra is not None
        assert moved.model_extra["mask"].is_meta
        assert moved.metadata == {"file_names": ["a.png"]}

    def test__init__when_no_targets__should_default_to_none(self) -> None:
        """Targets are optional: inference batches have none."""
        batch = Batch[TensorTargets](x={"image": torch.zeros(2, 3)})

        assert batch.targets is None
        assert batch.metadata is None

    def test__init__when_parametrized__should_accept_the_task_supervision_shape(self) -> None:
        """The supervision type is the task's choice: here one dict of boxes per image."""
        per_image = [{"boxes": torch.zeros(2, 4)}, {"boxes": torch.zeros(0, 4)}]

        batch = Batch[list[dict[str, torch.Tensor]]](
            x={"pixel_values": torch.zeros(2, 3, 4, 4)}, targets=per_image
        )

        assert batch.targets is not None
        assert len(batch.targets) == len(per_image)

    def test__init__when_parametrized__should_reject_the_wrong_shape(self) -> None:
        """Validation follows the parameter."""
        with pytest.raises(ValueError, match="targets"):
            Batch[list[dict[str, torch.Tensor]]](x={}, targets={"label": torch.zeros(2)})

    def test__batch_size__should_be_the_leading_dimension_of_the_first_input(self) -> None:
        """Lightning weighs the logged values by the number of samples."""
        image = torch.zeros(3, 2)
        batch = Batch[TensorTargets](x={"image": image, "mask": torch.zeros(5)})

        assert batch.batch_size == image.shape[0]

    def test__batch_size__when_no_inputs__should_raise_value_error(self) -> None:
        """A batch without inputs has no sample to count."""
        batch = Batch[TensorTargets](x={})

        with pytest.raises(ValueError, match="no batch size"):
            _ = batch.batch_size
