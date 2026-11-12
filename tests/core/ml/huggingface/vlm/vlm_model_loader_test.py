"""Tests for ``VlmModelLoader``, without downloading anything."""

from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from transformers import AutoModelForImageTextToText, AutoProcessor

from baseline_core.core.ml.huggingface.vlm.vlm_model_loader import VlmModelConfig, VlmModelLoader


class TestVlmModelLoader:
    """Tests for the loading arguments."""

    @pytest.fixture
    def from_pretrained(self, mocker: MockerFixture) -> tuple[MagicMock, MagicMock]:
        """The patched ``from_pretrained`` of the model and processor auto classes."""
        model = mocker.patch.object(AutoModelForImageTextToText, "from_pretrained")
        processor = mocker.patch.object(AutoProcessor, "from_pretrained")
        return model, processor

    def test__load__should_load_safetensors_without_remote_code(
        self, from_pretrained: tuple[MagicMock, MagicMock]
    ) -> None:
        """The config's kwargs reach ``from_pretrained``; the safety flags are always set."""
        model_from_pretrained, processor_from_pretrained = from_pretrained
        config = VlmModelConfig(
            model_id="Qwen/Qwen3.8-27B", load_kwargs={"dtype": "bfloat16", "device_map": "cuda"}
        )

        model, processor = VlmModelLoader(config).load()

        model_from_pretrained.assert_called_once_with(
            "Qwen/Qwen3.8-27B",
            trust_remote_code=False,
            use_safetensors=True,
            dtype="bfloat16",
            device_map="cuda",
        )
        processor_from_pretrained.assert_called_once_with(
            "Qwen/Qwen3.8-27B", trust_remote_code=False
        )
        assert model is model_from_pretrained.return_value.eval.return_value
        assert processor is processor_from_pretrained.return_value

    def test__load_kwargs__when_quantization__should_build_the_bitsandbytes_config(self) -> None:
        """The quantization block becomes a ``BitsAndBytesConfig``."""
        config = VlmModelConfig(
            model_id="Qwen/Qwen3.8-27B",
            load_kwargs={"device_map": "cuda"},
            quantization={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"},
        )

        load_kwargs = VlmModelLoader(config).load_kwargs()

        assert load_kwargs["device_map"] == "cuda"
        assert load_kwargs["quantization_config"].load_in_4bit is True
        assert load_kwargs["quantization_config"].bnb_4bit_quant_type == "nf4"
