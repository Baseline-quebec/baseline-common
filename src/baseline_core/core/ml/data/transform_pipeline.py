"""Composable transforms configured from plain dictionaries."""

import hashlib
import json
import typing
from collections.abc import Callable
from types import ModuleType
from typing import Any, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from baseline_core.core.ml.dynamic_import import DynamicImport


class TransformConfig(BaseModel):
    """A transform class, its constructor parameters and when it applies.

    Attributes:
        name: Class name, looked up in the pipeline's module.
        params: Constructor keyword arguments; a dtype name becomes a ``torch.dtype``.
        transforms: Nested steps, passed as ``transforms`` (``RandomApply``); flags ignored.
        apply_on_eval: Whether the step also runs outside training; ``False`` for augmentations.
        apply_on_the_fly: Whether the step runs on each read; ``False`` caches its result once.
    """

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    transforms: list["TransformConfig"] = Field(default_factory=list)
    apply_on_eval: bool = True
    apply_on_the_fly: bool = True

    def build(self, module: ModuleType) -> Callable[[Any], Any]:
        """Instantiate the transform and the steps nested in it.

        Args:
            module: Module exposing the transform classes.

        Returns:
            The transform; ``ImportError`` if ``module`` has no class ``name``.
        """
        transform_class = DynamicImport.resolve_class(module, self.name)
        hints = self.constructor_hints(transform_class)
        params = {
            name: self.resolve_param(value, hints.get(name)) for name, value in self.params.items()
        }

        if self.transforms:
            params["transforms"] = [nested.build(module) for nested in self.transforms]

        transform: Callable[[Any], Any] = transform_class(**params)
        return transform

    @staticmethod
    def constructor_hints(transform_class: type[Any]) -> dict[str, Any]:
        """Type hints of the constructor of ``transform_class``, empty when they cannot be read.

        Args:
            transform_class: Class to inspect.

        Returns:
            Annotation per parameter name.
        """
        try:
            return typing.get_type_hints(transform_class.__init__)
        except (NameError, TypeError):
            return {}

    @staticmethod
    def resolve_param(value: Any, annotation: Any) -> Any:  # noqa: ANN401
        """Turn a dtype name into the ``torch.dtype`` the constructor expects.

        Args:
            value: Parameter value from the configuration.
            annotation: Constructor annotation of the parameter, or ``None``.

        Returns:
            The ``torch.dtype`` named by ``value`` if ``annotation`` accepts one, else ``value``.

        Raises:
            ValueError: If ``value`` names no ``torch.dtype``.
        """
        if not isinstance(value, str) or not TransformConfig.accepts(annotation, torch.dtype):
            return value

        dtype = getattr(torch, value, None)

        if isinstance(dtype, torch.dtype):
            return dtype

        error_msg = f"'{value}' is not a torch dtype."
        raise ValueError(error_msg)

    @staticmethod
    def accepts(annotation: Any, expected: type[Any]) -> bool:  # noqa: ANN401
        """Whether ``annotation`` is ``expected`` or a union or container including it.

        Args:
            annotation: Type annotation to inspect.
            expected: Type looked for.

        Returns:
            ``True`` if ``expected`` appears in ``annotation``.
        """
        if annotation is expected:
            return True

        return any(TransformConfig.accepts(arg, expected) for arg in typing.get_args(annotation))


class TransformPipeline(BaseModel):
    """Ordered transforms applied one after the other; callable like a single transform.

    Attributes:
        module: Module exposing the transform classes, as a dotted path or a module object.
        transforms: Steps, in order.
    """

    module: str | ModuleType = Field(exclude=True)
    transforms: list[TransformConfig] = Field(default_factory=list)

    _steps: list[tuple[TransformConfig, Callable[[Any], Any]]] = PrivateAttr(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("module", mode="before")
    @classmethod
    def import_module(cls, value: str | ModuleType) -> ModuleType:
        """Import the module when given as a dotted path."""
        return DynamicImport.ensure_module(value)

    @model_validator(mode="after")
    def build_steps(self) -> Self:
        """Instantiate every transform, failing fast on unknown class names."""
        module = DynamicImport.ensure_module(self.module)
        self._steps = [(config, config.build(module)) for config in self.transforms]
        return self

    @model_validator(mode="after")
    def validate_cached_steps(self) -> Self:
        """Cached steps come before every live step and run on every split.

        Raises:
            ValueError: If a cached step follows a live one or is skipped outside training.
        """
        live_step = None

        for config in self.transforms:
            if config.apply_on_the_fly:
                live_step = live_step or config.name
            elif live_step is not None:
                error_msg = (
                    f"Step '{config.name}' is cached (apply_on_the_fly=False) but follows the "
                    f"live step '{live_step}'; cached steps come first."
                )
                raise ValueError(error_msg)
            elif not config.apply_on_eval:
                error_msg = (
                    f"Step '{config.name}' is cached but skipped outside training "
                    "(apply_on_eval=False); a cached result must be the same on every split."
                )
                raise ValueError(error_msg)

        return self

    @property
    def cache_key(self) -> str | None:
        """Hash of the module and configs of the cached steps; ``None`` when there are none."""
        cached = [
            config.model_dump(mode="json")
            for config in self.transforms
            if not config.apply_on_the_fly
        ]

        if not cached:
            return None

        module_name = DynamicImport.ensure_module(self.module).__name__
        payload = json.dumps([module_name, cached], sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def prepare(self, data: Any) -> Any:  # noqa: ANN401
        """Run ``data`` through the cached steps only.

        Args:
            data: Value to transform.

        Returns:
            The value the live steps start from.
        """
        for config, step in self._steps:
            if config.apply_on_the_fly:
                break

            data = step(data)

        return data

    def __call__(
        self,
        data: Any,  # noqa: ANN401
        *,
        training: bool = True,
        prepared: bool = False,
    ) -> Any:  # noqa: ANN401
        """Run ``data`` through every step in order.

        Args:
            data: Value to transform.
            training: Whether training-only steps (``apply_on_eval=False``) run.
            prepared: Whether ``data`` already went through ``prepare``; only live steps run.

        Returns:
            The transformed value.
        """
        for config, step in self._steps:
            if prepared and not config.apply_on_the_fly:
                continue

            if training or config.apply_on_eval:
                data = step(data)

        return data
