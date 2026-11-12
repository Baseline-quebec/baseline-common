"""Load OmegaConf/Hydra configs into Pydantic models, resolving class names and references."""

import inspect
from types import ModuleType, UnionType
from typing import Any, Union, get_args, get_origin

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from baseline_core.core.ml.dynamic_import import DynamicImport


class ConfigLoader:
    """Turn a config section into a validated Pydantic model, resolving names and references."""

    @classmethod
    def to_pydantic[T: BaseModel](
        cls,
        config: DictConfig,
        model_class: type[T],
        module: ModuleType | None = None,
        key: str | None = None,
    ) -> T:
        """Validate a section of ``config`` against ``model_class``.

        Args:
            config: Full configuration.
            model_class: Pydantic model to validate against.
            module: Module in which ``type[...]`` fields are resolved by class name.
            key: Dotted path of the section (``"training.optimizer"``), or ``None`` for all.

        Returns:
            The validated model.

        Raises:
            KeyError: If ``key`` does not exist in ``config``.
        """
        section = OmegaConf.select(config, key) if key is not None else config

        if section is None:
            error_msg = f"Config key '{key}' not found."
            raise KeyError(error_msg)

        data = OmegaConf.to_container(section, resolve=True)
        resolved = cls._resolve(data, model_class, {}, module)
        return model_class.model_validate(resolved)

    @classmethod
    def _resolve(
        cls,
        data: Any,  # noqa: ANN401
        annotation: Any,  # noqa: ANN401
        references: dict[str, dict[str, Any]],
        module: ModuleType | None,
    ) -> Any:  # noqa: ANN401
        """Walk ``data`` along ``annotation``, resolving class names and ``id_`` references.

        Args:
            data: Raw config value.
            annotation: Type expected for ``data``.
            references: Entries seen so far, keyed by ``id_``; filled in place.
            module: Module in which class names are resolved.

        Returns:
            ``data`` with class names replaced by classes and references by their entry.
        """
        origin, args = get_origin(annotation), get_args(annotation)

        if origin is list and args and isinstance(data, list):
            return [cls._resolve(item, args[0], references, module) for item in data]

        if origin in {Union, UnionType}:
            models = [arg for arg in args if isinstance(arg, type) and issubclass(arg, BaseModel)]

            if len(models) == 1:
                return cls._resolve(data, models[0], references, module)

            return data

        if origin is type and isinstance(data, str):
            if module is None:
                error_msg = f"A module is required to resolve class '{data}'."
                raise ValueError(error_msg)

            return DynamicImport.resolve_class(module, data)

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return cls._resolve_model(data, annotation, references, module)

        return data

    @classmethod
    def _resolve_model(
        cls,
        data: Any,  # noqa: ANN401
        model_class: type[BaseModel],
        references: dict[str, dict[str, Any]],
        module: ModuleType | None,
    ) -> Any:  # noqa: ANN401
        """Resolve a value expected to be ``model_class``: an ``id_`` reference or a dict.

        Args:
            data: Raw config value.
            model_class: Model expected for ``data``.
            references: Entries seen so far, keyed by ``id_``; filled in place.
            module: Module in which class names are resolved.

        Returns:
            The referenced entry, the resolved dict, or the built subclass if abstract.

        Raises:
            ValueError: If ``data`` references an ``id_`` not defined earlier.
        """
        if isinstance(data, str):
            if data not in references:
                error_msg = f"Reference '{data}' is not defined earlier in the config."
                raise ValueError(error_msg)

            return references[data]

        if isinstance(data, dict):
            if inspect.isabstract(model_class):
                return cls._resolve_subclass(data, model_class, references, module)

            for name, field in model_class.model_fields.items():
                if name in data:
                    data[name] = cls._resolve(data[name], field.annotation, references, module)

            if "id_" in data:
                references[data["id_"]] = data

        return data

    @classmethod
    def _resolve_subclass(
        cls,
        data: dict[str, Any],
        model_class: type[BaseModel],
        references: dict[str, dict[str, Any]],
        module: ModuleType | None,
    ) -> BaseModel:
        """Build the concrete subclass of an abstract model named by the ``name`` entry.

        Args:
            data: Config entry, e.g. ``{name: AdamWOptimizer, lr: 1.0e-4}``.
            model_class: Abstract model expected for ``data``.
            references: Entries seen so far, keyed by ``id_``; filled in place.
            module: Module in which ``name`` is resolved.

        Returns:
            The validated subclass instance.

        Raises:
            ValueError: If ``data`` has no ``name`` or no module is given.
            TypeError: If ``name`` is not a subclass of ``model_class``.
        """
        if "name" not in data or module is None:
            error_msg = (
                f"'{model_class.__name__}' is abstract: give a 'name' entry naming a subclass "
                "and a module to resolve it in."
            )
            raise ValueError(error_msg)

        fields = {key: value for key, value in data.items() if key != "name"}
        concrete = DynamicImport.resolve_class(module, data["name"])

        if not issubclass(concrete, model_class):
            error_msg = f"'{data['name']}' is not a subclass of '{model_class.__name__}'."
            raise TypeError(error_msg)

        resolved = cls._resolve_model(fields, concrete, references, module)
        return concrete.model_validate(resolved)
