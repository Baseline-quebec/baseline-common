"""Resolve classes named in configuration files."""

import importlib
from types import ModuleType
from typing import Any


class DynamicImport:
    """Imports modules and classes named by strings."""

    @staticmethod
    def ensure_module(module: str | ModuleType) -> ModuleType:
        """Import ``module`` when given as a dotted path, return it unchanged otherwise.

        Args:
            module: Dotted module path or an already imported module.

        Returns:
            The module object.
        """
        return importlib.import_module(module) if isinstance(module, str) else module

    @staticmethod
    def resolve_class(module: str | ModuleType, name: str) -> type[Any]:
        """Return class ``name`` from ``module``.

        Args:
            module: Dotted module path or module object exposing the class.
            name: Class name.

        Returns:
            The class object.

        Raises:
            ImportError: If the module has no attribute ``name``.
            TypeError: If that attribute is not a class.
        """
        resolved = DynamicImport.ensure_module(module)
        candidate = getattr(resolved, name, None)

        if candidate is None:
            error_msg = f"'{name}' not found in module '{resolved.__name__}'."
            raise ImportError(error_msg)

        if not isinstance(candidate, type):
            error_msg = f"'{resolved.__name__}.{name}' is not a class."
            raise TypeError(error_msg)

        return candidate
