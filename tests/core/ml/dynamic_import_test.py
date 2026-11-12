"""Tests for class resolution from configuration strings."""

import collections
import math

import pytest

from baseline_core.core.ml.dynamic_import import DynamicImport


class TestEnsureModule:
    """Tests for ``DynamicImport.ensure_module``."""

    def test__ensure_module__when_dotted_path__should_import_it(self) -> None:
        """A string is imported."""
        module = DynamicImport.ensure_module("math")

        assert module is math

    def test__ensure_module__when_module__should_return_it_unchanged(self) -> None:
        """A module object passes through."""
        module = DynamicImport.ensure_module(math)

        assert module is math


class TestResolveClass:
    """Tests for ``DynamicImport.resolve_class``."""

    def test__resolve_class__when_class_exists__should_return_it(self) -> None:
        """The class object is returned, whether the module is a string or an object."""
        from_string = DynamicImport.resolve_class("collections", "OrderedDict")
        from_module = DynamicImport.resolve_class(collections, "OrderedDict")

        assert from_string is collections.OrderedDict
        assert from_module is collections.OrderedDict

    def test__resolve_class__when_name_missing__should_raise_import_error(self) -> None:
        """An unknown name is an ``ImportError`` naming the module."""
        with pytest.raises(ImportError, match="'Missing' not found in module 'math'"):
            DynamicImport.resolve_class("math", "Missing")

    def test__resolve_class__when_attribute_is_not_a_class__should_raise_type_error(self) -> None:
        """A constant or function is rejected."""
        with pytest.raises(TypeError, match="is not a class"):
            DynamicImport.resolve_class(math, "pi")
