"""Tests for loading OmegaConf configs into Pydantic models."""

from types import ModuleType
from typing import Any, ClassVar

import pytest
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from baseline_core.core.ml.config_loader import ConfigLoader
from baseline_core.core.ml.training import optimizer
from baseline_core.core.ml.training.optimizer import OptimizerFactory


class UserKind(BaseModel):
    """Domain class resolved by name from the module."""


class ArticleKind(BaseModel):
    """Domain class resolved by name from the module."""


class Node(BaseModel):
    """Config entry others can reference through ``id_``."""

    id_: str
    kind: type[BaseModel]


class Edge(BaseModel):
    """Config entry referencing two nodes by id."""

    src: Node
    dst: Node


class Graph(BaseModel):
    """Root config: nodes first, then edges referencing them."""

    nodes: list[Node]
    edges: list[Edge]


class TestConfigLoader:
    """Tests for ``ConfigLoader.to_pydantic``."""

    EXPECTED_NODE_COUNT: ClassVar[int] = 2

    @pytest.fixture
    def domain_module(self) -> ModuleType:
        """Module exposing the classes a config may name."""
        module = ModuleType("fake_domain")
        module.__dict__.update({"UserKind": UserKind, "ArticleKind": ArticleKind})
        return module

    @pytest.fixture
    def graph_config(self) -> dict[str, Any]:
        """Config with class names and ``id_`` references to resolve."""
        return {
            "graph": {
                "nodes": [
                    {"id_": "User", "kind": "UserKind"},
                    {"id_": "Article", "kind": "ArticleKind"},
                ],
                "edges": [{"src": "User", "dst": "Article"}],
            }
        }

    def test__to_pydantic__when_class_names_and_references__should_resolve_both(
        self, domain_module: ModuleType, graph_config: dict[str, Any]
    ) -> None:
        """Strings become classes where ``type[...]`` is expected and nodes where a model is."""
        config = OmegaConf.create(graph_config)

        graph = ConfigLoader.to_pydantic(config, Graph, module=domain_module, key="graph")

        assert len(graph.nodes) == self.EXPECTED_NODE_COUNT
        assert graph.nodes[0].kind is UserKind
        assert graph.edges[0].src == graph.nodes[0]
        assert graph.edges[0].dst == graph.nodes[1]

    def test__to_pydantic__when_no_key__should_validate_whole_config(
        self, domain_module: ModuleType, graph_config: dict[str, Any]
    ) -> None:
        """Without ``key`` the root of the config is validated."""
        config = OmegaConf.create(graph_config["graph"])
        assert isinstance(config, DictConfig)

        graph = ConfigLoader.to_pydantic(config, Graph, module=domain_module)

        assert len(graph.nodes) == self.EXPECTED_NODE_COUNT

    def test__to_pydantic__when_key_missing__should_raise_key_error(
        self, domain_module: ModuleType
    ) -> None:
        """A missing section is a ``KeyError`` naming the key."""
        config = OmegaConf.create({"other": {}})

        with pytest.raises(KeyError, match="Config key 'graph' not found"):
            ConfigLoader.to_pydantic(config, Graph, module=domain_module, key="graph")

    def test__to_pydantic__when_class_missing__should_raise_import_error(
        self, domain_module: ModuleType, graph_config: dict[str, Any]
    ) -> None:
        """An unknown class name is an ``ImportError``."""
        graph_config["graph"]["nodes"][0]["kind"] = "GhostKind"
        config = OmegaConf.create(graph_config)

        with pytest.raises(ImportError, match="'GhostKind' not found"):
            ConfigLoader.to_pydantic(config, Graph, module=domain_module, key="graph")

    def test__to_pydantic__when_no_module__should_raise_value_error(
        self, graph_config: dict[str, Any]
    ) -> None:
        """Class names cannot be resolved without a module."""
        config = OmegaConf.create(graph_config)

        with pytest.raises(ValueError, match="module is required"):
            ConfigLoader.to_pydantic(config, Graph, key="graph")

    def test__to_pydantic__when_reference_undefined__should_raise_value_error(
        self, domain_module: ModuleType, graph_config: dict[str, Any]
    ) -> None:
        """An edge pointing at an unknown node id is a ``ValueError``."""
        graph_config["graph"]["edges"][0]["src"] = "GhostNode"
        config = OmegaConf.create(graph_config)

        with pytest.raises(ValueError, match="Reference 'GhostNode' is not defined"):
            ConfigLoader.to_pydantic(config, Graph, module=domain_module, key="graph")


class TestConfigLoaderPolymorphism:
    """Tests for abstract fields built as the subclass named in the config."""

    @pytest.fixture
    def optimizer_config(self) -> DictConfig:
        """An optimizer factory naming its optimizer and scheduler classes."""
        config = OmegaConf.create({
            "optimizer": {"name": "AdamWOptimizer", "lr": 0.001, "weight_decay": 0.01},
            "scheduler": {"name": "PlateauScheduler", "metric": "val_loss"},
        })
        assert isinstance(config, DictConfig)
        return config

    def test__to_pydantic__when_abstract_field__should_build_named_subclass(
        self, optimizer_config: DictConfig
    ) -> None:
        """``name`` picks the subclass; optional abstract fields are resolved too."""
        factory = ConfigLoader.to_pydantic(optimizer_config, OptimizerFactory, module=optimizer)

        assert isinstance(factory.optimizer, optimizer.AdamWOptimizer)
        assert factory.optimizer.weight_decay == pytest.approx(0.01)
        assert isinstance(factory.scheduler, optimizer.PlateauScheduler)

    def test__to_pydantic__when_abstract_field_without_name__should_raise_value_error(
        self, optimizer_config: DictConfig
    ) -> None:
        """An abstract model cannot be built without a subclass name."""
        del optimizer_config.optimizer["name"]

        with pytest.raises(ValueError, match="'AbstractOptimizer' is abstract"):
            ConfigLoader.to_pydantic(optimizer_config, OptimizerFactory, module=optimizer)

    def test__to_pydantic__when_name_is_not_a_subclass__should_raise_type_error(
        self, optimizer_config: DictConfig
    ) -> None:
        """A class of the module outside the hierarchy is rejected."""
        optimizer_config.optimizer["name"] = "PlateauScheduler"

        with pytest.raises(TypeError, match="not a subclass of 'AbstractOptimizer'"):
            ConfigLoader.to_pydantic(optimizer_config, OptimizerFactory, module=optimizer)
