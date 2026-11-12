"""Which attributes of a graph's nodes a model reads as inputs."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict


class Feature(BaseModel):
    """One attribute of a node read as a model input (e.g. a product's ``price``)."""

    model_config = ConfigDict(frozen=True)

    name: str
    transform: Callable[[Any], Any] | None = None


class GraphFeatureConfig(BaseModel):
    """The features read from each node type, keyed by the schema's ``NodeConfig.id_``."""

    model_config = ConfigDict(frozen=True)

    features: dict[str, list[Feature]]
