"""A model written once against a schema type, with no data."""

from abc import ABC, abstractmethod

from pydantic import BaseModel

from baseline_core.core.optimization.problem_formulation.model import Model


class AbstractModelDefinition[S: BaseModel](ABC):
    """Names its schema and posts its rules in ``define``; the data arrives only when bound."""

    schema: type[S]
    """The type of the object the model will be bound to, set on the subclass."""

    @abstractmethod
    def define(self, model: Model[S]) -> None:
        """Post every rule.

        Args:
            model: The model to post on, its selectors typed by ``schema``.
        """
