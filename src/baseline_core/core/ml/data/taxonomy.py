"""The labels the project knows, with the descriptions given to annotators and models."""

from collections import Counter
from pathlib import Path
from typing import Literal, Self

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, model_validator


class LabelDefinition(BaseModel):
    """One label of the taxonomy.

    Attributes:
        name: Label name used in annotations.
        description: What the label covers, as told to annotators and to the teacher model.
        status: ``"retained"`` once confirmed for the product, ``"candidate"`` before.
    """

    name: str
    description: str
    status: Literal["retained", "candidate"]


class Taxonomy(BaseModel):
    """Every label the project knows, with its description.

    Attributes:
        labels: Label definitions, unique by name; their order defines the label ids.
    """

    labels: list[LabelDefinition] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_names(self) -> Self:
        """Label names are unique."""
        duplicates = sorted(name for name, n in Counter(self.names).items() if n > 1)

        if duplicates:
            error_msg = f"Taxonomy label names must be unique, got duplicates: {duplicates}"
            raise ValueError(error_msg)

        return self

    @classmethod
    def load(cls, path: Path) -> "Taxonomy":
        """Read the taxonomy from a YAML file with a ``labels`` list.

        Args:
            path: YAML file.

        Returns:
            The taxonomy.
        """
        return cls.model_validate(OmegaConf.to_container(OmegaConf.load(path), resolve=True))

    def subset(self, names: list[str]) -> "Taxonomy":
        """Keep the labels in ``names``, in that order.

        Args:
            names: Label names.

        Returns:
            The smaller taxonomy.

        Raises:
            ValueError: If a name is not in the taxonomy.
        """
        by_name = {label.name: label for label in self.labels}
        unknown = sorted(set(names) - set(by_name))

        if unknown:
            error_msg = f"Labels {unknown} are not in the taxonomy {sorted(by_name)}."
            raise ValueError(error_msg)

        return Taxonomy(labels=[by_name[name] for name in names])

    @property
    def names(self) -> list[str]:
        """Label names, in order."""
        return [label.name for label in self.labels]

    @property
    def id2label(self) -> dict[int, str]:
        """Label name per 0-based id, as Hugging Face models expect."""
        return dict(enumerate(self.names))

    @property
    def descriptions(self) -> dict[str, str]:
        """Description per label name."""
        return {label.name: label.description for label in self.labels}
