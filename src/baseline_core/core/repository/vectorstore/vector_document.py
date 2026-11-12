"""VectorDocument — the document a vector store indexes, whatever the provider."""

from collections.abc import Callable
from inspect import cleandoc
from typing import Annotated, Any, ClassVar

from loguru import logger
from pydantic import BaseModel, ConfigDict


class VectorStoreProperty:
    """How a vector store treats a document field: indexed, embedded, or shown in a prompt."""

    def __init__(
        self,
        *,
        searchable: bool = False,
        filterable: bool = False,
        is_vector: bool = False,
        embedding_template: str | Callable[[Any], str] | None = None,
        prompt_template: str | Callable[[Any], str] | None = None,
    ) -> None:
        """Bind how the field is indexed, and the templates its text is embedded and prompted by.

        Args:
            searchable: Whether the field is indexed for keyword search.
            filterable: Whether the field is indexed for filters.
            is_vector: Whether the field holds the document's vector.
            embedding_template: Its text in the embedded text; ``{value}`` stands for the value.
            prompt_template: Its text in the prompt; ``{value}`` stands for the value.
        """
        self.searchable = searchable
        self.filterable = filterable
        self.is_vector = is_vector

        self.embedding_template = (
            (lambda value: embedding_template.format(value=value))
            if isinstance(embedding_template, str)
            else embedding_template
        )

        self.prompt_template = (
            (lambda value: prompt_template.format(value=value))
            if isinstance(prompt_template, str)
            else prompt_template
        )


class VectorDocument(BaseModel):
    """A document a vector store indexes: its id, its vector, the text it embeds and prompts.

    Attributes:
        id_: The document's id.
    """

    model_config = ConfigDict(from_attributes=True)

    id_: Annotated[str, VectorStoreProperty(filterable=True)]

    vector: Annotated[list[float] | None, VectorStoreProperty(is_vector=True)] = None

    __collection_name__: ClassVar[str]
    __vector_field_name__: ClassVar[str | None] = None
    __embedding_prefix__: ClassVar[str] = ""

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        """Find the subclass's vector field, refusing more than one."""
        super().__init_subclass__(**kwargs)

        vector_fields = [
            (field_name, metadata)
            for field_name, field_info in cls.model_fields.items()
            if field_info.metadata
            for metadata in field_info.metadata
            if isinstance(metadata, VectorStoreProperty) and metadata.is_vector
        ]

        if len(vector_fields) > 1:
            error_message = (
                f"Only one vector field allowed in {cls.__name__}, found {len(vector_fields)}"
            )
            logger.error(error_message)
            raise ValueError(error_message)

        cls.__vector_field_name__ = vector_fields[0][0] if vector_fields else None

    @property
    def prompt(self) -> str:
        """Prompt to be used by the LLM for recommending documents."""
        prompt_texts = [
            vectorstore_property.prompt_template(getattr(self, field_name))
            for field_name, field_info in type(self).model_fields.items()
            if field_info.metadata
            and (
                vectorstore_property := next(
                    (
                        metadata
                        for metadata in field_info.metadata
                        if isinstance(metadata, VectorStoreProperty)
                    ),
                    None,
                )
            )
            and vectorstore_property.prompt_template
        ]
        return "\n".join(prompt_texts)

    @property
    def text_to_embed(self) -> str:
        """Text embedding sequence.

        Returns:
            Fields with an embedding template are concatenated into a single string.
        """
        embedding_texts = [
            vectorstore_property.embedding_template(getattr(self, field_name))
            for field_name, field_info in type(self).model_fields.items()
            if field_info.metadata
            and getattr(self, field_name) is not None
            and (
                vectorstore_property := next(
                    (
                        metadata
                        for metadata in field_info.metadata
                        if isinstance(metadata, VectorStoreProperty)
                    ),
                    None,
                )
            )
            and vectorstore_property.embedding_template
        ]
        raw_text = "\n".join(embedding_texts)

        if self.__embedding_prefix__:
            return f"{self.__embedding_prefix__}{raw_text}"

        return raw_text


class RetrievedDocument(BaseModel):
    """A document a search retrieved, with its relevance score and its rank."""

    content: VectorDocument
    score: float
    rank: int

    @property
    def prompt(self) -> str:
        """The document as an LLM prompt reads it, with its rank and relevance score."""
        return cleandoc(
            f"""
            <document rank="{self.rank + 1}" relevance-score="{self.score:.2f}">
                {self.content.prompt}
            </document>
            """
        )
