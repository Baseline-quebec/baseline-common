"""DocumentExtractionOrchestrator: one file, from its bytes to what each of its sections names."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from baseline_core.core.llm.document_extraction.classification.classification_service import (
    ClassificationService,
)
from baseline_core.core.llm.document_extraction.document_parser import DocumentParser
from baseline_core.core.llm.document_extraction.extraction.extraction_service import (
    ExtractionService,
)
from baseline_core.core.llm.document_extraction.indexed_document import (
    FileMetadata,
    IndexedDocument,
)
from baseline_core.core.llm.document_extraction.indexing.indexing_service import IndexingService
from baseline_core.core.llm.document_extraction.section_entities import SectionEntities


class SourceFile(BaseModel):
    """One file handed to the pipeline: its bytes, and what its collection knows of it."""

    model_config = ConfigDict(frozen=True)

    content: bytes = Field(..., repr=False, title="Raw bytes of the document")
    name: str = Field("", title="File name, so a damaged document can be named in the log")
    document_id: str = Field(
        "", title="The collection's id; the checksum when empty, which tells two collections apart"
    )
    source: str | None = Field(
        None, title="The URL or path the file was collected from — what a link to it prints"
    )
    dated: date | None = Field(
        None, title="The date the collection knows for the document; sections fall back on it"
    )


class DocumentExtractionOrchestrator[Out: SectionEntities]:
    """Reads one file: parse, index, classify, extract, every stage on the same tree."""

    def __init__(
        self,
        parser: DocumentParser,
        indexing: IndexingService,
        classification: ClassificationService,
        extraction: ExtractionService[Out],
    ) -> None:
        """Bind the parser and the three services, as ``core/llm/document_extraction`` groups them.

        Args:
            parser: Parser of a file, whatever its format, configured by the project.
            indexing: Service reading the plan the document prints.
            classification: Service settling uncertain sections and folding back split ones.
            extraction: Service reading what each section names, in the project's schema.
        """
        self._parser = parser
        self._indexing = indexing
        self._classification = classification
        self._extraction = extraction

    async def run(self, file: SourceFile) -> IndexedDocument:
        """Parse one file, index the sections it prints, then read what each of them names.

        Args:
            file: The file's bytes and what its collection knows of it.

        Returns:
            What the file says and its resolved sections.
        """
        document = self._parser.parse(file.content)
        located = await self._indexing.index(document)
        located = await self._classification.settle(located)
        sections = await self._extraction.extract(located, document, file.content)

        # After extraction: what a section names is the evidence it was split off the one above.
        sections = await self._classification.repair(sections)
        metadata = FileMetadata.from_layout(document, file.name, file.source)

        return IndexedDocument(
            document_id=file.document_id or metadata.checksum or file.name,
            file=metadata,
            dated=file.dated,
            sections=sections,
            text=document.text,
        )
