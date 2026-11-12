"""What one run of the record-matching agent reads: the record as written, and the candidates."""

from pydantic import BaseModel, ConfigDict, Field


class WrittenRecord(BaseModel):
    """An organization as a document wrote it, in the document's own words."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="The name as written, e.g. 'ACME IND. (MTL)'.")
    code: str | None = Field(None, description="The document's own code for it, if any.")
    address: str | None = Field(None, description="The address as written, if any.")


class CandidateRecord(BaseModel):
    """A known organization of the reference system that the written one may be."""

    model_config = ConfigDict(frozen=True)

    id_: str = Field(description="Its id in the reference system; the only id a match may name.")
    name: str
    address: str | None = None


class RecordMatchingDeps(BaseModel):
    """Everything one decision reads: the written record and the candidates near it."""

    model_config = ConfigDict(frozen=True)

    written: WrittenRecord
    candidates: list[CandidateRecord]
