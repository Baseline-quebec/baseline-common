"""Unit tests for the Qdrant writer: the write contract, and embedding only what changed."""

from unittest.mock import patch

import pytest
from pydantic_ai import Embedder
from qdrant_client import AsyncQdrantClient

from baseline_core.core.repository.abstract_repository import (
    DuplicateRecordError,
    RecordNotFoundError,
)
from baseline_core.core.repository.query_filters import EqualityFilter
from baseline_core.core.repository.vectorstore.qdrant.qdrant_read_repository import (
    QdrantReadRepository,
)
from baseline_core.core.repository.vectorstore.qdrant.qdrant_write_repository import (
    QdrantWriteRepository,
)
from tests.core.repository.vectorstore.qdrant.conftest import Machine


class TestInsert:
    """An insert embeds new documents and refuses an id already held."""

    @staticmethod
    async def test__an_insert__creates_the_collection_and_the_document(
        machine_reads: QdrantReadRepository[Machine],
        machine_writes: QdrantWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """No collection is created beforehand."""
        inserted = await machine_writes.insert_one(fleet[0])

        assert inserted == "m-1"
        assert await machine_reads.get_by_id("m-1") == fleet[0]

    @staticmethod
    async def test__many__are_inserted_in_order(
        machine_writes: QdrantWriteRepository[Machine], fleet: list[Machine]
    ) -> None:
        """Their ids, as given."""
        inserted = await machine_writes.insert_many(fleet)

        assert inserted == ["m-1", "m-2", "m-3", "m-4", "m-5"]

    @staticmethod
    async def test__an_id_held__is_refused(
        machine_writes: QdrantWriteRepository[Machine], fleet: list[Machine]
    ) -> None:
        """An insert never replaces; an upsert does."""
        await machine_writes.insert_one(fleet[0])

        with pytest.raises(DuplicateRecordError, match="m-1"):
            await machine_writes.insert_many([fleet[1], fleet[0]])

    @staticmethod
    async def test__an_id_given_twice__is_refused_before_any_write(
        machine_reads: QdrantReadRepository[Machine],
        machine_writes: QdrantWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """Nothing of the batch is written."""
        with pytest.raises(DuplicateRecordError, match="m-2"):
            await machine_writes.insert_many([fleet[1], fleet[0], fleet[1]])

        assert await machine_reads.count_by_filters([]) == 0

    @staticmethod
    async def test__a_large_insert__embeds_by_batches(
        machine_writes: QdrantWriteRepository[Machine], embedder: Embedder
    ) -> None:
        """120 machines, three batches of at most 50."""
        machines = [Machine(id_=f"m-{number}", name=f"Lathe {number}") for number in range(120)]

        with patch.object(embedder, "embed_documents", wraps=embedder.embed_documents) as embed:
            await machine_writes.insert_many(machines)

        assert [len(call.args[0]) for call in embed.call_args_list] == [50, 50, 20]

    @staticmethod
    async def test__a_datetime__is_stored_as_iso_text(
        stocked: QdrantReadRepository[Machine], qdrant_client: AsyncQdrantClient
    ) -> None:
        """A datetime payload index reads RFC 3339 text."""
        points = await qdrant_client.retrieve("machines", [stocked.point_id("m-1")])

        assert points[0].payload is not None
        assert points[0].payload["serviced"] == "2026-09-01T00:00:00Z"
        assert "vector" not in points[0].payload


class TestUpdate:
    """An update replaces the payload, and re-embeds only a changed text to embed."""

    @staticmethod
    async def test__an_update__replaces_the_document(
        stocked: QdrantReadRepository[Machine],
        machine_writes: QdrantWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """The new notes are read back."""
        await machine_writes.update_one(fleet[0].model_copy(update={"notes": "oiled"}))

        found = await stocked.get_by_id("m-1")

        assert found is not None
        assert found.notes == "oiled"

    @staticmethod
    async def test__an_unchanged_text__is_not_embedded_again(
        stocked: QdrantReadRepository[Machine],
        machine_writes: QdrantWriteRepository[Machine],
        embedder: Embedder,
        fleet: list[Machine],
    ) -> None:
        """Notes are not embedded; a name is."""
        with patch.object(embedder, "embed_documents", wraps=embedder.embed_documents) as embed:
            await machine_writes.update_one(fleet[0].model_copy(update={"notes": "oiled"}))
            payload_only = embed.call_count
            await machine_writes.update_one(fleet[0].model_copy(update={"name": "Lathe Z"}))

        assert payload_only == 0
        assert embed.call_count == 1
        assert await stocked.count_by_filters([EqualityFilter(field="name", value="Lathe Z")]) == 1

    @staticmethod
    async def test__a_document_not_held__is_refused(
        stocked: QdrantReadRepository[Machine], machine_writes: QdrantWriteRepository[Machine]
    ) -> None:
        """An update never inserts."""
        with pytest.raises(RecordNotFoundError, match="m-9"):
            await machine_writes.update_one(Machine(id_="m-9", name="Drill"))

        assert await stocked.get_by_id("m-9") is None

    @staticmethod
    async def test__an_upsert__inserts_or_replaces(
        stocked: QdrantReadRepository[Machine],
        machine_writes: QdrantWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """A new drill, and the first lathe renamed; the last of an id repeated is kept."""
        renamed = fleet[0].model_copy(update={"name": "Lathe Z"})

        upserted = await machine_writes.upsert_many([
            Machine(id_="m-9", name="Drill"),
            fleet[0],
            renamed,
        ])

        assert upserted == ["m-9", "m-1", "m-1"]
        assert await stocked.get_by_id("m-9") == Machine(id_="m-9", name="Drill")
        assert await stocked.get_by_id("m-1") == renamed

    @staticmethod
    async def test__an_upsert__creates_the_collection(
        machine_reads: QdrantReadRepository[Machine],
        machine_writes: QdrantWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """The first write may be an upsert."""
        await machine_writes.upsert_one(fleet[0])

        assert await machine_reads.get_by_id("m-1") == fleet[0]


class TestDelete:
    """A delete removes documents by id or by filters, and refuses what would surprise."""

    @staticmethod
    async def test__an_id__deletes_its_document(
        stocked: QdrantReadRepository[Machine],
        machine_writes: QdrantWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """The others stay."""
        deleted = await machine_writes.delete_one("m-1")

        assert deleted == "m-1"
        assert await stocked.get_by_id("m-1") is None
        assert await stocked.count_by_filters([]) == len(fleet) - 1

    @staticmethod
    async def test__an_id_not_held__is_refused(
        stocked: QdrantReadRepository[Machine],
        machine_writes: QdrantWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """Nothing is deleted when one id is unknown."""
        with pytest.raises(RecordNotFoundError, match="m-9"):
            await machine_writes.delete_many(["m-1", "m-9"])

        assert await stocked.count_by_filters([]) == len(fleet)

    @staticmethod
    async def test__filters__delete_every_match_past_one_page(
        machine_reads: QdrantReadRepository[Machine],
        machine_writes: QdrantWriteRepository[Machine],
    ) -> None:
        """More matches than one page of results."""
        worn = [Machine(id_=f"w-{number}", name="Lathe", kind="worn") for number in range(150)]
        await machine_writes.insert_many([*worn, Machine(id_="kept", name="Press")])
        worn_filter = EqualityFilter(field="kind", value="worn")

        deleted = await machine_writes.delete_by_filters([worn_filter])

        assert deleted == len(worn)
        assert await machine_reads.count_by_filters([worn_filter]) == 0
        assert await machine_reads.get_by_id("kept") is not None

    @staticmethod
    async def test__no_filter__is_refused(
        stocked: QdrantReadRepository[Machine],
        machine_writes: QdrantWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """No filter matches everything: that is delete_all, never a slip."""
        with pytest.raises(ValueError, match="delete_all"):
            await machine_writes.delete_by_filters([])

        assert await stocked.count_by_filters([]) == len(fleet)

    @staticmethod
    async def test__all__empties_the_collection(
        stocked: QdrantReadRepository[Machine],
        machine_writes: QdrantWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """Five deleted, none left."""
        deleted = await machine_writes.delete_all()

        assert deleted == len(fleet)
        assert await stocked.count_by_filters([]) == 0
