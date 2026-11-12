"""Unit tests for the Weaviate writer: the write contract, and embedding only what changed."""

import uuid

import pytest
from pydantic_ai import Embedder

from baseline_common.core.repository.abstract_repository import RecordNotFoundError
from baseline_common.core.repository.vectorstore.weaviate.weaviate_read_repository import (
    WeaviateReadRepository,
)
from baseline_common.core.repository.vectorstore.weaviate.weaviate_write_repository import (
    WeaviateWriteRepository,
)
from tests.core.repository.vectorstore.weaviate.conftest import Machine, WeaviateDouble


class TestInsert:
    """An insert embeds each document and stores it under a UUID derived from its id_."""

    @staticmethod
    async def test__many__are_inserted_in_order(
        weaviate_double: WeaviateDouble,
        machine_reads: WeaviateReadRepository[Machine],
        machine_writes: WeaviateWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """Their ids, as given, each read back as it was written."""
        inserted = await machine_writes.insert_many(fleet)

        assert inserted == ["m-1", "m-2", "m-3"]
        assert len(weaviate_double.objects) == 3
        assert await machine_reads.get_by_id("m-2") == fleet[1]

    @staticmethod
    async def test__the_vector__is_sent_beside_the_properties(
        weaviate_double: WeaviateDouble,
        machine_writes: WeaviateWriteRepository[Machine],
        fleet: list[Machine],
        embedder: Embedder,
    ) -> None:
        """The vector field is never a property; unset fields are left out."""
        expected = await embedder.embed_query([fleet[1].text_to_embed])

        await machine_writes.insert_one(fleet[1])

        held = weaviate_double.by_id("m-2")
        assert held.properties == {
            "id_": "m-2",
            "name": "Press B",
            "kind": "press",
            "mode": "auto",
            "tags": ["north"],
        }
        assert held.vector["default"] == list(expected.embeddings[0])
        assert held.uuid == uuid.uuid5(uuid.NAMESPACE_DNS, "m-2")

    @staticmethod
    async def test__an_id_held__is_refused_naming_it(
        machine_writes: WeaviateWriteRepository[Machine], fleet: list[Machine]
    ) -> None:
        """Weaviate refuses a UUID it holds; the insert says so instead of logging it."""
        await machine_writes.insert_one(fleet[0])

        with pytest.raises(ValueError, match=r"refused 1 of 2: m-1 \(id already exists\)"):
            await machine_writes.insert_many([fleet[1], fleet[0]])


class TestUpdate:
    """An update replaces documents held, re-embedding only a changed text."""

    @staticmethod
    async def test__a_changed_text__is_embedded_again(
        weaviate_double: WeaviateDouble,
        machine_writes: WeaviateWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """The name is embedded: a new name sends a new vector."""
        await machine_writes.insert_one(fleet[0])
        renamed = fleet[0].model_copy(update={"name": "Lathe Z"})

        updated = await machine_writes.update_one(renamed)

        assert updated == "m-1"
        update = weaviate_double.collection.data.update.await_args
        assert update is not None
        assert update.kwargs["vector"] is not None
        assert weaviate_double.by_id("m-1").properties["name"] == "Lathe Z"

    @staticmethod
    async def test__an_unchanged_text__keeps_the_stored_vector(
        weaviate_double: WeaviateDouble,
        machine_writes: WeaviateWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """Hours are not embedded: no vector is sent, so Weaviate keeps its own."""
        await machine_writes.insert_one(fleet[0])
        stored = weaviate_double.by_id("m-1").vector["default"]

        await machine_writes.update_one(fleet[0].model_copy(update={"hours": 40.0}))

        update = weaviate_double.collection.data.update.await_args
        assert update is not None
        assert update.kwargs["vector"] is None
        held = weaviate_double.by_id("m-1")
        assert (held.properties["hours"], held.vector["default"]) == (40.0, stored)

    @staticmethod
    async def test__a_missing_id__is_refused_before_any_write(
        weaviate_double: WeaviateDouble,
        machine_writes: WeaviateWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """Nothing of the batch is written when one document is not held."""
        await machine_writes.insert_one(fleet[0])
        renamed = fleet[0].model_copy(update={"name": "Lathe Z"})

        with pytest.raises(RecordNotFoundError, match="m-2"):
            await machine_writes.update_many([renamed, fleet[1]])

        weaviate_double.collection.data.update.assert_not_awaited()
        assert weaviate_double.by_id("m-1").properties["name"] == "Lathe A"


class TestUpsert:
    """An upsert inserts what is new and replaces what is held."""

    @staticmethod
    async def test__new_and_held__are_both_written(
        machine_reads: WeaviateReadRepository[Machine],
        machine_writes: WeaviateWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """The held one is replaced, the new one inserted."""
        await machine_writes.insert_one(fleet[0])
        renamed = fleet[0].model_copy(update={"name": "Lathe Z"})

        upserted = await machine_writes.upsert_many([renamed, fleet[1]])

        assert upserted == ["m-1", "m-2"]
        assert await machine_reads.get_by_id("m-1") == renamed
        assert await machine_reads.get_by_id("m-2") == fleet[1]

    @staticmethod
    async def test__an_id_given_twice__keeps_the_last(
        machine_reads: WeaviateReadRepository[Machine],
        machine_writes: WeaviateWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """The contract keeps the last of a repeated id, as the other backends do."""
        renamed = fleet[0].model_copy(update={"name": "Lathe Z"})

        upserted = await machine_writes.upsert_many([fleet[0], renamed])

        assert upserted == ["m-1", "m-1"]
        assert await machine_reads.get_by_id("m-1") == renamed


class TestDelete:
    """A delete removes documents by the UUIDs their id_ values derive."""

    @staticmethod
    async def test__one__is_deleted_by_its_id(
        weaviate_double: WeaviateDouble,
        machine_writes: WeaviateWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """The others stay."""
        await machine_writes.insert_many(fleet)

        deleted = await machine_writes.delete_one("m-2")

        assert deleted == "m-2"
        assert sorted(item.properties["id_"] for item in weaviate_double.objects.values()) == [
            "m-1",
            "m-3",
        ]

    @staticmethod
    async def test__many__return_their_ids(
        weaviate_double: WeaviateDouble,
        machine_writes: WeaviateWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """The id_ values, as a delete_one returns, not Weaviate's UUIDs."""
        await machine_writes.insert_many(fleet)

        deleted = await machine_writes.delete_many(["m-1", "m-3"])

        assert deleted == ["m-1", "m-3"]
        assert list(weaviate_double.objects) == [str(uuid.uuid5(uuid.NAMESPACE_DNS, "m-2"))]

    @staticmethod
    async def test__a_missing_id__is_refused_naming_its_uuid(
        machine_writes: WeaviateWriteRepository[Machine], fleet: list[Machine]
    ) -> None:
        """A document not deleted is an error, not a silent skip."""
        await machine_writes.insert_one(fleet[0])
        missing = str(uuid.uuid5(uuid.NAMESPACE_DNS, "m-9"))

        with pytest.raises(ValueError, match=f"Failed to delete 1/2.*{missing}"):
            await machine_writes.delete_many(["m-1", "m-9"])

    @staticmethod
    async def test__all__empties_the_collection(
        weaviate_double: WeaviateDouble,
        machine_writes: WeaviateWriteRepository[Machine],
        fleet: list[Machine],
    ) -> None:
        """Every id_ held comes back."""
        await machine_writes.insert_many(fleet)

        deleted = await machine_writes.delete_all()

        assert sorted(deleted) == ["m-1", "m-2", "m-3"]
        assert weaviate_double.objects == {}

    @staticmethod
    async def test__all_of_an_empty_collection__deletes_nothing(
        weaviate_double: WeaviateDouble, machine_writes: WeaviateWriteRepository[Machine]
    ) -> None:
        """No delete is sent."""
        assert await machine_writes.delete_all() == []
        weaviate_double.collection.data.delete_many.assert_not_awaited()
