"""Gateway — reads and writes one record type of an HTTP API, through one object."""

from pydantic import BaseModel

from baseline_common.framework.repository.abstract_repository import AbstractRepository
from baseline_common.framework.repository.http.abstract_query_dialect import AbstractQueryDialect
from baseline_common.framework.repository.http.client.http_client import HttpClient
from baseline_common.framework.repository.http.read_gateway import ReadGateway
from baseline_common.framework.repository.http.write_gateway import (
    WriteGateway,
    WriteGatewayConfig,
)


class Gateway[T: BaseModel](WriteGateway[T], ReadGateway[T], AbstractRepository[T]):
    """Reads and writes one record type of an HTTP API; the model's aliases are the API's fields."""

    def __init__(
        self,
        client: HttpClient,
        content_type: type[T],
        dialect: AbstractQueryDialect,
        config: WriteGatewayConfig,
        *,
        primary_key: str = "id",
    ) -> None:
        """Bind the client, the model, how the API is queried and where it is written.

        Args:
            client: The client bound to the API.
            content_type: The model every record is read as and written from.
            dialect: How the API is queried.
            config: Where the API takes the writes, and how they are shaped.
            primary_key: The model field holding a record's id.
        """
        ReadGateway.__init__(self, client, content_type, dialect, primary_key=primary_key)
        WriteGateway.__init__(self, client, content_type, config)
