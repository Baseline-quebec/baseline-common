"""Resilience settings of one HttpClient: timeouts, retries and connection pooling."""

from pydantic import BaseModel, ConfigDict, Field


class HttpConfig(BaseModel):
    """Timeouts, retries and pooling of one HttpClient."""

    model_config = ConfigDict(frozen=True)

    connect_timeout: float = 5.0
    read_timeout: float = 30.0
    write_timeout: float = 10.0
    pool_timeout: float = 5.0

    retry_total: int = Field(default=3, ge=0)
    """Attempts after the first one; 0 sends every request once."""

    retry_backoff: float = Field(default=0.5, ge=0.0)
    """Base delay in seconds, doubled at every attempt."""

    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)
    """Statuses worth a second attempt: the server is busy or briefly down."""

    retry_methods: tuple[str, ...] = ("HEAD", "GET", "PUT", "DELETE", "OPTIONS")
    """Idempotent methods only: a POST retried after a lost response may be applied twice."""

    respect_retry_after: bool = True
    retry_after_cap: float = Field(default=60.0, ge=0.0)
    """Longest ``Retry-After`` honored, so a rate limit cannot park a worker for an hour."""

    max_connections: int = 20
    max_keepalive_connections: int = 10
    keepalive_expiry: float = 30.0

    raise_for_status: bool = True
    """Whether a 4xx or 5xx answer raises ``httpx.HTTPStatusError`` by default."""
