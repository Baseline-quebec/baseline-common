"""Resilience settings of one HttpClient: timeouts, retries and connection pooling."""

from pydantic import BaseModel, ConfigDict, Field


class HttpConfig(BaseModel):
    """Timeouts, retries and pooling of one HttpClient."""

    model_config = ConfigDict(frozen=True)
    """Frozen: a client keeps the settings it was built with."""

    connect_timeout: float = 5.0
    """Seconds to open a connection."""
    read_timeout: float = 30.0
    """Seconds to wait for each chunk of an answer."""
    write_timeout: float = 10.0
    """Seconds to send each chunk of a request."""
    pool_timeout: float = 5.0
    """Seconds to wait for a free connection in the pool."""

    retry_total: int = Field(default=3, ge=0)
    """Attempts after the first one; 0 sends every request once."""

    retry_backoff: float = Field(default=0.5, ge=0.0)
    """Base delay in seconds, doubled at every attempt."""

    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)
    """Statuses worth a second attempt: the server is busy or briefly down."""

    retry_methods: tuple[str, ...] = ("HEAD", "GET", "PUT", "DELETE", "OPTIONS")
    """Idempotent methods only: a POST retried after a lost response may be applied twice."""

    respect_retry_after: bool = True
    """Whether a retry waits as long as the answer's ``Retry-After`` asks."""
    retry_after_cap: float = Field(default=60.0, ge=0.0)
    """Longest ``Retry-After`` honored, so a rate limit cannot park a worker for an hour."""

    max_connections: int = 20
    """Connections open at once, at most."""
    max_keepalive_connections: int = 10
    """Idle connections kept for reuse, at most."""
    keepalive_expiry: float = 30.0
    """Seconds an idle connection is kept."""

    raise_for_status: bool = True
    """Whether a 4xx or 5xx answer raises ``httpx.HTTPStatusError`` by default."""
