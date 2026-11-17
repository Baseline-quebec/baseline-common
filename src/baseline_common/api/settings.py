"""API-level settings loaded from environment variables and ``.env``."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Top-level configuration of the API."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    app_name: str = "api"
    """The app's name: the title of its schema, and what its health check answers."""
    log_level: str = "INFO"
    """The lowest level the app logs."""

    api_host: str = "0.0.0.0"  # noqa: S104 - bind all interfaces for the containerised API
    """The interface the server binds."""
    api_port: int = 8000
    """The port the server listens on."""


@lru_cache
def get_api_settings() -> Settings:
    """Return the cached application settings.

    Returns:
        Settings loaded from the environment and the ``.env`` file.
    """
    return Settings()
