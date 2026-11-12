"""API-level settings loaded from environment variables and ``.env``."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Top-level configuration of the API."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    app_name: str = "api"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"  # noqa: S104 - bind all interfaces for the containerised API
    api_port: int = 8000

    frontend_directory: Path | None = None
    """A built web UI served at the root; None serves the API alone."""

    validation_locations: dict[str, str] = {
        "body": "corps de la requête",
        "body.file": "fichier",
        "body.files": "fichiers",
    }
    """The French label of a request location, keyed by its dotted path; a project adds its own."""
