"""The versions the API reports and is mounted under."""

from importlib.metadata import version
from typing import ClassVar


class ApiVersion:
    """The release of the package serving the API, and the version its schema is mounted under."""

    APP: ClassVar[str] = version(__name__.split(".", maxsplit=1)[0])
    """Version of the package serving the API, read from whatever package this code was copied into."""
    SCHEMA: ClassVar[str] = "v1"
    """Version of the schema the API serves; bumped only on a breaking change, apart from APP."""
