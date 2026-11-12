"""The versions the API reports and is mounted under."""

from importlib.metadata import version
from typing import ClassVar


class ApiVersion:
    """The release of the package serving the API, and the version its schema is mounted under."""

    # Read from the installed package's metadata, whatever package this code was copied into.
    APP: ClassVar[str] = version(__name__.split(".", maxsplit=1)[0])
    # Bumped only on a breaking API change; decoupled from APP.
    SCHEMA: ClassVar[str] = "v1"
