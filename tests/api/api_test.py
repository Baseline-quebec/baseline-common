from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from baseline_common.api.api import Api
from baseline_common.api.api_version import ApiVersion
from baseline_common.api.settings import Settings


class TestApi:
    def test__the_root_app__boots_the_mounted_version_and_its_container(self) -> None:
        """A mount never runs its own lifespan: the root enters it, so v1 resolves from Dishka."""
        app = Api(Settings(app_name="probe"))

        with TestClient(app) as client:
            response = client.get(f"/{ApiVersion.SCHEMA}/health")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"status": "ok", "name": "probe", "version": ApiVersion.APP}

    def test__a_frontend_directory__is_served_at_the_root_and_revalidated(
        self, tmp_path: Path
    ) -> None:
        """The UI answers at the root, told to revalidate, and the API stays under its version."""
        (tmp_path / "index.html").write_text("<p>ui</p>")
        app = Api(Settings(app_name="probe", frontend_directory=tmp_path))

        with TestClient(app) as client:
            page = client.get("/")
            health = client.get(f"/{ApiVersion.SCHEMA}/health")

        assert page.text == "<p>ui</p>"
        assert page.headers["Cache-Control"] == "no-cache"
        assert health.status_code == status.HTTP_200_OK

    def test__no_frontend_directory__serves_the_api_alone(self) -> None:
        """Without a UI, the root answers 404 rather than a page."""
        app = Api(Settings(app_name="probe"))

        with TestClient(app) as client:
            response = client.get("/")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test__a_missing_frontend_directory__is_refused_at_construction(
        self, tmp_path: Path
    ) -> None:
        """A UI named but absent fails when the app is built, not on the first page asked."""
        settings = Settings(frontend_directory=tmp_path / "missing")

        with pytest.raises(RuntimeError, match="does not exist"):
            Api(settings)
