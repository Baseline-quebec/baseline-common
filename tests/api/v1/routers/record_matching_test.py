from fastapi import status
from fastapi.testclient import TestClient

from baseline_common.models.record_matching import MatchRequest


class TestMatch:
    def test__when_a_written_name_is_posted__returns_its_matching_candidate(
        self, client: TestClient
    ) -> None:
        """The route resolves the service from the container and answers its decision."""
        request = {
            "name": "ACME Ind. Inc. (MTL)",
            "candidates": [{"id_": "7", "name": "Acme Ind"}, {"id_": "8", "name": "Beta"}],
        }

        response = client.post("/record-matching", json=request)

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"verdict": "match", "record_ids": ["7"]}

    def test__when_the_service_is_injected__returns_a_schema_without_parameters(
        self, client: TestClient
    ) -> None:
        """A dependency from the container never shows in the schema as a parameter."""
        operation = client.get("/openapi.json").json()["paths"]["/record-matching"]["post"]

        assert "parameters" not in operation

    def test__when_a_name_is_blank__returns_a_422_in_french(self, client: TestClient) -> None:
        """The bounds of the request are checked before the service runs, in French."""
        response = client.post("/record-matching", json={"name": "  ", "candidates": []})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert response.json()["message"] == "La valeur est trop courte. (body → name)"

    def test__when_too_many_candidates_are_sent__returns_a_422_in_french(
        self, client: TestClient
    ) -> None:
        """A request comparing more candidates than allowed is refused before any comparison."""
        known = {"id_": "1", "name": "Atelier Dupuis"}
        request = {"name": "Dupuis", "candidates": [known] * (MatchRequest.CANDIDATES_MAX + 1)}

        response = client.post("/record-matching", json=request)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert response.json()["message"] == "La liste est trop longue. (body → candidates)"
