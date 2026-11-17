from fastapi import status
from fastapi.testclient import TestClient


class TestMatch:
    def test__a_written_name__is_matched_to_its_candidate(self, client: TestClient) -> None:
        """The route resolves the service from the container and answers its decision."""
        request = {
            "name": "ACME Ind. Inc. (MTL)",
            "candidates": [{"id_": "7", "name": "Acme Ind"}, {"id_": "8", "name": "Beta"}],
        }

        response = client.post("/record-matching", json=request)

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"verdict": "match", "record_ids": ["7"]}

    def test__the_injected_service__is_not_a_request_parameter(self, client: TestClient) -> None:
        """A dependency from the container never shows in the schema as a parameter."""
        operation = client.get("/openapi.json").json()["paths"]["/record-matching"]["post"]

        assert "parameters" not in operation
