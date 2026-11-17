from fastapi import status
from fastapi.testclient import TestClient

from baseline_common.models.record_matching import MatchRequest


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

    def test__a_blank_name__is_refused_in_french(self, client: TestClient) -> None:
        """The bounds of the request are checked before the service runs, in French."""
        response = client.post("/record-matching", json={"name": "  ", "candidates": []})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert response.json()["message"] == "La valeur est trop courte. (body → name)"

    def test__too_many_candidates__are_refused_in_french(self, client: TestClient) -> None:
        """A request comparing more candidates than allowed is refused before any comparison."""
        known = {"id_": "1", "name": "Atelier Dupuis"}
        request = {"name": "Dupuis", "candidates": [known] * (MatchRequest.CANDIDATES_MAX + 1)}

        response = client.post("/record-matching", json=request)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert response.json()["message"] == "La liste est trop longue. (body → candidates)"
