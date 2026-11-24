from typing import Any

import pytest

from baseline_common.models.record_matching import MatchRequest, MatchVerdict
from baseline_common.services.record_matching.crm_adapter import CrmOrganization
from baseline_common.services.record_matching.record_matching_service import RecordMatchingService


@pytest.fixture
def account() -> dict[str, Any]:
    """Return a CRM account as the CRM's API answers it.

    Returns:
        The account's JSON body.
    """
    return {
        "id": 4021,
        "Account": {"Legal_Name": "Atelier Dupuis inc.", "Trade_Name": "Dupuis"},
        "Billing_Address": {
            "Street": "12 rue des Forges",
            "Unit": None,
            "City": "Trois-Rivières",
            "Province": "QC",
            "Postal_Code": "G9A 1A1",
        },
    }


class TestCrmOrganization:
    def test__when_a_crm_account_is_parsed__is_read_in_the_domain_s_fields(
        self, account: dict[str, Any]
    ) -> None:
        """The id is read as text, the name from its nested path, the address as one line."""
        candidate = CrmOrganization.model_validate(account)

        assert candidate.id_ == "4021"
        assert candidate.name == "Atelier Dupuis inc."
        assert candidate.address == "12 rue des Forges, Trois-Rivières QC G9A 1A1"

    def test__when_an_address_has_no_locality__is_its_street_line(
        self, account: dict[str, Any]
    ) -> None:
        """A missing part is left out, with the comma that would precede it."""
        account["Billing_Address"] = {"Street": "12 rue des Forges", "Unit": "B"}

        candidate = CrmOrganization.model_validate(account)

        assert candidate.address == "12 rue des Forges B"

    @pytest.mark.parametrize("address", [None, {}, {"Street": None, "City": ""}])
    def test__when_an_address_is_empty__is_none(
        self, account: dict[str, Any], address: dict[str, str | None] | None
    ) -> None:
        """An address with no part filled is no address."""
        account["Billing_Address"] = address

        candidate = CrmOrganization.model_validate(account)

        assert candidate.address is None

    def test__when_a_crm_organization_is_a_candidate__the_service_matches_it(
        self, account: dict[str, Any]
    ) -> None:
        """The service reads the adapter's organizations without knowing the CRM."""
        request = MatchRequest(
            name="ATELIER DUPUIS (TR)", candidates=[CrmOrganization.model_validate(account)]
        )

        decision = RecordMatchingService().match(request)

        assert decision.verdict is MatchVerdict.MATCH
        assert decision.record_ids == ["4021"]
