"""CrmOrganization — an adapter reading an organization from a CRM account, as its API answers it."""

from typing import ClassVar

from pydantic import AliasPath, ConfigDict, Field, field_validator

from baseline_common.models.organization import Organization


class CrmOrganization(Organization):
    """An organization read from a CRM account: its id a number, its name nested, its address in parts."""

    model_config = ConfigDict(coerce_numbers_to_str=True)

    ADDRESS_LINE: ClassVar[list[str]] = ["Street", "Unit"]
    """The address parts written before the comma, in order."""

    LOCALITY: ClassVar[list[str]] = ["City", "Province", "Postal_Code"]
    """The address parts written after the comma, in order."""

    id_: str = Field(validation_alias="id")
    name: str = Field(validation_alias=AliasPath("Account", "Legal_Name"))
    address: str | None = Field(None, validation_alias="Billing_Address")

    @field_validator("address", mode="before")
    @classmethod
    def join_address(cls, parts: dict[str, str | None] | None) -> str | None:
        """Write the address the CRM keeps in parts on one line; None when no part is filled.

        Args:
            parts: The CRM's billing address, part by part; None when it has none.

        Returns:
            The address on one line, e.g. ``12 rue des Forges, Trois-Rivières QC G9A 1A1``.
        """
        if not parts:
            return None

        groups = [
            " ".join(part for name in names if (part := parts.get(name)))
            for names in [cls.ADDRESS_LINE, cls.LOCALITY]
        ]

        return ", ".join(group for group in groups if group) or None
