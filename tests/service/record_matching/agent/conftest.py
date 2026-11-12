"""A written organization and two candidates near it."""

import pytest

from baseline_core.service.record_matching.agent.deps import (
    CandidateRecord,
    RecordMatchingDeps,
    WrittenRecord,
)


@pytest.fixture
def deps() -> RecordMatchingDeps:
    """ACME as an invoice wrote it, and two organizations at the same address."""
    return RecordMatchingDeps(
        written=WrittenRecord(name="ACME IND. (MTL)", code="C-0042", address="10 rue du Port"),
        candidates=[
            CandidateRecord(id_="org-1", name="Acme Industries inc.", address="10 rue du Port"),
            CandidateRecord(id_="org-2", name="Port Logistique", address="10 rue du Port"),
        ],
    )
