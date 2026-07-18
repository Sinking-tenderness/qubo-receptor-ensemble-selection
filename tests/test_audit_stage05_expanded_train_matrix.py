import pytest

from scripts.audit_stage05_expanded_train_matrix import verify_seed_evidence


def test_verify_seed_evidence_rejects_wrong_count() -> None:
    with pytest.raises(ValueError, match="evidence count differs"):
        verify_seed_evidence([], 3, 1280, 160, 8, 0)
