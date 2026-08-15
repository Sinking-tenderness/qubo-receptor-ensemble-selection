import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_stage32c_pparg_md_pair_failure import stable_ranks, transform


def test_transform_uses_frozen_training_cdf() -> None:
    reference = np.asarray([[1.0], [1.0], [3.0]])
    values = np.asarray([[0.0], [1.0], [2.0], [4.0]])
    assert np.allclose(transform(reference, values)[:, 0], [0.125, 0.375, 0.625, 0.875])


def test_stable_ranks_match_manifest_order_for_ties() -> None:
    assert np.array_equal(stable_ranks(np.asarray([2.0, 1.0, 1.0, 3.0])), [3, 1, 2, 4])


def test_stage32c_result_and_audit_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    result_path = root / "data/stage32c_pparg_md_pair_failure_diagnostic_result.json"
    audit_path = root / "data/stage32c_pparg_md_pair_failure_diagnostic_audit.json"
    if not result_path.exists() or not audit_path.exists():
        return
    result = json.loads(result_path.read_text(encoding="ascii"))
    audit = json.loads(audit_path.read_text(encoding="ascii"))
    assert result["mechanism"]["nonselective_decoy_promotion"] is True
    assert result["decision"]["stop_frozen_min_aggregation_md_pair_efficacy_route"] is True
    assert result["decision"]["locked_test_authorized"] is False
    assert audit["status"] == "stage32c_pparg_md_pair_failure_diagnostic_audit_ok"
    assert all(audit["checks"].values())
