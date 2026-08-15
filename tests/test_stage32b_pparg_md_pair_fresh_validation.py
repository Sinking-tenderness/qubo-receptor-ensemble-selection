import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_stage32b_pparg_md_pair_fresh_validation import enrichment_factor, transform_with_reference
from scripts.experimental.unidock.run_stage32b_pparg_md_pair_fresh_validation import validate_config
from scripts.freeze_stage32b_pparg_fresh_validation_inputs import identity_sha256


def test_frozen_cdf_transform_uses_training_reference_only() -> None:
    reference = np.asarray([[1.0], [1.0], [3.0]])
    validation = np.asarray([[0.0], [1.0], [2.0], [4.0]])
    assert np.allclose(transform_with_reference(reference, validation)[:, 0], [0.125, 0.375, 0.625, 0.875])


def test_enrichment_factor_recovers_perfect_top_one_percent() -> None:
    labels = np.asarray([1] * 10 + [0] * 90)
    scores = np.arange(100, dtype=float)
    assert np.isclose(enrichment_factor(scores, labels, 0.01), 10.0)


def test_stage32b_frozen_counts_and_boundaries() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/stage32b_pparg_md_pair_fresh_validation.json").read_text(encoding="ascii"))
    validate_config(config)
    assert config["expected"] == {"receptor_count": 2, "ligand_count": 1576, "seed_count": 3, "batch_count": 6, "score_row_count": 9456, "locked_test_rows": 0}
    assert config["confirmation_analysis"]["confirmation_gate"]["minimum_primary_bedroc20_gain"] == 0.02
    assert config["evidence_timing"]["fresh_validation_docking_scores_known_before_freeze"] is False
    assert config["evidence_timing"]["locked_test_rows_permitted"] is False


def test_stage32b_selection_and_validation_identity_are_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    selection = json.loads((root / "data/stage32b_pparg_md_pair_train_selection.json").read_text(encoding="ascii"))
    assert selection["selected_singleton"]["receptor_ids"] == ["PPARG_MD_01124_2I4J"]
    assert selection["selected_pair"]["receptor_ids"] == ["PPARG_MD_00524_2ATH", "PPARG_MD_01124_2I4J"]
    assert selection["train_pair_minus_single_robust"] > 0.02
    rows = list(__import__("csv").DictReader((root / "data/processed/stage32b_pparg_fresh_validation_source_manifest.csv").open(encoding="utf-8")))
    assert len(rows) == 1576
    assert Counter(row["label"] for row in rows) == Counter({"active": 75, "decoy": 1501})
    assert {row["selection_role"] for row in rows} == {"fresh_validation"}
    assert identity_sha256([row["ligand_id"] for row in rows]) == "E1CD98D5833FCC150D654E6681D7E63A3156DD7E8C04A3676823E3B978A2265C"


def test_stage32b_result_and_audit_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    result_path = root / "data/stage32b_pparg_md_pair_fresh_validation_result.json"
    audit_path = root / "data/stage32b_pparg_md_pair_fresh_validation_audit.json"
    if not result_path.exists() or not audit_path.exists():
        return
    result = json.loads(result_path.read_text(encoding="ascii"))
    audit = json.loads(audit_path.read_text(encoding="ascii"))
    assert result["confirmation_gate"]["passed"] is False
    assert result["decision"]["locked_test_authorized"] is False
    assert audit["status"] == "stage32b_pparg_md_pair_fresh_validation_audit_ok"
    assert all(audit["checks"].values())
