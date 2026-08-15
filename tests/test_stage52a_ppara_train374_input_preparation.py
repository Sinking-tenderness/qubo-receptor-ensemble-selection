import json
from collections import Counter
from pathlib import Path

from scripts.experimental.unidock.build_stage52a_ppara_train374_input_bundle import (
    bundle_paths,
)
from scripts.experimental.unidock.prepare_development_ligand_inputs import (
    read_csv,
    row_signature,
    run,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage52a_ppara_train374_unidock_input_preparation.json"


def test_stage52_preregistration_freezes_complete_exploratory_grid():
    prereg = json.loads(
        (
            ROOT
            / "configs/stage52_ppara_posthoc_exploratory_development_preregistration.json"
        ).read_text()
    )
    assert prereg["authorization"]["stage51_confirmatory_gate_pass"] is False
    assert prereg["frozen_receptors"]["receptor_count"] == 20
    assert len(prereg["frozen_receptors"]["receptor_ids"]) == 20
    assert prereg["development_panel"]["ligand_count"] == 374
    assert prereg["production"]["seed_count"] == 3
    assert prereg["production"]["expected_receptor_ligand_seed_count"] == 22440
    assert prereg["future_method_comparison"]["bedroc_alpha"] == 20.0


def test_stage52a_audit_only_exposes_only_train_panel():
    result = run(CONFIG, ROOT, audit_only=True, resume=False, overwrite=False)
    assert result["status"] == "audit_only_ok"
    assert result["ligand_count"] == 374
    assert result["label_counts"] == {"active": 187, "decoy": 187}
    assert result["fresh_validation_rows_read"] == 0
    assert result["locked_test_rows_read"] == 0
    assert result["future_pair_count"] == 22440


def test_stage52a_bundle_excludes_protected_splits_and_supports_resume():
    paths = bundle_paths(ROOT)
    lowered = [path.lower() for path in paths]
    assert not any("fresh_validation" in path for path in lowered)
    assert not any("locked_test" in path for path in lowered)
    assert not any("data/protected" in path for path in lowered)
    runner = (
        ROOT
        / "scripts/experimental/unidock/run_stage52a_ppara_train374_input_preparation_remote.sh"
    ).read_text()
    assert "--resume" in runner
    assert "--audit-only" in runner
    assert "AUTO_POWEROFF" in runner


def test_stage52a_row_signature_is_identity_and_config_bound():
    row = read_csv(ROOT / "data/processed/stage49_ppara_train374_ligand_manifest.csv")[0]
    first = row_signature(row, "A" * 64)
    assert first == row_signature(row, "A" * 64)
    assert first != row_signature(row, "B" * 64)
    changed = dict(row)
    changed["canonical_smiles"] += "C"
    assert first != row_signature(changed, "A" * 64)
    rows = read_csv(ROOT / "data/processed/stage49_ppara_train374_ligand_manifest.csv")
    assert Counter(row["label"] for row in rows) == Counter(active=187, decoy=187)


def test_stage52a_independent_audit_authorizes_exploratory_production():
    audit = json.loads(
        (
            ROOT
            / "data/stage52a_ppara_train374_unidock_inputs_independent_audit.json"
        ).read_text()
    )
    assert audit["status"] == "stage52a_ppara_train374_inputs_independent_audit_ok"
    assert audit["ligand_count"] == 374
    assert audit["valid_checkpoint_count"] == 374
    assert audit["invalid_pdbqt_count"] == 0
    assert audit["closure_pseudoatom_ligand_count"] == 0
    assert audit["decision"]["stage52b_exploratory_production_authorized"] is True
    assert audit["decision"]["stage51_confirmatory_status_changed"] is False
