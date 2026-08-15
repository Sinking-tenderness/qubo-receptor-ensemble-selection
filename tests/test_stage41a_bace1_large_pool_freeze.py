from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.freeze_stage41a_bace1_large_pool import portable_source_path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="ascii"))


def load_manifest() -> list[dict[str, str]]:
    path = ROOT / "data/processed/stage41a_bace1_large_pool_manifest.csv"
    with path.open(newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


def test_portable_source_path_rebases_historical_checkout() -> None:
    relative = "data/stage21c_bace1_structural_selection_summary.json"
    historical = f"D:/old/location/qubo-receptor-ensemble-selection/{relative}"
    assert portable_source_path(ROOT, historical) == ROOT / relative


def test_stage41a_freezes_all_preparation_ready_structures() -> None:
    result = load_json("data/stage41a_bace1_large_pool_freeze_result.json")
    assert result["status"] == "stage41a_bace1_large_pool_frozen"
    assert result["counts"]["preparation_ready_count"] == 49
    assert result["counts"]["frozen_receptor_count"] == 49
    assert result["counts"]["total_state_count_k1_to_k6"] == 16_122_225
    assert result["counts"]["state_count_by_k"]["3"] == 18_424
    assert result["development_ligand_protocol"]["maximum_pair_count_if_all_49_pass"] == 39_102
    assert all(
        result["data_boundary"][key] == 0
        for key in (
            "development_ligand_rows_read",
            "fresh_validation_rows_read",
            "locked_test_rows_read",
            "docking_scores_read",
            "new_docking_jobs",
            "quantum_hardware_jobs",
        )
    )


def test_stage41a_manifest_is_complete_portable_and_reference_first() -> None:
    rows = load_manifest()
    assert len(rows) == 49
    assert len({row["conformer_id"] for row in rows}) == 49
    assert rows[0]["pdb_id"] == "3L5D"
    assert sum(row["is_reference"] == "True" for row in rows) == 1
    assert all(not Path(row["mmcif_path"]).is_absolute() for row in rows)
    assert all(not Path(row["aligned_protein_pdb_path"]).is_absolute() for row in rows)


def test_stage41a_independent_audit_passes() -> None:
    audit = load_json("data/stage41a_bace1_large_pool_freeze_audit.json")
    assert audit["status"] == "stage41a_bace1_large_pool_freeze_audit_ok"
    assert all(audit["checks"].values())
    assert len(audit["coordinate_file_checks"]) == 49
    assert all(audit["coordinate_file_checks"].values())
