import csv
import json
from pathlib import Path


def test_stage18i_final16_is_unique_and_structurally_selected() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = json.loads(
        (root / "data/stage18i_pparg_exploratory_final16_selection_summary.json").read_text(
            encoding="utf-8"
        )
    )
    with (
        root / "data/processed/stage18i_pparg_exploratory_final16_prepared_receptor_manifest.csv"
    ).open(encoding="utf-8-sig", newline="") as handle:
        prepared = list(csv.DictReader(handle))

    assert summary["status"] == "stage18i_pparg_exploratory_final16_selected"
    assert summary["stage18e_confirmatory_gate"] == "closed_failed_14_of_24"
    assert summary["stage18h_exploratory_recovery_gate"] == "passed_7_of_8"
    assert [row["conformer_id"] for row in summary["selected_recovery_additions"]] == [
        "PPARG_2P4Y_aligned",
        "PPARG_3FUR_aligned",
    ]
    assert len(summary["final_receptor_ids"]) == 16
    assert len(set(summary["final_receptor_ids"])) == 16
    assert [row["conformer_id"] for row in prepared] == summary["final_receptor_ids"]
    assert all(row["status"] == "ok" for row in prepared)
    assert summary["data_use_audit"]["redocking_rmsd_magnitudes_used_for_selection"] is False
    assert summary["data_use_audit"]["benchmark_docking_scores_read"] == 0
