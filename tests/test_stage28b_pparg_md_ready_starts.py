import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_openmm_equilibration_smoke import load_smoke_config


ROOT = Path(__file__).resolve().parents[1]


def rows(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_stage28b_preflight_selection_is_complete_and_label_independent() -> None:
    summary = json.loads(
        (ROOT / "data/stage28b_pparg_md_ready_start_selection_summary.json").read_text(encoding="ascii")
    )
    assert summary["status"] == "stage28b_pparg_md_ready_start_selection_ok"
    assert summary["counts"] == {
        "coordinate_eligible": 157,
        "md_ineligible": 145,
        "md_ready": 12,
        "md_ready_distance_pairs": 66,
        "selected": 8,
    }
    assert summary["seed_conformer_id"] == "PPARG_2HFP_aligned"
    assert summary["quarantined_stage28_conformer_ids"] == ["PPARG_2GTK_reference"]
    assert all(value == 0 for value in summary["data_boundary"].values())


def test_stage28b_selected_starts_pass_every_topology_gate() -> None:
    selected = rows("data/processed/stage28b_pparg_md_ready_selected8.csv")
    audit = {row["conformer_id"]: row for row in rows("data/processed/stage28b_pparg_md_ready_preflight_audit.csv")}
    assert [row["conformer_id"] for row in selected] == [
        "PPARG_2HFP_aligned",
        "PPARG_8CPI_aligned",
        "PPARG_3TY0_aligned",
        "PPARG_2ATH_aligned",
        "PPARG_3D6D_aligned",
        "PPARG_2PRG_aligned",
        "PPARG_3GBK_aligned",
        "PPARG_2I4J_aligned",
    ]
    for row in selected:
        record = audit[row["conformer_id"]]
        assert record["status"] == "md_ready"
        assert record["exclusion_reasons"] == ""
        assert int(record["internal_gap_count"]) == 0
        assert 1.1 <= float(record["minimum_adjacent_peptide_cn_distance_angstrom"])
        assert float(record["maximum_adjacent_peptide_cn_distance_angstrom"]) <= 1.7
        assert record["last_residue_missing_atoms"] == ""


def test_stage28b_uses_an_isolated_run_root_and_valid_openmm_configs() -> None:
    config = json.loads(
        (ROOT / "configs/stage28b_pparg_md_ready_multistart_md_ensemble.json").read_text(encoding="ascii")
    )
    starts = rows(config["runtime"]["start_manifest"])
    assert config["runtime"]["run_root"].startswith("results/runs/stage28b_")
    assert config["amendment"]["quarantined_frames_permitted"] is False
    assert len(starts) == 8
    assert all("stage28_pparg_multistart_md/" not in row["system_manifest"] for row in starts)
    for row in starts:
        equilibration = load_smoke_config(ROOT / row["equilibration_config"])
        production = load_smoke_config(ROOT / row["production_config"])
        assert str(equilibration["experiment_id"]).startswith("stage28b-pparg-md-ready-")
        assert str(production["experiment_id"]).startswith("stage28b-pparg-md-ready-")
