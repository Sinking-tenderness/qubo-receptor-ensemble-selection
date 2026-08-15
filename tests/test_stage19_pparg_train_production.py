import json
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.experimental.unidock.build_stage19c_pparg_train668_production_bundle import (
    CONFIG,
    bundle_paths,
)


def test_stage19b_train_inputs_are_complete_and_protected() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = json.loads(
        (root / "data/stage19b_pparg_train668_unidock_input_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["status"] == "stage19b_pparg_train668_unidock_inputs_ok"
    assert summary["ligand_count"] == 668
    assert summary["label_counts"] == {"active": 334, "decoy": 334}
    assert summary["failed_ligand_count"] == 0
    assert summary["closure_pseudoatom_ligand_count"] == 0
    assert summary["data_boundary"]["fresh_validation_rows_read"] == 0
    assert summary["data_boundary"]["test_rows_read"] == 0


def test_isolated_stage19c_bundle_supports_audit_only(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    paths = bundle_paths(root)
    lowered = [path.lower() for path in paths]
    assert not any("fresh_validation" in path for path in lowered)
    assert not any("locked_test" in path for path in lowered)
    assert not any("selected_ligand_panel_manifest" in path for path in lowered)

    for relative in paths:
        source = root / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.experimental.unidock.run_stage19c_pparg_train668_production",
            "--config",
            CONFIG,
            "--audit-only",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "audit_only_ok"
    assert result["target_id"] == "PPARG"
    assert result["receptor_count"] == 16
    assert result["ligand_count"] == 668
    assert result["expected_batch_count"] == 48
    assert result["expected_pair_count"] == 32064
    assert result["validation_rows"] == 0
    assert result["test_rows"] == 0
    assert result["stage18e_confirmatory_gate"] == "closed_failed_14_of_24"
