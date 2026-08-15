import json
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.experimental.unidock.build_stage51_ppara_large_pool_cognate_redocking_bundle import (
    CONFIG,
    bundle_paths,
)
from scripts.experimental.unidock.run_stage51_ppara_large_pool_cognate_redocking import (
    common,
    validate_inputs,
)


ROOT = Path(__file__).resolve().parents[1]


def test_stage50_independent_audit_authorizes_redocking():
    audit = json.loads(
        (ROOT / "data/stage50_ppara_large_pool_inputs_independent_audit.json").read_text()
    )
    assert audit["status"] == "stage50_ppara_large_pool_inputs_independent_audit_ok"
    assert audit["prepared_receptor_count"] == 60
    assert audit["technical_preparation_failure_count"] == 4
    assert audit["maximum_existing_atom_displacement_angstrom"] == 0.0
    assert audit["cognate_redocking_authorized"] is True


def test_stage51_audit_freezes_complete_three_seed_grid():
    config = common.read_json(ROOT / "configs/stage51_ppara_large_pool_cognate_redocking.json")
    receptors, cases, audit = validate_inputs(ROOT, config)
    assert len(receptors) == len(cases) == 60
    assert audit["frozen_receptor_count"] == 64
    assert audit["technical_preparation_failure_count"] == 4
    assert audit["seed_count"] == 3
    assert audit["expected_redocking_pair_count"] == 180
    assert audit["ligand_labels_read"] == 0
    assert audit["fresh_validation_rows_read"] == 0


def test_stage51_bundle_and_runner_guards():
    summary = json.loads(
        (
            ROOT
            / "data/stage51a_ppara_large_pool_cognate_redocking_amendment01_bundle_summary.json"
        ).read_text()
    )
    runner = (
        ROOT
        / "scripts/experimental/unidock/run_stage51_ppara_large_pool_cognate_redocking_remote.sh"
    ).read_text()
    assert summary["status"] == "ok"
    assert summary["gpu_pair_count"] == 180
    assert summary["gpu_required_for_execution"] is True
    assert "AUTO_POWEROFF" in runner
    assert "--resume" in runner


def test_stage51_independent_audit_preserves_failed_gate() -> None:
    audit = json.loads(
        (
            ROOT
            / "data/stage51_ppara_large_pool_cognate_redocking_independent_audit.json"
        ).read_text()
    )
    assert audit["status"] == "stage51_ppara_large_pool_gate_failure_independently_confirmed"
    assert audit["execution_integrity"]["completed_batch_count"] == 180
    assert audit["execution_integrity"]["audit_error_count"] == 0
    assert audit["gate"]["passed_receptor_count"] == 20
    assert audit["gate"]["minimum_passing_receptor_count"] == 24
    assert audit["gate"]["technical_gate_pass"] is False
    assert audit["decision"]["confirmatory_development_panel_docking_authorized"] is False


def test_stage51_isolated_bundle_supports_audit_only(tmp_path: Path) -> None:
    paths = bundle_paths(ROOT)
    assert "scripts/prepare_receptor.py" in paths

    for relative in paths:
        source = ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.experimental.unidock.run_stage51_ppara_large_pool_cognate_redocking",
            "--config",
            CONFIG,
            "--root",
            ".",
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
    assert result["target_id"] == "PPARA"
    assert result["receptor_count"] == 60
    assert result["expected_redocking_pair_count"] == 180
    assert result["fresh_validation_rows_read"] == 0
    assert result["test_rows_read"] == 0
