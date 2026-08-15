import json
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.experimental.unidock.build_stage14d_fa10_cognate_redocking_bundle import (
    CONFIG,
    bundle_paths,
)
from scripts.experimental.unidock.run_stage14d_fa10_cognate_redocking import (
    summarize_gate,
    truth,
)


def test_truth_accepts_json_and_csv_boolean_values() -> None:
    assert truth(True)
    assert truth("True")
    assert not truth(False)
    assert not truth("False")


def test_receptor_gate_requires_two_successes_and_median_threshold() -> None:
    rows = [
        {
            "conformer_id": "R1",
            "top_ranked_rmsd_angstrom": rmsd,
            "top_ranked_pose_success": success,
        }
        for rmsd, success in ((1.0, True), (1.8, "True"), (4.0, False))
    ]

    result = summarize_gate(rows, ["R1"], 2.0, 2)[0]

    assert result["successful_seed_count"] == 2
    assert result["median_top_ranked_rmsd_angstrom"] == 1.8
    assert result["gate_pass"] is True


def test_isolated_bundle_file_set_supports_audit_only(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    paths = bundle_paths(root)
    assert not any("fresh_validation" in path.lower() for path in paths)
    assert not any("stage13f_egfr" in path.lower() for path in paths)

    for relative in paths:
        source = root / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.experimental.unidock.run_stage14d_fa10_cognate_redocking",
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
    assert result["target_id"] == "FA10"
    assert result["receptor_count"] == 16
    assert result["case_count"] == 16
    assert result["seed_count"] == 3
    assert result["expected_redocking_pair_count"] == 48
    assert result["ligand_labels_read"] == 0
    assert result["benchmark_docking_scores_read"] == 0
