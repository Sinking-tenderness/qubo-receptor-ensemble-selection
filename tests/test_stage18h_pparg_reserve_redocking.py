import json
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.experimental.unidock.build_stage18h_pparg_reserve_redocking_bundle import (
    CONFIG,
    bundle_paths,
)
from scripts.experimental.unidock.run_stage18h_pparg_reserve_redocking import (
    summarize_gate,
)


def test_reserve_gate_requires_two_successes_and_median_threshold() -> None:
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


def test_isolated_bundle_supports_posthoc_audit_only(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    paths = bundle_paths(root)
    lowered = [path.lower() for path in paths]
    assert not any("fresh_validation" in path for path in lowered)
    assert not any("stage09_mk14" in path for path in lowered)
    assert not any("3wj4_tby" in path for path in lowered)

    for relative in paths:
        source = root / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.experimental.unidock.run_stage18h_pparg_reserve_redocking",
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
    assert result["experiment_class"] == "posthoc_exploratory_reserve_recovery"
    assert result["stage18e_confirmatory_gate"] == "closed_failed_14_of_24"
    assert result["receptor_count"] == 8
    assert result["case_count"] == 8
    assert result["seed_count"] == 3
    assert result["expected_redocking_pair_count"] == 24
    assert result["ligand_labels_read"] == 0
    assert result["benchmark_docking_scores_read"] == 0
