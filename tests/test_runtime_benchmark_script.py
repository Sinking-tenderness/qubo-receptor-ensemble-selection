from __future__ import annotations

import re
import shutil
import subprocess
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_adaptive_vs_exhaustive_runtime_benchmark.sh"

EXPECTED_CONFIGS = {
    "MK14": {
        "adaptive": "mk14_adaptive_rp09_remote.json",
        "fixed": [f"mk14_fixed_k{k}_remote.json" for k in range(1, 7)],
    },
    "PPARG": {
        "adaptive": "pparg_adaptive_remote.json",
        "fixed": [f"pparg_fixed_k{k}_remote.json" for k in range(1, 7)],
    },
    "BACE1": {
        "adaptive": "bace1_adaptive_remote.json",
        "fixed": [f"bace1_fixed_k{k}_remote.json" for k in range(1, 7)],
    },
    "ESR1": {
        "adaptive": "esr1_adaptive_remote.json",
        "fixed": [f"esr1_fixed_k{k}_remote.json" for k in range(1, 7)],
    },
    "PPARA": {
        "adaptive": "ppara_adaptive_remote.json",
        "fixed": [f"ppara_fixed_k{k}_remote.json" for k in range(1, 7)],
    },
}


def _config_specs(script_text: str) -> list[tuple[str, str, str]]:
    return re.findall(
        r'"([A-Z0-9]+)\|(adaptive|fixed)\|(configs/experiments/[^" ]+)"',
        script_text,
    )


def test_runtime_script_freezes_the_35_configuration_comparison() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    specs = _config_specs(text)

    assert len(specs) == 35
    assert "bace1_adaptive_diag_remote.json" not in text
    assert "set -euo pipefail" in text
    assert "/usr/bin/time" in text
    assert "--from build_problem" in text
    assert "--to persist" in text
    assert "--resume" not in text

    for target, config_set in EXPECTED_CONFIGS.items():
        target_specs = [path for name, mode, path in specs if name == target and mode == "adaptive"]
        assert target_specs == [f"configs/experiments/{config_set['adaptive']}"]
        fixed_specs = [path for name, mode, path in specs if name == target and mode == "fixed"]
        assert fixed_specs == [f"configs/experiments/{name}" for name in config_set["fixed"]]


def test_runtime_script_emits_both_runtime_csv_reports() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "all_configurations.csv" in text
    assert "adaptive_vs_exhaustive.csv" in text
    assert "exhaustive_total_seconds" in text
    assert "speedup" in text
    assert "status" in text


def test_runtime_script_keeps_adaptive_manifest_columns_aligned() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'fixed_k = "adaptive"' in text
    assert '    fixed_k = ""' not in text


def test_runtime_script_has_valid_bash_syntax_when_bash_is_available() -> None:
    bash = shutil.which("bash")
    if os.name == "nt" or bash is None or not SCRIPT.is_file():
        pytest.skip("bash is not available in the local Windows environment")

    result = subprocess.run(
        [bash, "-n", str(SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
