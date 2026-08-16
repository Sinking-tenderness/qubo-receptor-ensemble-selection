"""Build a deterministic core bundle for the Stage 22 coverage QUBO screen."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import read_json  # noqa: F401 (deduped)
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_stage05_mk14_remote_bundle import write_bundle




def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    config = read_json(root / "configs/stage22_structural_state_coverage_qubo.json")
    paths = {
        "configs/stage22_structural_state_coverage_qubo.json",
        "scripts/run_stage22_structural_state_coverage_qubo.py",
        "scripts/audit_stage22_structural_state_coverage_qubo.py",
        "scripts/build_stage22_structural_state_coverage_qubo_bundle.py",
        "scripts/diagnose_stage22_beam_baseline.py",
        "scripts/diagnose_stage22_global_milp.py",
        "scripts/run_stage21_structure_aware_qubo.py",
        "scripts/__init__.py",
        "tests/test_stage22_structural_state_coverage_qubo.py",
        "data/stage22_structural_state_coverage_qubo_result.json",
        "data/stage22_structural_state_coverage_qubo_audit.json",
        "data/stage22_structural_state_coverage_qubo_model_record.json",
        "data/stage22_beam_baseline_diagnostic.json",
        "data/stage22_global_milp_diagnostic.json",
        "reports/stage-22/structural_state_coverage_qubo.md",
        "reports/stage-22/search_diagnostics.md",
        "pyproject.toml",
    }
    for spec in config["targets"].values():
        for relative in spec["inputs"].values():
            paths.add(str(relative).replace("\\", "/"))
    for relative in config["outputs"].values():
        paths.add(str(relative).replace("\\", "/"))
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_bundle(args.root, args.output, bundle_paths(args.root))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
