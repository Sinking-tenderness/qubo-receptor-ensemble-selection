"""Build the deterministic Stage 19g set-function landscape core bundle."""

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

try:
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
    from scripts.prepare_receptor import file_sha256
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
    from scripts.prepare_receptor import file_sha256


CONFIG = Path("configs/stage19g_cross_target_set_function_landscape.json")
AUDIT = Path("data/stage19g_cross_target_set_function_landscape_audit.json")
FIXED_PATHS = (
    "scripts/audit_stage19g_set_function_landscape.py",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/build_stage19g_set_function_landscape_bundle.py",
    "scripts/diagnose_stage12a_mk14_qubo_objective_adequacy.py",
    "scripts/diagnose_stage19e_cross_target_qubo_v2.py",
    "scripts/diagnose_stage19g_set_function_landscape.py",
    "scripts/normalized_receptor_qubo.py",
    "scripts/prepare_receptor.py",
    "scripts/run_stage05_mk14_method_gate.py",
    "scripts/run_stage05_mk14_uncertainty_qubo_gate.py",
    "scripts/screen_stage10_mk14_expanded16_qubo_greedy.py",
    "tests/test_stage19g_set_function_landscape.py",
)




def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    config = read_json(root / CONFIG)
    result_path = root / str(config["outputs"]["result_json"])
    result = read_json(result_path)
    audit = read_json(root / AUDIT)
    if result["status"] != "stage19g_cross_target_set_function_landscape_complete":
        raise ValueError("Stage 19g result status differs")
    if result["decision"]["cross_target_route"] != (
        "no_cross_target_efficacy_qubo_route_authorized"
    ):
        raise ValueError("Stage 19g cross-target route differs")
    if result["decision"]["bace1_method_amendment_authorized"] is not False:
        raise ValueError("Stage 19g bundle cannot authorize BACE1")
    if audit["status"] != "stage19g_cross_target_set_function_landscape_audit_ok":
        raise ValueError("Stage 19g audit did not pass")
    if audit["result"]["sha256"] != file_sha256(result_path):
        raise ValueError("Stage 19g audit identifies another result")

    source_config_path = root / str(config["inputs"]["stage19e_config"]["path"])
    source_config = read_json(source_config_path)
    paths = [CONFIG.as_posix(), AUDIT.as_posix(), *FIXED_PATHS]
    paths.extend(
        str(descriptor["path"]) for descriptor in config["inputs"].values()
    )
    for target in source_config["targets"].values():
        paths.extend(
            str(descriptor["path"])
            for descriptor in target["inputs"].values()
        )
    paths.extend(
        str(descriptor["path"])
        for descriptor in result["outputs"].values()
    )
    paths.extend(
        (
            str(config["outputs"]["result_json"]),
            str(config["outputs"]["report_md"]),
        )
    )
    normalized = sorted(set(path.replace("\\", "/") for path in paths))
    forbidden = ("fresh_validation", "locked_test", "bace1_docking")
    if any(marker in path.lower() for path in normalized for marker in forbidden):
        raise ValueError("Stage 19g bundle contains a protected-panel path")
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = write_bundle(root, args.output, bundle_paths(root))
    result.update(
        {
            "operation": "Stage 19g cross-target set-function landscape core results",
            "target_ids": ["MK14", "PPARG"],
            "experiment_class": "posthoc_cross_target_train_only_diagnostic",
            "cross_target_efficacy_qubo_route_authorized": False,
            "bace1_method_amendment_authorized": False,
            "new_docking_jobs": 0,
            "fresh_validation_rows": 0,
            "test_rows": 0,
            "bace1_docking_rows": 0,
        }
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
