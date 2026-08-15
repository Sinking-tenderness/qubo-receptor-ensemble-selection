"""Build the deterministic Stage 19f stable-pair QUBO core bundle."""

from __future__ import annotations

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


CONFIG = Path("configs/stage19f_cross_target_stable_pair_qubo.json")
AUDIT = Path("data/stage19f_cross_target_stable_pair_qubo_audit.json")
FIXED_PATHS = (
    "scripts/audit_stage19f_stable_pair_qubo.py",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/build_stage19f_stable_pair_qubo_bundle.py",
    "scripts/diagnose_stage12a_mk14_qubo_objective_adequacy.py",
    "scripts/diagnose_stage19e_cross_target_qubo_v2.py",
    "scripts/diagnose_stage19f_stable_pair_qubo.py",
    "scripts/normalized_receptor_qubo.py",
    "scripts/prepare_receptor.py",
    "scripts/run_stage05_mk14_method_gate.py",
    "scripts/run_stage05_mk14_uncertainty_qubo_gate.py",
    "scripts/screen_stage10_mk14_expanded16_qubo_greedy.py",
    "tests/test_stage19f_stable_pair_qubo.py",
)


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    config = read_json(root / CONFIG)
    result_path = root / str(config["outputs"]["result_json"])
    result = read_json(result_path)
    audit = read_json(root / AUDIT)
    if result["status"] != (
        "stage19f_stable_pair_qubo_not_supported_do_not_amend_bace1"
    ):
        raise ValueError("Stage 19f result status differs")
    if audit["status"] != "stage19f_cross_target_stable_pair_qubo_audit_ok":
        raise ValueError("Stage 19f audit did not pass")
    if audit["result"]["sha256"] != file_sha256(result_path):
        raise ValueError("Stage 19f audit identifies another result")
    if result["gate"]["bace1_amendment_authorized"] is not False:
        raise ValueError("Stage 19f bundle cannot authorize BACE1")

    source_config_path = root / str(config["inputs"]["stage19e_config"]["path"])
    source_config = read_json(source_config_path)
    paths = [CONFIG.as_posix(), AUDIT.as_posix(), *FIXED_PATHS]
    paths.extend(
        str(descriptor["path"])
        for descriptor in config["inputs"].values()
    )
    paths.extend(
        str(descriptor["path"])
        for descriptor in source_config["prior_records"].values()
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
        raise ValueError("Stage 19f bundle contains a protected panel path")
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
            "operation": "Stage 19f cross-target stable-pair QUBO core results",
            "target_ids": ["MK14", "PPARG"],
            "experiment_class": "posthoc_cross_target_train_only_development",
            "bace1_amendment_authorized": False,
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
