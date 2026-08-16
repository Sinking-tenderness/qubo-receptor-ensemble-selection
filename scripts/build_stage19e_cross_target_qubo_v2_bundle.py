"""Build the deterministic Stage 19e core-results bundle."""

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


CONFIG = Path("configs/stage19e_cross_target_qubo_v2_nested_diagnostic.json")
AUDIT = Path("data/stage19e_cross_target_qubo_v2_nested_diagnostic_audit.json")
FIXED_PATHS = (
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/build_stage19e_cross_target_qubo_v2_bundle.py",
    "scripts/diagnose_stage19e_cross_target_qubo_v2.py",
    "scripts/audit_stage19e_cross_target_qubo_v2.py",
    "tests/test_stage19e_cross_target_qubo_v2.py",
)




def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    config = read_json(root / CONFIG)
    result_path = root / config["outputs"]["result_json"]
    result = read_json(result_path)
    audit = read_json(root / AUDIT)
    if result["status"] != (
        "stage19e_quadratic_v2_not_supported_do_not_amend_bace1"
    ):
        raise ValueError("Stage 19e result status differs")
    if audit["status"] != (
        "stage19e_cross_target_qubo_v2_nested_diagnostic_audit_ok"
    ):
        raise ValueError("Stage 19e audit did not pass")
    if audit["result"]["sha256"] != file_sha256(result_path):
        raise ValueError("Stage 19e audit identifies another result")
    if result["gate"]["bace1_v2_amendment_authorized"] is not False:
        raise ValueError("Stage 19e bundle cannot authorize BACE1 v2")

    paths = [CONFIG.as_posix(), AUDIT.as_posix(), *FIXED_PATHS]
    paths.extend(
        str(descriptor["path"])
        for descriptor in config["prior_records"].values()
    )
    for target in config["targets"].values():
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
    forbidden = ("fresh_validation", "locked_test")
    if any(marker in path.lower() for path in normalized for marker in forbidden):
        raise ValueError("Stage 19e bundle contains a protected panel path")
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
            "operation": "Stage 19e MK14+PPARG nested QUBO v2 diagnostic core results",
            "target_ids": ["MK14", "PPARG"],
            "experiment_class": "posthoc_cross_target_train_only_development",
            "bace1_v2_amendment_authorized": False,
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
