"""Build a deterministic core-results bundle for Stage 19d."""

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


CONFIG = Path("configs/stage19d_pparg_train668_frozen_comparator_analysis.json")
AUDIT = Path("data/stage19d_pparg_train668_frozen_comparator_audit.json")
FIXED_PATHS = (
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/build_stage19d_pparg_train_comparator_bundle.py",
    "scripts/audit_stage19d_pparg_train_comparators.py",
    "scripts/evaluate_stage19d_pparg_train_comparators.py",
    "tests/test_stage19d_pparg_train_comparators.py",
)




def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    config = read_json(root / CONFIG)
    result = read_json(root / config["outputs"]["result_json"])
    audit = read_json(root / AUDIT)
    frozen = read_json(root / config["outputs"]["frozen_methods_json"])

    if result["status"] != "stage19d_pparg_train_only_comparison_complete":
        raise ValueError("Stage 19d result is not complete")
    if audit["status"] != (
        "stage19d_pparg_train668_frozen_comparator_audit_ok"
    ):
        raise ValueError("Stage 19d independent audit did not pass")
    if audit["result"]["sha256"] != file_sha256(
        root / config["outputs"]["result_json"]
    ):
        raise ValueError("Stage 19d audit identifies another result")
    if any(int(value) != 0 for value in result["data_boundary"].values()):
        raise ValueError("Stage 19d crossed the train-only boundary")

    paths = [CONFIG.as_posix(), AUDIT.as_posix(), *FIXED_PATHS]
    paths.extend(
        str(descriptor["path"]) for descriptor in config["inputs"].values()
    )
    paths.extend(
        str(descriptor["path"]) for descriptor in result["outputs"].values()
    )
    paths.extend(
        (
            str(config["outputs"]["result_json"]),
            str(config["outputs"]["report_md"]),
        )
    )
    paths.extend(
        str(evidence["model"]["path"])
        for evidence in frozen["models"].values()
    )
    normalized = sorted(set(path.replace("\\", "/") for path in paths))
    forbidden = ("fresh_validation", "locked_test")
    if any(marker in path.lower() for path in normalized for marker in forbidden):
        raise ValueError("Stage 19d result bundle contains a protected panel path")
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
            "operation": "Stage 19d PPARG Train-668 frozen comparator core results",
            "target_id": "PPARG",
            "experiment_class": "posthoc_exploratory_train_only",
            "ligand_count": 668,
            "receptor_count": 16,
            "robust_method_count": 16,
            "primary_only_method_count": 2,
            "validation_rows": 0,
            "test_rows": 0,
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
