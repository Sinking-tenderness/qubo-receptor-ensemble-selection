"""Build a deterministic reproduction bundle for the Stage51b diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_stage05_mk14_remote_bundle import write_bundle
from scripts.diagnose_stage51b_ppara_redocking_bias import read_csv, read_json


CONFIG = "configs/stage51b_ppara_redocking_bias_diagnostic.json"
FIXED_PATHS = (
    CONFIG,
    "scripts/diagnose_stage51b_ppara_redocking_bias.py",
    "scripts/build_stage51b_ppara_redocking_bias_bundle.py",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "data/stage51b_ppara_redocking_bias_diagnostic_result.json",
    "data/stage51b_ppara_redocking_bias_diagnostic_audit.json",
    "data/processed/stage51b_ppara_redocking_bias_diagnostic.csv",
    "data/processed/stage51b_ppara_chemotype_clusters.csv",
    "reports/stage-51/ppara_redocking_bias_diagnostic.md",
    "tests/test_stage51b_ppara_redocking_bias.py",
)


def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    config = read_json(root / CONFIG)
    paths = list(FIXED_PATHS)
    paths.extend(
        str(descriptor["path"])
        for descriptor in dict(config["inputs"]).values()
    )
    paths.extend(
        str(descriptor["path"])
        for descriptor in dict(config["implementation"]).values()
    )
    case_manifest = root / str(
        dict(config["inputs"])["stage50_case_manifest"]["path"]
    )
    paths.extend(row["reference_sdf"] for row in read_csv(case_manifest))
    normalized = sorted(set(path.replace("\\", "/") for path in paths))
    forbidden = ("fresh_validation", "locked_test", "development_panel_score")
    if any(marker in path.lower() for path in normalized for marker in forbidden):
        raise ValueError("Stage51b bundle crossed a protected data boundary")
    if any(int(value) != 0 for value in dict(config["data_boundary"]).values()):
        raise ValueError("Stage51b bundle data boundary is not closed")
    for relative in normalized:
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = write_bundle(root, args.output, bundle_paths(root))
    diagnostic = read_json(
        root / "data/stage51b_ppara_redocking_bias_diagnostic_result.json"
    )
    result.update(
        {
            "operation": "Stage51b PPARA post-hoc label-free redocking-bias diagnostic reproduction bundle",
            "target_id": "PPARA",
            "prepared_receptor_count": 60,
            "passing_receptor_count": 20,
            "stable_receptor_count": 18,
            "confirmatory_gate_remains_failed": True,
            "exploratory_branch_candidate": diagnostic["decision"][
                "exploratory_twenty_receptor_branch_candidate"
            ],
            "activity_labels_read": 0,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
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
