"""Build the 15-receptor manifest admitted by Stage 08 and Stage 08b."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .select_stage08_mk14_expanded16_pool import (
        checked_record,
        file_sha256,
        read_csv,
        read_json,
        write_csv,
        write_json,
    )
    from .experimental.unidock.run_stage08b_mk14_replacement_redocking import (
        receptor_row_from_first_round,
    )
except ImportError:
    from select_stage08_mk14_expanded16_pool import (
        checked_record,
        file_sha256,
        read_csv,
        read_json,
        write_csv,
        write_json,
    )
    from experimental.unidock.run_stage08b_mk14_replacement_redocking import (
        receptor_row_from_first_round,
    )


def run_build(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    root = Path.cwd().resolve()
    config = read_json(config_path)
    boundary = dict(config["data_boundary"])
    if any(int(value) != 0 for value in boundary.values()):
        raise ValueError("Stage 08c manifest build crossed a data boundary")
    inputs = dict(config["inputs"])
    paths = {
        key: checked_record(value)
        for key, value in inputs.items()
        if isinstance(value, dict)
    }
    stage08_failure = read_json(paths["stage08_failure_adjudication"])
    stage08b_failure = read_json(paths["stage08b_failure_adjudication"])
    if stage08_failure.get("status") != "stage08_redocking_gate_failed_two_receptors":
        raise ValueError("Stage 08 failure adjudication status differs")
    if stage08b_failure.get("status") != "stage08b_replacement_gate_failed_one_receptor":
        raise ValueError("Stage 08b failure adjudication status differs")
    expected = dict(config["expected"])
    stage08_admitted = [str(value) for value in expected["stage08_admitted_receptor_ids"]]
    stage08b_admitted = [str(value) for value in expected["stage08b_admitted_receptor_ids"]]
    if set(stage08_admitted) != set(str(value) for value in stage08_failure["admitted_receptor_ids"]):
        raise ValueError("Stage 08 admitted receptor set differs")
    if stage08b_admitted != [str(value) for value in stage08b_failure["admitted_receptor_ids"]]:
        raise ValueError("Stage 08b admitted receptor set differs")
    excluded = [str(value) for value in expected["excluded_receptor_ids"]]
    observed_failed = [
        *[str(value) for value in stage08_failure["failed_receptor_ids"]],
        *[str(value) for value in stage08b_failure["failed_receptor_ids"]],
    ]
    if excluded != observed_failed:
        raise ValueError("cumulative failed receptor set differs")

    existing = read_csv(paths["existing8_receptor_manifest"])
    if len(existing) != int(expected["existing_receptor_count"]):
        raise ValueError("existing receptor count differs")
    eligible_by_id = {
        row["conformer_id"]: row for row in read_csv(paths["eligible_pool"])
    }
    stage08_config = read_json(paths["stage08_config"])
    stage08b_config = read_json(paths["stage08b_config"])
    stage08_cases = {
        str(case["conformer_id"]): dict(case) for case in stage08_config["cases"]
    }
    stage08b_cases = {
        str(case["conformer_id"]): dict(case) for case in stage08b_config["cases"]
    }
    stage08_root = root / str(inputs["stage08_run_directory"])
    stage08b_root = root / str(inputs["stage08b_run_directory"])
    admitted_rows: list[dict[str, object]] = []
    for receptor_id in stage08_admitted:
        row = receptor_row_from_first_round(
            root,
            stage08_root,
            eligible_by_id[receptor_id],
            stage08_cases[receptor_id],
        )
        row["source_pool"] = "stage08_redocking_admitted"
        admitted_rows.append(row)
    for receptor_id in stage08b_admitted:
        row = receptor_row_from_first_round(
            root,
            stage08b_root,
            eligible_by_id[receptor_id],
            stage08b_cases[receptor_id],
        )
        row["source_pool"] = "stage08b_replacement_redocking_admitted"
        admitted_rows.append(row)

    fields = list(existing[0])
    if any(set(row) != set(fields) for row in admitted_rows):
        raise ValueError("admitted receptor manifest schema differs")
    current: list[dict[str, object]] = [dict(row) for row in existing]
    current.extend({field: row[field] for field in fields} for row in admitted_rows)
    current_ids = [str(row["conformer_id"]) for row in current]
    if current_ids != [str(value) for value in expected["current_receptor_ids"]]:
        raise ValueError(f"current receptor order differs: {current_ids}")
    if len(current_ids) != int(expected["current_receptor_count"]):
        raise ValueError("current receptor count differs")
    if set(excluded).intersection(current_ids):
        raise ValueError("a failed receptor entered the current manifest")

    outputs = dict(config["outputs"])
    manifest_path = Path(str(outputs["current_receptor_manifest_csv"]))
    summary_path = Path(str(outputs["summary_json"]))
    if not overwrite and (manifest_path.exists() or summary_path.exists()):
        raise FileExistsError("Stage 08c current-manifest outputs already exist")
    write_csv(manifest_path, current)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage08c_current15_manifest_ok",
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "current_receptor_count": len(current),
        "current_receptor_ids": current_ids,
        "excluded_receptor_ids": excluded,
        "data_boundary": {str(key): int(value) for key, value in boundary.items()},
        "outputs": {
            "current_receptor_manifest_csv": {
                "path": manifest_path.as_posix(),
                "sha256": file_sha256(manifest_path),
            }
        },
        "next_gate": "select one nonredundant structural replacement and run the unchanged three-seed cognate-redocking gate",
    }
    write_json(summary_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_build(args.config, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
