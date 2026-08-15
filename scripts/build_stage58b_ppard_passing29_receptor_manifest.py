"""Freeze the Stage57-passing PPARD receptor manifest for Pilot-96 production."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(root: Path, manifest_output: Path, summary_output: Path) -> dict[str, object]:
    root = root.resolve()
    stage57_path = root / "data/stage57_ppard_cognate_redocking_summary.json"
    receptor_path = root / "data/processed/stage57_ppard_prepared_receptor_manifest.csv"
    gate_path = root / "data/processed/stage57_ppard_receptor_gate_results.csv"
    stage57 = read_json(stage57_path)
    receptors = read_csv(receptor_path)
    gates = read_csv(gate_path)
    if stage57["status"] != "stage57_ppard_cognate_redocking_gate_ok":
        raise ValueError("Stage57 PPARD redocking gate did not pass")
    if int(stage57["passed_receptor_count"]) != 29:
        raise ValueError("Stage57 PPARD passing count differs")
    prepared_ids = [row["conformer_id"] for row in receptors]
    gate_by_id = {row["conformer_id"]: row for row in gates}
    passing_ids = [
        receptor_id
        for receptor_id in prepared_ids
        if gate_by_id[receptor_id]["gate_pass"] == "True"
    ]
    if passing_ids != [
        str(row["conformer_id"])
        for row in stage57["receptor_gate_results"]
        if bool(row["gate_pass"])
    ]:
        raise ValueError("Stage57 PPARD passing order differs")
    if len(passing_ids) != 29:
        raise ValueError("Stage58b PPARD passing receptor count differs")

    receptor_by_id = {row["conformer_id"]: row for row in receptors}
    output_rows: list[dict[str, object]] = []
    for receptor_id in passing_ids:
        receptor = receptor_by_id[receptor_id]
        gate = gate_by_id[receptor_id]
        pdbqt = root / receptor["receptor_pdbqt"]
        if (
            receptor["status"] != "ok"
            or not pdbqt.is_file()
            or sha256(pdbqt) != receptor["receptor_pdbqt_sha256"].upper()
        ):
            raise ValueError(f"Stage58b receptor differs: {receptor_id}")
        if gate["successful_seed_count"] != "3" or gate["seed_count"] != "3":
            raise ValueError(f"Stage58b receptor is not a stable 3-of-3 pass: {receptor_id}")
        output_rows.append(
            {
                **receptor,
                "stage57_gate_pass": True,
                "stage57_seed_count": 3,
                "stage57_successful_seed_count": 3,
                "stage57_median_top_ranked_rmsd_angstrom": gate[
                    "median_top_ranked_rmsd_angstrom"
                ],
                "stage57_maximum_top_ranked_rmsd_angstrom": gate[
                    "maximum_top_ranked_rmsd_angstrom"
                ],
            }
        )

    manifest_output = (
        manifest_output if manifest_output.is_absolute() else root / manifest_output
    )
    summary_output = summary_output if summary_output.is_absolute() else root / summary_output
    write_csv(manifest_output, output_rows)
    result = {
        "schema_version": "1.0",
        "freeze_id": "stage58b-ppard-stage57-passing29-receptor-freeze-v1",
        "status": "stage58b_ppard_passing29_receptor_manifest_ok",
        "selection_rule": "retain every Stage57 cognate-redocking gate pass in frozen Stage57 receptor order",
        "receptor_count": len(output_rows),
        "receptor_ids": passing_ids,
        "all_receptors_passed_three_of_three_seeds": True,
        "data_boundary": {
            "pilot_docking_scores_read": 0,
            "pilot_ligand_labels_read": 0,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
        },
        "inputs": {
            "stage57_summary": {"path": stage57_path.relative_to(root).as_posix(), "sha256": sha256(stage57_path)},
            "stage57_receptor_manifest": {"path": receptor_path.relative_to(root).as_posix(), "sha256": sha256(receptor_path)},
            "stage57_gate_results": {"path": gate_path.relative_to(root).as_posix(), "sha256": sha256(gate_path)},
        },
        "output": {
            "path": manifest_output.relative_to(root).as_posix(),
            "sha256": sha256(manifest_output),
        },
        "next_gate": "dock the frozen Pilot-96 against all 29 receptors with three frozen seeds",
        "decision_boundary": "This freeze uses only the preregistered Stage57 technical gate and no Pilot-96 docking outcome. It cannot establish enrichment, complementarity, QUBO superiority, or quantum advantage.",
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("data/processed/stage58b_ppard_stage57_passing29_receptor_manifest.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/stage58b_ppard_stage57_passing29_receptor_manifest_summary.json"),
    )
    args = parser.parse_args()
    run(args.root, args.manifest_output, args.summary_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
