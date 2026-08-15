"""Stream-audit the Stage61b PPARD diagnostics archive without extracting poses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


RUN_ROOT = "results/runs/stage61b_ppard_remaining144_unidock113_production"
CONFIG_PATH = "configs/stage61b_ppard_remaining144_unidock113_production.json"
LIGAND_MANIFEST_PATH = (
    "data/processed/stage61a_ppard_remaining144_unidock_pdbqt_manifest.csv"
)
SUMMARY_PATH = f"{RUN_ROOT}/summary.json"
AUDIT_PATH = "data/stage61b_ppard_remaining144_unidock113_production_audit.json"
AMENDMENT_PATH = "data/stage61b_ppard_progress_descriptor_amendment01.json"
SCORES_PATH = f"{RUN_ROOT}/scores.csv"
BATCH_RUNS_PATH = f"{RUN_ROOT}/batch_runs.csv"
MEDIAN_MATRIX_PATH = f"{RUN_ROOT}/primary_median_score_matrix.csv"
MINIMUM_MATRIX_PATH = f"{RUN_ROOT}/sensitivity_minimum_score_matrix.csv"
PROGRESS_PATH = f"{RUN_ROOT}/progress.json"

EXPECTED_ARCHIVE_MEMBER_COUNT = 12_899
EXPECTED_BATCH_COUNT = 87
EXPECTED_LIGAND_COUNT = 144
EXPECTED_RECEPTOR_COUNT = 29
EXPECTED_SEED_COUNT = 3
EXPECTED_PAIR_COUNT = 12_528

VINA_RESULT = re.compile(
    r"^REMARK\s+VINA\s+RESULT:\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)"
)
COORDINATE_WARNING = re.compile(
    r"^t\.coords\.size\(\)=(\d+), out\[0\]\.coords\.size\(\)=(\d+)$"
)
OUTPUT_CONTAINER_WARNING = "WARNING: in add_to_output_container"
COORDINATE_SIZE_MISMATCH = "t.coords.size()="


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_csv_bytes(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def read_json_bytes(data: bytes) -> dict[str, object]:
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def atom_signature_bytes(data: bytes) -> tuple[int, dict[str, int]]:
    atom_types: Counter[str] = Counter()
    atom_count = 0
    for raw_line in data.decode("utf-8", errors="replace").splitlines():
        if not raw_line.startswith(("ATOM  ", "HETATM")):
            continue
        fields = raw_line.split()
        if not fields:
            continue
        atom_count += 1
        atom_types[fields[-1]] += 1
    if atom_count == 0:
        raise ValueError("PDBQT contains no atom records")
    return atom_count, dict(sorted(atom_types.items()))


def pose_score_bytes(data: bytes) -> tuple[float, int]:
    scores: list[float] = []
    for raw_line in data.decode("utf-8", errors="replace").splitlines():
        match = VINA_RESULT.match(raw_line.strip())
        if match is None:
            continue
        score = float(match.group(1))
        if not math.isfinite(score):
            raise ValueError("pose contains a non-finite score")
        scores.append(score)
    if not scores:
        raise ValueError("pose contains no Vina result")
    return scores[0], len(scores)


def classify_warning_text(text: str, pose_failure_count: int) -> dict[str, object]:
    lines = [line.strip() for line in text.splitlines()]
    output_lines = [
        line for line in lines if line.startswith(OUTPUT_CONTAINER_WARNING)
    ]
    coordinate_lines = [
        line for line in lines if line.startswith(COORDINATE_SIZE_MISMATCH)
    ]
    other_warning_lines = [
        line
        for line in lines
        if "WARNING" in line and not line.startswith(OUTPUT_CONTAINER_WARNING)
    ]
    coordinate_pairs: list[list[int]] = []
    invalid_coordinate_lines: list[str] = []
    for line in coordinate_lines:
        match = COORDINATE_WARNING.fullmatch(line)
        if match is None:
            invalid_coordinate_lines.append(line)
            continue
        first = int(match.group(1))
        second = int(match.group(2))
        coordinate_pairs.append([first, second])
        if second - first != 1:
            invalid_coordinate_lines.append(line)
    event_count = max(len(output_lines), len(coordinate_lines))
    any_warning = bool(
        event_count or other_warning_lines or invalid_coordinate_lines
    )
    approved_shape = (
        event_count > 0
        and len(output_lines) == len(coordinate_lines)
        and not other_warning_lines
        and not invalid_coordinate_lines
    )
    resolved = not any_warning or (approved_shape and pose_failure_count == 0)
    return {
        "known_warning_event_count": event_count if approved_shape else 0,
        "unresolved_warning_event_count": 0 if resolved else max(1, event_count),
        "output_container_warning_count": len(output_lines),
        "coordinate_size_warning_count": len(coordinate_lines),
        "coordinate_size_pairs": coordinate_pairs,
        "other_warning_lines": other_warning_lines,
        "invalid_coordinate_lines": invalid_coordinate_lines,
        "pose_integrity_failure_count": pose_failure_count,
        "status": "resolved" if resolved else "unresolved",
    }


def safe_archive_members(
    archive: Path,
) -> tuple[list[tarfile.TarInfo], dict[str, tarfile.TarInfo]]:
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
    names = [member.name for member in members]
    if len(set(names)) != len(names):
        raise ValueError("diagnostics archive contains duplicate paths")
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive path: {member.name}")
        if not member.isfile():
            raise ValueError(f"archive contains a non-file member: {member.name}")
    return members, {member.name: member for member in members}


def read_selected_members(
    archive: Path, member_names: set[str]
) -> dict[str, bytes]:
    values: dict[str, bytes] = {}
    with tarfile.open(archive, "r:gz") as handle:
        for name in sorted(member_names):
            extracted = handle.extractfile(name)
            if extracted is None:
                raise FileNotFoundError(f"archive member is missing: {name}")
            values[name] = extracted.read()
    return values


def input_ligand_signatures(
    input_root: Path, ligand_rows: list[dict[str, str]]
) -> dict[str, tuple[int, dict[str, int]]]:
    signatures: dict[str, tuple[int, dict[str, int]]] = {}
    for row in ligand_rows:
        ligand_id = row["ligand_id"]
        path = (input_root / row["pdbqt_path"]).resolve()
        try:
            path.relative_to(input_root)
        except ValueError as error:
            raise ValueError(f"ligand input leaves the input root: {ligand_id}") from error
        data = path.read_bytes()
        if sha256_bytes(data) != row["pdbqt_sha256"].upper():
            raise ValueError(f"ligand input hash differs: {ligand_id}")
        signatures[ligand_id] = atom_signature_bytes(data)
    if len(signatures) != len(ligand_rows):
        raise ValueError("ligand manifest contains duplicate IDs")
    return signatures


def compare_batch_rows(
    batch_rows: list[dict[str, str]],
    aggregate_by_key: dict[tuple[str, str, str], dict[str, str]],
) -> None:
    compared = (
        "target_id",
        "base_seed",
        "label",
        "selection_role",
        "gpu_score",
        "pose_count",
        "status",
        "output_pose_path",
        "output_pose_sha256",
        "input_atom_count",
        "output_atom_count",
        "atom_count_match",
        "atom_types_match",
        "single_pose_match",
        "pose_integrity_status",
    )
    for row in batch_rows:
        key = (row["seed_id"], row["receptor_id"], row["ligand_id"])
        aggregate = aggregate_by_key.get(key)
        if aggregate is None:
            raise ValueError(f"batch row is absent from aggregate scores: {key}")
        if any(row[field] != aggregate[field] for field in compared):
            raise ValueError(f"batch and aggregate rows differ: {key}")


def compare_matrix(
    matrix_rows: list[dict[str, str]],
    aggregate_rows: list[dict[str, str]],
    ligand_rows: list[dict[str, str]],
    receptor_ids: list[str],
    aggregation: str,
) -> None:
    if [row["ligand_id"] for row in matrix_rows] != [
        row["ligand_id"] for row in ligand_rows
    ]:
        raise ValueError(f"{aggregation} matrix ligand order differs")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in aggregate_rows:
        grouped[(row["ligand_id"], row["receptor_id"])].append(
            float(row["gpu_score"])
        )
    ligand_by_id = {row["ligand_id"]: row for row in ligand_rows}
    for row in matrix_rows:
        ligand = ligand_by_id[row["ligand_id"]]
        if row["label"] != ligand["label"] or row["selection_role"] != ligand[
            "selection_role"
        ]:
            raise ValueError(f"{aggregation} matrix metadata differs")
        for receptor_id in receptor_ids:
            values = grouped[(row["ligand_id"], receptor_id)]
            if len(values) != EXPECTED_SEED_COUNT:
                raise ValueError(f"{aggregation} matrix seed count differs")
            expected = (
                statistics.median(values) if aggregation == "median" else min(values)
            )
            if abs(float(row[receptor_id]) - expected) > 1e-12:
                raise ValueError(
                    f"{aggregation} matrix value differs: "
                    f"{row['ligand_id']}/{receptor_id}"
                )


def stream_batch_payloads(
    archive: Path,
    aggregate_by_pose: dict[str, dict[str, str]],
    aggregate_by_key: dict[tuple[str, str, str], dict[str, str]],
    input_signatures: dict[str, tuple[int, dict[str, int]]],
) -> dict[str, object]:
    summaries: dict[tuple[str, str], dict[str, object]] = {}
    batch_score_hashes: dict[tuple[str, str], str] = {}
    batch_log_hashes: dict[tuple[str, str], str] = {}
    warning_audits: dict[tuple[str, str], dict[str, object]] = {}
    batch_row_counts: Counter[tuple[str, str]] = Counter()
    pose_counts: Counter[tuple[str, str]] = Counter()
    seen_poses: set[str] = set()

    batch_pattern = re.compile(
        rf"^{re.escape(RUN_ROOT)}/batches/([^/]+)/([^/]+)/(.+)$"
    )
    with tarfile.open(archive, "r|gz") as handle:
        for member in handle:
            match = batch_pattern.match(member.name)
            if match is None:
                continue
            seed_id, receptor_id, tail = match.groups()
            key = (seed_id, receptor_id)
            extracted = handle.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(member.name)
            if tail == "batch_summary.json":
                summaries[key] = read_json_bytes(extracted.read())
            elif tail == "scores.csv":
                data = extracted.read()
                rows = read_csv_bytes(data)
                compare_batch_rows(rows, aggregate_by_key)
                if len(rows) != EXPECTED_LIGAND_COUNT:
                    raise ValueError(f"batch row count differs: {key}")
                if len({row["ligand_id"] for row in rows}) != len(rows):
                    raise ValueError(f"batch contains duplicate ligand IDs: {key}")
                batch_score_hashes[key] = sha256_bytes(data)
                batch_row_counts[key] = len(rows)
            elif tail == "unidock.log":
                data = extracted.read()
                batch_log_hashes[key] = sha256_bytes(data)
                warning_audits[key] = classify_warning_text(
                    data.decode("utf-8", errors="replace"), 0
                )
            elif tail.startswith("poses/") and tail.endswith("_out.pdbqt"):
                data = extracted.read()
                row = aggregate_by_pose.get(member.name)
                if row is None:
                    raise ValueError(f"unexpected pose output: {member.name}")
                if member.name in seen_poses:
                    raise ValueError(f"duplicate pose output: {member.name}")
                if sha256_bytes(data) != row["output_pose_sha256"].upper():
                    raise ValueError(f"pose hash differs: {member.name}")
                score, model_count = pose_score_bytes(data)
                if abs(score - float(row["gpu_score"])) > 1e-9:
                    raise ValueError(f"pose score differs: {member.name}")
                if model_count != 1 or int(row["pose_count"]) != 1:
                    raise ValueError(f"pose count differs: {member.name}")
                output_count, output_types = atom_signature_bytes(data)
                input_count, input_types = input_signatures[row["ligand_id"]]
                if output_count != input_count or output_types != input_types:
                    raise ValueError(f"pose atom signature differs: {member.name}")
                stored_checks = (
                    int(row["input_atom_count"]) == input_count,
                    int(row["output_atom_count"]) == output_count,
                    row["atom_count_match"] == "True",
                    row["atom_types_match"] == "True",
                    row["single_pose_match"] == "True",
                    row["pose_integrity_status"] == "ok",
                    row["status"] == "ok",
                )
                if not all(stored_checks):
                    raise ValueError(f"stored pose audit differs: {member.name}")
                pose_counts[key] += 1
                seen_poses.add(member.name)

    expected_batch_keys = {
        (row["seed_id"], row["receptor_id"]) for row in aggregate_by_key.values()
    }
    if set(summaries) != expected_batch_keys:
        raise ValueError("batch summary grid differs")
    if set(batch_score_hashes) != expected_batch_keys:
        raise ValueError("batch score-file grid differs")
    if set(batch_log_hashes) != expected_batch_keys:
        raise ValueError("batch log grid differs")
    if seen_poses != set(aggregate_by_pose):
        raise ValueError("pose path grid differs")

    known_warning_count = 0
    for key in sorted(expected_batch_keys):
        summary = summaries[key]
        if summary.get("status") != "ok":
            raise ValueError(f"batch summary status differs: {key}")
        if (summary.get("seed_id"), summary.get("receptor_id")) != key:
            raise ValueError(f"batch summary identity differs: {key}")
        if int(summary["ligand_count"]) != EXPECTED_LIGAND_COUNT:
            raise ValueError(f"batch summary ligand count differs: {key}")
        if batch_row_counts[key] != EXPECTED_LIGAND_COUNT:
            raise ValueError(f"batch score row count differs: {key}")
        if pose_counts[key] != EXPECTED_LIGAND_COUNT:
            raise ValueError(f"batch pose count differs: {key}")
        if batch_score_hashes[key] != str(summary["scores_sha256"]).upper():
            raise ValueError(f"batch score hash differs: {key}")
        if batch_log_hashes[key] != str(summary["log_sha256"]).upper():
            raise ValueError(f"batch log hash differs: {key}")
        expected_pose_audit = {
            "audited_pose_count": EXPECTED_LIGAND_COUNT,
            "failure_count": 0,
            "mismatches": [],
            "status": "ok",
        }
        if dict(summary["pose_integrity_audit"]) != expected_pose_audit:
            raise ValueError(f"batch pose audit differs: {key}")
        if warning_audits[key] != dict(summary["warning_adjudication"]):
            raise ValueError(f"batch warning audit differs: {key}")
        known_warning_count += int(
            warning_audits[key]["known_warning_event_count"]
        )

    return {
        "batch_count": len(summaries),
        "pose_count": len(seen_poses),
        "known_warning_event_count": known_warning_count,
        "unresolved_warning_event_count": sum(
            int(value["unresolved_warning_event_count"])
            for value in warning_audits.values()
        ),
    }


def run(archive: Path, input_root: Path, output: Path) -> dict[str, object]:
    archive = archive.resolve()
    input_root = input_root.resolve()
    members, members_by_name = safe_archive_members(archive)
    if len(members) != EXPECTED_ARCHIVE_MEMBER_COUNT:
        raise ValueError("diagnostics archive member count differs")

    selected_names = {
        CONFIG_PATH,
        LIGAND_MANIFEST_PATH,
        SUMMARY_PATH,
        AUDIT_PATH,
        AMENDMENT_PATH,
        SCORES_PATH,
        BATCH_RUNS_PATH,
        MEDIAN_MATRIX_PATH,
        MINIMUM_MATRIX_PATH,
        PROGRESS_PATH,
    }
    missing = sorted(selected_names - set(members_by_name))
    if missing:
        raise FileNotFoundError(f"diagnostics archive members are missing: {missing}")
    selected = read_selected_members(archive, selected_names)

    config = read_json_bytes(selected[CONFIG_PATH])
    summary = read_json_bytes(selected[SUMMARY_PATH])
    source_audit = read_json_bytes(selected[AUDIT_PATH])
    amendment = read_json_bytes(selected[AMENDMENT_PATH])
    progress = read_json_bytes(selected[PROGRESS_PATH])
    ligand_rows = read_csv_bytes(selected[LIGAND_MANIFEST_PATH])
    aggregate_rows = read_csv_bytes(selected[SCORES_PATH])
    batch_runs = read_csv_bytes(selected[BATCH_RUNS_PATH])
    median_rows = read_csv_bytes(selected[MEDIAN_MATRIX_PATH])
    minimum_rows = read_csv_bytes(selected[MINIMUM_MATRIX_PATH])

    if summary.get("status") != "stage61b_ppard_remaining144_unidock_matrix_ok":
        raise ValueError("source summary status differs")
    if source_audit.get("status") != (
        "independent_stage61b_ppard_remaining144_unidock_matrix_audit_ok"
    ):
        raise ValueError("source independent-audit status differs")
    if amendment.get("status") != "stage61b_progress_descriptor_amendment01_ok":
        raise ValueError("progress amendment status differs")
    if progress.get("status") != "stage61b_production_complete":
        raise ValueError("progress status differs")
    if sha256_bytes(selected[CONFIG_PATH]) != str(summary["config"]["sha256"]).upper():
        raise ValueError("config hash differs")
    ligand_descriptor = dict(config["inputs"])["ligand_manifest"]
    if sha256_bytes(selected[LIGAND_MANIFEST_PATH]) != str(
        ligand_descriptor["sha256"]
    ).upper():
        raise ValueError("ligand manifest hash differs")

    output_members = {
        "scores_csv": SCORES_PATH,
        "batch_runs_csv": BATCH_RUNS_PATH,
        "median_matrix_csv": MEDIAN_MATRIX_PATH,
        "minimum_matrix_csv": MINIMUM_MATRIX_PATH,
        "progress_json": PROGRESS_PATH,
    }
    for key, path in output_members.items():
        descriptor = dict(summary["outputs"])[key]
        if descriptor["path"] != path:
            raise ValueError(f"summary output path differs: {key}")
        if sha256_bytes(selected[path]) != str(descriptor["sha256"]).upper():
            raise ValueError(f"summary output hash differs: {key}")
        if len(selected[path]) != int(descriptor["size_bytes"]):
            raise ValueError(f"summary output size differs: {key}")

    if len(ligand_rows) != EXPECTED_LIGAND_COUNT:
        raise ValueError("ligand count differs")
    if Counter(row["label"] for row in ligand_rows) != Counter(
        {"active": 72, "decoy": 72}
    ):
        raise ValueError("ligand labels differ")
    signatures = input_ligand_signatures(input_root, ligand_rows)

    if len(aggregate_rows) != EXPECTED_PAIR_COUNT:
        raise ValueError("aggregate pair count differs")
    aggregate_by_key = {
        (row["seed_id"], row["receptor_id"], row["ligand_id"]): row
        for row in aggregate_rows
    }
    aggregate_by_pose = {row["output_pose_path"]: row for row in aggregate_rows}
    if len(aggregate_by_key) != len(aggregate_rows):
        raise ValueError("aggregate scores contain duplicate keys")
    if len(aggregate_by_pose) != len(aggregate_rows):
        raise ValueError("aggregate scores contain duplicate pose paths")
    seed_ids = sorted({row["seed_id"] for row in aggregate_rows})
    receptor_ids = list(dict.fromkeys(row["receptor_id"] for row in aggregate_rows))
    ligand_ids = {row["ligand_id"] for row in aggregate_rows}
    if len(seed_ids) != EXPECTED_SEED_COUNT:
        raise ValueError("seed count differs")
    if len(receptor_ids) != EXPECTED_RECEPTOR_COUNT:
        raise ValueError("receptor count differs")
    if ligand_ids != set(signatures):
        raise ValueError("aggregate ligand grid differs")

    stream_result = stream_batch_payloads(
        archive, aggregate_by_pose, aggregate_by_key, signatures
    )
    if stream_result["batch_count"] != EXPECTED_BATCH_COUNT:
        raise ValueError("stream-audited batch count differs")
    if stream_result["pose_count"] != EXPECTED_PAIR_COUNT:
        raise ValueError("stream-audited pose count differs")
    if int(stream_result["unresolved_warning_event_count"]) != 0:
        raise ValueError("stream audit found unresolved warnings")

    if len(batch_runs) != EXPECTED_BATCH_COUNT or any(
        row["status"] != "ok"
        or int(row["unresolved_warning_event_count"]) != 0
        or int(row["pose_integrity_failure_count"]) != 0
        for row in batch_runs
    ):
        raise ValueError("batch ledger technical gate differs")
    compare_matrix(
        median_rows, aggregate_rows, ligand_rows, receptor_ids, "median"
    )
    compare_matrix(
        minimum_rows, aggregate_rows, ligand_rows, receptor_ids, "minimum"
    )
    if int(summary["known_warning_event_count"]) != int(
        stream_result["known_warning_event_count"]
    ):
        raise ValueError("known-warning total differs")
    if int(summary["unresolved_warning_event_count"]) != 0:
        raise ValueError("summary reports unresolved warnings")
    if int(summary["pose_integrity_failure_count"]) != 0:
        raise ValueError("summary reports pose-integrity failures")

    raw_target_ids = sorted({row["target_id"] for row in aggregate_rows})
    raw_target_mismatch_count = sum(
        row["target_id"] != "PPARD" for row in aggregate_rows
    )
    result = {
        "schema_version": "1.0",
        "audit_id": "stage61b-ppard-diagnostics-archive-local-stream-audit-v1",
        "status": "stage61b_ppard_diagnostics_archive_local_stream_audit_ok",
        "archive": {
            "path": str(archive),
            "sha256": sha256_file(archive),
            "size_bytes": archive.stat().st_size,
            "member_count": len(members),
            "duplicate_path_count": 0,
            "unsafe_path_count": 0,
            "non_file_member_count": 0,
        },
        "dimensions": {
            "seed_count": len(seed_ids),
            "receptor_count": len(receptor_ids),
            "ligand_count": len(ligand_rows),
            "batch_count": int(stream_result["batch_count"]),
            "pair_count": len(aggregate_rows),
            "pose_count": int(stream_result["pose_count"]),
            "label_counts": dict(sorted(Counter(row["label"] for row in ligand_rows).items())),
        },
        "independent_recomputations": {
            "input_ligand_hashes_exact": True,
            "aggregate_and_batch_rows_exact": True,
            "pose_sha256_exact_count": EXPECTED_PAIR_COUNT,
            "pose_score_exact_count": EXPECTED_PAIR_COUNT,
            "single_pose_exact_count": EXPECTED_PAIR_COUNT,
            "atom_count_and_type_signature_exact_count": EXPECTED_PAIR_COUNT,
            "batch_score_and_log_hashes_exact_count": EXPECTED_BATCH_COUNT,
            "median_matrix_exact": True,
            "minimum_matrix_exact": True,
        },
        "technical_gate": {
            "known_warning_event_count": int(
                stream_result["known_warning_event_count"]
            ),
            "unresolved_warning_event_count": 0,
            "pose_integrity_failure_count": 0,
            "status": "pass",
        },
        "metadata_adjudication": {
            "raw_target_id_values": raw_target_ids,
            "expected_target_id": "PPARD",
            "raw_target_id_mismatch_count": raw_target_mismatch_count,
            "status": "known_metadata_only_issue_requiring_stage61c_amendment",
            "scientific_score_or_pose_effect": False,
        },
        "data_boundary": {
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
        },
        "interpretation_boundary": (
            "This local audit establishes the byte-level and pose-level integrity "
            "of the Stage61b diagnostics archive. It does not establish QUBO "
            "efficacy, independent target validation, quantum execution, speedup, "
            "or quantum advantage."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.archive, args.input_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
