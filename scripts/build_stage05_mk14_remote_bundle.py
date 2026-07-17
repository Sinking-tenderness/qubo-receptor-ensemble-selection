"""Build a deterministic Linux-ready MAPK14 remote execution bundle."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


FIXED_PATHS = (
    "configs/stage05_mk14_development_e16_seed0_linux.json",
    "configs/stage05_mk14_development_e16_seed1_linux.json",
    "configs/stage05_mk14_development_e16_seed2_linux.json",
    "configs/stage05_mk14_development_protocol_freeze.json",
    "configs/stage05_mk14_development_seed_aggregation.json",
    "configs/stage05_mk14_development_method_gate_preregistration.json",
    "configs/stage05_mk14_search_ladder_e16_cpu2.txt",
    "data/processed/stage05_mk14_receptor_preparation_manifest.csv",
    "data/processed/stage05_mk14_development_120a120d_pdbqt_manifest.csv",
    "data/stage05_mk14_development_input_preparation_summary.json",
    "environment/bin/vina_1.2.7_linux_x86_64",
    "scripts/run_md_receptor_ligand_benchmark.py",
    "scripts/batch_vina_docking.py",
    "scripts/batch_vina_docking_parallel.py",
    "scripts/build_score_matrix.py",
    "scripts/prepare_receptor.py",
    "scripts/aggregate_seed_replicates.py",
    "scripts/audit_stage05_development_matrix.py",
    "scripts/run_stage05_mk14_development_remote.sh",
    "reports/stage-05/mk14_development_protocol_freeze_and_input_preparation.md",
    "reports/stage-05/mk14_development_method_gate_preregistration.md",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_paths(root: Path, manifest: str, column: str) -> list[str]:
    path = root / manifest
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or column not in rows[0]:
        raise ValueError(f"manifest is empty or missing {column}: {manifest}")
    return [row[column].replace("\\", "/") for row in rows]


def stage05_bundle_paths(root: Path) -> list[str]:
    paths = list(FIXED_PATHS)
    paths.extend(
        manifest_paths(
            root,
            "data/processed/stage05_mk14_receptor_preparation_manifest.csv",
            "receptor_pdbqt",
        )
    )
    paths.extend(
        manifest_paths(
            root,
            "data/processed/stage05_mk14_development_120a120d_pdbqt_manifest.csv",
            "pdbqt_path",
        )
    )
    return sorted(set(paths))


def normalized_tar_info(path: Path, arcname: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(arcname)
    info.size = path.stat().st_size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mode = 0o755 if arcname.endswith((".sh", "vina_1.2.7_linux_x86_64")) else 0o644
    return info


def write_bundle(root: Path, output: Path, relative_paths: list[str]) -> dict[str, object]:
    root = root.resolve()
    normalized = sorted(set(path.replace("\\", "/") for path in relative_paths))
    missing = [path for path in normalized if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"bundle inputs are missing: {missing}")
    manifest = "".join(
        f"{file_sha256(root / path)}  {path}\n" for path in normalized
    ).encode("ascii")
    if b"\r" in manifest:
        raise ValueError("bundle manifest unexpectedly contains CR characters")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, mtime=0
        ) as gzip_handle:
            with tarfile.open(fileobj=gzip_handle, mode="w") as archive:
                for relative in normalized:
                    path = root / relative
                    info = normalized_tar_info(path, relative)
                    with path.open("rb") as source:
                        archive.addfile(info, source)
                manifest_info = tarfile.TarInfo("bundle_manifest.sha256")
                manifest_info.size = len(manifest)
                manifest_info.mtime = 0
                manifest_info.uid = 0
                manifest_info.gid = 0
                manifest_info.uname = "root"
                manifest_info.gname = "root"
                manifest_info.mode = 0o644
                archive.addfile(manifest_info, io.BytesIO(manifest))
    return {
        "schema_version": "1.0",
        "status": "ok",
        "source_file_count": len(normalized),
        "archive_entry_count": len(normalized) + 1,
        "archive": output.as_posix(),
        "archive_size_bytes": output.stat().st_size,
        "archive_sha256": file_sha256(output).upper(),
        "manifest_line_ending": "LF",
        "deterministic_metadata": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_bundle(args.root, args.output, stage05_bundle_paths(args.root))
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
