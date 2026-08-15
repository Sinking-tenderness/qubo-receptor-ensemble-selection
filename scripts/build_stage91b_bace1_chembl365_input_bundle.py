import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


FILES = [
    "configs/stage91b_bace1_chembl365_unidock_input_preparation.json",
    "data/stage91_bace1_group_robust_rescue_preregistration_result.json",
    "data/stage91b_bace1_development_manifest_freeze.json",
    "data/processed/stage91b_bace1_chembl365_development_ligand_manifest.csv",
    "scripts/__init__.py",
    "scripts/experimental/__init__.py",
    "scripts/batch_prepare_ligand_pdbqt.py",
    "scripts/prepare_ligand_3d_sdf.py",
    "scripts/experimental/unidock/__init__.py",
    "scripts/experimental/unidock/prepare_stage42b_bace1_train266_inputs.py",
    "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
    "scripts/experimental/unidock/prepare_stage91b_bace1_chembl365_inputs.py",
    "scripts/experimental/unidock/audit_stage91b_bace1_chembl365_inputs.py",
    "scripts/experimental/unidock/run_stage91b_bace1_chembl365_input_preparation_remote.sh",
]


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest().upper()


def add_bytes(archive, name, value, mode=0o644):
    info = tarfile.TarInfo(name)
    info.size = len(value)
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    archive.addfile(info, io.BytesIO(value))


def run(root, output):
    members = []
    for relative in FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        members.append((relative, path.read_bytes()))
    manifest = "".join(
        f"{sha256_bytes(value)}  {relative}\n" for relative, value in members
    ).encode("ascii")
    instructions = b"""# Stage91b remote execution\n\nThis bundle prepares and independently audits only the 365 frozen BACE1 development ligands. It does not run docking and contains no confirmation or locked-test ligand.\n\nUse the existing qubo-unidock-stage08 environment. Run scripts/experimental/unidock/run_stage91b_bace1_chembl365_input_preparation_remote.sh with AUTO_POWEROFF=1.\n"""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", fileobj=raw_handle, mode="wb", mtime=0
        ) as gzip_handle:
            with tarfile.open(
                fileobj=gzip_handle, mode="w", format=tarfile.PAX_FORMAT
            ) as archive:
                for relative, value in members:
                    mode = 0o755 if relative.endswith(".sh") else 0o644
                    add_bytes(archive, relative, value, mode)
                add_bytes(archive, "bundle_manifest.sha256", manifest)
                add_bytes(archive, "EXECUTION.md", instructions)
    result = {
        "schema_version": "1.0",
        "status": "ok",
        "archive": str(output),
        "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest().upper(),
        "archive_size_bytes": output.stat().st_size,
        "source_file_count": len(members),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("deliverables/stage91b_bace1_chembl365_unidock_inputs_external_input_v1.tar.gz"),
    )
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
