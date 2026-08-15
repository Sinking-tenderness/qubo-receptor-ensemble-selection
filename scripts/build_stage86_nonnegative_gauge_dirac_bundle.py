"""Build deterministic Stage86 external-input and returned-result archives."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


def input_paths(root: Path) -> list[Path]:
    values = [
        "configs/stage77_quantum_hardware_interface_gate.json",
        "configs/stage84_mixed_radix_dirac_iqp_gate.json",
        "configs/stage86_nonnegative_gauge_dirac_rescue.json",
        "data/stage72_constraint_native_cqm_model_record.json",
        "data/stage85a_qci_dirac3_failure_adjudication.json",
        "data/stage86_nonnegative_gauge_dirac_rescue_prepared.json",
        "data/stage86_nonnegative_gauge_dirac_rescue_audit.json",
        "environment/stage79_qci_dirac3.yml",
        "reports/stage-86/nonnegative_gauge_external_execution.md",
        "results/runs/stage74_larger_k_solver_scaling/workload_metrics.csv",
        "results/runs/stage74_larger_k_solver_scaling/cell_comparison.csv",
        "results/runs/stage74_larger_k_solver_scaling/solver_trials.csv",
        "scripts/run_stage75_explicit_variable_k_cqm.py",
        "scripts/run_stage81_dirac_global_qubo_formulation_gate.py",
        "scripts/run_stage84_mixed_radix_dirac_iqp_gate.py",
        "scripts/prepare_stage86_nonnegative_gauge_dirac_rescue.py",
        "scripts/experimental/quantum/run_stage85_mixed_radix_dirac_calibration.py",
        "scripts/experimental/quantum/run_stage86_nonnegative_gauge_dirac_rescue.py",
        "scripts/build_stage86_nonnegative_gauge_dirac_bundle.py",
    ]
    paths = [root / value for value in values]
    paths.extend(
        sorted(
            (root / "results/runs/stage86_nonnegative_gauge_dirac_rescue/instance").glob("*.json")
        )
    )
    return paths


def result_paths(root: Path) -> list[Path]:
    paths = [
        root / "configs/stage86_nonnegative_gauge_dirac_rescue.json",
        root / "data/stage86_nonnegative_gauge_dirac_rescue_prepared.json",
    ]
    output = root / "external_results/stage86_nonnegative_gauge_dirac_rescue"
    if output.is_dir():
        paths.extend(sorted(path for path in output.rglob("*") if path.is_file()))
    for name in ("stage86_qci_preflight.log", "stage86_qci_rescue.log"):
        path = root / name
        if path.is_file():
            paths.append(path)
    return paths


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def build(root: Path, output: Path, kind: str) -> dict[str, object]:
    paths = input_paths(root) if kind == "input" else result_paths(root)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage86 bundle is missing files: {missing}")
    unique = sorted(set(path.resolve() for path in paths), key=lambda path: path.as_posix())
    manifest = "".join(
        f"{sha256(path)}  {path.relative_to(root.resolve()).as_posix()}\n"
        for path in unique
    ).encode("ascii")
    prefix = (
        "stage86_nonnegative_gauge_dirac_rescue_external_input_v1"
        if kind == "input"
        else "stage86_nonnegative_gauge_dirac_rescue_results_v1"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in unique:
                    relative = path.relative_to(root.resolve()).as_posix()
                    add_bytes(archive, f"{prefix}/{relative}", path.read_bytes())
                add_bytes(archive, f"{prefix}/bundle_manifest.sha256", manifest)
    return {
        "schema_version": "1.0",
        "status": "ok",
        "kind": kind,
        "archive": str(output),
        "archive_sha256": sha256(output),
        "archive_size_bytes": output.stat().st_size,
        "source_file_count": len(unique),
        "archive_entry_count": len(unique) + 1,
        "deterministic_metadata": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("input", "results"), required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    record = build(root, Path(args.output).resolve(), args.kind)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
