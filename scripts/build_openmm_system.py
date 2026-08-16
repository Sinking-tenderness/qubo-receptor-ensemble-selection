"""Build and audit an OpenMM solvated system without running dynamics."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import argparse
import json
from pathlib import Path

from qubo_receptor_ensemble.io import file_sha256 as sha256  # noqa: F401
from qubo_receptor_ensemble.md import (  # noqa: F401
    REQUIRED_TOP_LEVEL_KEYS,
    load_protocol,
    topology_counts,
    validate_protocol,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--solvated-pdb-output", type=Path, required=True)
    parser.add_argument("--system-xml-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    starting = protocol["starting_structure"]
    assert isinstance(starting, dict)
    input_pdb = Path(str(starting["pdb_path"]))
    if not input_pdb.is_file():
        raise FileNotFoundError(input_pdb)
    output_paths = [args.manifest_output, args.solvated_pdb_output, args.system_xml_output]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"output exists; use --overwrite after review: {existing}")

    try:
        import openmm
        from openmm import XmlSerializer, unit
        from openmm.app import HBonds, PME, ForceField, Modeller, PDBFile
    except ImportError as exc:
        raise RuntimeError(
            "OpenMM is not available. Create the separate environment from "
            "environment/stage03_openmm.yml before building the MD system."
        ) from exc

    protonation = protocol["protonation"]
    force_field_settings = protocol["force_field"]
    solvation = protocol["solvation"]
    assert isinstance(protonation, dict)
    assert isinstance(force_field_settings, dict)
    assert isinstance(solvation, dict)
    pdb = PDBFile(str(input_pdb))
    force_field = ForceField(*force_field_settings["protein_and_water_xml"])
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(force_field, pH=float(protonation["target_ph"]))
    modeller.addSolvent(
        force_field,
        model=str(force_field_settings["water_geometry_model"]),
        padding=float(solvation["padding_nm"]) * unit.nanometer,
        ionicStrength=float(solvation["ionic_strength_molar"]) * unit.molar,
        neutralize=bool(solvation["neutralize"]),
        positiveIon=str(solvation["positive_ion"]),
        negativeIon=str(solvation["negative_ion"]),
    )
    system = force_field.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=float(force_field_settings["nonbonded_cutoff_nm"]) * unit.nanometer,
        constraints=HBonds,
    )
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.solvated_pdb_output.open("w", encoding="ascii") as handle:
        PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
    args.system_xml_output.write_text(XmlSerializer.serialize(system), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "experiment_id": protocol["experiment_id"],
        "status": "ok",
        "operation": "solvated OpenMM system build only; no minimization, equilibration, or production dynamics were run",
        "protocol_path": args.protocol.as_posix(),
        "input_pdb": input_pdb.as_posix(),
        "input_pdb_sha256": sha256(input_pdb),
        "openmm_version": openmm.version.version,
        "input_topology": topology_counts(pdb.topology),
        "solvated_topology": topology_counts(modeller.topology),
        "outputs": {
            "solvated_pdb": args.solvated_pdb_output.as_posix(),
            "solvated_pdb_sha256": sha256(args.solvated_pdb_output),
            "system_xml": args.system_xml_output.as_posix(),
            "system_xml_sha256": sha256(args.system_xml_output),
        },
    }
    args.manifest_output.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
