"""Build and audit an OpenMM solvated system without running dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment_id",
    "starting_structure",
    "protonation",
    "force_field",
    "solvation",
    "dynamics",
    "planned_outputs",
}


def load_protocol(path: Path) -> dict[str, object]:
    protocol = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a JSON object")
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(protocol))
    if missing:
        raise ValueError(f"protocol is missing required keys: {', '.join(missing)}")
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: dict[str, object]) -> None:
    starting = protocol["starting_structure"]
    protonation = protocol["protonation"]
    force_field = protocol["force_field"]
    solvation = protocol["solvation"]
    dynamics = protocol["dynamics"]
    outputs = protocol["planned_outputs"]
    if not all(isinstance(section, dict) for section in (starting, protonation, force_field, solvation, dynamics, outputs)):
        raise ValueError("protocol sections must be JSON objects")
    if not str(starting.get("pdb_path", "")).endswith(".pdb"):
        raise ValueError("starting_structure.pdb_path must name a PDB file")
    if float(protonation.get("target_ph", 0.0)) <= 0.0:
        raise ValueError("protonation.target_ph must be positive")
    xml = force_field.get("protein_and_water_xml")
    if not isinstance(xml, list) or len(xml) < 2 or not all(isinstance(item, str) for item in xml):
        raise ValueError("force_field.protein_and_water_xml must contain protein and water XML files")
    for key in ("padding_nm", "ionic_strength_molar"):
        if float(solvation.get(key, -1.0)) < 0.0:
            raise ValueError(f"solvation.{key} must be non-negative")
    for key in ("temperature_kelvin", "pressure_bar", "timestep_fs", "friction_per_ps", "frame_stride_ps"):
        if float(dynamics.get(key, 0.0)) <= 0.0:
            raise ValueError(f"dynamics.{key} must be positive")
    if int(dynamics.get("seed", 0)) <= 0:
        raise ValueError("dynamics.seed must be a positive integer")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def topology_counts(topology: object) -> dict[str, int]:
    residues = list(topology.residues())
    atoms = list(topology.atoms())
    return {
        "chain_count": sum(1 for _ in topology.chains()),
        "residue_count": len(residues),
        "atom_count": len(atoms),
        "water_residue_count": sum(residue.name in {"HOH", "WAT"} for residue in residues),
        "sodium_ion_count": sum(residue.name in {"NA", "Na+"} for residue in residues),
        "chloride_ion_count": sum(residue.name in {"CL", "Cl-"} for residue in residues),
    }


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
