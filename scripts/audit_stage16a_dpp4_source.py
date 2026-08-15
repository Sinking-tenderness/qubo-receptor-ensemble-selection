"""Audit the preregistered DPP4 DUD-E source and 2I78 reference."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolAlign, rdMolDescriptors

from scripts.audit_external_target_intake import audit_ism, file_sha256
from scripts.audit_stage15_hivpr_source import (
    neutralized_heavy_molecule,
    pdb_atom_audit,
    read_json,
    verified,
)


def official_reference_audit(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    chain_ca_counts = {
        chain: sum(
            line.startswith("ATOM  ")
            and line[21:22] == chain
            and line[12:16].strip() == "CA"
            for line in lines
        )
        for chain in ("A", "B", "C", "D")
    }
    ligand = [
        line
        for line in lines
        if line.startswith("HETATM")
        and line[17:20].strip() == "KIQ"
        and line[21:22] == "B"
        and line[22:26].strip() == "901"
    ]
    resolution = None
    for line in lines:
        if line.startswith("REMARK   2 RESOLUTION.") and "ANGSTROMS" in line:
            resolution = float(line.split()[3])
            break
    if chain_ca_counts != {"A": 726, "B": 726, "C": 726, "D": 726}:
        raise ValueError("2I78 protein chain lengths differ")
    if len(ligand) != 31:
        raise ValueError("2I78 KIQ ligand differs")
    if resolution is None or not math.isclose(resolution, 2.5, abs_tol=1e-6):
        raise ValueError("2I78 resolution differs")
    return {
        "protein_auth_asym_ids": ["A", "B", "C", "D"],
        "protein_ca_counts": chain_ca_counts,
        "reference_auth_asym_id": "B",
        "reference_sequence_range": [39, 764],
        "cognate_ligand_comp_id": "KIQ",
        "cognate_ligand_auth_asym_id": "B",
        "cognate_ligand_auth_seq_id": 901,
        "cognate_ligand_heavy_atom_count": len(ligand),
        "resolution_angstrom": resolution,
    }


def ca_coordinate_set(path: Path, chain: str | None) -> set[tuple[float, float, float]]:
    output: set[tuple[float, float, float]] = set()
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        if chain is not None and line[21:22] != chain:
            continue
        output.add(
            (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
        )
    return output


def receptor_coordinate_provenance(dude_pdb: Path, official_pdb: Path) -> dict[str, object]:
    dude = ca_coordinate_set(dude_pdb, None)
    official_a = ca_coordinate_set(official_pdb, "A")
    official_b = ca_coordinate_set(official_pdb, "B")
    matched_b = dude & official_b
    matched_a_only = (dude - matched_b) & official_a
    unmatched = dude - matched_b - matched_a_only
    if len(dude) != 573 or len(matched_b) != 530 or len(matched_a_only) != 43 or unmatched:
        raise ValueError("DUD-E DPP4 receptor coordinate provenance differs")
    return {
        "dude_ca_count": len(dude),
        "exact_2i78_chain_b_ca_count": len(matched_b),
        "exact_2i78_chain_a_donor_ca_count": len(matched_a_only),
        "unmatched_ca_count": len(unmatched),
        "interpretation": "DUD-E supplies a label-free catalytic-region hybrid assembled entirely from exact 2I78 chain-B and chain-A coordinates",
    }


def ligand_identity_audit(dude_mol2: Path, official_sdf: Path) -> dict[str, object]:
    dude = Chem.MolFromMol2Block(
        dude_mol2.read_text(encoding="utf-8", errors="replace"),
        removeHs=False,
        sanitize=True,
    )
    official = Chem.MolFromMolBlock(
        official_sdf.read_text(encoding="utf-8", errors="replace"),
        removeHs=False,
        sanitize=True,
    )
    if dude is None or official is None:
        raise ValueError("DPP4 reference ligand parsing failed")
    dude_heavy = neutralized_heavy_molecule(dude)
    official_heavy = neutralized_heavy_molecule(official)
    dude_elements = Counter(atom.GetSymbol() for atom in dude_heavy.GetAtoms())
    official_elements = Counter(atom.GetSymbol() for atom in official_heavy.GetAtoms())
    dude_bonds = Counter(str(bond.GetBondType()) for bond in dude_heavy.GetBonds())
    official_bonds = Counter(str(bond.GetBondType()) for bond in official_heavy.GetBonds())
    if (
        dude_heavy.GetNumAtoms() != 31
        or official_heavy.GetNumAtoms() != 31
        or dude_elements != official_elements
        or dude_bonds != official_bonds
    ):
        raise ValueError("DPP4 reference ligand heavy-atom graphs differ")
    rmsd = float(
        rdMolAlign.CalcRMS(
            dude_heavy,
            official_heavy,
            maxMatches=1_000_000,
            symmetrizeConjugatedTerminalGroups=True,
        )
    )
    if not math.isclose(rmsd, 0.0, abs_tol=1e-6):
        raise ValueError("DUD-E and RCSB DPP4 ligand coordinates differ")
    return {
        "heavy_atom_count": dude_heavy.GetNumAtoms(),
        "heavy_atom_elements": dict(sorted(dude_elements.items())),
        "heavy_atom_bond_types": dict(sorted(dude_bonds.items())),
        "dude_formula": rdMolDescriptors.CalcMolFormula(dude),
        "official_formula": rdMolDescriptors.CalcMolFormula(official),
        "dude_formal_charge": Chem.GetFormalCharge(dude),
        "official_formal_charge": Chem.GetFormalCharge(official),
        "protonation_difference_only": True,
        "charge_neutralized_fixed_frame_rmsd_angstrom": rmsd,
        "heavy_atom_coordinate_identity": True,
    }


def audit_ism_with_frozen_exclusions(
    path: Path, specification: dict[str, object]
) -> dict[str, object]:
    exclusions = {
        int(row["line_number"]): dict(row)
        for row in specification.get("allowed_invalid_rows", [])
    }
    ids: list[str] = []
    canonical: Counter[str] = Counter()
    charged_count = 0
    multi_fragment_count = 0
    heavy_counts: list[int] = []
    molecular_weights: list[float] = []
    observed_exclusions: list[dict[str, object]] = []
    raw_row_count = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw_row_count += 1
            fields = stripped.split()
            exclusion = exclusions.get(line_number)
            if exclusion is not None:
                if (
                    len(fields) < 2
                    or fields[0] != str(exclusion["smiles"])
                    or fields[1] != str(exclusion["source_id"])
                ):
                    raise ValueError(
                        f"frozen DPP4 source exclusion differs at line {line_number}"
                    )
                observed_exclusions.append(exclusion)
                continue
            if len(fields) < 2:
                raise ValueError(f"DPP4 ISM row has fewer than two fields: {line_number}")
            smiles, molecule_id = fields[0], fields[1]
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f"unexpected DPP4 ISM parse failure: {line_number}")
            ids.append(molecule_id)
            canonical[Chem.MolToSmiles(molecule, isomericSmiles=True)] += 1
            charged_count += int(Chem.GetFormalCharge(molecule) != 0)
            multi_fragment_count += int(len(Chem.GetMolFrags(molecule)) > 1)
            heavy_counts.append(molecule.GetNumHeavyAtoms())
            molecular_weights.append(float(Descriptors.MolWt(molecule)))

    source_id_counts = Counter(ids)
    duplicate_source_ids = sorted(
        molecule_id
        for molecule_id, count in source_id_counts.items()
        if count > 1
    )
    if raw_row_count != int(specification["row_count"]):
        raise ValueError("DPP4 raw ISM row count differs")
    if len(observed_exclusions) != len(exclusions):
        raise ValueError("DPP4 frozen source exclusions are incomplete")
    if len(ids) != int(specification["valid_row_count"]):
        raise ValueError("DPP4 valid ISM row count differs")
    if (
        len(source_id_counts) != int(specification["valid_unique_source_id_count"])
        or len(duplicate_source_ids)
        != int(specification["valid_duplicate_source_id_count"])
        or max(source_id_counts.values())
        != int(specification["maximum_source_id_multiplicity"])
    ):
        raise ValueError("DPP4 valid ISM source-ID multiplicity differs")

    duplicate_canonical_rows = sum(
        count - 1 for count in canonical.values() if count > 1
    )
    return {
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "raw_row_count": raw_row_count,
        "valid_row_count": len(ids),
        "excluded_invalid_row_count": len(observed_exclusions),
        "excluded_invalid_rows": observed_exclusions,
        "unique_source_molecule_id_count": len(source_id_counts),
        "duplicate_source_molecule_id_count": len(duplicate_source_ids),
        "duplicate_source_molecule_row_count": sum(
            source_id_counts[value] for value in duplicate_source_ids
        ),
        "maximum_source_id_multiplicity": max(source_id_counts.values()),
        "duplicate_source_molecule_ids": duplicate_source_ids,
        "recommended_internal_id_rule": (
            "target_label_plus_source_line_number; preserve source molecule ID as metadata"
        ),
        "rdkit_parsed_count": len(heavy_counts),
        "rdkit_failure_count_after_frozen_exclusion": 0,
        "unique_canonical_smiles_count": len(canonical),
        "duplicate_canonical_row_count": duplicate_canonical_rows,
        "charged_molecule_count": charged_count,
        "multi_fragment_molecule_count": multi_fragment_count,
        "heavy_atom_count_min": min(heavy_counts),
        "heavy_atom_count_max": max(heavy_counts),
        "molecular_weight_min": min(molecular_weights),
        "molecular_weight_max": max(molecular_weights),
    }


def run(config_path: Path, root: Path) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = dict(config["implementation"])
    if file_sha256(Path(__file__)) != str(implementation["sha256"]).upper():
        raise ValueError("Stage 16a implementation SHA-256 differs")
    dependency = dict(implementation["dependency"])
    verified(root, dependency)

    prereg_path = verified(root, dict(config["preregistration"]))
    preregistration = read_json(prereg_path)
    if preregistration["preregistration_id"] != "stage16-dpp4-replacement-exploratory-20260731-v1":
        raise ValueError("DPP4 replacement preregistration differs")
    if any(
        int(preregistration["data_boundary"][key]) != 0
        for key in (
            "DPP4_benchmark_docking_scores_read",
            "DPP4_fresh_validation_rows_read",
            "DPP4_test_rows_read",
        )
    ):
        raise ValueError("DPP4 preregistration crossed a protected boundary")

    dude = dict(config["dude_source"])
    archive_path = verified(root, dict(dude["archive"]))
    active_spec = dict(dude["actives"])
    decoy_spec = dict(dude["decoys"])
    active_path = verified(root, active_spec)
    decoy_path = verified(root, decoy_spec)
    active_audit = audit_ism(
        active_path,
        int(active_spec["row_count"]),
        int(active_spec["unique_source_id_count"]),
        int(active_spec["duplicate_source_id_count"]),
        int(active_spec["maximum_source_id_multiplicity"]),
    )
    decoy_audit = audit_ism_with_frozen_exclusions(decoy_path, decoy_spec)
    active_audit["path"] = active_path.relative_to(root).as_posix()
    decoy_audit["path"] = decoy_path.relative_to(root).as_posix()

    reference = dict(config["reference_structure"])
    dude_receptor = verified(root, dict(reference["dude_receptor_pdb"]))
    dude_ligand = verified(root, dict(reference["dude_crystal_ligand_mol2"]))
    official_pdb = verified(root, dict(reference["official_pdb"]))
    official_cif = verified(root, dict(reference["official_mmcif"]))
    official_sdf = verified(root, dict(reference["official_ligand_sdf"]))

    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage16a_dpp4_source_audit_ok",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "target": config["target"],
        "dude_source": {
            "archive": {
                "path": archive_path.relative_to(root).as_posix(),
                "sha256": file_sha256(archive_path),
            },
            "actives": active_audit,
            "decoys": decoy_audit,
        },
        "reference_structure": {
            "pdb_id": "2I78",
            "dude_receptor": {
                "path": dude_receptor.relative_to(root).as_posix(),
                "sha256": file_sha256(dude_receptor),
                **pdb_atom_audit(dude_receptor),
            },
            "official_pdb": {
                "path": official_pdb.relative_to(root).as_posix(),
                "sha256": file_sha256(official_pdb),
                **official_reference_audit(official_pdb),
            },
            "official_mmcif": {
                "path": official_cif.relative_to(root).as_posix(),
                "sha256": file_sha256(official_cif),
            },
            "dude_receptor_coordinate_provenance": receptor_coordinate_provenance(
                dude_receptor, official_pdb
            ),
            "ligand_identity": {
                "dude_mol2_path": dude_ligand.relative_to(root).as_posix(),
                "dude_mol2_sha256": file_sha256(dude_ligand),
                "official_sdf_path": official_sdf.relative_to(root).as_posix(),
                "official_sdf_sha256": file_sha256(official_sdf),
                **ligand_identity_audit(dude_ligand, official_sdf),
            },
        },
        "data_boundary": {
            "ligand_identity_and_source_labels_read": True,
            "docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "next_gate": "run the frozen P27487 X-ray metadata discovery and coordinate-only DPP4 receptor audit",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    output_path = root / str(config["output_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
