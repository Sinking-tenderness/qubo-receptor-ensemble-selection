"""Audit the frozen ESR1 DUD-E source, panel allocation, and 1SJ0 identity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    from scripts.audit_external_target_intake import audit_ism, file_sha256
    from scripts.audit_stage15_hivpr_source import (
        neutralized_heavy_molecule,
        pdb_atom_audit,
        read_json,
        verified,
    )
except ModuleNotFoundError:
    from audit_external_target_intake import audit_ism, file_sha256
    from audit_stage15_hivpr_source import (
        neutralized_heavy_molecule,
        pdb_atom_audit,
        read_json,
        verified,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty ESR1 allocation")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def scaffold_for(molecule: Chem.Mol) -> str:
    copy = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(copy)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=copy, includeChirality=False
    )
    return scaffold or Chem.MolToSmiles(copy, canonical=True)


def allocate_active_panels(
    path: Path, allocation: dict[str, object]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    seed = str(allocation["hash_seed"])
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if len(fields) < 2:
                raise ValueError(f"ESR1 active row is incomplete: {line_number}")
            molecule = Chem.MolFromSmiles(fields[0])
            if molecule is None:
                raise ValueError(f"ESR1 active SMILES does not parse: {line_number}")
            source_id = fields[1]
            rows.append(
                {
                    "source_line_number": line_number,
                    "source_molecule_id": source_id,
                    "source_smiles": fields[0],
                    "canonical_smiles": Chem.MolToSmiles(
                        molecule, isomericSmiles=True
                    ),
                    "scaffold_smiles": scaffold_for(molecule),
                    "allocation_rank_sha256": hashlib.sha256(
                        f"{seed}|{line_number}|{source_id}".encode("ascii")
                    ).hexdigest(),
                }
            )
    if len(rows) != int(allocation["source_active_count"]):
        raise ValueError("ESR1 active source row count differs")
    if len({str(row["source_molecule_id"]) for row in rows}) != len(rows):
        raise ValueError("ESR1 active source IDs are not unique")
    if len({str(row["canonical_smiles"]) for row in rows}) != len(rows):
        raise ValueError("ESR1 canonical active SMILES are not unique")
    scaffold_counts = Counter(str(row["scaffold_smiles"]) for row in rows)
    if max(scaffold_counts.values()) != int(
        allocation["maximum_scaffold_group_size"]
    ):
        raise ValueError("ESR1 active scaffold grouping differs")

    ranked = sorted(
        rows,
        key=lambda row: (
            str(row["allocation_rank_sha256"]),
            int(row["source_line_number"]),
        ),
    )
    sizes = {
        "development_train": int(allocation["train_active_count"]),
        "fresh_validation": int(allocation["fresh_validation_active_count"]),
        "locked_test": int(allocation["locked_test_active_count"]),
    }
    selected_count = sum(sizes.values())
    if selected_count != int(allocation["selected_active_count"]):
        raise ValueError("ESR1 selected active count differs")
    if len(ranked) - selected_count != int(
        allocation["unallocated_source_surplus_count"]
    ):
        raise ValueError("ESR1 active-source surplus differs")
    boundaries = (
        sizes["development_train"],
        sizes["development_train"] + sizes["fresh_validation"],
        selected_count,
    )
    for index, row in enumerate(ranked):
        if index < boundaries[0]:
            row["selection_role"] = "development_train"
            row["split"] = "train"
        elif index < boundaries[1]:
            row["selection_role"] = "fresh_validation"
            row["split"] = "validation"
        elif index < boundaries[2]:
            row["selection_role"] = "locked_test"
            row["split"] = "test"
        else:
            row["selection_role"] = "unallocated_source_surplus"
            row["split"] = ""
    split_scaffolds: dict[str, set[str]] = {}
    for row in ranked:
        if row["split"]:
            split_scaffolds.setdefault(str(row["scaffold_smiles"]), set()).add(
                str(row["split"])
            )
    if any(len(splits) > 1 for splits in split_scaffolds.values()):
        raise ValueError("ESR1 active scaffold crosses a panel boundary")
    summary = {
        "source_active_count": len(rows),
        "selected_active_count": selected_count,
        "unallocated_source_surplus_count": len(rows) - selected_count,
        "unique_canonical_smiles_count": len(
            {str(row["canonical_smiles"]) for row in rows}
        ),
        "scaffold_group_count": len(scaffold_counts),
        "maximum_scaffold_group_size": max(scaffold_counts.values()),
        "panel_counts": sizes,
        "scaffold_disjoint": True,
    }
    return sorted(rows, key=lambda row: int(row["source_line_number"])), summary


def ca_coordinate_set(path: Path, chain: str | None) -> set[tuple[float, float, float]]:
    values: set[tuple[float, float, float]] = set()
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        if chain is not None and line[21:22] != chain:
            continue
        values.add(
            (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
        )
    return values


def protein_frame_provenance(dude_pdb: Path, official_pdb: Path) -> dict[str, object]:
    dude = ca_coordinate_set(dude_pdb, None)
    official = ca_coordinate_set(official_pdb, "A")
    exact = dude & official
    if (len(dude), len(official), len(exact)) != (213, 243, 213):
        raise ValueError("ESR1 DUD-E/RCSB CA-coordinate provenance differs")
    return {
        "dude_ca_count": len(dude),
        "official_chain_a_ca_count": len(official),
        "exact_coordinate_intersection_count": len(exact),
        "dude_only_ca_count": len(dude - official),
        "official_only_ca_count": len(official - dude),
        "interpretation": "DUD-E retains a prepared 1SJ0 chain-A subset; all 213 retained C-alpha coordinates are exactly in the official frame.",
    }


def official_reference_audit(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    chain_a_ca = sum(
        line.startswith("ATOM  ")
        and line[21:22] == "A"
        and line[12:16].strip() == "CA"
        for line in lines
    )
    ligand_atoms = [
        line
        for line in lines
        if line.startswith("HETATM")
        and line[17:20].strip() == "E4D"
        and line[21:22] == "A"
        and line[22:26].strip() == "600"
    ]
    resolution = next(
        (
            float(line.split()[3])
            for line in lines
            if line.startswith("REMARK   2 RESOLUTION.") and "ANGSTROMS" in line
        ),
        None,
    )
    if chain_a_ca != 243 or len(ligand_atoms) != 33:
        raise ValueError("1SJ0 chain-A or E4D records differ")
    if resolution is None or not math.isclose(resolution, 1.9, abs_tol=1e-6):
        raise ValueError("1SJ0 resolution differs")
    return {
        "reference_auth_asym_id": "A",
        "reference_chain_a_ca_count": chain_a_ca,
        "cognate_ligand_comp_id": "E4D",
        "cognate_ligand_auth_asym_id": "A",
        "cognate_ligand_auth_seq_id": 600,
        "cognate_ligand_heavy_atom_count": len(ligand_atoms),
        "resolution_angstrom": resolution,
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
        raise ValueError("ESR1 reference-ligand parsing failed")
    dude_heavy = neutralized_heavy_molecule(dude)
    official_heavy = neutralized_heavy_molecule(official)
    dude_elements = Counter(atom.GetSymbol() for atom in dude_heavy.GetAtoms())
    official_elements = Counter(atom.GetSymbol() for atom in official_heavy.GetAtoms())
    dude_bonds = Counter(str(bond.GetBondType()) for bond in dude_heavy.GetBonds())
    official_bonds = Counter(str(bond.GetBondType()) for bond in official_heavy.GetBonds())
    if (
        dude_heavy.GetNumAtoms() != 33
        or official_heavy.GetNumAtoms() != 33
        or dude_elements != official_elements
        or dude_bonds != official_bonds
    ):
        raise ValueError("ESR1 reference-ligand heavy-atom graphs differ")
    rmsd = float(
        rdMolAlign.CalcRMS(
            dude_heavy,
            official_heavy,
            maxMatches=1_000_000,
            symmetrizeConjugatedTerminalGroups=True,
        )
    )
    if not math.isclose(rmsd, 0.0, abs_tol=1e-6):
        raise ValueError("DUD-E and RCSB ESR1 ligand coordinates differ")
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


def run(config_path: Path, root: Path) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    if file_sha256(Path(__file__)) != str(config["implementation"]["sha256"]).upper():
        raise ValueError("Stage 20a implementation SHA-256 differs")
    preregistration_path = verified(root, dict(config["preregistration"]))
    preregistration = read_json(preregistration_path)
    if (
        preregistration["preregistration_id"]
        != "stage20-esr1-independent-exploratory-20260801-v1"
    ):
        raise ValueError("ESR1 preregistration differs")

    dude = dict(config["dude_source"])
    archive_path = verified(root, dict(dude["archive"]))
    active_path = verified(root, dict(dude["actives"]))
    decoy_path = verified(root, dict(dude["decoys"]))
    active_audit = audit_ism(active_path, 383, 383, 0, 1)
    decoy_audit = audit_ism(decoy_path, 20685, 20663, 22, 2)
    if decoy_audit["unique_canonical_smiles_count"] != 20675:
        raise ValueError("ESR1 decoy canonical-SMILES count differs")
    allocation_rows, allocation_summary = allocate_active_panels(
        active_path, dict(config["active_allocation"])
    )

    reference = dict(config["reference_structure"])
    dude_receptor = verified(root, dict(reference["dude_receptor_pdb"]))
    dude_ligand = verified(root, dict(reference["dude_crystal_ligand_mol2"]))
    official_pdb = verified(root, dict(reference["official_pdb"]))
    official_cif = verified(root, dict(reference["official_mmcif"]))
    official_sdf = verified(root, dict(reference["official_ligand_sdf"]))
    allocation_path = root / str(config["outputs"]["active_allocation_csv"])
    result_path = root / str(config["outputs"]["summary_json"])
    write_csv(allocation_path, allocation_rows)

    for audit in (active_audit, decoy_audit):
        audit["path"] = (root / str(audit["path"])).relative_to(root).as_posix()
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage20a_esr1_source_and_active_allocation_ok",
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
        "active_allocation": {
            **allocation_summary,
            "path": allocation_path.relative_to(root).as_posix(),
            "sha256": file_sha256(allocation_path),
        },
        "reference_structure": {
            "pdb_id": "1SJ0",
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
            "protein_frame_provenance": protein_frame_provenance(
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
            "coordinate_pool_selected": False,
            "docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
            "pparg_stage19c_enrichment_rows_read": 0,
        },
        "next_gate": "run frozen ESR1 RCSB metadata discovery, coordinate audit, and deterministic structural max-min selection",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
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
