"""Download and audit one official AlphaFold DB structure record."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "qubo-receptor-ensemble-selection/0.1"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def audit_pdb_text(text: str) -> dict[str, object]:
    atom_lines = [line for line in text.splitlines() if line.startswith("ATOM  ")]
    if not atom_lines:
        raise ValueError("downloaded AlphaFold PDB has no ATOM records")
    residues = {
        (line[21:22].strip(), line[22:26].strip(), line[26:27].strip(), line[17:20].strip())
        for line in atom_lines
    }
    plddt_values = []
    for line in atom_lines:
        try:
            plddt_values.append(float(line[60:66]))
        except ValueError as exc:
            raise ValueError("invalid pLDDT B-factor field in AlphaFold PDB") from exc
    return {
        "atom_count": len(atom_lines),
        "residue_count": len(residues),
        "chain_ids": sorted({line[21:22].strip() for line in atom_lines}),
        "plddt_mean": round(sum(plddt_values) / len(plddt_values), 3),
        "plddt_min": min(plddt_values),
        "plddt_max": max(plddt_values),
    }


def select_entry(entries: list[dict[str, object]], accession: str) -> dict[str, object]:
    matches = [
        entry
        for entry in entries
        if entry.get("isUniProt") is True and entry.get("uniprotAccession") == accession
    ]
    if not matches:
        raise ValueError(f"AlphaFold DB returned no canonical UniProt entry for {accession}")
    canonical = [entry for entry in matches if entry.get("entryId") == f"AF-{accession}-F1"]
    if len(canonical) != 1:
        raise ValueError(f"could not select one canonical AlphaFold entry for {accession}")
    return canonical[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accession", required=True, help="Canonical UniProt accession, e.g. P24941")
    parser.add_argument("--pdb-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (args.pdb_output.exists() or args.manifest_output.exists()) and not args.overwrite:
        raise FileExistsError("output exists; use --overwrite only after reviewing provenance")

    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{args.accession}"
    entries = json.loads(fetch_bytes(api_url).decode("utf-8"))
    entry = select_entry(entries, args.accession)
    pdb_url = str(entry["pdbUrl"])
    pdb_bytes = fetch_bytes(pdb_url)
    if pdb_bytes.lstrip().lower().startswith(b"<html"):
        raise ValueError("AlphaFold download returned HTML rather than PDB data")
    pdb_text = pdb_bytes.decode("ascii")
    audit = audit_pdb_text(pdb_text)
    args.pdb_output.parent.mkdir(parents=True, exist_ok=True)
    args.pdb_output.write_bytes(pdb_bytes)
    manifest = {
        "schema_version": "1.0",
        "source": "AlphaFold Protein Structure Database official API",
        "downloaded_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "api_url": api_url,
        "uniprot_accession": args.accession,
        "entry_id": entry["entryId"],
        "model_entity_id": entry["modelEntityId"],
        "model_created_date": entry["modelCreatedDate"],
        "model_version": entry["latestVersion"],
        "pdb_url": pdb_url,
        "cif_url": entry["cifUrl"],
        "plddt_json_url": entry["plddtDocUrl"],
        "pae_json_url": entry["paeDocUrl"],
        "local_pdb_path": args.pdb_output.as_posix(),
        "pdb_sha256": sha256(args.pdb_output),
        "pdb_audit": audit,
        "interpretation_note": "AlphaFold pLDDT is a confidence signal, not evidence of a ligand-bound pocket state.",
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
