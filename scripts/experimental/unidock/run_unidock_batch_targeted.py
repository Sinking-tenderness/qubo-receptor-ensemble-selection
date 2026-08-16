"""Target-aware wrapper for the frozen legacy Uni-Dock batch helper."""

from __future__ import annotations

from typing import Any

from scripts.experimental.unidock import run_unidock_gpu_equivalence as legacy


def infer_target_id(ligands: list[dict[str, str]]) -> str:
    """Return the single non-empty target identifier represented by a batch."""
    target_ids = {
        str(row.get("target_id", "")).strip()
        for row in ligands
        if str(row.get("target_id", "")).strip()
    }
    if not ligands or len(target_ids) != 1:
        raise ValueError(
            "a Uni-Dock batch must contain one non-empty target_id; "
            f"observed {sorted(target_ids)}"
        )
    return next(iter(target_ids))


def apply_target_id(
    rows: list[dict[str, object]], target_id: str
) -> list[dict[str, object]]:
    """Replace the legacy MK14 label without changing any docking value."""
    if not target_id:
        raise ValueError("target_id must be non-empty")
    return [{**row, "target_id": target_id} for row in rows]


def run_batch(*args: Any, **kwargs: Any) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run the frozen helper and repair its historical target metadata bug."""
    if len(args) >= 5:
        ligands = args[4]
    else:
        ligands = kwargs.get("ligands")
    if not isinstance(ligands, list):
        raise TypeError("run_batch requires a ligand list")
    target_id = infer_target_id(ligands)
    rows, summary = legacy.run_batch(*args, **kwargs)
    corrected = apply_target_id(rows, target_id)
    return corrected, {**summary, "target_id": target_id}
