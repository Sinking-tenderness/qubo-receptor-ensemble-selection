"""Audit the DUD-E ESR2 reference against the frozen wild-type protocol rule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verified(root: Path, record: dict[str, str]) -> Path:
    path = root / record["path"]
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"input hash mismatch: {record['path']}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def line_count(path: Path) -> int:
    with path.open(encoding="utf-8-sig") as handle:
        return sum(1 for _ in handle)


def reference_row(rows: list[dict[str, str]], target_id: str, pdb_id: str) -> dict[str, str]:
    matches = [row for row in rows if row["target_id"] == target_id and row["pdb_id"] == pdb_id]
    if len(matches) != 1:
        raise ValueError("ESR2 reference metadata row is ambiguous")
    return matches[0]


def report(result: dict[str, Any]) -> str:
    reference = result["reference_metadata"]
    return "\n".join(
        [
            "# Stage109: ESR2 参考复合物身份门",
            "",
            "## 目的",
            "",
            "按既有的结果盲候选排序检查 ESR2 是否能进入野生型受体构象协议。本阶段仅核对活性/诱饵文件行数，不解析或使用分子结构/标签值；也不下载 RCSB 坐标或运行对接。",
            "",
            "## 结果",
            "",
            "- DUD-E ESR2 包的共晶配体标签为 `OHT_101_2FSZ`，即参考复合物 `2FSZ`。",
            f"- RCSB 元数据显示 `2FSZ` 的突变数为 `{reference['mutation_count']}`：`{reference['pdbx_mutation_note']}`。",
            f"- 其元数据状态为 `{reference['status']}`，排除原因：`{reference['exclusion_reasons']}`。",
            f"- ESR2 元数据合格结构数为 `{result['counts']['metadata_eligible_receptor_count']}`，虽超过池规模门，但参考本身不合格。",
            "",
            "## 决定",
            "",
            "**No-Go。** 不能在已看到 `2FSZ` 含突变后更换参考、降低野生型条件，或把同一来源配体迁移到另一个构象框架。本阶段在坐标下载、配体分配和对接前停止。",
            "",
        ]
    )


def run(root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads((root / config_path).read_text(encoding="utf-8"))
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    ranking = config["ranking_provenance"]
    completed = json.loads((root / ranking["completed_ppard_result"]["path"]).read_text(encoding="utf-8"))
    selection = json.loads((root / ranking["ppard_selection_record"]["path"]).read_text(encoding="utf-8"))
    if completed["status"] != "stage62_ppard_train240_frozen_nested_qubo_complete":
        raise ValueError("PPARD completion record differs")
    if selection["selection"]["eligible_target_order"] != ["PPARA", "PPARD", "ESR2"]:
        raise ValueError("unexpected historical target order")
    if any(value != 0 for value in config["data_boundary"].values()):
        boundary = config["data_boundary"]
        allowed = {
            "source_active_lines_counted": 367,
            "source_decoy_lines_counted": 20199,
            "source_label_values_parsed_or_used": 0,
            "rcsb_coordinate_downloads": 0,
            "docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        }
        if boundary != allowed:
            raise ValueError("Stage109 protected-data boundary differs")
    if line_count(inputs["dude_actives"]) != int(config["data_boundary"]["source_active_lines_counted"]):
        raise ValueError("ESR2 active source line count differs")
    if line_count(inputs["dude_decoys"]) != int(config["data_boundary"]["source_decoy_lines_counted"]):
        raise ValueError("ESR2 decoy source line count differs")

    rows = read_csv(inputs["historical_metadata"])
    target = config["target"]
    reference = reference_row(rows, target["target_id"], target["dude_reference_structure"])
    receptor_count = sum(row["target_id"] == target["target_id"] and row["status"] == "metadata_eligible" for row in rows)
    mutation_count = int(reference["mutation_count"])
    rule = config["frozen_reference_rules"]
    reference_is_wild_type = mutation_count == int(rule["required_mutation_count"])
    reference_is_eligible = reference["status"] == "metadata_eligible"
    pool_passes = receptor_count >= int(rule["minimum_metadata_eligible_receptor_count"])
    gate_passes = reference_is_wild_type and reference_is_eligible and pool_passes
    if gate_passes:
        raise ValueError("a passing Stage109 requires an independently reviewed ESR2 protocol before release")

    result = {
        "schema_version": "1.0",
        "status": "stage109_esr2_reference_identity_no_go",
        "evidence_status": "outcome-unseen source and public-metadata integrity gate; not an enrichment or QUBO result",
        "config": {"path": str(config_path).replace("\\", "/"), "sha256": sha256(root / config_path)},
        "target": target,
        "ranking_provenance": {
            "historical_eligible_order": selection["selection"]["eligible_target_order"],
            "completed_before_esr2": ["PPARA", "PPARD"],
            "esr2_next_unused_target": True,
        },
        "source_archive": {"path": config["inputs"]["dude_archive"]["path"], "sha256": sha256(inputs["dude_archive"]), "size_bytes": inputs["dude_archive"].stat().st_size},
        "reference_metadata": {
            "pdb_id": reference["pdb_id"],
            "dude_ligand_label": target["dude_reference_ligand_name"],
            "status": reference["status"],
            "exclusion_reasons": reference["exclusion_reasons"],
            "mutation_count": mutation_count,
            "pdbx_mutation_note": reference["pdbx_mutation_note"],
            "resolution_angstrom": float(reference["resolution_angstrom"]),
            "reference_sequence_coverage": float(reference["reference_sequence_coverage"]),
            "entity_sequence_coverage": float(reference["entity_sequence_coverage"]),
        },
        "counts": {"metadata_eligible_receptor_count": receptor_count, "minimum_required_receptor_count": int(rule["minimum_metadata_eligible_receptor_count"])},
        "gate_checks": {"reference_is_wild_type": reference_is_wild_type, "reference_is_metadata_eligible": reference_is_eligible, "receptor_pool_passes": pool_passes, "gate_passes": False},
        "data_boundary": config["data_boundary"],
        "decision": {
            "source_identity_gate_passed": False,
            "rcsb_coordinate_download_authorized": False,
            "ligand_panel_allocation_authorized": False,
            "cognate_redocking_authorized": False,
            "production_docking_authorized": False,
            "fresh_validation_authorized": False,
            "locked_test_authorized": False,
            "quantum_hardware_authorized": False,
            "next_action": rule["failure_action"],
        },
    }
    output = config["outputs"]
    summary_path = root / output["summary_json"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    report_path = root / output["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage109_esr2_reference_identity_gate.json"))
    args = parser.parse_args()
    result = run(args.root.resolve(), args.config)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
