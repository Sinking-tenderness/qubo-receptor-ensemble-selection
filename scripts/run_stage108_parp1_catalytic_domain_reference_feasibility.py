"""Audit whether PARP1 catalytic-domain metadata can support the frozen pool size."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def require_hash(root: Path, record: dict[str, str]) -> Path:
    path = root / record["path"]
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"input hash mismatch: {record['path']}")
    return path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def eligible(row: dict[str, str], rule: dict[str, Any]) -> bool:
    return (
        int(row["mutation_count"]) == int(rule["required_mutation_count"])
        and float(row["resolution_angstrom"]) <= float(rule["maximum_resolution_angstrom"])
        and int(rule["minimum_sample_sequence_length"]) <= int(row["sample_sequence_length"]) <= int(rule["maximum_sample_sequence_length"])
        and float(row["entity_sequence_coverage"]) >= float(rule["minimum_entity_sequence_coverage"])
        and int(row["qualifying_ligand_count"]) >= int(rule["minimum_qualifying_ligand_count"])
        and not str(row[rule["forbidden_title_pattern_field"]]).strip()
    )


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def report(record: dict[str, Any]) -> str:
    count = record["counts"]["metadata_eligible_count"]
    minimum = record["pool_gate"]["minimum_receptors_for_contact_state_protocol"]
    reference = record["provisional_reference"]["pdb_id"]
    return "\n".join(
        [
            "# Stage108: PARP1 催化结构域参考框架可行性审计",
            "",
            "## 目的",
            "",
            "本阶段只重审公开元数据：若将结构范围明确定义为 PARP1 催化结构域，是否仍有足够的野生型共晶结构支撑既定的 16 受体接触状态覆盖协议。没有下载坐标、读取配体面板或运行对接。",
            "",
            "## 结果",
            "",
            f"- 元数据合格的催化结构域结构：`{count}` 个。",
            f"- 按预先登记的最低分辨率规则得到的暂定参考：`{reference}`。",
            f"- 已冻结协议要求的最小构象池：`{minimum}` 个。",
            "- 结论：**No-Go**。`10 < 16`，因此不能把该小池伪装成原 16 受体确认实验。",
            "",
            "## 解释",
            "",
            "Stage107 的全长覆盖率规则不适合催化结构域；本阶段以实体覆盖率检查催化域结构的完整性，得到可解释的结构来源。但池规模仍不够。未来可以另行提出一个小规模 PARP1 探索协议，然而它必须重新定义问题、比较基线和成功门槛，不能用于挽救或确认现有 16 受体结论。",
            "",
            "## 锁定边界",
            "",
            "本阶段不释放坐标下载、配体制备、重对接、生产对接、新鲜验证、锁定测试或量子硬件。",
            "",
        ]
    )


def run(root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads((root / config_path).read_text(encoding="utf-8"))
    parent_path = require_hash(root, config["parent_stage"])
    metadata_path = require_hash(root, config["input_metadata"])
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent["status"] != "stage107_parp1_contact_state_metadata_intake_no_go":
        raise ValueError("Stage108 requires the frozen Stage107 No-Go result")
    if any(value != 0 for value in config["data_boundary"].values()):
        raise ValueError("Stage108 must begin with an empty protected-data boundary")

    rows = read_rows(metadata_path)
    selected = sorted(
        (row for row in rows if eligible(row, config["metadata_eligibility"])),
        key=lambda row: (float(row["resolution_angstrom"]), row["pdb_id"]),
    )
    if not selected:
        raise ValueError("catalytic-domain audit found no eligible structures")

    minimum = int(config["pool_gate"]["minimum_receptors_for_contact_state_protocol"])
    gate_passes = len(selected) >= minimum
    outputs = config["outputs"]
    fieldnames = list(rows[0])
    write_csv(root / outputs["eligible_candidates_csv"], selected, fieldnames)
    result = {
        "schema_version": "1.0",
        "status": "stage108_parp1_catalytic_domain_reference_feasibility_no_go" if not gate_passes else "stage108_parp1_catalytic_domain_reference_feasibility_pool_gate_passed",
        "evidence_status": "posthoc scope-definition feasibility audit using only previously collected public metadata; not an efficacy result, trained rule, or confirmation experiment",
        "config": {"path": str(config_path).replace("\\", "/"), "sha256": sha256(root / config_path)},
        "parent_stage": {"path": config["parent_stage"]["path"], "sha256": config["parent_stage"]["sha256"], "status": parent["status"]},
        "scope": config["scope"],
        "counts": {"input_metadata_count": len(rows), "metadata_eligible_count": len(selected), "minimum_required_count": minimum, "pool_gate_passes": gate_passes},
        "eligible_pdb_ids": [row["pdb_id"] for row in selected],
        "provisional_reference": {"pdb_id": selected[0]["pdb_id"], "resolution_angstrom": float(selected[0]["resolution_angstrom"]), "selection_status": "metadata-only provisional reference; coordinate work remains locked"},
        "pool_gate": config["pool_gate"],
        "data_boundary": config["data_boundary"],
        "decision": {
            "coordinate_structural_audit_authorized": False,
            "ligand_preparation_authorized": False,
            "redocking_authorized": False,
            "production_docking_authorized": False,
            "fresh_validation_released": False,
            "locked_test_released": False,
            "quantum_hardware_authorized": False,
            "next_action": "Stop PARP1 under the frozen 16-receptor protocol. A future smaller-pool study requires a separately reviewed protocol and is not a confirmation of this branch.",
        },
    }
    if gate_passes:
        raise ValueError("A passed pool gate needs a separately reviewed protocol; do not auto-release later work")
    summary_path = root / outputs["summary_json"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    report_path = root / outputs["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage108_parp1_catalytic_domain_reference_feasibility.json"))
    args = parser.parse_args()
    record = run(args.root.resolve(), args.config)
    print(json.dumps(record, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
