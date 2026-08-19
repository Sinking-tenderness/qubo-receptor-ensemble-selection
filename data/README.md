# 数据说明

仓库内 `data/` 只保留小型输入、处理清单和必要说明。Stage102A 的大规模原始配体、受体和运行结果位于 `DATA_ROOT` 指定的外部数据包。

通过 `run_experiment.py --data-root` 使用，不复制成仓库内第二套数据目录。

## 当前主流程输入

完整配置从外部数据根的 raw 目录读取。FA10 和 EGFR 的配体源分别位于：

- `data/raw/external_targets/fa10_dude/fa10/actives_final.ism` 和
  `decoys_final.ism`；
- `data/raw/external_targets/egfr_dude/egfr/actives_final.ism` 和 `decoys_final.ism`。

同一目标目录还提供参考受体 PDB 和晶体配体；`data/raw/rcsb/fa10/`、`data/raw/rcsb/egfr/` 提供 RCSB 结构池。`prepare` 阶段从这些 raw 输入重新生成配体中间数据、受体 PDB/PDBQT、序列对齐结果、受体 manifest，并根据晶体配体坐标计算本次运行的 docking box。它不依赖仓库中旧的 processed manifest 或固定 `common_box.json`。

这些路径相对于 `--data-root` 解析。默认实际选择为 FA10 13 个受体、EGFR 12 个受体，以及每个目标 120 个 active 和 480 个 decoy。

## 旧回放数据

仓库中已有的 `data/processed/stage102a_*matrix.csv` 仅供旧 schema 2.0 replay
或回归 fixture 使用，不是默认完整实验的输入。完整模式从冻结配体分配或
`.ism` 结构源和受体 manifest 开始，并默认重新 docking。

运行器会自动记录文件信息和运行 provenance。用户运行步骤不需要手工计算或核对 SHA-256。大规模 docking pose、轨迹、环境文件和结果归档不应加入仓库 Git 历史。
