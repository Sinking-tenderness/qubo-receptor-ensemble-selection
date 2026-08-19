# 完整实验配置

此目录包含 schema `3.0` 的完整实验配置。Uni-Dock 需要在 Linux 环境中执行。环境创建、仓库安装和检查步骤见[完整实验流程](../../docs/experiment_workflow_zh.md)第 1 节；大规模输入和结果位于 `DATA_ROOT` 指定的数据根目录。

基本调用：

```bash
REPO_ROOT=/path/to/qubo-receptor-ensemble-selection
DATA_ROOT=/path/to/qubo_receptor_ensemble_experiment_data_20260815
cd "$REPO_ROOT"
python scripts/run_experiment.py plan \
  --config configs/experiments/stage102a_fa10_full.json \
  --data-root "$DATA_ROOT"
```

配置约定：

- `workflow_mode: "full"` 从配体结构和受体 manifest 开始；示例通过
  `selection.ordering: "scaffold_hash_allocation"` 从 raw DUD-E ISM 按历史
  Stage102A 原则重新分配配体，但仍会重新进行 3D/PDBQT 准备和 docking；
- `docking.redock: true` 是默认值，`engine: "unidock"` 是默认引擎；Uni-Dock
  需要在 Linux 环境中运行；
- `docking.executable: "unidock"` 从当前 shell 的 `PATH` 查找 Uni-Dock；
- `selection` 控制受体数、配体数和 active/decoy 配额；
- `selection.ordering` 可选 `scaffold_hash_allocation`、
  `preselected_manifest`、`manifest_order` 或 `seeded_sample`；
  `scaffold_hash_allocation` 从 `sources.active_ism` 和 `sources.decoy_ism`
  按历史 Stage102A 规则确定性分配，`preselected_manifest` 要求
  `sources.ligand_manifest`，其余两者也从 raw ISM 选择；
- `paths` 控制每个阶段输出和从中间阶段继续时的前置文件位置；
- `start_stage`/`end_stage` 可以在配置中设置，也可以由 CLI 的 `--from`/`--to`
  覆盖；
- `sources` 和 `paths` 中的相对路径相对于 `--data-root`，不是相对于仓库根目录。

VinaCPU 不是另一套流程。复制配置后只需将 `docking.engine` 改为 `vina_cpu`，
并将 `docking.executable` 改为 Linux 环境中的可执行文件路径，同时保留与目标
匹配的 box 和搜索参数。

`workflow_mode: "reference_replay"` 只用于明确的已有 score table/matrix 重放；
不能把它作为默认完整实验的别名。
