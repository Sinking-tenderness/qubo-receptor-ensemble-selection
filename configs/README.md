# 配置说明

当前新实验使用 `configs/experiments/*.json` 的 schema `3.0`。旧的
`configs/pipelines/*.json` 是 schema `2.0` 的 matrix replay 配置，保留用于
兼容，不是默认主流程。

## 完整实验配置

| 配置 | 受体数 | 配体数 | active / decoy | 默认 engine | 默认 k |
|---|---:|---:|---:|---|---:|
| `experiments/stage102a_fa10_full.json` | 13 | 600 | 120 / 480 | Uni-Dock | 2 |
| `experiments/stage102a_egfr_full.json` | 12 | 600 | 120 / 480 | Uni-Dock | 1 |

完整实验在 Linux 环境中执行。首次建立 Conda 环境、安装仓库包和检查依赖，请按[完整实验流程](../docs/experiment_workflow_zh.md)第 1 节执行。运行时使用以下路径变量：

```bash
REPO_ROOT=/path/to/qubo-receptor-ensemble-selection
DATA_ROOT=/path/to/qubo_receptor_ensemble_experiment_data_20260815
cd "$REPO_ROOT"
test -d "$DATA_ROOT/data/raw"
```

路径相对于命令行 `--data-root` 解析。默认配置中的 `docking.executable` 为 `unidock`，从激活环境的 `PATH` 查找。

源数据、准备结果和 docking 结果可以全部位于该仓库外部的数据包，不需要
复制进仓库。

## 关键字段

```json
{
  "schema_version": "3.0",
  "workflow_mode": "full",
  "start_stage": "prepare",
  "end_stage": "persist",
  "selection": {
    "receptor_count": 13,
    "ligand_count": 600,
    "label_counts": {"active": 120, "decoy": 480},
    "ordering": "preselected_manifest"
  },
  "docking": {
    "redock": true,
    "engine": "unidock",
    "executable": "unidock",
    "seeds": [20260821, 20260822, 20260823]
  }
}
```

`sources` 和 `paths` 中的相对路径都相对于 `--data-root`。Stage102A 完整配置
直接读取 raw 目录中的 active/decoy `.ism`、参考受体 PDB、晶体配体和 RCSB
结构目录；`prepare` 阶段据此生成本次运行的配体 manifest、对齐受体 PDB/PDBQT、
受体 manifest 和 docking box。Stage102A 默认使用
`scaffold_hash_allocation`，复现历史的 source ID/canonical SMILES/scaffold 分组
和确定性配额选择；`manifest_order` 仍可显式用于按源文件顺序选择。
`preselected_manifest` 只适用于明确冻结输入的其他实验或 replay。

`redock` 默认为 `true`。完整模式下不能关闭它；关闭 redock 必须显式改为
`workflow_mode: "reference_replay"`。支持的 engine 为 `unidock` 和 `vina_cpu`。默认 `executable` 为 `unidock`，从激活环境的 `PATH` 查找。

## 阶段起点和前置路径

可以在配置中设置 `start_stage`/`end_stage`，也可以用命令行 `--from`/`--to`
覆盖。被跳过阶段的输入必须在 `paths` 中声明：

- `prepare`：raw active/decoy `.ism`、参考受体 PDB、晶体配体和 RCSB 目录；
- `dock`：`prepared_ligand_manifest`、`selected_receptor_manifest`、`docking_box`；
- `aggregate`：上述 manifest 和 `score_tables`；
- `build_problem`：`primary_matrix` 和受体 manifest；
- `solve`：`problem`；
- `evaluate`：`selection`；
- `persist`：`evaluation`。

运行器只检查和使用当前配置指定的路径，不隐式读取仓库旧的矩阵。文件
provenance 由程序自动记录，不要求用户手工填 SHA-256。

## 命令

```bash
REPO_ROOT=/path/to/qubo-receptor-ensemble-selection
DATA_ROOT=/path/to/qubo_receptor_ensemble_experiment_data_20260815
cd "$REPO_ROOT"

CONFIG=configs/experiments/stage102a_fa10_full.json
python scripts/run_experiment.py validate --config "$CONFIG" --data-root "$DATA_ROOT"
python scripts/run_experiment.py plan --config "$CONFIG" --data-root "$DATA_ROOT"
python scripts/run_experiment.py run --config "$CONFIG" --data-root "$DATA_ROOT"
```

## QUBO 目标和方法比较

完整实验默认使用 `BEDROC20`：

```json
"problem": {
  "strategy": "qubo",
  "utility_metric": "bedroc",
  "bedroc_alpha": 20.0
}
```

需要比较多种方法时，使用 `problem.mode: "compare"` 和 `problem.methods`。可直接复用已经完成的 `aggregate` 结果，从 `build_problem` 开始运行：

```bash
CONFIG=configs/experiments/stage102a_method_comparison_template.json
python scripts/run_experiment.py validate --config "$CONFIG" --data-root "$DATA_ROOT"
python scripts/run_experiment.py run --config "$CONFIG" --data-root "$DATA_ROOT" --from build_problem --to persist
```

结果写入 `results/runs/<experiment>/methods/<method_id>/`，根目录同时生成 `method_capabilities.json` 和 `comparison.json`。结构、MD、辅助变量或硬件方法缺少依赖时会标记为 `unsupported_for_input`，不会静默降级为普通 QUBO。

详见 [实验流程](../docs/experiment_workflow_zh.md)。
