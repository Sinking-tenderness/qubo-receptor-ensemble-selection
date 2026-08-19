# Linux 完整实验流程

从实验数据准备开始执行完整的受体集合选择流程。Uni-Dock 需要在 Linux 环境中执行。

## 1. 固定路径和运行环境

本文示例使用以下路径：

```text
仓库：/mnt/e/Quant/qubo-receptor-ensemble-selection
数据根目录：/mnt/e/Quant/qubo_receptor_ensemble_experiment_data_20260815
运行环境：qubo-unidock
```

`--data-root` 是外部实验数据的根目录。配置文件中的 `sources` 和 `paths`
字段使用相对路径时，均相对于该目录解析；运行结果默认也写入该数据包，
不会自动写入仓库内的第二套大规模数据目录。

开始前激活已配置好的运行环境：

```bash
conda activate qubo-unidock
cd /mnt/e/Quant/qubo-receptor-ensemble-selection
python --version
command -v unidock
```

## 2. 完整流程入口

当前完整实验的唯一主入口是 `scripts/run_experiment.py`。FA10 和 EGFR
分别使用以下配置：

```text
configs/experiments/stage102a_fa10_full.json
configs/experiments/stage102a_egfr_full.json
```

典型的 FA10 执行流程如下：

```bash
cd /mnt/e/Quant/qubo-receptor-ensemble-selection

REPO_ROOT=/mnt/e/Quant/qubo-receptor-ensemble-selection
DATA_ROOT=/mnt/e/Quant/qubo_receptor_ensemble_experiment_data_20260815
CONFIG="$REPO_ROOT/configs/experiments/stage102a_fa10_full.json"

python scripts/run_experiment.py validate \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT"

python scripts/run_experiment.py plan \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT"

python scripts/run_experiment.py run \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT"
```

完整流程依次执行：

```text
prepare -> dock -> aggregate -> build_problem -> solve -> evaluate -> persist
```

各阶段职责如下：

1. `prepare`：只读取 `data/raw` 中的 DUD-E ISM、DUD-E 参考受体、晶体配体和
   RCSB 结构池。它生成来源配体 manifest、配体 3D 结构/PDBQT、RCSB PDB、
   对齐 PDB、受体 PDBQT、受体 manifest，并根据晶体配体坐标计算本次运行的
   docking box。
2. `dock`：调用当前 `qubo-unidock` 环境中的 Uni-Dock 重新 docking。
3. `aggregate`：聚合多个 seed 的 score table，生成 primary 和 sensitivity
   矩阵。
4. `build_problem`：使用本次生成的 primary matrix 构造 QUBO。
5. `solve`：调用配置中的 solver 求解受体子集。
6. `evaluate`：在配置指定的数据划分上计算评估指标。
7. `persist`：保存选择结果、评估结果、配置快照和运行 manifest。

默认选择目标是 `BEDROC20`，即 `utility_metric: "bedroc"` 和
`bedroc_alpha: 20.0`。ROC-AUC 仍会作为辅助指标记录，但不参与默认的
QUBO 选择或自适应 `k` 决策。

如果需要在同一套 docking 矩阵上尝试多个历史方法，可以把 `problem` 写成
`mode: "compare"` 并列出 `methods`。每种方法分别写入
`methods/<method_id>/problem.json`、`selection.json` 和 `evaluation.json`，同时
在运行根目录生成 `method_capabilities.json` 与 `comparison.json`。这种比较从
`build_problem` 开始即可，不需要重新 prepare、dock 或 aggregate。

`validate` 只检查输入，`plan` 只展开计划；二者都不会启动结构准备或
docking。

### 2.1 Stage102A 配体选择

两个 Stage102A 完整配置默认使用：

```json
"selection": {
  "ordering": "scaffold_hash_allocation",
  "allocation": {
    "hash_namespace": "STAGE102A",
    "outer_fold_count": 5,
    "minimum_label_counts_per_outer_fold": {
      "active": 20,
      "decoy": 80
    }
  }
}
```

`prepare` 直接从 raw `.ism` 读取配体，并按历史 Stage102A 原则进行无分数分配：
按 source molecule ID、canonical SMILES 和无手性 Bemis-Murcko scaffold 建连通组，
使用确定性 hash 排序选择 exact 120 个 active 和 480 个 decoy，排除与已选
active scaffold 重叠的 decoy，再把组分配到 5 个 outer fold。这个阶段不会读取
docking score、旧矩阵或旧的 processed manifest。选择结果写入
`source_ligands.csv`，审计摘要写入 `source_ligand_allocation_summary.json`。

`ordering: "manifest_order"` 仍然可用。启用它时，程序按 active/decoy 的 raw
`.ism` 行顺序截取配置的配额，不执行 scaffold hash 分配；它不会影响默认的
Stage102A 配置。

## 3. 示例数据

Stage102A 示例的数据和默认规模如下：

| 目标 | 受体构象数 | 配体总数 | active | decoy | docking seed 数 |
|---|---:|---:|---:|---:|---:|
| FA10 | 13 | 600 | 120 | 480 | 3 |
| EGFR | 12 | 600 | 120 | 480 | 3 |

两个示例都从外部数据包的 raw 目录读取配体和受体。配置中的路径例如：

```text
data/raw/external_targets/fa10_dude/fa10/actives_final.ism
data/raw/external_targets/fa10_dude/fa10/decoys_final.ism
data/raw/external_targets/fa10_dude/fa10/receptor.pdb
data/raw/external_targets/fa10_dude/fa10/crystal_ligand.mol2
data/raw/rcsb/fa10/
```

FA10 的 RCSB CIF 候选也可以位于该目录下的 `coordinate_pool/`；程序会递归
发现 `.cif`/`.pdb`，同一结构 ID 优先使用 CIF。EGFR 配置使用对应的
`data/raw/external_targets/egfr_dude/egfr/` 和 `data/raw/rcsb/egfr/`。这些路径都在 `DATA_ROOT`
下解析，而不是在仓库根目录下寻找。

## 4. Docking 配置

默认配置使用本机 Uni-Dock，并重新执行 docking：

```json
"docking": {
  "redock": true,
  "engine": "unidock",
  "executable": "unidock",
  "seeds": [20260821, 20260822, 20260823]
}
```

`executable: "unidock"` 表示从当前 Linux shell 的 `PATH` 查找可执行文件。
如果 `command -v unidock` 找不到它，应先修复 `qubo-unidock` 环境或在配置
中填写 Linux 环境中的绝对路径。

完整模式的 box 只配置计算规则，不填写固定中心坐标：

```json
"box": {
  "method": "ligand_bounds",
  "padding": 5.0,
  "minimum_size": [22.0, 22.0, 28.0]
}
```

`prepare` 会用 raw `crystal_ligand.mol2`/`.sdf` 的坐标计算中心和尺寸，写入
`docking_box.json`，随后把该结果传给 docking adapter。`allow_bad_res` 只在
明确配置时启用，并会在受体准备审计中记录 Meeko 删除的非模板残基。

VinaCPU 仍可作为显式的替代适配器，但不是当前默认流程：

```json
"docking": {
  "redock": true,
  "engine": "vina_cpu",
  "executable": "/path/to/vina"
}
```

不同 engine 的 score table 不得在同一次聚合中混用。完整模式要求
`docking.redock` 为 `true`；只有显式使用
`workflow_mode: "reference_replay"` 并提供已有 score table 或 matrix 时，
才允许关闭 redock。

## 5. 从中间阶段继续

可以在 JSON 中设置起止阶段，也可以用命令行覆盖：

```bash
python scripts/run_experiment.py run \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --from aggregate \
  --to persist
```

从中间阶段开始时，运行器只使用配置中明确声明的前置路径，不会自动搜索
仓库或数据包中的旧矩阵：

| 起始阶段 | 必须存在的前置路径 |
|---|---|
| `prepare` | raw `.ism`、参考受体 PDB、晶体配体和 RCSB 结构目录 |
| `dock` | `paths.prepared_ligand_manifest`、`paths.selected_receptor_manifest` |
| `aggregate` | 上述两个 manifest、`paths.score_tables` |
| `build_problem` | `paths.primary_matrix`、`paths.selected_receptor_manifest` |
| `solve` | `paths.problem` |
| `evaluate` | `paths.selection` |
| `persist` | `paths.evaluation` |

`--from` 和 `--to` 必须遵守配置声明的 canonical 阶段顺序。

## 6. 输出、续跑和覆盖

默认运行目录由配置中的 `paths.run_directory` 指定，并相对于
`DATA_ROOT` 解析。典型结构如下：

```text
results/runs/stage102a_fa10_full_local/
  source_ligands.csv
  prepared_ligands.csv
  receptors/
    source_pdb/
    aligned_pdb/
    prepared/
  receptor_preparation_audit.json
  selected_receptors.csv
  docking_box.json
  score_tables/
  matrices/
    aggregated_long.csv
    primary_median_matrix.csv
    sensitivity_minimum_matrix.csv
  problem.json
  selection.json
  evaluation.json
  config.snapshot.json
  manifest.json
  summary.json
```

中断后可以在同一运行环境中续跑：

```bash
python scripts/run_experiment.py run \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --resume
```

已有输出默认不会覆盖。只有确认需要重写当前运行目录时才使用
`--overwrite`。文件 SHA-256 和其他 provenance 由程序自动记录，不需要手工
计算或填写。

`docking_box.json` 的中心来自 raw 晶体配体坐标包围盒中心，尺寸为
`max(坐标范围 + 2 * padding, minimum_size)`。它记录晶体配体路径和 SHA-256；
`docking` 阶段只使用这个本次运行生成的 box，不接受旧的 `common_box.json`
或配置中预先填写的六个坐标值。

## 7. 旧入口边界

`scripts/run_pipeline.py` 和 `configs/pipelines/*.json` 是 schema `2.0` 的
matrix replay 兼容入口。它们从已有 score matrix 开始，不负责配体准备和
docking，也不是本文的默认完整实验入口。

旧入口的 Linux 执行方式见 [配置说明](../configs/pipelines/README.md)。新实验
统一使用：

```text
scripts/run_experiment.py
configs/experiments/*.json
```

## 8. 开发者检查

在仓库根目录执行轻量检查：

```bash
cd /mnt/e/Quant/qubo-receptor-ensemble-selection
conda activate qubo-unidock

python scripts/run_experiment.py --help
python -m pytest -q --basetemp /tmp/qubo-receptor-ensemble-selection-pytest
git diff --check
```

回归测试不会启动 600 个配体乘多受体乘多 seed 的生产 docking。完整运行前
应先执行 `validate` 和 `plan`，确认数据根目录、配置、engine 和输出目录。

## 9. 科研边界

- FA10 和 EGFR 当前是 development/train 案例，不是独立确认实验；
- 受体数量、`k`、QUBO 权重和 solver 参数不能根据 locked test 结果回调；
- 不把经典 exact QUBO 结果解释为量子优势；
- 不因为 docking 失败自动替换受体、修改 box 或回调参数；
- 不将不同 docking engine 的结果混合为同一实验矩阵。
