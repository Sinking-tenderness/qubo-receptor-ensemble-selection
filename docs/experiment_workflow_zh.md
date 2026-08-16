# 实验流程指南（环境构建 → 数据准备 → 运行代码）

> 适用仓库：`qubo-receptor-ensemble-selection`（dev_ylj 分支，重构后状态）。
> 默认运行环境：**Linux**（命令均为 bash 语法）。历史 stage 脚本已从工作区移除，git 历史中完整保留。

---

## 1. 仓库结构

```
仓库根
├── src/qubo_receptor_ensemble/   # 可复用核心（io/pdb/screening/qubo/docking/matrix/preparation/ligand/md/metrics）
├── scripts/                      # 41 个命令行入口（workflow.py 是目录）
├── configs/                      # 预注册实验配置 JSON
├── data/                         # 实验数据（本地，不入 git）
├── results/                      # 小规模结果表（JSON/CSV）
├── environment/                  # conda 环境定义
├── tests/                        # 活代码测试 + 数据记录验证
├── reports/                      # stage 报告与交接文档（历史记录）
└── conftest.py                   # pytest 根配置（src 免安装可导入）
```

## 2. 数据与 git 边界

- 大数据不进入 git：`data/`（原始/处理数据、对接输出、轨迹）、`results/runs/`、`*.tar.gz`、`*.dcd` 等已在 `.gitignore` 中排除。
- 新增数据放在本地 `data/` 或外部数据包目录，不要 `git add`。
- git 只保存：代码、配置（configs/）、小结果表、文档、测试。
- 实验配置在运行任何代码之前写好并提交 git；配置中记录输入文件的 SHA-256，脚本运行时校验输入一致性，输入变化会报错。

## 3. 环境构建

### 3.1 主环境

```bash
conda env create -f environment/environment.yml
conda activate qubo-receptor-ensemble
python -m pip install -e .
```

- 环境名 `qubo-receptor-ensemble`，Python 3.11，含 numpy/pandas/scipy/sklearn/rdkit/xgboost 3.1.1/dimod/meeko/prody（见 `environment/environment.yml`）。
- `environment.yml` 内 `pip: -e ..` 的相对路径取决于执行目录，activate 后手动执行 `python -m pip install -e .`。
- 验证：

```bash
python -c "import qubo_receptor_ensemble; print(qubo_receptor_ensemble.__version__)"
python scripts/workflow.py list
python -m pytest -q
```

- 无 conda 环境时，仓库根的 `conftest.py` 会把 `src/` 加入 sys.path，`python -m pytest` 可直接运行。

### 3.2 专用环境

| 环境文件 | 用途 |
| --- | --- |
| `stage03_openmm.yml` | OpenMM 分子动力学 |
| `stage05_unidock_gpu.yml` 等 stage*.yml | Uni-Dock GPU 对接 |
| `stage78_dwave_advantage2.yml` / `stage79_qci_dirac3.yml` | 量子硬件 PoC（已收敛） |
| `qaoa-environment.yml` | QAOA 仿真 |

```bash
conda env create -f environment/stage03_openmm.yml
```

## 4. 数据准备

### 4.1 数据包解压

数据包是 tar.gz，位于本地 `data/`（不入 git）或外部完整数据包目录：

```
data/abl1.tar.gz   data/braf.tar.gz   data/cdk2.tar.gz   data/egfr.tar.gz
```

解压：

```bash
cd data
tar -xzf abl1.tar.gz
```

解压后每个靶点目录包含：

```
abl1/
├── receptor.pdb           # 受体结构
├── crystal_ligand.mol2    # 共晶配体（重对接验证用）
├── actives_final.ism      # DUD-E 活性 SMILES（一行一个）
├── decoys_final.ism       # DUD-E 诱饵 SMILES（一行一个）
└── actives_final.mol2.gz / decoys_final.mol2.gz / *.sdf.gz   # 3D 结构
```

### 4.2 配体清单（ligands.csv）

`check_ligand_smiles.py` 的输入是配体清单 CSV，核心列：

```
ligand_id,smiles,label,source,target_id
```

清单从 `.ism` 文件构建，使用 `scripts/` 下的两个脚本：

```bash
python scripts/make_dude_subset.py --help
python scripts/build_dude_ligand_manifest.py --help
```

- `make_dude_subset.py`：按靶点从 `.ism` 抽活性/诱饵子集（可指定数量与种子）。
- `build_dude_ligand_manifest.py`：把 `actives_final.ism` / `decoys_final.ism` 合成为
  `ligand_id,smiles,label,source,target_id` 清单 CSV。

自建配体时，自行准备含 5 列的 CSV（`source` 填来源库名，`target_id` 填靶点标识）。
格式参照仓库历史样例 `data/processed/stage05_mk14_development_120a120d.csv`。

### 4.3 配体准备（SMILES → 3D SDF → PDBQT）

```bash
# ① SMILES 审计（RDKit 解析、去重、理化性质统计）
python scripts/check_ligand_smiles.py --input ligands.csv --output audited.csv --summary summary.json

# ② 显式氢 3D SDF（ETKDG 嵌入 + MMFF/UFF 优化）
python scripts/prepare_ligand_3d_sdf.py --input audited.csv --sdf-dir sdf/ --manifest prep3d.csv --seed 20260709

# ③ 并行 PDBQT 制备（Meeko，可续跑）
python scripts/batch_prepare_ligand_pdbqt_parallel.py --input-manifest prep3d.csv --pdbqt-dir pdbqt/ --output-manifest pdbqt_manifest.csv --workers 4 --resume
```

### 4.4 受体（PDB → 对齐 → PDBQT）

```bash
# ① Kabsch 刚性对齐（Cα 匹配）
python scripts/align_receptor_structure.py --reference ref.pdb --mobile mobile.pdb --output aligned.pdb --summary-output align.json

# ② 清理并参数化（ProDy 去水/杂原子 + Meeko 受体）
python scripts/prepare_receptor.py --input-pdb aligned.pdb --chain A --protein-only-output protein.pdb --prepared-pdb-output prepared.pdb --pdbqt-output receptor.pdbqt --summary-output summary.json
```

### 4.5 共晶重对接验证

```bash
python scripts/evaluate_redocking_rmsd.py --case-id demo --reference-sdf ligand.sdf --docked-pdbqt docked.pdbqt --pose-table-output poses.csv --summary-output rmsd.json
```

对称修正重原子 RMSD ≤ 2.0 Å 视为成功。

## 5. 运行流水线

supported 流程统一从 `workflow.py` 出发：

```bash
python scripts/workflow.py list
python scripts/workflow.py show dock-vina
python scripts/workflow.py run dock-vina -- --help
python scripts/workflow.py run dock-vina -- <参数...>
```

### 5.1 对接（Uni-Dock）

对接引擎为 **Uni-Dock 1.1.3**（GPU）。环境与安装：

```bash
conda env create -f environment/stage05_unidock_gpu.yml   # 环境 qubo-unidock，含 unidock=1.1.3
conda activate qubo-unidock
```

生产入口在 `scripts/experimental/unidock/`，每个靶点一个生产执行器（`run_stageXX_<target>_production.py`），
以 MK14 Train-696 为例：

```bash
python scripts/experimental/unidock/run_stage09_mk14_train696_production.py \
  --config configs/stage09_mk14_train696_unidock113_production.json \
  --root . --unidock <unidock可执行文件> --resume
```

参数：`--config`（预注册配置）、`--root`（仓库根）、`--unidock`（可执行路径）、`--resume`、
`--seed-id`、`--receptor-id`、`--audit-only`、`--finalize-only`。

Uni-Dock 三个搜索 profile：`fast` / `balance` / `detail`，在配置的 `search` 段指定；
生产矩阵使用确认过的 profile（如 enhanced 1024/80），主评价指标 BEDROC20。

CPU 参考：早期官方矩阵（CDK2/MK14 早期 stage）由 AutoDock Vina 1.2.7 生成，
`scripts/batch_vina_docking_parallel.py` 仍可用作 CPU 基准；Uni-Dock 与 Vina 是不同引擎，
分数不混用。

### 5.2 打分矩阵

```bash
# 单 seed：代表分（pose_rank_1 或 min_score）+ 长表/宽矩阵
python scripts/build_score_matrix.py --score-table scores.csv --long-output long.csv --matrix-output matrix.csv --summary-output summary.json --representative pose_rank_1

# 多 seed 聚合（中位数/最小值矩阵）
python scripts/aggregate_seed_replicates.py --config configs/xxx_aggregate.json
```

### 5.3 筛选评估

```bash
python scripts/evaluate_virtual_screening.py --score-table scores.csv --ranking-output ranked.csv --metrics-output metrics.json --top-fractions 0.01 0.05 --bedroc-alpha 20
```

指标：ROC-AUC（pairwise）、PR-AUC、BEDROC20、EF1%/5%/10%、bootstrap 95% CI。

### 5.4 受体组合选择

```bash
# 原型：穷举小 QUBO 子集（train 分片内）
python scripts/solve_qubo_receptor_subset.py --matrix matrix.csv --split-manifest split.csv --receptor r1 r2 r3 --output result.json --target-size 2

# 正式：嵌套 scaffold CV 拟合选择方法（fit-ensemble）
python scripts/run_development_scaffold_cv_gate.py --config configs/xxx_cv_gate.json
```

选择方法只在 train 分片内拟合；locked test 标签在预注册门通过前锁定。

### 5.5 分子动力学

```bash
python scripts/build_openmm_system.py --protocol configs/xxx_md.json --manifest-output build.json --solvated-pdb-output solvated.pdb --system-xml-output system.xml
python scripts/run_openmm_equilibration.py --config configs/xxx_equil.json --overwrite
python scripts/run_openmm_production.py --config configs/xxx_prod.json --resume
python scripts/analyze_md_trajectory.py --config configs/xxx_traj_qc.json
```

### 5.6 完整最小示例

```bash
conda activate qubo-receptor-ensemble
python -m pip install -e .

# 数据包 → 配体清单
tar -xzf data/abl1.tar.gz -C data/
python scripts/build_dude_ligand_manifest.py --help   # 按脚本参数生成 ligands.csv

# 配体准备
python scripts/check_ligand_smiles.py --input ligands.csv --output audited.csv --summary audit.json
python scripts/prepare_ligand_3d_sdf.py --input audited.csv --sdf-dir sdf --manifest sdf3d.csv
python scripts/batch_prepare_ligand_pdbqt_parallel.py --input-manifest sdf3d.csv --pdbqt-dir pdbqt --output-manifest pdbqt.csv --workers 4

# 受体准备
python scripts/prepare_receptor.py --input-pdb data/abl1/receptor.pdb --chain A --protein-only-output protein.pdb --prepared-pdb-output prep.pdb --pdbqt-output receptor.pdbqt --summary-output receptor.json

# 对接（Uni-Dock，GPU）
conda activate qubo-unidock
python scripts/experimental/unidock/run_stage09_mk14_train696_production.py --config configs/stage09_mk14_train696_unidock113_production.json --root . --unidock unidock --resume

# 矩阵 + 评估
python scripts/build_score_matrix.py --score-table scores.csv --long-output long.csv --matrix-output matrix.csv --summary-output matrix.json
python scripts/evaluate_virtual_screening.py --score-table long.csv --ranking-output ranked.csv --metrics-output metrics.json

# 组合选择（train 分片内）
python scripts/solve_qubo_receptor_subset.py --matrix matrix.csv --split-manifest split.csv --receptor r1 r2 r3 --output qubo.json --target-size 2
```

## 6. 验证与回归

```bash
python -m pytest -q
python -m compileall scripts src
python scripts/workflow.py list
```

## 7. 恢复历史脚本

```bash
git log --oneline -- scripts/
git show eb958ea^:scripts/run_stage111_thrb_identity_adjudication.py > scripts/run_stage111_thrb_identity_adjudication.py
```

## 8. 科研纪律

1. 不用已看过的外层结果继续调 QUBO 系数、k 阈值或接触状态阈值，避免验证集变成训练集。
2. 各靶点的身份与参考结构以预注册配置为准，不确定时先查交接文档。
3. 没有新的外部价值实例前，不租量子硬件、不主张量子优势。

详见 `reports/handover/successor_quickstart_20260815_zh.md`。
