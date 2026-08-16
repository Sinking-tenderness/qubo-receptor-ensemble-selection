# 实验流程指南（环境构建 → 数据准备 → 运行代码）

> 适用仓库：`qubo-receptor-ensemble-selection`（dev_ylj 分支，重构后状态）。
> 本文描述**当前可用的活代码**流程：`src/` 是可复用核心，`scripts/` 是 41 个命令行入口，
> 历史 stage 脚本已从工作区移除（git 历史中完整保留，可随时恢复）。

---

## 1. 仓库结构（当前状态）

```
仓库根
├── src/qubo_receptor_ensemble/   # 可复用核心（io/pdb/screening/qubo/docking/matrix/preparation/ligand/md/metrics）
├── scripts/                      # 41 个命令行入口（workflow.py 是目录）
├── configs/                      # 355 个预注册实验配置 JSON（含期望哈希）
├── data/                         # 实验数据（tar.gz 包 + 解压目录 + processed/）
├── results/                      # 小规模结果表（JSON/CSV）
├── environment/                  # conda 环境定义（15 个 yml）
├── tests/                        # 86 个测试（活代码测试 + 数据记录验证）
├── reports/                      # stage 报告与交接文档（历史记录，不参与运行）
└── conftest.py                   # pytest 根配置（src 免安装可导入）
```

---

## 2. 环境构建

### 2.1 主环境（推荐）

```powershell
# 在仓库根目录执行
conda env create -f environment/environment.yml
conda activate qubo-receptor-ensemble
python -m pip install -e .
```

要点：

- 主环境名 `qubo-receptor-ensemble`，Python 3.11，包含 numpy/pandas/scipy/sklearn/
  rdkit/xgboost 3.1.1/dimod/meeko/prody 等（见 `environment/environment.yml`）。
- `environment.yml` 内有一行 `pip: -e ..`，其相对路径取决于执行 conda 命令时的
  工作目录——**不要依赖它**，activate 后手动执行 `python -m pip install -e .` 最稳妥。
- 装完验证：

```powershell
python -c "import qubo_receptor_ensemble; print(qubo_receptor_ensemble.__version__)"
python scripts/workflow.py list      # 应列出全部 supported 流水线
python -m pytest -q                  # 86 个测试（本机缺数据/依赖时部分会失败，属环境问题）
```

> 注意：本机若无法创建 conda 环境（如沙箱/网络限制），可用 `conftest.py` 的
> 机制直接运行：仓库根下 `python -m pytest` 会自动把 `src/` 加入 sys.path。

### 2.2 专用环境

按实验需要创建（都放在 `environment/` 下）：

| 环境文件 | 用途 |
| --- | --- |
| `stage03_openmm.yml` | OpenMM 分子动力学（MD 构建/平衡/生产） |
| `stage05_unidock_gpu.yml` 等 stage*.yml | Uni-Dock GPU 对接（GPU 机） |
| `stage78_dwave_advantage2.yml` / `stage79_qci_dirac3.yml` | 量子硬件 PoC（已收敛，暂不需要） |
| `qaoa-environment.yml` | QAOA 仿真 |

```powershell
conda env create -f environment/stage03_openmm.yml
```

### 2.3 实验记录约定

- 实验配置（configs/*.json）在**跑任何代码之前**写好并提交 git（预注册）。
- 配置里钉住：输入路径、参数、随机种子、期望 SHA-256。不要在一个已有打分矩阵
  的流程中替换对接引擎或参数。
- 被钉死的哈希对应脚本内容——修改脚本会使校验失败，这是设计而非故障。

---

## 3. 数据准备

### 3.1 数据包

- 仓库内 `data/`：`abl1/ braf/ cdk2/ egfr/ processed/` 及对应 `*.tar.gz`。
- 完整数据包（历史交付包，65,454 条目）见交接文档
  `reports/handover/quantum_receptor_ensemble_handover_20260815.md`。
- 大文件不入 git：新增数据放在 `data/` 下但不要 `git add`（.gitignore 已排除 raw/processed 等）。

### 3.2 配体准备（SMILES → 3D SDF → PDBQT）

输入：配体清单 CSV，必须含列 `ligand_id, smiles, label, source, target_id`。

```powershell
# ① 审计 SMILES（RDKit 解析、去重、理化性质统计）
python scripts/check_ligand_smiles.py --input ligands.csv --output audited.csv --summary summary.json

# ② 生成显式氢 3D SDF（ETKDG 嵌入 + MMFF/UFF 优化）
python scripts/prepare_ligand_3d_sdf.py --input audited.csv --sdf-dir sdf/ --manifest prep3d.csv --seed 20260709

# ③ 并行制备 PDBQT（Meeko，可断点续跑）
python scripts/batch_prepare_ligand_pdbqt_parallel.py --input-manifest prep3d.csv --pdbqt-dir pdbqt/ --output-manifest pdbqt_manifest.csv --workers 4 --resume
```

### 3.3 受体准备（PDB → 对齐 → PDBQT）

```powershell
# ① 刚性对齐到参考坐标框（Kabsch，Cα 匹配）
python scripts/align_receptor_structure.py --reference ref.pdb --mobile mobile.pdb --output aligned.pdb --summary-output align.json

# ② 清理并参数化受体（ProDy 去水/杂原子 + Meeko 制备受体）
python scripts/prepare_receptor.py --input-pdb aligned.pdb --chain A --protein-only-output protein.pdb --prepared-pdb-output prepared.pdb --pdbqt-output receptor.pdbqt --summary-output summary.json
```

### 3.4 共晶重对接验证（协议校验）

```powershell
python scripts/evaluate_redocking_rmsd.py --case-id demo --reference-sdf ligand.sdf --docked-pdbqt docked.pdbqt --pose-table-output poses.csv --summary-output rmsd.json
```

- 对称修正重原子 RMSD ≤ 2.0 Å 视为成功；先跑共晶重对接，再跑正式筛选。

---

## 4. 运行实验流水线

所有 supported 流程都能从 `workflow.py` 出发：

```powershell
python scripts/workflow.py list              # 查看全部
python scripts/workflow.py show dock-vina    # 查看某条
python scripts/workflow.py run dock-vina -- --help   # 查看参数
python scripts/workflow.py run dock-vina -- <参数...> # 运行
```

### 4.1 对接（打分生成）

```powershell
# CPU AutoDock Vina 1.2.7（官方打分，生产路径）
python scripts/batch_vina_docking_parallel.py `
  --manifest pdbqt_manifest.csv --vina-exe <vina路径> --receptor receptor.pdbqt --receptor-id r1 `
  --config box.cfg --output-dir poses/ --log-dir logs/ --score-table scores.csv `
  --workers 8 --base-seed 20260709 --resume

# GPU Uni-Dock（实验性；只用于消耗过的训练组诊断，不混入官方矩阵）
python scripts/experimental/unidock/run_unidock_gpu_equivalence.py --help
```

box.cfg 键：`center_x/y/z, size_x/y/z, exhaustiveness, num_modes[, cpu]`。

### 4.2 打分矩阵（长表 → 配体×受体矩阵）

```powershell
# 单 seed：选代表分（pose_rank_1 或 min_score），建长表+宽矩阵
python scripts/build_score_matrix.py --score-table scores.csv --long-output long.csv --matrix-output matrix.csv --summary-output summary.json --representative pose_rank_1

# 多 seed 聚合（审计哈希 → 中位数/最小值矩阵 → primary/sensitivity）
python scripts/aggregate_seed_replicates.py --config configs/xxx_aggregate.json
```

### 4.3 筛选评估（指标）

```powershell
python scripts/evaluate_virtual_screening.py --score-table scores.csv --ranking-output ranked.csv --metrics-output metrics.json --top-fractions 0.01 0.05 --bedroc-alpha 20
```

指标：ROC-AUC（pairwise）、PR-AUC、BEDROC20、EF1%/5%/10%、bootstrap 95% CI。

### 4.4 受体组合选择（QUBO / 经典基线）

```powershell
# 原型：穷举求解小 QUBO 子集（开发集 train 分片内）
python scripts/solve_qubo_receptor_subset.py --matrix matrix.csv --split-manifest split.csv --receptor r1 r2 r3 --output result.json --target-size 2

# 正式：嵌套 scaffold CV 拟合选择方法（fit-ensemble，主入口）
python scripts/run_development_scaffold_cv_gate.py --config configs/xxx_cv_gate.json
```

> 纪律：选择方法只在 **train 分片**内拟合；locked test 标签在预注册门通过前锁定，
> 不得用外层结果调 QUBO 系数 / k 阈值（见交接文档红线）。

### 4.5 分子动力学（可选：生成受体构象池）

```powershell
python scripts/build_openmm_system.py --protocol configs/xxx_md.json --manifest-output build.json --solvated-pdb-output solvated.pdb --system-xml-output system.xml
python scripts/run_openmm_equilibration.py --config configs/xxx_equil.json --overwrite
python scripts/run_openmm_production.py --config configs/xxx_prod.json --resume
python scripts/analyze_md_trajectory.py --config configs/xxx_traj_qc.json
```

### 4.6 完整最小示例（一条龙）

```powershell
# 1) 环境
conda activate qubo-receptor-ensemble
python -m pip install -e .

# 2) 数据
python scripts/check_ligand_smiles.py --input ligands.csv --output audited.csv --summary audit.json
python scripts/prepare_ligand_3d_sdf.py --input audited.csv --sdf-dir sdf --manifest sdf3d.csv
python scripts/batch_prepare_ligand_pdbqt_parallel.py --input-manifest sdf3d.csv --pdbqt-dir pdbqt --output-manifest pdbqt.csv --workers 4
python scripts/prepare_receptor.py --input-pdb aligned.pdb --chain A --protein-only-output protein.pdb --prepared-pdb-output prep.pdb --pdbqt-output receptor.pdbqt --summary-output receptor.json

# 3) 对接（CPU Vina）
python scripts/batch_vina_docking_parallel.py --manifest pdbqt.csv --vina-exe vina --receptor receptor.pdbqt --receptor-id r1 --config box.cfg --output-dir poses --log-dir logs --score-table scores.csv --workers 8

# 4) 矩阵 + 评估
python scripts/build_score_matrix.py --score-table scores.csv --long-output long.csv --matrix-output matrix.csv --summary-output matrix.json
python scripts/evaluate_virtual_screening.py --score-table long.csv --ranking-output ranked.csv --metrics-output metrics.json

# 5) 组合选择（开发集）
python scripts/solve_qubo_receptor_subset.py --matrix matrix.csv --split-manifest split.csv --receptor r1 r2 r3 --output qubo.json --target-size 2
```

---

## 5. 验证与回归

```powershell
python -m pytest -q                 # 全部保留测试
python -m compileall scripts src     # 语法检查
python scripts/workflow.py list      # 目录完整性
```

- 本机缺 xgboost/openmm/gemmi/dimod 等依赖时，对应测试会以 error 形式跳过——先建好环境再判断。
- 沙箱/受限环境下 tmp_path 相关测试可能报 PermissionError，属环境限制。

## 6. 已删除脚本的恢复

历史 stage 脚本已从工作区移除，git 历史完整保留：

```powershell
git log --oneline -- scripts/                 # 找到删除提交（eb958ea 等）
git show eb958ea^:scripts/run_stage111_thrb_identity_adjudication.py > scripts/run_stage111_thrb_identity_adjudication.py
```

## 7. 红线（必须遵守）

1. 不要用已看过的 EGFR/FA10/MK14/PPARG 外层结果继续调 QUBO 系数、k 阈值或接触状态阈值。
2. DUD-E `THRB` 实为 thrombin/F2（P00734，参考 PDB 1YPE），不是甲状腺受体；旧结构记录不可复用。
3. 没有新的外部价值实例前，不租量子硬件、不写"量子优势"。

详见 `reports/handover/successor_quickstart_20260815_zh.md`。
