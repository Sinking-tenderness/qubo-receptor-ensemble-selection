# 上次组会后的科研进展总结

## 1. 研究目标

最终目标是从大规模 AF2/MD 蛋白构象池中，使用 QUBO 选择少量互补、非冗余的
受体构象，在控制 docking 成本的同时改善虚拟筛选早期富集率。

当前尚未进入 QUBO 建模，主要工作是建立可靠的单构象 docking 基线和
`ligand × receptor conformer` 数据接口。

## 2. 上次组会后的主要进展

### 2.1 完成 1AQ1-STU Redocking

- 靶点：人源 CDK2；
- 结构：1AQ1 chain A；
- 共晶配体：STU；
- Vina：1.2.7；
- 最优 docking score：-13.87 kcal/mol；
- mode 1 重原子 RMSD：0.225 Å。

结果说明当前受体准备、配体准备、docking box 和搜索参数能够在该体系中复现
接近共晶结构的 pose，但不能证明 Vina score 等于真实结合自由能，也不能证明
虚拟筛选排序一定可靠。

### 2.2 建立小型 Active/Decoy 数据集

- 数据来源：DUD-E CDK2；
- 子集：10 actives + 50 decoys；
- RDKit 解析：60/60 成功；
- canonical duplicate：0；
- Meeko PDBQT：60/60 成功；
- `CDK2_D0036` 有三维优化 warning，已保留并记录。

DUD-E decoys 不是实验确认的 inactive，因此高排名 decoy 只能视为候选假阳性。

### 2.3 完成 1AQ1 批量 Docking

- ligand：60；
- docking success：60；
- docking failed：0；
- score rows：599；
- top 10：4 actives，6 decoys。

结果表明 actives 和 decoys 不能仅靠 Vina score 完全分开。

### 2.4 完成虚拟筛选评价

| 指标 | 点估计 | Bootstrap 95% CI |
| --- | ---: | --- |
| ROC-AUC | 0.644 | 0.411-0.840 |
| PR-AUC | 0.459 | 0.134-0.748 |
| BEDROC, alpha=20 | 0.653 | 0.056-0.920 |
| EF1% | 6.0 | 0.000-12.000 |
| EF5% | 4.0 | 0.000-8.571 |

当前结果在小数据集上优于随机，但置信区间较宽。EF1% 实际只由 top 1 ligand
决定，暂时不能作为稳健结论。

### 2.5 完成 Top-hit 和 Pose 分析

重点检查了两个 active 和两个高排名 decoy：

- actives：`CDK2_A0009`、`CDK2_A0010`；
- decoys：`CDK2_D0022`、`CDK2_D0036`。

四个分子都进入结合口袋，且未发现明显重原子碰撞。D0022 的伸展方向和接触
残基模式与 active poses 差异较大，因此被列为候选假阳性，但不能据此证明其
真实无活性。

### 2.6 建立 Score Matrix 接口

已将 docking long table 转换为：

- 每个 ligand-receptor pair 一行的代表分数表；
- 每个 receptor conformer 一列的 wide score matrix。

当前矩阵只有 `CDK2_1AQ1_A_prepared` 一列。该接口将用于后续多构象比较和
QUBO 构象选择。

### 2.7 新增第二个受体构象 1HCL

1HCL 是分辨率 1.80 Å 的 apo CDK2 结构：

- 与 1AQ1 共同 Cα：277；
- 对齐后全共同 Cα RMSD：0.666 Å；
- 共晶 STU 位于对齐后的 1HCL ATP pocket 内；
- 使用与 1AQ1 相同的 docking box。

1HCL 存在两处 alternate location：

- GLN131：A/B occupancy = 0.80/0.20；
- SER264：A/B occupancy = 0.66/0.37。

首个基线显式选择主占有 A 状态。最终 1HCL receptor PDBQT 包含：

- 294 residues；
- 2875 atoms；
- 509 hydrogen-like atoms；
- 0 HETATM；
- 与 1AQ1 相同的电荷模型和 AutoDock atom types。

目前 1HCL 已准备完成，但尚未运行 ligand docking。

## 3. 当前已形成的能力

- receptor/ligand preparation；
- 共晶配体 redocking 和 pose RMSD；
- active/decoy batch docking；
- ROC-AUC、PR-AUC、EF、BEDROC 和 bootstrap CI；
- top-hit pose/contact 分析；
- receptor 刚体对齐和一致化准备；
- `ligand × receptor conformer` score matrix 接口；
- 失败、warning、参数、随机种子和软件版本记录。

当前自动化测试为 `5 passed`，关键模块已形成独立 Git commits 并推送。

## 4. 当前问题

1. 数据集只有 60 个分子，early-enrichment 指标不稳定。
2. DUD-E decoys 不是实验确认的 inactive，存在 benchmark bias。
3. 1HCL 尚未 docking，目前还没有真正的双构象性能比较。
4. 水、质子化、altloc 和缺失残基等准备选择可能影响结果。
5. Vina 可能过度奖励较大、疏水、带电或高柔性的分子。
6. docking box 来源于 1AQ1-STU，可能对 holo pocket 存在一定偏向。

## 5. 下一步计划

1. 使用 2 个 active 和 2 个 decoy 验证 1HCL docking。
2. 验证通过后，将同一批 60 个 ligand 全量对接到 1HCL。
3. 计算 1HCL 的 AUC、EF、BEDROC 和 bootstrap CI。
4. 生成 1AQ1/1HCL 两列 score matrix。
5. 比较两个构象的性能、active coverage、decoy 排序和互补性。
6. 建立 minimum、mean、single-best 等简单 ensemble baselines。
7. 再扩展更多 PDB、AF2 或 MD 构象，并进入稀疏构象选择和 QUBO 建模。

## 6. 建议与老师讨论

1. 后续继续使用 DUD-E，还是加入 ChEMBL、BindingDB 或独立测试集？
2. 先完成 1AQ1/1HCL 双构象验证，还是立即扩大 PDB 构象池？
3. 第一版是否统一删除水，后续再做关键结构水敏感性分析？
4. ligand 数据何时扩大，训练、验证和测试集如何划分？
5. QUBO 优先优化 early enrichment、active coverage、构象互补性、冗余，
   还是 docking cost？
6. 预计使用多少 ligands 和 receptor conformers，可用计算资源有多少？
7. 课题重点是提高富集率，还是在更低计算成本下维持相近性能？
8. 是否有独立实验数据、合作数据或外部测试集用于最终验证？

## 7. 当前结论

上次组会后，课题已从结构和软件准备阶段推进到可运行、可评价、可审计的单构象
虚拟筛选基线，并完成第二个 CDK2 receptor conformer 的对齐和准备。

下一阶段将实证比较不同 receptor conformers 的 active/decoy 排序差异，为后续
稀疏构象选择和 QUBO 建模生成真实的多构象 score matrix。
