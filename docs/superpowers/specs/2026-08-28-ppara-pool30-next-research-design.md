# PPARA pool30 后续研究设计

日期：2026-08-28

文档性质：研究设计与决策门槛，不是新的实验结果报告。

## 1. 研究目标

在 PPARA 从 15 个候选构象扩展到 30 个候选构象后，QUBO 仍未在当前 full-data 回放中优于 direct BEDROC `greedy`。下一步研究不再把“增加构象数量”本身当作目标，而是回答三个可证伪的问题：

1. 当前 QUBO objective 是否真的与最终关心的 BEDROC20 排序一致？
2. 候选池从 30 个构象继续扩大时，exact 求解的时间、内存和可审计范围在哪里？
3. 如果目标函数确实对齐，QUBO 选择优势能否在未参与选择的独立配体面板上复现？

本设计不预设 QUBO 必须成功。若结果表明目标函数没有对齐，或优势只存在于复用同一矩阵的 full-data 回放，应停止继续扩大候选池，转而修正评价口径或目标函数。

## 2. 已知证据和边界

PPARA 是当前优先目标，因为既有 15 受体 V5 对比中，QUBO 与 direct BEDROC `greedy` 的差距小于 PPARG 等目标，适合做“扩大候选池能否显出差异”的压力测试。30 受体 fixed `k=4` 补充实验的结果为：

| 方法 | BEDROC20 | ROC-AUC | PR-AUC | EF1% | EF5% | EF10% |
|---|---:|---:|---:|---:|---:|---:|
| QUBO | 0.781674 | **0.914705** | 0.710738 | 3.333 | 4.000 | 3.750 |
| greedy | **0.795533** | 0.914375 | **0.712710** | 3.333 | **4.333** | **3.917** |
| linear | **0.795533** | 0.914375 | **0.712710** | 3.333 | **4.333** | **3.917** |
| single | 0.796052 | 0.872882 | 0.675664 | **4.167** | 4.333 | 3.750 |

QUBO 相对 `greedy` 的 BEDROC20 差值为 `-0.013859`，ROC-AUC 差值为 `+0.000330`，PR-AUC 差值为 `-0.001971`。QUBO 集合是 `2NPA + 5HYK + 6KB0 + 6KB5`；`greedy/linear` 集合是 `5HYK + 6KB0 + 6KB2 + 6KB5`。

本结果的边界必须保留：

- 选择和评价均复用同一个 30 受体矩阵，属于 `full_data/development-only` 回放；
- `locked_test_rows_read=0`，没有独立测试证据；
- 使用的是 classical exact backend，没有量子硬件或量子模拟器的对照；
- 一次 fixed `k=4` 不能证明 30 候选池在所有 `k`、所有目标函数或所有蛋白质上都无效。

## 3. 核心假设

### H1：目标函数对齐

当前 `basic_utility` QUBO 的 objective 排序应当与独立计算的训练 BEDROC20 或 `mean_score` 聚合 BEDROC20 有正向秩相关。如果相关性弱或方向相反，继续增加候选构象不会解决根本问题。

### H2：规模代价是组合增长，而非仅变量线性增长

30 个二元选择变量的 QUBO 表面上只有 30 个变量和 435 个成对项，但固定基数集合的数量为：

| k | `C(30,k)` |
|---:|---:|
| 1 | 30 |
| 2 | 435 |
| 3 | 4,060 |
| 4 | 27,405 |
| 5 | 142,506 |
| 6 | 593,775 |

这些数字是候选集合数量上界，不等同于具体 solver 的实际搜索节点数；它们用于解释 exact 搜索的指数级最坏情况，并作为规模记录的统一基准。

### H3：样本外优势才是有效优势

只有在目标函数通过对齐诊断后，且 QUBO 在独立配体面板上相对 `greedy` 的主要指标差值达到预先冻结的门槛，才可继续投入更大候选池或新的 docking 预算。门槛具体数值目前为“待确认”，不能在看到结果后再调整。

## 4. 阶段 A：30 候选池目标函数诊断

### A1. 固定 `k=4` 的全组合审计

直接在已下载的 30 列 primary matrix 上枚举 `C(30,4)=27,405` 个集合，不重新 docking。对每个集合同时计算：

- QUBO objective 值；
- 训练集单受体效用和冗余项；
- 以 `mean_score` 聚合后的 BEDROC20；
- 以其他预先注册指标聚合后的 ROC-AUC、PR-AUC 和 EF；
- 集合成员、与 QUBO 解的集合交并比、与 direct `greedy` 解的集合交并比。

输出至少包括：Spearman 和 Kendall 秩相关、QUBO 最优集合在真实 BEDROC 排序中的名次、真实最优集合与 QUBO 集合的 objective 差距、真实指标 regret、top-N 集合重叠率。相关性和 regret 的验收阈值需要在运行前冻结，当前为“待确认”；建议先注册“秩相关达到正向且 regret 不超过预设容忍值”的二元门，而不是事后挑选一个有利指标。

### A2. 目标函数敏感性

仅在 A1 完成后，对现有矩阵做小范围、预注册的消融：

- 去掉冗余项；
- 保留冗余项但改变其符号或候选权重；
- 比较 `mean_score` 与稳健聚合（如 median 或预先选定的尾部稳健统计）。

消融只用于解释“为什么 objective 与最终指标不一致”，不允许根据同一 full-data 结果反复调权重后宣称验证成功。权重网格、主指标和选择规则必须在运行前记录，具体网格为“待确认”。

### A 阶段决策门

- 通过：至少一个冻结的 objective 版本在 A1 中显示稳定的正向对齐，并且 QUBO 解的真实指标 regret 在预设范围内；进入阶段 B 的规模测试和阶段 C 的独立验证准备。
- 不通过：停止扩大 PPARA 候选池；将研究问题改写为目标函数、聚合方式或数据划分问题，不再把 solver 规模作为主要卖点。

## 5. 阶段 B：规模与求解器诊断

### B1. 固定 k 的分级运行

在同一 30 受体矩阵上运行 `k=1..6`。每个 k 都记录：候选组合数量、实际求解节点（若 solver 提供）、wall time、峰值 RSS、CPU 时间、结果 objective、是否证明全局最优、运行中止原因和工件哈希。运行顺序和并发数要固定，避免把服务器瞬时负载误当成规模趋势。

exact 求解应设有运行前冻结的时间和内存上限，具体上限为“待确认”。达到上限时立即停止并保留 partial log，不允许以未证明的结果替代 exact 最优。

### B2. 可审计近似求解

当 `C(30,k)` 超出 exact 预算时，使用预先指定的 beam search、局部搜索或分块策略作为挑战者。每个近似结果必须标注 `best-known`，同时记录：随机种子、初始解、beam 宽度或迭代次数、重复次数、最好和最差 objective、与 exact 可解小规模结果的偏差。

近似 solver 的结果只能回答“在固定资源下能找到什么”，不能回答“已找到全局最优”，也不能直接用于宣称量子优势。

### B 阶段决策门

- exact 在目标 k 范围内可承受：保留 exact 作为审计基线，再比较近似方法的速度和 regret。
- exact 不可承受但近似结果稳定：将研究重点转为可证明或可复现实用求解器，并把 classical baseline 作为主要参照。
- exact 和近似都无法在冻结资源内稳定复现：停止继续扩大候选池，先降低问题维度或改用分层候选生成。

## 6. 阶段 C：独立配体面板验证

阶段 C 只有在阶段 A 通过后才启动。选择数据必须与 A、B 使用的 30 受体矩阵隔离：优先使用事先冻结且未读入当前选择流程的 PPARA locked-test rows；若该面板不存在或已被使用，则新建独立配体 panel。面板来源、active/decoy 数量、scaffold 分层和 docking 预算均为“待确认”，需在运行前记录。

在训练面板上选择集合，在独立面板上用同一套 `mean_score` 聚合评价：

- QUBO；
- direct BEDROC `greedy`；
- `linear`；
- `single`。

冻结的主要指标为 BEDROC20，ROC-AUC、PR-AUC、EF1%、EF5% 和 EF10% 为辅助指标。报告配对差值、按 scaffold 的差值分布、bootstrap 置信区间、最坏 scaffold/折表现和选择集合稳定性。独立验证中不能根据结果重新选择 `k`、冗余权重或 solver。

### C 阶段决策门

- 通过：QUBO 相对 `greedy` 在主要指标上达到预注册的正向差值，且置信区间和最坏分层结果没有显示不可接受的下行；才考虑新目标或更大的候选池。
- 不通过但 objective 对齐：说明瓶颈可能在数据量、聚合方式或 solver 近似误差，继续做机制诊断，不宣称方法优势。
- 不通过且 objective 不对齐：停止扩大规模，优先重做目标函数和数据划分设计。

## 7. 最小可行执行包

### 输入材料

- 本地结果：`E:\Quant\remote_runs\ppara_pool30_fixed_k4_remote`；
- 远程原始矩阵：`/root/autodl-tmp/qubo_data_root/results/runs/ppara_pool30_adaptive_remote/matrices/primary_median_matrix.csv`；
- 30 受体 `problem.json`、`config.snapshot.json`、`selection.json` 和基线比较 CSV；
- 当前仓库提交：`397cc78`。

### 输出工件

- `pool30_objective_enumeration.csv`：每个 `k=4` 集合的 objective 和真实指标；
- `pool30_objective_diagnostics.json`：秩相关、regret、top-N 重叠和配置哈希；
- `pool30_solver_scaling.csv`：k、组合数、时间、峰值内存和最优性证明状态；
- `pool30_solver_scaling.json`：运行限制、失败原因和 best-known 说明；
- 独立验证的 `summary_by_target.csv`、`paired_comparisons.csv` 和 protocol metadata；
- 一份只引用上述工件的研究报告，不改写历史结果汇总。

### 验收标准

- 选择与评价的数据边界清楚，full-data 回放和独立验证不混写；
- 每个 exact 结果都有最优性证明或明确的失败状态；
- 每个近似结果都有 `best-known` 标签和可重放随机种子；
- 所有门槛在运行前冻结，未根据结果回调；
- 没有把 classical exact/heuristic 结果写成量子优势；
- 不修改 `E:\Quant\docs\qubo_receptor_ensemble_experiment_results_zh.md`。

## 8. 不做什么

- 不先把候选池扩大到 60 或 100 再寻找优势；
- 不在同一 full-data 矩阵上反复调权重后选择最好的一次作为证据；
- 不重新 docking 已经存在且已核验的 PPARA 15 个旧受体；
- 不把 adaptive 与 exhaustive 的一次性运行时间差异解释成渐近复杂度结论；
- 不把 exact solver 找到的 classical 最优解释成量子计算优势。

## 9. 待确认事项

- A 阶段秩相关和 regret 的数值门槛；
- B 阶段 exact 的 wall-time、峰值内存和并发上限；
- 近似 solver 的具体算法、重复次数和停止条件；
- C 阶段独立配体 panel 的来源、规模和 scaffold 分层；
- 独立验证的预注册效应量和置信区间判定规则。

在上述事项冻结前，不启动新的大规模 docking，也不把 PPARA pool30 结果升级为方法学结论。
