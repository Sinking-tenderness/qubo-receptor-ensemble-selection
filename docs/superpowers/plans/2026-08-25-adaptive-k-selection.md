# Adaptive k 全候选选择实现计划

> **面向 AI 代理的工作说明：** 按 TDD 执行本计划。每个行为先写回归测试并确认失败，再写最小实现；完成后运行目标测试和相关集成测试。

**目标：** 修复当前自适应 k 的顺序早停，使 k=1 只是搜索起点；所有候选 k 都能基于相对 k=1 的 OOF 证据参与选择，在证据支持时选择 k=2 或 k=3。

**范围：** `adaptive_cardinality.py` 的实际远程实验链路、完整流程接入、相关配置/工作流说明和测试。保持 QUBO surrogate、docking 数据和通用 `k_selection.py` 不变。

## 当前行为基线

- `estimate_adaptive_cardinality()` 已在每个 inner fold 求解所有候选 k。
- 当前只生成相邻转换 `1->2`、`2->3`。
- `select_adaptive_k()` 要求转换严格顺序，并在首个失败转换处停止。
- `build_problem_stage()` 将 `selected_k` 写入 `problem_config.target_size`，并传播 `adaptive_cardinality.json` 到后续阶段。
- 远程自适应配置默认候选为 `[1, 2, 3]`，当前显式使用 `lower_quantile: 0.025`。

## 设计决策

1. 对所有候选生成所有有序两两转换，至少包含 `1->2`、`1->3` 和 `2->3`。
2. 选择逻辑只将 `from_k == 1` 的转换作为最终候选证据；其他转换保留作诊断。
3. 候选通过条件保持现有门控语义：bootstrap 下置信界大于 `minimum_effect`，且 rescue contrast 严格为正。第一版 `minimum_effect` 默认为 0，不根据本轮结果后验调节。
4. 在通过候选中按 bootstrap 下置信界最大者选择；下置信界相同或差异不足时偏好较小 k。没有更大候选通过时返回 1。
5. 保留 `lower_quantile` 兼容入口，默认单侧 95% 下界使用 0.05；配置和工作流说明明确该语义。
6. 在 artifact 中保存每个转换的完整诊断摘要，不能因某个转换失败而丢弃后续转换。

## 实现任务

### 1. 选择器回归测试

**文件：** `tests/test_adaptive_cardinality.py`

- [x] 将当前“失败后不评估 k=3”的测试改为断言：`1->2` 失败时仍评估 `1->3`，且 `1->3` 通过时选择 k=3。
- [x] 增加全转换诊断测试，断言 `1->2`、`1->3`、`2->3` 均出现在决策 artifact 中。
- [x] 增加无候选通过时返回 k=1 的测试。
- [x] 增加多个候选通过时按风险分数选择、平分时选择较小 k 的测试。

### 2. 估计器回归测试

**文件：** `tests/test_adaptive_cardinality.py`

- [x] 扩展现有 estimator 测试，断言每个候选 k 每个 inner fold 都被求解。
- [x] 断言 estimator 输出所有两两转换，而不是只输出到首个失败点。
- [x] 保留 utility metric、aggregation、progress 和 deterministic bootstrap 的现有断言。

### 3. 最小选择器实现

**文件：** `src/qubo_receptor_ensemble/adaptive_cardinality.py`

- [x] 抽取候选转换统计计算，允许转换不再满足连续的 `from_k == previous_to_k` 约束。
- [x] 按候选 k 构造相对起点 k=1 的转换，并另行构造其余两两诊断转换。
- [x] 修改 `select_adaptive_k()`：扫描全部转换，过滤起点相对证据，计算通过候选，按 bootstrap LCB 选择结果；不再 `break`。
- [x] 保持 `AdaptiveCardinalityDecision` 的 `selected_k`、`need_multi_conformation` 和现有字段兼容。
- [x] 为 diagnostics 增加选择参考和候选通过状态，保持 JSON 可序列化。
- [x] 修正默认 `lower_quantile` 为单侧 95% 语义的 0.05，并保留显式旧值可用。

### 4. 流程和配置接入

**文件：**
- `src/qubo_receptor_ensemble/experiment.py`
- `configs/experiments/mk14_adaptive_remote.json`
- `configs/experiments/pparg_adaptive_remote.json`
- `configs/experiments/stage102a_egfr_adaptive.json`
- `configs/experiments/stage102a_fa10_adaptive.json`

- [x] 保持 `selected_k -> target_size -> problem/selection/evaluation` 的传播链路不变。
- [x] 让当前可复用的 adaptive 配置显式表达新的置信分位点语义；保留历史 remote snapshot 不变。
- [x] 不改变实验候选范围和 QUBO 求解后端。

### 5. 集成测试和文档

**文件：**
- `tests/test_adaptive_cardinality_integration.py`
- `docs/experiment_workflow_zh.md`

- [x] 在完整 workflow 测试中验证所有转换 artifact 能从 build_problem 传播到 selection/evaluation。
- [x] 更新文档：k=1 是搜索起点；选择器评估全部候选；失败候选不阻塞后续候选。
- [x] 明确当前结果仍属于 development/inner-fold 选择证据，不等同于 locked test 结论。

## 验证命令

1. `pytest -q tests/test_adaptive_cardinality.py`
2. `pytest -q tests/test_adaptive_cardinality_integration.py tests/test_full_experiment_config.py`
3. `pytest -q tests/test_k_selection.py tests/test_full_workflow_end_to_end.py`
4. 需要时运行完整测试集，确认未引入跨模块回归。

## 明确不做

- 不修改 `src/qubo_receptor_ensemble/k_selection.py`。
- 不修改 `src/qubo_receptor_ensemble/qubo.py` 或 surrogate 目标。
- 不重新执行 docking、量子硬件实验或读取 outer/locked-test 标签。
- 不覆盖或恢复工作区已有的无关未提交变更。
