# 给下一位同学的交接说明

## 先读这三句话

1. 本项目不是“量子已经战胜经典虚拟筛选”的项目；目前没有这个证据，也不能这样写。
2. 已经得到的可靠发现是：多构象组合有时有用，但高度依赖具体蛋白。FA10 是稳定的双构象正例，EGFR 是明确的单构象反例。
3. 当前最值得继续的问题是：能否在对接前，通过共晶配体的口袋接触状态，判断一个新蛋白是否值得做多构象组合筛选。

完整历史和所有数值见 [quantum_receptor_ensemble_handover_20260815.md](quantum_receptor_ensemble_handover_20260815.md)。本文件只讲如何安全接手。

## 你接到的东西

| 内容 | 位置 | 用途 |
| --- | --- | --- |
| GitHub 代码 | `https://github.com/Sinking-tenderness/qubo-receptor-ensemble-selection` | 脚本、配置、测试、轻量审计结果与报告。 |
| 全量数据包 | `D:\量子×蛋白质\qubo_receptor_ensemble_experiment_data_20260815.tar.gz` | 原始/处理数据、对接输出、分析目录、历史交付包、受体/配体和报告。 |
| 数据包校验 | `D:\量子×蛋白质\qubo_receptor_ensemble_experiment_data_20260815.tar.gz.sha256` | SHA-256：`910624b47ef5f87231f14c169921dbeef6ac1f0523c8209fd679d3fdfcf488ee`。 |
| 数据包清单 | `D:\量子×蛋白质\qubo_receptor_ensemble_experiment_data_20260815.manifest.txt` | 共 65,454 个归档条目。 |
| 主交接文档 | `reports/handover/quantum_receptor_ensemble_handover_20260815.md` | Stage19-112 的逐段结果、允许主张和后续门槛。 |

## 当前可信结论

- Uni-Dock `v1.1.3` 的三种子生产流程可用；主评价指标为 `BEDROC20`。
- 旧 QUBO 在 PPARG 中不如直接贪心和 RF：`0.813` vs `0.858` vs `0.969`。这不是计算失败，而是方法不具备普适迁移性。
- FA10 的固定双构象组合在三种子上稳定提高 BEDROC20，平均相对单构象 `+0.0324`；EGFR 添加第二构象反而下降。
- 结构距离最大不等于筛选互补。FA10 正例的共晶接触状态有中等但清晰差异；EGFR 反例没有得到正向筛选收益。
- QUBO 对“质量下限、预算和冗余”这类约束表达很好，但在大多数可精确验证的小问题上，强经典算法已经找到同样的最优解。
- 量子硬件 PoC 曾成功运行，但局部修补问题对经典法也很容易；全局约束模型又不能稳定保证可行解。因此量子硬件目前不是下一步。

## 三条不能碰的红线

1. **不要用已经看过的 EGFR、FA10、MK14、PPARG 外层结果继续调 QUBO 系数、$k$ 阈值或接触状态阈值。** 这会把验证集变成训练集。
2. **不要把 THRB 当作 thyroid hormone receptor beta。** DUD-E `THRB` 实际是 thrombin/F2（UniProt `P00734`，参考 PDB `1YPE`）；旧的 19 个结构记录属于错误蛋白，不能复用。
3. **不要在没有新的外部价值实例前租量子硬件或把“能运行 QUBO”写成量子优势。**

## 第一周应该做什么

### 第 1 天：复现最小审计

在仓库根目录运行：

```powershell
$PY = 'C:\Users\MM\anaconda3\envs\qubo-receptor-ensemble\python.exe'
$env:CONDA_DEFAULT_ENV = 'qubo-receptor-ensemble'

& $PY scripts\run_stage111_thrb_identity_adjudication.py --root .
& $PY scripts\audit_stage111_thrb_identity_adjudication.py --root .
& $PY scripts\run_stage112_historical_candidate_pool_amendment01.py --root .
& $PY scripts\audit_stage112_historical_candidate_pool_amendment01.py --root .
& $PY -m pytest -q
```

当前基准是 `905 passed, 1 skipped`。若缺少 XGBoost，先执行：

```powershell
& $PY -m pip install "xgboost>=3.1,<3.2"
```

### 第 2-3 天：写并冻结凝血酶预注册

在任何下载或对接之前，建立新的 `Stage113` 配置。必须写清：

- 靶点与来源身份：DUD-E `THRB` = thrombin/F2 = `P00734`；
- 野生型参考结构门和参考口袋定义；
- 结构池门：至少 32 个元数据合格候选，之后至少 16 个通过坐标和重对接门的受体；
- 比较方法：单受体、线性 Top-$k$、直接贪心、强经典优化和精确/MILP；
- 主指标 `BEDROC20`、三种子和成功判据；
- 信息边界：确认/测试标签在相应门通过前锁定。

预注册应先由另一人审阅并提交 Git，再进行数据下载。

### 第 4-5 天：只做结果未知的结构元数据审计

通过预注册后才可以：

1. 下载并核验 DUD-E 来源身份；
2. 对 RCSB 结构做野生型、非共价、同链口袋和结构质量筛选；
3. 记录候选数是否达到 32；
4. 未达到就记录 No-Go 并停止，不降低门槛；
5. 达到才进入坐标、重对接与小规模 Uni-Dock 开发矩阵。

## 什么时候可以重启量子硬件

只有同时满足下列条件：

- 冻结的组合规则在未参与选择的数据上优于预先指定的强经典基线；
- QUBO 的最优解与贪心/一步交换确实不同，且这种差异带来外部 BEDROC20 价值；
- 小规模可由枚举或 MILP 给出最优证书，大规模又确实存在稳定的经典局部陷阱。

缺少任意一条，都先继续经典机制验证，而非使用 D-Wave、QCI 或其他量子硬件。

## 交接时最重要的判断

项目的价值不在于强行得到“QUBO 更强”，而在于已经用多蛋白、强基线、嵌套验证和硬件正反例划清了边界。下一位同学只要保持这个纪律，就有机会把“何时存在功能互补、何时不应多选构象”做成可信的机制研究；反过来，绕过门槛追逐漂亮数值会使现有证据链失去价值。
