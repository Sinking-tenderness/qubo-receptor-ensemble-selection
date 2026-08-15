# Stage98 跨蛋白受体互补性分析

本分析只使用已经完成的真实 Uni-Dock 矩阵，不新增对接、不使用量子硬件。complementarity 选择器只读取对接分数和受体间秩相关性；active/decoy 或 high/low 标签只在最终 BEDROC 评价时使用。

## 预注册门槛

- 蛋白数量：5。
- BEDROC：alpha=20。
- 通过条件：至少 3 个蛋白的 k=3 相对 k=1 提升 >= 0.02，平均提升 >= 0.02，最差蛋白 >= -0.02。

## 结果

- k=3 平均提升：`-0.046388`。
- 最差蛋白提升：`-0.180018`。
- 达到 +0.02 的蛋白数：`1/5`。
- Go/No-Go：`NO-GO`。

| Target | k=1 complementarity BEDROC | k=3 complementarity BEDROC | Gain | >=0.02 |
|---|---:|---:|---:|---|
| MK14 | 0.357420 | 0.399234 | 0.041814 | True |
| PPARG | 0.782673 | 0.602655 | -0.180018 | False |
| BACE1 | 0.895646 | 0.907516 | 0.011871 | False |
| PPARA | 0.768580 | 0.695415 | -0.073165 | False |
| PPARD | 0.758528 | 0.726087 | -0.032441 | False |

## 解释

如果 Go/No-Go 为 NO-GO，则不能再通过增加蛋白、调 diversity weight 或新增 QUBO 形式来追逐同一结论；应将论文定位为受体组合效用、QUBO/CQM 表达和量子硬件边界研究。监督 oracle_train 只用于估计标签可提供的上限，不是可部署方法。
