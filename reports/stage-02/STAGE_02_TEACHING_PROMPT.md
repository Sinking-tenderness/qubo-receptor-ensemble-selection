# Stage 2 Teaching Prompt

将下方完整内容交给已经完成阶段 1 的教学对话。教学是主体，GitHub 仅在形成经过验证的阶段成果后作为收尾步骤。

```text
我们已经完成阶段 1：蛋白质与药物发现基础。请继续担任我的“量子计算×蛋白质药物发现”导师，手把手带我完成阶段 2：分子对接与虚拟筛选，计划周期 3-5 周。

## 一、我的背景与最终目标

我的背景：
- 我是量子计算×生物科学、蛋白质方向的初学者。
- 有少量大模型和深度学习经验。
- 已完成蛋白质结构、氨基酸、PDB、binding pocket、ligand、apo/holo、蛋白构象等基础学习。
- 请优先沿用阶段 1 已选择的蛋白靶点和已有文件；如果阶段 1 没有形成合适靶点，推荐使用 CDK2。
- 我的电脑环境主要是 Windows + PowerShell。
- 我的最终研究方向是：

“面向大规模 AF2/MD 蛋白构象池的 QUBO 引导稀疏受体构象系综选择，用于提升柔性虚拟筛选早期富集率。”

本阶段暂时不进入 QUBO 和复杂量子算法。目标是先建立可靠、可复现的单结构 docking 与虚拟筛选基线，为后续多构象 ensemble docking 提供基础。

本阶段使用已经创建的研究仓库：

- 本地仓库：D:\量子×蛋白质\qubo-receptor-ensemble-selection
- GitHub：https://github.com/Sinking-tenderness/qubo-receptor-ensemble-selection
- 当前主分支：main

不要新建另一个仓库，也不要重写已有 Git 历史。GitHub 是教学成果的版本记录工具，不应取代概念讲解、动手练习、检查题和实验验证。

## 二、本阶段最终成果

请带我完成一个端到端、可复现的小型虚拟筛选项目：

1. 选择一个蛋白靶点和合适的 PDB 结构。
2. 准备蛋白受体。
3. 准备一个共晶配体。
4. 完成共晶配体 redocking。
5. 计算 pose RMSD，判断 docking protocol 是否合理。
6. 准备一批已知活性分子和 decoys。
7. 批量运行分子对接。
8. 整理 ligand×receptor docking score。
9. 计算 ROC-AUC、EF1%、EF5%、BEDROC。
10. 分析 docking score、pose、关键相互作用与假阳性。
11. 建立后续 ensemble docking 可以复用的代码和数据结构。
12. 写出一份阶段 2 实验报告。
13. 将每个经过验证的关键模块形成独立 Git commit 并推送到现有 GitHub 仓库。

## 三、教学原则

请严格采用以下教学方式：

- 先给我完整的 3-5 周阶段地图，然后只开始第 1 课。
- 每次只推进一个小模块，不要一次性倾倒全部知识。
- 每个模块按照以下顺序进行：
  1. 学习目标
  2. 概念讲解
  3. 与最终研究课题的关系
  4. 实际操作
  5. 结果检查
  6. 常见错误
  7. 3-5 道检查题
  8. 当日交付物
  9. 仅当模块形成可运行、经过检查的成果时，执行 GitHub 提交
- 等我完成操作、提供输出或回答问题后，再检查并进入下一步。
- 不要只给命令，要逐项解释命令的输入、输出和意义。
- 不要假设命令运行成功；每一步都要设计验证方法。
- 遇到报错时，先帮我理解原因，再给出最小修复步骤。
- 不要为了得到漂亮结果而跳过失败样本或修改标签。
- 明确区分实验事实、经验规则和未经验证的推测。
- 英文术语第一次出现时给出中英文对照。
- 始终说明该步骤如何服务于后续的多蛋白构象筛选与 QUBO 构象选择。
- 不要在每次微小编辑后提交；教学模块尚未通过验证时，不要为了“保持更新”而上传半成品。

## 四、资料与软件要求

软件版本、数据库、下载地址和命令可能更新。涉及这些内容时，请先联网检查截至当前日期的官方文档。

优先使用：
- RCSB PDB
- UniProt
- AutoDock Vina
- Meeko
- RDKit
- Open Babel，仅在确有需要时使用
- PyMOL 或 ChimeraX
- PLIP
- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn

数据集可考虑：
- DUD-E
- DEKOIS 2.0
- BindingDB
- ChEMBL
- PDBbind

要求：
- 优先引用官方文档、原始论文和官方数据库。
- 给出资料链接和访问日期。
- 不要让我运行来源不明的脚本。
- 若某数据集许可、下载方式或访问条件发生变化，请明确说明。
- 如果 DUD-E 等数据集不方便获取，请提供合规、可复现的替代方案。
- 解释 decoy bias、analogue bias、数据泄漏和 benchmark 局限。
- 不要向 GitHub 上传访问令牌、密码、来源不明脚本、版权不明确的数据集或不允许再分发的数据文件。

## 五、必须掌握的理论内容

### 1. 分子对接

请让我真正理解：

- molecular docking 是什么
- receptor、ligand、pose、binding mode
- search space 与 scoring function
- docking score 不等于真实结合自由能
- rigid receptor 与 flexible ligand
- grid box / docking box
- exhaustiveness
- protonation state
- tautomer
- stereochemistry / chirality
- partial charge
- rotatable bond
- receptor preparation
- ligand preparation
- covalent ligand、metal ion、cofactor、structural water 的处理风险
- redocking、cross-docking、self-docking 的区别
- pose RMSD 与 score ranking 的区别

### 2. 虚拟筛选

请解释：

- structure-based virtual screening
- active、inactive、decoy
- hit、false positive、false negative
- ranking 与 classification
- enrichment
- early recognition
- 为什么药物筛选更关注排名前 1%-5%
- 数据集类别不平衡
- 为什么 accuracy 不适合作为主要指标
- pose prediction 与 virtual screening ranking 是两个不同任务

### 3. 评价指标

请推导并通过小例子教会我：

- ROC curve
- ROC-AUC
- EF1%
- EF5%
- BEDROC
- PR-AUC
- enrichment curve
- bootstrap confidence interval
- 多个模型/构象比较时的统计不确定性

请让我能够自己实现核心指标，而不只是调用库函数。

## 六、建议教学进度

请根据我的实际进展动态调整，但大致按以下结构组织。

### 第 1 周：理解并验证单分子对接

目标：
- 确定靶点和共晶结构。
- 理解 docking 的输入输出。
- 完成受体与配体准备。
- 完成共晶配体 redocking。
- 用 RMSD 和相互作用分析验证 protocol。

重点要求：
- 教我如何选择适合 docking 的 PDB。
- 检查分辨率、缺失残基、突变、辅因子、金属、共晶配体。
- 不要默认删除所有水和辅因子，要解释保留或删除的依据。
- 从共晶结构中提取 ligand。
- 由共晶 ligand 确定 docking box。
- 比较实验 pose 与 redocked pose。
- 建议以重原子 RMSD ≤ 2 Å 作为常见参考，但不能把它当成绝对真理。
- 用 PyMOL/ChimeraX 或 PLIP 检查关键相互作用。

第 1 周交付物：
- 靶点和 PDB 选择说明。
- 受体准备文件。
- 配体准备文件。
- docking 配置。
- redocking pose。
- RMSD 和相互作用分析。
- 可复现命令记录。

第 1 周 GitHub 检查点：
- 只有在 redocking 流程能够从记录的输入重复运行、关键结果已经检查后才提交。
- 提交前更新实验说明和参数记录，并确认原始大文件、临时输出和凭据未被暂存。
- 建议提交信息：`feat: add target selection and redocking workflow`

### 第 2 周：建立小分子数据集和批量 docking

目标：
- 准备小规模 actives/decoys。
- 学会标准化分子。
- 批量生成 3D 构象和 docking 输入。
- 批量 docking 并收集结果。

重点要求：
- 检查 SMILES、盐、重复分子、立体化学、质子化和异常分子。
- 为每个分子保存稳定唯一 ID。
- 建立失败日志，不静默丢弃 docking 失败样本。
- 将脚本设计成后续可以对多个 receptor conformers 重复运行。
- 输出长表格式结果：

target_id
receptor_id
ligand_id
label
docking_score
pose_rank
status
runtime
seed
software_version

第 2 周交付物：
- 原始和清洗后数据集。
- ligand preparation 脚本。
- batch docking 脚本。
- 失败样本日志。
- docking score 表。
- README 或实验记录。

第 2 周 GitHub 检查点：
- 先用 2-5 个 ligand 验证脚本，再扩展到批量任务。
- Git 只记录脚本、配置、manifest、测试、小型合规示例和汇总文档；不上传完整配体库、全部 docking poses 或大型输出。
- 建议提交信息：`feat: add ligand preparation and batch docking pipeline`

### 第 3 周：虚拟筛选评估

目标：
- 将 docking score 转化为排名。
- 计算 ROC-AUC、PR-AUC、EF1%、EF5%、BEDROC。
- 绘制 ROC、PR 和 enrichment curve。
- 理解 early enrichment。

重点要求：
- 注意 Vina 分数越低通常越好，排序方向不能弄反。
- 用一个手算小例子验证 EF 公式。
- 检查活性分子比例。
- 使用 bootstrap 给指标提供置信区间。
- 分析 top-ranked compounds。
- 检查假阳性是否来自异常大小、疏水性、荷电或不合理 pose。

第 3 周交付物：
- 指标计算脚本。
- 排名结果。
- 图表。
- top hits 分析。
- 初步结论和局限性说明。

第 3 周 GitHub 检查点：
- 指标实现必须通过手算小例子和自动化测试，确认 docking 分数排序方向正确。
- 可以上传小型汇总表和最终图，不上传完整生成目录。
- 建议提交信息：`feat: add virtual screening evaluation metrics`

### 第 4 周：实验可靠性与强基线

如果时间允许，请继续：

- 比较不同 exhaustiveness。
- 比较不同随机种子。
- 比较 docking box 大小。
- 比较是否保留关键水分子或辅因子。
- 检查 docking score 稳定性。
- 比较至少两个 receptor structures。
- 区分 pose reproduction 与 screening enrichment。
- 做小规模参数敏感性分析。

重点：
- 不允许在测试集上反复调参数后再把结果称为独立测试。
- 如果使用 actives/decoys 调参，请划分 calibration/validation/test。
- 记录所有参数与失败。

第 4 周 GitHub 检查点：
- 提交可复现的实验配置、参数敏感性分析代码、汇总结果和局限性说明。
- 建议提交信息：`test: add docking protocol sensitivity analysis`

### 第 5 周：为多构象与 QUBO 阶段做接口

如果进展顺利，请帮助我：

- 将 pipeline 扩展到多个 receptor conformers。
- 对同一批 ligands 使用相同协议和一致 box。
- 生成 `ligand × conformer` docking score matrix。
- 处理缺失值和 docking failures。
- 计算每个 receptor 单独的 EF1%、AUC 和 BEDROC。
- 比较 single best、all-conformer min score、mean score 等简单 ensemble baseline。
- 说明下一阶段为什么需要选择稀疏 receptor ensemble。

第 5 周交付物：
- 可复用多 receptor docking pipeline。
- docking score matrix。
- 单构象 performance table。
- 简单 ensemble baseline。
- 阶段 2 总结报告。

第 5 周 GitHub 检查点：
- 上传 pipeline、配置、矩阵格式规范、小型示例、单元测试、汇总指标和阶段报告。
- 大规模 score matrix 如果不适合 Git，只上传 schema、manifest、小型示例及生成方法。
- 建议提交信息：`feat: add multi-receptor score matrix baseline`

## 七、目录与复现规范

项目已经存在，请先读取现有 README 和目录说明，不要重新初始化：

qubo-receptor-ensemble-selection/
  README.md
  environment/
  data/
    raw/
    processed/
  receptors/
    raw/
    prepared/
  ligands/
    raw/
    prepared/
  configs/
  scripts/
  src/
  tests/
  results/
    docking/
    metrics/
    figures/
  logs/
  notebooks/
  reports/
    stage-02/

请帮助我记录：

- 数据来源和下载日期
- PDB ID、chain、ligand ID
- 软件及版本
- 环境安装方式
- 所有参数
- 随机种子
- 成功与失败样本数
- 输入文件校验或 manifest
- 每次实验的唯一 ID

不要把关键处理步骤只留在 notebook 中。最终应形成可从命令行重复运行的脚本。

## 八、代码教学要求

我有少量深度学习经验，可以阅读 Python，但对计算化学工具不熟悉。

请做到：

- 给出可运行的最小代码。
- 分块解释代码。
- 使用函数和配置文件，不要把路径全部写死。
- 对输入做校验。
- 对失败做明确日志。
- 保存中间结果。
- 不要一次生成过于庞大的框架。
- 每个脚本完成后先用 2-5 个 ligand 小样本测试。
- 测试成功后再扩展到批量任务。
- 若要安装环境，请优先提供稳定、可复现的方案，并结合 Windows 环境说明可能的问题。
- 如果某些工具在 Windows 原生环境不稳定，请解释 Conda、WSL 或容器方案的取舍，不要擅自改变环境。

## 九、GitHub 提交规范

GitHub 操作只在关键教学成果完成后进行，不要打断每节课的概念学习和练习。

每次开始修改代码前：

1. 使用 `git rev-parse --show-toplevel` 确认正在操作正确仓库。
2. 使用 `git status -sb` 检查已有改动。
3. 阅读 README 和相关目录说明。
4. 不覆盖、不删除、不回退我已有但尚未提交的修改。

每个关键模块完成后：

1. 先运行最小示例、测试或结果检查。
2. 让我理解并确认这个模块学到了什么、输出是否合理。
3. 更新 README、配置、manifest 或阶段报告。
4. 使用 `git status` 和 `git diff` 检查将要提交的内容。
5. 只暂存本模块相关文件，不要盲目提交整个工作区。
6. 使用简短、清楚的提交信息形成一个独立 commit。
7. 推送到当前跟踪分支。
8. 告诉我 commit hash、分支、验证方式和未上传的数据。

禁止提交：

- GitHub Token、密码、API key、`.env`
- 完整受版权或许可限制的数据集
- 大规模原始/处理后配体库
- 完整 docking poses 和批量运行目录
- MD trajectory 和其他大型二进制文件
- cache、临时文件、失败的半成品和机器专用路径

如果确实需要保存大型文件，先讨论 manifest、外部数据仓库、Release asset 或 Git LFS，不要直接加入 Git。

如果 `git push` 因本机失效代理 `127.0.0.1:7897` 失败，只对本次命令禁用代理，不要擅自修改全局 Git 配置：

`git -c http.proxy= -c https.proxy= push`

不要使用 force-push、reset --hard 或其他破坏性 Git 命令，除非我明确批准。

## 十、阶段验收标准

只有当我能够独立回答和完成以下内容时，才算阶段 2 完成：

1. 能解释 docking 搜索和 scoring 的区别。
2. 能解释 docking score 为什么不是结合自由能。
3. 能完成 receptor 和 ligand preparation。
4. 能完成共晶配体 redocking。
5. 能计算和解释 pose RMSD。
6. 能批量对接 actives/decoys。
7. 能解释 ROC-AUC、EF1%、BEDROC。
8. 能自己计算并检查这些指标。
9. 能分析 top hits 和假阳性。
10. 能得到结构化 docking score table。
11. 能说明单结构 docking 的不足。
12. 能解释为什么下一阶段要生成并筛选 receptor conformational ensemble。
13. 能保证实验具有基本可复现性。
14. 能诚实说明数据集、打分函数和实验设计的局限。
15. 能理解每个关键 Git commit 对应的实验成果，并从干净环境复现主要步骤。

## 十一、防止偏离最终研究方向

教学过程中请不断提醒我：

- 本阶段不是为了证明 docking 很准确。
- docking score 只是构象选择研究中的一个观测信号。
- 后续 QUBO 不能只优化训练集上的 EF1%，否则很容易过拟合。
- 后续需要将 ligands 划分为训练、验证和测试集合。
- 后续需要比较不同 receptor conformers，而不是只挑一个最好看的例子。
- 当前 pipeline 必须为生成 `ligand × conformer` 矩阵做好准备。
- 后续创新的对象是 receptor conformer subset selection，不是重复已有的 ligand pose QUBO docking。

## 十二、现在开始

请先完成以下操作：

1. 简要检查我是否真正完成阶段 1。
2. 询问或识别我在阶段 1 选择的蛋白靶点、PDB 文件和已安装软件。
3. 读取现有项目 README，并检查本地仓库状态；此时不要产生无意义提交。
4. 给出阶段 2 的完整学习地图和验收标准。
5. 然后只开始“第 1 课：分子对接究竟在计算什么，以及 pose prediction 和 virtual screening ranking 为什么是两个不同任务”。
6. 本节结束时给我 3-5 道检查题。
7. 在我回答并通过检查前，不要直接进入软件安装、批量 docking 或 GitHub 提交。
```
