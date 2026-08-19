# Canonical Pipeline Configurations

当前目录包含 schema `2.0` 的兼容配置。它们从已经接收的 Stage102A
primary median score matrix 开始，不执行配体准备或 docking：

- `stage102a_fa10_development_selection.json`：FA10，固定 `k=2`；
- `stage102a_egfr_development_selection.json`：EGFR，固定 `k=1`。

两份配置只允许 `train`，把 `test` 声明为 locked，并将输出写入独立的
`results/pipeline_runs/` 子目录。

## 配置结构

配置必须包含：

- `schema_version`、`experiment_id`、`target_id`、`purpose`；
- 按顺序排列的 `prepare`、`build_problem`、`solve`、`evaluate`、`persist`；
- 带 `path` 和 `sha256` 的 `inputs`；
- `data_policy`，其中 `evaluate_locked_test` 必须为 `false`；
- `prepare`、`problem`、`solve`、`evaluate` 和 `outputs`。

路径相对于 `scripts/run_pipeline.py --root` 解析。所有输入在任何阶段运行前
都要通过存在性和 SHA-256 校验。运行器会写入配置快照、阶段 manifest 和顶层
`manifest.json`。

## Linux 执行

```bash
REPO_ROOT=/path/to/qubo-receptor-ensemble-selection
cd "$REPO_ROOT"

python scripts/run_pipeline.py validate --config configs/pipelines/stage102a_fa10_development_selection.json --root .
python scripts/run_pipeline.py plan --config configs/pipelines/stage102a_fa10_development_selection.json --root .
python scripts/run_pipeline.py run --config configs/pipelines/stage102a_fa10_development_selection.json --root .
```

EGFR 使用同名 EGFR 配置。`plan` 和 `run --dry-run` 只生成 planned manifest，
不能当作真实选择结果。

## 求解边界

当前注册的 `exact` 后端支持 `qubo` 和 `normalized_qubo`；`greedy` 支持基础
`qubo`。固定卡数配置使用 `exact`，适用于当前 12/13 个受体的小规模开发
问题。canonical runner 不执行 docking，也不自动选择或解锁测试集。
