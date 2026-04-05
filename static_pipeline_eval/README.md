# ORT Static Pipeline Eval

`ORT/static_pipeline_eval` 是 ORT monorepo 下的一个子项目，已通过 subtree 迁入主仓库。它用来把 `single_op_stage1_mlp` 已经训练好的单算子性能预测结果，重新组织成整图级静态流水线时间线，并评估预测的 end-to-end makespan 与真实 branch-parallel 执行时间之间的误差。

它解决的问题不是“单个算子预测准不准”，而是“当这些算子带着真实依赖关系和 branch 并发规则被排成一条静态时间线之后，整图误差会变成什么样”。

## 1. 项目目标

这个项目的目标有三件事：

1. 复用已有单算子预测结果，不重新训练模型。
2. 将 ORT DLRM branch-parallel 图按照静态变量和 DAG 依赖关系进行离线排程。
3. 输出整图误差、embedding 启动顺序验证结果，以及后续可能需要黑盒校准的误差热点。

当前实现的重点是 embedding branch 的静态排程规则，因为 `inter_threads` 会直接决定 embedding lookup 阶段同时允许多少条 branch 并行执行。

## 2. v1 排程器在做什么

`v1` 排程器本质上是一个离线静态执行模拟器。

输入是：

- 每个 op 的预测时长 `pred_us`
- 每个 combo 的 `op_shapes_*.csv`
- 每个 combo 的真实 timeline `branch_parallel_op_timeline.csv`
- 每个 combo 的静态变量，例如 `batch_size`、`num_indices_per_lookup`、`inter_threads`

输出是：

- 预测的整图 makespan
- 真实整图 makespan
- 二者误差
- embedding 分支启动顺序与并发度检查

它不是简单把所有 op 的时长相加，而是显式建模：

- 哪些节点有依赖关系
- 哪些 embedding branch 可以并发
- `tail` 必须在 `bottom + embedding pool` 结束后才能开始

## 3. 排流水线流程

整个流程可以拆成 6 步：

### Step 1. 读取单算子预测结果

默认读取：

- `models/combined/combined_predictions_test.csv`

每一行代表一个 op 的预测结果，核心字段包括：

- `case_id`
- `combo`
- `op_idx`
- `pred_us`
- `target_us`

### Step 2. 从 `op_shapes` 重建图结构

每个 combo 对应一个 `op_shapes_*.csv`。这个文件不是节点表，而是按 tensor edge 展开的结构化描述。

项目会先对以下字段去重得到节点：

- `node_idx`
- `node_name`
- `op_type`

然后根据输入 tensor 的 `producer_node` 重建前驱边，忽略：

- `initializer`
- `graph_input`
- `Constant`

这样得到的图是一个非 Constant 的 DAG，用来表示整图的静态依赖关系。

### Step 3. 折叠 8 条 embedding branch

`v1` 对 8 条 embedding branch 应用专门规则。每条 branch 会把：

- `/emb_lX/Gather`
- `/emb_lX/Reshape`
- `/emb_lX/ReduceSum`

折叠成一个复合 branch task。

这样做的原因是：

- 这 3 个 op 在 branch-parallel runner 中天然属于同一条 branch
- branch 的槽位占用应当覆盖整个 `Gather -> ReduceSum` 生命周期
- 单独按 node 级排程不容易反映实际 branch 槽位竞争

### Step 4. 按 `inter_threads` 做 branch 槽位排程

对 8 条 embedding branch，`v1` 固定使用以下规则：

1. 启动顺序固定为 `0 -> 7`
2. `inter_threads` 是同时可用的 branch 槽位数
3. 一条 branch 从 `Gather` 开始占槽
4. 到该 branch 的 `ReduceSum` 结束时释放槽位
5. 后续 branch 按 FIFO 在最早释放的槽位上补入

这对应于项目当前已经验证过的 branch-parallel 执行语义。

### Step 5. 对 bottom 和 tail 做静态时间线拼接

`v1` 不把整图粗暴拆成“所有 op 最长路”，而是按 runner 语义组织：

- `bottom` 与 embedding branches 同时启动
- `tail` 只有在 `bottom` 和所有 embedding branch 都完成后才能开始
- `tail` 内部仍然保留 `op_shapes` 中的真实 DAG 依赖

这一步的结果是整张图的预测时间线，从而得到预测 makespan。

### Step 6. 从真实 timeline 提取整图真值

真实时间来自：

- `branch_parallel_op_timeline.csv`

真值提取规则与 `single_op_stage1_mlp` 保持一致：

1. 对每个 batch 计算整图 span
2. 丢弃最早的一个 batch
3. 对剩余 batch span 求均值

如果某个 artifact 中节点因为上游稳定性过滤缺失，就会进入 partial report；如果一个 artifact 没有做 drop，例如 `300_iter_quick_nodrop`，则可能所有 combo 都是 full graph。

## 4. 当前目录结构

```text
ORT/static_pipeline_eval/
├── AGENTS.md
├── AGENT_WORKLOG.md
├── README.md
├── STATIC_PIPELINE_SCHEDULER_PAPER.md
├── run_static_pipeline_eval.py
├── static_pipeline_eval/
│   ├── __init__.py
│   ├── artifact_loader.py
│   ├── graph_contract.py
│   └── schedule_engine.py
└── tests/
    ├── conftest.py
    ├── test_run_static_pipeline_eval.py
    └── test_schedule_engine.py
```

## 5. 核心模块说明

### `run_static_pipeline_eval.py`

项目的公开 CLI 入口。负责：

- 读取 artifact
- 构建 combo 级图结构
- 调用静态排程器
- 提取真实 timeline 真值
- 生成报告

### `static_pipeline_eval/artifact_loader.py`

负责统一读取：

- `classed_dataset_full.csv`
- `combined_predictions_test.csv`
- `sweep_summary.csv`
- `op_shapes_*.csv`
- `branch_parallel_op_timeline.csv`

### `static_pipeline_eval/schedule_engine.py`

负责：

- DAG 重建
- coverage 分类
- embedding branch 折叠
- FIFO + slot 调度
- 整图 predicted makespan 计算
- embedding 顺序和并发检查

### `static_pipeline_eval/graph_contract.py`

定义跨模块共享的数据结构，例如：

- `ComboSpec`
- `OpNode`
- `BranchTask`
- `CoverageSummary`
- `TimingSummary`
- `ScheduleResult`

## 6. 使用方法

### 6.1 使用默认 artifact 运行

默认 artifact 是：

- `ORT/single_op_stage1_mlp/artifacts/latest/classed_op_mlp_test_78910_analytical_5_200_iter_quick`

运行命令：

```bash
cd /data/qc/dlrm/ORT/static_pipeline_eval
python3 run_static_pipeline_eval.py --run-name v1_validation
```

### 6.2 指定其他兼容 artifact 运行

例如运行 `300_iter_quick_nodrop`：

```bash
cd /data/qc/dlrm/ORT/static_pipeline_eval
python3 run_static_pipeline_eval.py \
  --artifact-root /data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/classed_op_mlp_test_78910_analytical_5_300_iter_quick_nodrop \
  --run-name v1_300_iter_quick_nodrop
```

CLI 参数：

- `--artifact-root`
  指向一个 schema 兼容的 `single_op_stage1_mlp` artifact 根目录
- `--ort-root`
  指向 ORT 根目录，默认是 `/data/qc/dlrm/ORT`
- `--run-name`
  指定输出目录名，结果会写到 `artifacts/latest/<run_name>/`

## 7. 输出文件说明

每次运行会在：

- `ORT/static_pipeline_eval/artifacts/latest/<run_name>/`

生成以下文件。

### `summary.json`

总览统计，包括：

- combo 数量
- full / partial 覆盖数量
- full-graph 误差统计
- partial 误差统计
- embedding 顺序语义检查结果

### `full_combo_metrics.csv`

只包含完整图 combo，每行对应一个完整图 E2E 评估样本。

核心字段：

- `predicted_e2e_us`
- `actual_e2e_us`
- `abs_error_us`
- `ape`

### `partial_combo_metrics.csv`

只包含 partial combo，每行对应一个“观测到的子图”误差样本。

注意：

- 这不是整图 E2E
- 文件中已经显式标记 `metric_scope=observed_subgraph_non_e2e`

### `embedding_order_check.csv`

用于验证静态排程假设是否与真实 timeline 对齐。

核心字段：

- `launch_order`
- `matches_fifo`
- `max_gather_concurrency`
- `matches_inter_threads`
- `handoff_gap_mean_us`

### `calibration_candidates.md`

总结 schedule-level 误差热点，给后续黑盒校准提供候选位置。

当前重点包括：

- embedding 槽位交接空隙
- embedding 复合 branch 残差
- join 后微尾段 bundle
- top MLP 末段波动

## 8. 当前已验证结果

### `v1_validation`

对应 artifact：

- `classed_op_mlp_test_78910_analytical_5_200_iter_quick`

结果摘要：

- total test combos: `331`
- full combos: `49`
- partial combos: `282`
- full-graph MAPE: `0.041985`

### `v1_300_iter_quick_nodrop`

对应 artifact：

- `classed_op_mlp_test_78910_analytical_5_300_iter_quick_nodrop`

结果摘要：

- total test combos: `331`
- full combos: `331`
- partial combos: `0`
- full-graph MAPE: `0.063821`

这个对比说明：

- 不 drop 异常点后，图覆盖更加完整
- 但整体 E2E 误差会变大，因为更多“难样本”被纳入整图评估

## 9. 方法假设与限制

当前 `v1` 的假设是“先把主骨架搭对”，因此它不会主动建模以下复杂行为：

- runtime 乱序调度
- work stealing
- cache / 带宽竞争引起的动态残差
- branch 切换时的额外调度开销拟合

也就是说，`v1` 是一个规则型静态排程器，而不是最终版黑盒校准系统。

如果未来 full-graph 误差依旧偏大，推荐优先在以下位置增加黑盒校准：

1. embedding branch 级 correction
2. branch handoff gap correction
3. join 后微尾段 bundle correction

## 10. 第四章统一实验目录

第四章相关的实验脚本、总控入口、图表构建和章节草稿，已经统一收口到：

- `ORT/static_pipeline_eval/chapter4_experiments`
- 统一产物目录：`ORT/static_pipeline_eval/artifacts/latest/chapter4_cpu`
- 统一章节草稿：`ORT/static_pipeline_eval/chapter4_cpu_experiments_draft.md`

推荐的一键复现入口是：

```bash
cd /data/qc/dlrm/ORT/static_pipeline_eval
python3 chapter4_experiments/run_all_chapter4_experiments.py
```

这条入口会顺序生成：

1. 平台与数据集统计
2. 单算子总体评估
3. 单算子分类评估
4. 单算子代表算子图
5. 单算子 OOD 泛化
6. 单算子基线与消融
7. 整图静态聚合评估
8. 整图简单求和基线
9. 典型时间线与关键路径
10. 图表汇总
11. 章节草稿

第四章脚本的固定入口与配置文件包括：

- `chapter4_experiments/chapter4_config.py`
- `chapter4_experiments/run_single_op_core_eval.py`
- `chapter4_experiments/run_single_op_ood_eval.py`
- `chapter4_experiments/run_single_op_ablation_eval.py`
- `chapter4_experiments/run_e2e_core_eval.py`
- `chapter4_experiments/run_e2e_sum_baseline.py`
- `chapter4_experiments/export_timeline_cases.py`
- `chapter4_experiments/build_chapter4_figures.py`
- `chapter4_experiments/write_chapter4_draft.py`
- `chapter4_experiments/run_all_chapter4_experiments.py`

## 11. 开发约束

这个目录下的 agent 工作流是强约束的：

1. 修改前先读 `AGENT_WORKLOG.md`
2. 修改后必须回写 `AGENT_WORKLOG.md`
3. 每次完成任务后都要在 `ORT/static_pipeline_eval` 子仓库中提交 git

详细规范见：

- `AGENTS.md`
- `AGENT_WORKLOG.md`
