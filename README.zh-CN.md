# ORT DLRM Sweep 与训练特征构建

English version: [README.md](./README.md)

该目录包含当前基于 ONNX Runtime 的 DLRM 工作流，用于：

- 运行带 profiling 的整图推理
- 生成按 batch 变化的 `op_shapes` 元数据
- 使用 DynamoRIO 构建并追踪单算子二进制
- 提取单算子 trace 特征
- 将 ORT CPU 线程 profiling 与 trace 特征合并为训练数据集
- 提取供下游建模使用的精简特征子集

## 主入口

使用 [run_ort_sweep.sh](./run_ort_sweep.sh) 作为标准的端到端工作流入口。

对于每个 `(batch_size, num_indices_per_lookup)` 组合，它会：

1. 运行启用 ORT profiling 的 [run_ort_dlrm.py](./run_ort_dlrm.py)
2. 写出组合专属的 `op_shapes_{batch}_{nip}.csv`
3. 在 `sweep_runs/onnx_profiles/` 下写出组合专属的 ORT profiling JSON
4. 从 profiling JSON 解析 CPU 线程使用情况
5. 基于推理阶段实际使用的改写后 ONNX 构建单算子二进制
6. 对所有单算子二进制运行 DynamoRIO trace
7. 提取单算子 trace 特征
8. 将 `op_shapes + cpu_thread_detail + trace_features` 合并成 `features/` 下的最终训练 CSV
9. 从最终训练 CSV 中提取 `features_selected/` 下的精简特征 CSV

当你希望复用同一套下游 sweep 流程，但需要切换不同推理前端时，使用 [run_ort_sweep_extensible.sh](./run_ort_sweep_extensible.sh)，例如切到 `branch_parallel` 模式。

当你只想做本地单次验证时，使用 [run_ort.sh](./run_ort.sh)。它支持：

- 标准整图 runner [run_ort_dlrm.py](./run_ort_dlrm.py)
- 手工 branch-parallel runner [run_ort_dlrm_branch_parallel.py](./run_ort_dlrm_branch_parallel.py)，通过 `MANUAL_BRANCH_PARALLEL=1` 开启

默认 sweep 范围为：

- `batch_size`: `32..2048`，步长 `32`
- `num_indices_per_lookup`: `100..1000`，步长 `50`

## 关键脚本

- [run_ort_sweep.sh](./run_ort_sweep.sh)
  端到端 batch sweep 驱动脚本。

- [run_ort_sweep_extensible.sh](./run_ort_sweep_extensible.sh)
  可扩展 sweep 驱动脚本，支持 `RUNNER_MODE=standard|branch_parallel`。

- [run_ort_dlrm.py](./run_ort_dlrm.py)
  使用 ORT 运行 DLRM 推理，导出 `op_shapes.csv`，并写出 ORT profiling JSON。

- [run_ort_dlrm_branch_parallel.py](./run_ort_dlrm_branch_parallel.py)
  将改写后的 DLRM 图拆成 `bottom`、`emb_l0..emb_l7` 和 `tail`，再通过多个 ORT session 并发执行 branch task。

- [extract_cpu_thread_usage.py](./onnx_operator_analysis/extract_cpu_thread_usage.py)
  解析 ORT profiling JSON，提取 CPU 线程调度统计。

- [extract_trace_features.py](./dynamorio_tracing/extract_trace_features.py)
  解析 DynamoRIO trace 输出，生成单算子 trace 特征 CSV。

- [build_training_features.py](./onnx_operator_analysis/build_training_features.py)
  将 ORT profiling 节点对齐到 `op_shapes`，按节点聚合 CPU 线程指标，并与 trace 特征合并。

- [select_feature_subset.py](./onnx_operator_analysis/select_feature_subset.py)
  从合并后的训练数据集中提取精简特征 CSV，并在需要时从对齐后的 CPU profiling CSV 回填 shape 字段。

- [visualize_ort_profile_timeline.py](./onnx_operator_analysis/visualize_ort_profile_timeline.py)
  将 ORT `Node` 事件可视化为 lane-based 时间线，并汇总算子并发情况。

## 整体目录结构

```text
ORT/
├── README.md
├── README.zh-CN.md
├── run_ort_dlrm.py
├── run_ort_dlrm_branch_parallel.py
├── run_ort.sh
├── run_ort_sweep.sh
├── run_ort_sweep_extensible.sh
├── op_shapes.csv
├── features/
│   ├── README.md
│   └── bs*_nip*.csv
├── features_selected/
│   └── bs*_nip*.csv
├── features_extensible/
│   └── bs*_nip*.csv
├── features_extensible_selected/
│   └── bs*_nip*.csv
├── sweep_runs/
│   ├── generated_onnx/
│   │   └── bs*_nip*/
│   ├── logs/
│   │   └── bs*_nip*/
│   ├── onnx_profiles/
│   │   └── bs*_nip*/
│   ├── op_shapes/
│   │   └── op_shapes_<batch>_<num_indices>.csv
│   └── sweep_summary.csv
├── sweep_runs_extensible/
│   ├── generated_onnx/
│   │   └── bs*_nip*/
│   ├── logs/
│   │   └── bs*_nip*/
│   ├── onnx_profiles/
│   │   └── bs*_nip*/
│   ├── op_shapes/
│   │   └── op_shapes_<batch>_<num_indices>.csv
│   └── sweep_summary.csv
├── onnx_operator_analysis/
│   ├── build_training_features.py
│   ├── extract_cpu_thread_usage.py
│   ├── visualize_ort_profile_timeline.py
│   └── *.md
└── dynamorio_tracing/
    ├── extract_trace_features.py
    ├── run_build_op_binaries.sh
    ├── run_drrio_all_ops.sh
    ├── out_per_op_bins_sweep/
    │   └── bs*_nip*/
    ├── drrio_traces_sweep/
    │   └── bs*_nip*/
    ├── trace_features_sweep/
    │   └── bs*_nip*.csv
    ├── out_per_op_bins_sweep_extensible/
    │   └── bs*_nip*/
    ├── drrio_traces_sweep_extensible/
    │   └── bs*_nip*/
    ├── trace_features_sweep_extensible/
    │   └── bs*_nip*.csv
    └── scripts/
        ├── generate_op_binaries.py
        ├── ort_per_op_trace.py
        ├── ort_with_markers.py
        └── single_op_runner.py
```

## 目录布局

- `sweep_runs/op_shapes/`
  每个组合对应的 `op_shapes_{batch}_{nip}.csv`

- `sweep_runs/onnx_profiles/<combo>/`
  整图 ORT profiling JSON 及其派生的 CPU 线程 CSV

- `dynamorio_tracing/out_per_op_bins_sweep/<combo>/`
  生成出的单算子二进制及 manifest

- `dynamorio_tracing/drrio_traces_sweep/<combo>/`
  原始 DynamoRIO trace 输出

- `dynamorio_tracing/trace_features_sweep/<combo>.csv`
  从 DynamoRIO trace 提取出的单算子 trace 特征

- `features/<combo>.csv`
  最终训练数据集，每行对应一个 ONNX 节点

- `features_selected/<combo>.csv`
  从 `features/<combo>.csv` 提取出的精简特征数据集

- [features/README.md](./features/README.md)
  `features/*.csv` 的字段字典，说明每类特征及各列含义

- `sweep_runs/sweep_summary.csv`
  每个组合一行的汇总表，包含关键输出路径

- `sweep_runs_extensible/onnx_profiles/<combo>/`
  extensible sweep 的 ORT profile 输出目录，包含 branch-parallel 合并后的 `ort_cann_profile_*.json`

- `features_extensible/<combo>.csv`
  extensible sweep 生成的最终训练数据集

- `features_extensible_selected/<combo>.csv`
  基于 extensible sweep 输出再提取出的精简特征数据集

- `sweep_runs_extensible/sweep_summary.csv`
  extensible sweep 的汇总表，每个组合一行，额外记录 `runner_mode`、`profile_json` 和 `effective_onnx_path`

## 依赖前提

该工作流默认以下环境可用：

- 可用的 Python 环境，包含 `onnxruntime`、`numpy` 和 `onnx`
- 如果 `USE_CANN=1`，则需要可用的 Ascend CANN 环境
- 配置路径下可用的 DynamoRIO
- 位于 `ONNX_PATH` 的 DLRM ONNX 模型

默认值为：

- `ASCEND_ENV_SH=/data/qc/Ascend/ascend-toolkit/set_env.sh`
- `ONNX_PATH=/data/qc/dlrm/dlrm_onnx/dlrm_s_pytorch.onnx`
- `DRRUN=/data/qc/simulator/DynamoRIO-AArch64-Linux-11.3.0-1/bin64/drrun`

## 快速开始

运行完整 sweep：

```bash
cd /data/qc/dlrm/ORT
bash run_ort_sweep.sh
```

只跑一个组合做验证：

```bash
cd /data/qc/dlrm/ORT
BATCH_START=32 BATCH_END=32 \
NUM_INDICES_START=100 NUM_INDICES_END=100 \
MAX_COMBOS=1 \
bash run_ort_sweep.sh
```

中断后续跑：

```bash
cd /data/qc/dlrm/ORT
RESUME=1 bash run_ort_sweep.sh
```

通过 extensible sweep 跑一个 branch-parallel 组合：

```bash
cd /data/qc/dlrm/ORT
RUNNER_MODE=branch_parallel \
PYTHON_BIN=/data/qc/anaconda3/envs/ort/bin/python \
BATCH_START=32 BATCH_END=32 \
NUM_INDICES_START=100 NUM_INDICES_END=100 \
MAX_COMBOS=1 \
INTRA_THREADS=4 \
INTER_THREADS=4 \
PARALLEL_BRANCHES=2 \
bash run_ort_sweep_extensible.sh
```

通过 `run_ort.sh` 本地跑一次 branch-parallel profile：

```bash
cd /data/qc/dlrm/ORT
MANUAL_BRANCH_PARALLEL=1 bash run_ort.sh
```

## 重要配置

[run_ort_sweep.sh](./run_ort_sweep.sh) 当前支持的常用环境变量：

- `NUM_BATCHES`, `WARMUP_BATCHES`
- `INTRA_THREADS`, `INTER_THREADS`
- `USE_CANN`, `DEVICE_ID`
- `NO_REPLACE_LOOP`
- `DISABLE_GRAPH_OPTIMIZATIONS`
- `USE_NUMACTL`, `NUMA_NODE`
- `RESUME`
- `EXTRACT_FEATURES`
- `SELECT_FEATURE_SUBSET`
- `SELECT_FEATURE_SUBSET_DUR_SOURCE=avg|min|max|sum`
- `START_IDX`, `MAX_OPS`

[run_ort_sweep_extensible.sh](./run_ort_sweep_extensible.sh) 额外支持的环境变量：

- `RUNNER_MODE=standard|branch_parallel`
- `PARALLEL_BRANCHES`
- `TAIL_INTRA_THREADS`
- `VERIFY_FULL_OUTPUT`
- `BRANCH_SUBMODEL_ROOT`
- `FORCE_CPU_OPS`
- `OUT_ROOT`, `PROFILE_ROOT`, `FEATURE_DATASET_ROOT`
- `FEATURE_SUBSET_ROOT`

当前工作流中的重要默认值：

- `NO_REPLACE_LOOP=0`
  保持启用 Loop 替换。

- `DISABLE_GRAPH_OPTIMIZATIONS=1`
  在整图 profiling 时关闭图优化，使 runtime 节点名尽量贴近 `op_shapes`。

- `EXTRACT_FEATURES=1`
  最终训练数据集依赖 DynamoRIO trace 特征。

- `SELECT_FEATURE_SUBSET=1`
  在生成 `features/*.csv` 后，sweep 还会继续输出 `features_selected/` 或 `features_extensible_selected/` 下的精简特征 CSV。

对于 `RUNNER_MODE=branch_parallel`：

- `INTER_THREADS` 仍然是通用的 sweep 级并发参数。
- `PARALLEL_BRANCHES` 是 branch runner 专用的覆盖参数。
- 实际 branch 并发度为 `PARALLEL_BRANCHES > 0 ? PARALLEL_BRANCHES : INTER_THREADS`。

## 单独运行各阶段

运行整图 ORT 推理，并导出 `op_shapes` 和 profiling JSON：

```bash
cd /data/qc/dlrm/ORT
python3 run_ort_dlrm.py \
  --onnx-path /data/qc/dlrm/dlrm_onnx/dlrm_s_pytorch.onnx \
  --batch-size 32 \
  --num-batches 3 \
  --warmup-batches 2 \
  --num-indices-per-lookup 100 \
  --shape-csv ./op_shapes.csv \
  --enable-profiling \
  --profile-dir ./onnx_operator_analysis \
  --disable-graph-optimizations
```

运行手工 branch-parallel 推理，同时导出拆分 profile 和合并后的 ORT profile：

```bash
cd /data/qc/dlrm/ORT
python3 run_ort_dlrm_branch_parallel.py \
  --onnx-path ./dlrm_onnx/dlrm_s_pytorch.onnx \
  --batch-size 32 \
  --num-batches 3 \
  --warmup-batches 2 \
  --num-indices-per-lookup 100 \
  --shape-csv ./op_shapes.csv \
  --enable-profiling \
  --profile-dir ./onnx_operator_analysis/branch_parallel \
  --intra-threads 4 \
  --inter-threads 4 \
  --parallel-branches 2 \
  --disable-graph-optimizations
```

从单个 ORT profile JSON 中解析 CPU 线程统计：

```bash
cd /data/qc/dlrm/ORT
PROFILE_DIR=./sweep_runs/onnx_profiles/bs32_nip100
PROFILE_JSON=$(find "$PROFILE_DIR" -maxdepth 1 -name 'ort_cann_profile*.json' | sort | tail -n 1)

python3 onnx_operator_analysis/extract_cpu_thread_usage.py \
  "$PROFILE_JSON" \
  --out-dir "$PROFILE_DIR"
```

对单个 ORT profile JSON 做算子并发时间线可视化：

```bash
cd /data/qc/dlrm/ORT
python3 onnx_operator_analysis/visualize_ort_profile_timeline.py \
  ./onnx_operator_analysis/ort_cann_profile_2026-03-11_15-32-04.json
```

对于 branch-parallel 运行，profile 目录中还可能包含：

- 拆分后的子图 profile JSON，例如 `bottom_profile_*.json`、`emb_l*_profile_*.json`
- 供 `extract_cpu_thread_usage.py` 兼容处理的合并 `ort_cann_profile_*.json`
- `branch_parallel_timeline.csv/html`
- `branch_parallel_op_timeline.csv/html`
- `branch_parallel_concurrency_segments.csv`
- `branch_parallel_op_concurrency_segments.csv`

基于已有产物构建最终训练 CSV：

```bash
cd /data/qc/dlrm/ORT
BATCH_SIZE=32
NUM_INDICES=100
COMBO=bs${BATCH_SIZE}_nip${NUM_INDICES}
PROFILE_DIR=./sweep_runs/onnx_profiles/$COMBO
PROFILE_STEM=$(basename "$(find "$PROFILE_DIR" -maxdepth 1 -name 'ort_cann_profile*.json' | sort | tail -n 1)" .json)
SHAPE_CSV=$(find ./sweep_runs/op_shapes -maxdepth 1 -name "op_shapes_${BATCH_SIZE}_${NUM_INDICES}.csv" | head -n 1)
TRACE_FEATURES=./dynamorio_tracing/trace_features_sweep/$COMBO.csv

python3 onnx_operator_analysis/build_training_features.py \
  --op-shapes "$SHAPE_CSV" \
  --cpu-detail "$PROFILE_DIR/${PROFILE_STEM}_cpu_thread_detail.csv" \
  --trace-features "$TRACE_FEATURES" \
  --aligned-cpu-detail-out "$PROFILE_DIR/${PROFILE_STEM}_cpu_thread_detail_aligned.csv" \
  --cpu-agg-out "$PROFILE_DIR/${PROFILE_STEM}_cpu_thread_node_aggregated.csv" \
  --unmatched-out "$PROFILE_DIR/${PROFILE_STEM}_cpu_thread_unmatched.csv" \
  --out ./features/$COMBO.csv \
  --batch-size "$BATCH_SIZE" \
  --num-indices-per-lookup "$NUM_INDICES"
```

基于已有 merged training CSV 提取精简特征 CSV：

```bash
cd /data/qc/dlrm/ORT
python3 onnx_operator_analysis/select_feature_subset.py \
  --input ./features/$COMBO.csv \
  --output ./features_selected/$COMBO.csv
```

## 输出语义

最终的 `features/<combo>.csv` 会保留 `op_shapes` 中每个 ONNX 节点各一行。

最终的 `features_selected/<combo>.csv` 也保持每个 ONNX 节点一行，但只保留供下游实验使用的精简特征子集。默认情况下，`dur_us` 来自 `cpu_dur_us_avg`，而 `input_type_shape` / `output_type_shape` 会在 merged dataset 中缺失时，从 `*_cpu_thread_detail_aligned.csv` 自动回填。

## 精简特征说明

sweep 还可以通过 [select_feature_subset.py](./onnx_operator_analysis/select_feature_subset.py) 额外生成 `features_selected/` 和 `features_extensible_selected/` 下的精简特征数据集。

对于每个组合，sweep 会额外写出：

- `features_selected/<combo>.csv`
- `features_selected/all_features.csv`
- `features_extensible_selected/<combo>.csv`
- `features_extensible_selected/all_features.csv`

精简特征数据集仍然保持每个 ONNX 节点一行，目前包含：

- 元数据：`batch_size`、`num_indices_per_lookup`、`node_name`、`op_type`、`trace_op_name`
- 算子 shape / size 字段：`input_type_shape`、`output_type_shape`、`output_size`、`activation_size`、`parameter_size`
- 指令与访存统计：`total_instructions`、`total_loads`、`total_stores`、`load_store_ratio`、`num_threads`
- reuse time 摘要：`reuse_time_mean` 以及所有已存在的 `reuse_time_bin_<n>_pct`
- reuse distance 摘要：`reuse_distance_mean`、`reuse_distance_median`、`reuse_distance_std`、`reuse_distance_unique_cache_lines_per_k_accesses`、`reuse_distance_instruction_accesses`、`reuse_distance_data_accesses`
- opcode mix 特征：`opc_branch_ratio`、`opc_fp_convert`、`opc_fp_load_simd`、`opc_fp_math`、`opc_fp_move`、`opc_fp_store_simd`、`opc_math`、`opc_simd`
- 耗时特征：`dur_us`

字段映射规则：

- `dur_us` 默认映射自 `cpu_dur_us_avg`；可以通过 `SELECT_FEATURE_SUBSET_DUR_SOURCE=avg|min|max|sum` 切换
- `output_size`、`activation_size`、`parameter_size` 来自 merged dataset 中 CPU 聚合后的 size 列
- `input_type_shape` 和 `output_type_shape` 会在需要时从 `*_cpu_thread_detail_aligned.csv` 回填
- `reuse_time_bin_<n>_pct` 会根据 merged feature 表头动态发现，因此如果上游提取器新增了更多 bin，这里的列数也会随之扩展

每一行可能包含：

- 标准化后的 `node_idx`、`node_name`、`op_type`
- 来自 DynamoRIO 的 trace 特征
- 来自 ORT 的按节点聚合 CPU 线程 profiling 特征
- `has_trace_features`
- `has_cpu_profile`
- `cpu_profile_missing_reason`

即使某些节点没有 CPU 线程调度统计，仍会保留在数据集中，尤其是 `Constant` 和其他轻量节点。这类行通常会是：

- `has_trace_features=1`
- `has_cpu_profile=0`
- `cpu_profile_missing_reason=constant_or_no_thread_stats`

## 对齐规则

要得到正确结果，下列内容必须针对同一组参数保持一致：

- `batch_size`
- `num_indices_per_lookup`
- `op_shapes_{batch}_{nip}.csv`
- ORT 推理实际使用的改写后 ONNX
- 基于该改写后 ONNX 构建出的单算子二进制

如果这些内容不一致，流程可能仍然能跑完，但最终数据集将不再严格对应目标模型运行。

## 常见问题

- `extract_features.log` 在运行时一直为空
  如果不是以非缓冲模式运行，Python 输出可能被缓冲。当前 sweep 已经对特征提取使用 `python3 -u`。

- 整图 profiling 的节点数与单算子二进制数量不一致
  ORT profiling 反映的是 runtime 执行事件，而单算子二进制来自 ONNX 图节点。`Constant` 节点以及 runtime 融合/优化后的节点都会导致差异。

- 有些行的 `has_cpu_profile=0`
  这是预期行为，表示该节点在 ORT profiling 中没有产出 `thread_scheduling_stats`，尤其常见于 `Constant` 节点。

- 单算子构建使用了错误的 ONNX
  当前 sweep 会从推理阶段解析出实际使用的改写后 ONNX，并将该路径传给单算子构建流程。

## 说明

- 推荐使用 `run_ort_sweep.sh` 作为可复现实验的数据集生成入口。
- 如果要接入新的推理前端而又不想影响后台还在运行的 `run_ort_sweep.sh`，优先在 `run_ort_sweep_extensible.sh` 中扩展。
- sweep 会为每组 `(batch_size, num_indices_per_lookup)` 输出独立产物，避免不同组合之间互相覆盖。
