# Roofline Op-Type Analysis

这个目录提供一个独立的 Roofline 分析入口，用 `dataset_all_no_trace` 这类单算子数据表来判断每类 `op_type` 更偏：

- `memory_bound`
- `near_ridge`
- `compute_bound`

分析脚本会优先复用数据表里已有的 `ana_*` / `feat_*` 列；如果输入缺少必要列，则会回退到项目已有的特征工程逻辑，自动补齐：

- `add_engineered_features`
- `add_operator_hardware_context`
- `add_analytical_hardware_software_features`

## 默认运行

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/roofline_op_type_analysis/analyze_roofline_op_types.py
```

默认输入：

- `/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/dataset_all_no_trace/dataset_full.csv`

默认输出：

- `/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/roofline_op_type_analysis`

## 常用参数

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/roofline_op_type_analysis/analyze_roofline_op_types.py \
  --input-csv /data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/dataset_all_no_trace/dataset_full.csv \
  --output-dir /tmp/roofline_op_type_analysis \
  --min-optype-count 50 \
  --ridge-band-low 0.8 \
  --ridge-band-high 1.25 \
  --thread-values 1 2 4
```

参数说明：

- `--hardware-profile`
  - 默认使用 `hardware_profile/kunpeng920_single_numa.yaml`
- `--min-optype-count`
  - 主图默认只展示样本数不低于这个阈值的 `op_type`
- `--ridge-band-low` / `--ridge-band-high`
  - 控制 `near_ridge` 的判定区间
- `--thread-values`
  - 默认自动读取输入里实际出现的线程数

## 分类口径

行级派生指标：

- `arithmetic_intensity = ana_compute_ops / feat_io_bytes_sum`
- `achieved_perf = ana_compute_ops / actual_us`
- `ridge_point = peak_fp32_ops_per_us / mem_bandwidth_bytes_per_us`
- `ridge_gap = arithmetic_intensity / ridge_point`

三分类规则：

- `ridge_gap < ridge_band_low` -> `memory_bound`
- `ridge_band_low <= ridge_gap <= ridge_band_high` -> `near_ridge`
- `ridge_gap > ridge_band_high` -> `compute_bound`

## 输出文件

```text
<output-dir>/
├── row_level_roofline.csv
├── op_type_thread_summary.csv
├── op_type_summary.csv
├── roofline_by_threads.png
├── op_type_bound_share.png
├── op_type_ridge_gap_heatmap.png
└── roofline_summary.json
```

含义说明：

- `row_level_roofline.csv`
  - 每条样本的 Roofline 指标和分类
- `op_type_thread_summary.csv`
  - `(op_type, num_threads)` 聚合点
- `op_type_summary.csv`
  - 每个 `op_type` 的总分类占比和 headline 标签
- `roofline_by_threads.png`
  - 按线程数分面的 Roofline 图
- `op_type_bound_share.png`
  - 每个 `op_type` 的三分类时长占比图
- `op_type_ridge_gap_heatmap.png`
  - `op_type x num_threads` 的聚合 `ridge_gap` 热图
- `roofline_summary.json`
  - 入口参数、总体分类占比、Top runtime `op_type` 标签和输出路径

Roofline 屋檐线绘制：
- y = min(peak_perf, bandwidth * x), 这里的x是算数强度，y是性能上界。

1. 计算屋顶`peak_perf`在[analyze_roofline_op_types.py (line 156)](/data/qc/dlrm/ORT/single_op_stage1_mlp/roofline_op_type_analysis/analyze_roofline_op_types.py:L156)到[analyze_roofline_op_types.py (line 168)](/data/qc/dlrm/ORT/single_op_stage1_mlp/roofline_op_type_analysis/analyze_roofline_op_types.py:L168):
```python
peak_fp32_ops_per_us =
    vector_sp_throughput
    * (simd_width_bits / 32.0)
    * 2.0
    * cpu_clock_ghz
    * 1e3
    * active_cores
``` 
含义：
- `vector_sp_throughput`：每周期每核心的单精度向量指令吞吐量，单位是指令数/cycle/core
- `simd_width_bits`：CPU SIMD 寄存器宽度，单位是bit；simd_width_bits / 32.0 就是每条指令能处理多少个单精度元素
- `2.0`：单精度每个元素需要 2 FLOP（乘加算子）
- `cpu_clock_ghz`：CPU 主频，单位是 GHz
- `1e3`：单位换算，GHz 转 MHz
- `active_cores`：活跃核心数

2. 带宽斜线``bandwidth * x``在图上是斜率为带宽的线，表示算数强度为x时性能上界。[analyze_roofline_op_types.py (line 168)](/data/qc/dlrm/ORT/single_op_stage1_mlp/roofline_op_type_analysis/analyze_roofline_op_types.py:L168)：
```python
mem_bandwidth_bytes_per_us = hw_memory_bandwidth_gb_s_total * 1e3
```
3. Roofline 左边那条斜线表示的是：
```python
Perf = BW * Arithmetic_Intensity
```

4. 拐点 ridge point = `peak_fp32_ops_per_us / mem_bandwidth_bytes_per_us`，表示算数强度达到这个值时，性能上界从带宽受限转为计算受限。

参数解释：
- `total_compute_ops`：算子实际执行的总计算量，单位是 FLOP 或 INTOPS，取决于算子类型
  - Gemm / MatMul：2 * M * N * K
  - ReduceSum: feat_reduction_work_items(规约合并的元素数量)
  - Relu/Add/Mul: feat_output_elements(这几个算子都是主元素算子，输出有多少元素，就需要做多少次核心运算)
  - sigmoid: 4*feat_output_elements (每个元素需要 4 次核心运算：取负、exp、加1、除法)
- `total_io_bytes `：算子实际执行的总 I/O 字节数 (output_size + activation_size + parameter_size)