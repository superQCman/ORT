# analyze_analytical_feature_correlation.py 使用说明

## 脚本作用

[analyze_analytical_feature_correlation.py](./analyze_analytical_feature_correlation.py) 用来分析 `classed_op_mlp` 数据集中 analytical proxy 特征与真实算子时延标签之间的相关性。

它的核心目标是回答两个问题：

1. `ana_calib_*` 这类 analytical 特征和真实时延 `label_operator_actual_dur_us` 的相关性有多强。
2. 这些特征在不同数据切分、不同模型组、以及不同 `op_type / case_id / combo` 维度下是否稳定。

脚本会直接读取已经准备好的 grouped dataset，而不是重新构建数据或训练模型。

## 输入与默认数据位置

默认输入根目录是：

- `../artifacts/latest/classed_op_mlp_test_analytical_5_200_iter`

脚本默认会从这里读取：

- `datasets/<model_group>/train.csv`
- `datasets/<model_group>/val.csv`
- `datasets/<model_group>/test.csv`
- `datasets/<model_group>/feature_columns.json`

默认分析的模型组是：

- `gather`

内置支持的 5 个默认模型组为：

- `gather`
- `layout_move`
- `view_meta`
- `mixed_balanced`
- `compute_dominant`

默认目标列：

- `label_operator_actual_dur_us`

默认 analytical 特征列：

- `ana_calib_mem_us`
- `ana_calib_compute_us`
- `ana_calib_total_us`


## 当前 `mixed_balanced` 的 analytical model 口径

这份相关性分析文档现在需要结合最新的 `mixed_balanced` contract 来看：

- 当前 `mixed_balanced` 训练输入里真正启用的 analytical 特征是 `ana_calib_total_us`
- `ana_calib_mem_us` 和 `ana_calib_compute_us` 仍然会导出到 grouped dataset 里，方便误差分析，但它们不再作为该异构组的统一训练输入
- 原因是 `mixed_balanced` 同时包含：
  - `ReduceSum`
  - `Relu`
  - `Add`
  - `Mul`
  - `Sigmoid`
- 这些算子的机理差异较大，统一用 `mem_us / compute_us` 两分法做组级输入不稳定；更合适的是使用按 `op_type` 细分公式生成的 `ana_calib_total_us`

当前这 5 个算子的 analytical model 口径如下。

### `ReduceSum`

`ReduceSum` 继续沿用既有 calibrated reduction 模型：

- `stream_bytes = activation_size + output_size`
- `add_ops = feat_reduction_work_items`
- `inner = feat_reduction_axes_product`
- `BW_reduce_inf = BW_peak * rho_copy_inf * kappa_reduce`
- `BW_reduce_eff(inner) = BW_reduce_inf * inner / (BW_reduce_inf * tau_reduce_start + inner)`
- `T_reduce = max(stream_bytes / BW_reduce_eff(inner), add_ops / PeakAdd(T))`

这里：

- `ana_calib_total_us = T_reduce`
- `ana_calib_mem_us` 对应 `stream_bytes / BW_reduce_eff(inner)`
- `ana_calib_compute_us` 对应 `add_ops / PeakAdd(T)`

### `Relu`

`Relu` 现在不再走旧的 `generic_mixed` fallback，而是单独按 unary elementwise、memory-dominant kernel 建模：

- `stream_bytes = feat_io_bytes_sum`
- `BW_relu_inf = BW_peak * rho_relu_inf`
- `BW_relu_eff(stream_bytes) = BW_relu_inf * stream_bytes / (BW_relu_inf * tau_relu_start + stream_bytes)`
- `T_relu = stream_bytes / BW_relu_eff(stream_bytes)`

这里：

- `ana_calib_total_us = T_relu`
- `ana_calib_mem_us = T_relu`
- `ana_calib_compute_us = 0`
- `ana_calib_overhead_us = 0`

### `Add`

`Add` 现在按 binary elementwise micro-kernel 建模，假设主要由小流量 bandwidth efficiency 和固定 kernel overhead 组成：

- `stream_bytes = feat_io_bytes_sum`
- `BW_add_inf = BW_peak * rho_add_inf`
- `BW_add_eff(stream_bytes) = BW_add_inf * stream_bytes / (BW_add_inf * tau_add_start + stream_bytes)`
- `T_add = tau_add + stream_bytes / BW_add_eff(stream_bytes)`

这里：

- `ana_calib_total_us = T_add`
- `ana_calib_mem_us = stream_bytes / BW_add_eff(stream_bytes)`
- `ana_calib_compute_us = 0`
- `ana_calib_overhead_us = tau_add`

### `Mul`

`Mul` 和 `Add` 类似，也按 binary elementwise micro-kernel 建模，但拥有独立的 calibrated 参数：

- `stream_bytes = feat_io_bytes_sum`
- `BW_mul_inf = BW_peak * rho_mul_inf`
- `BW_mul_eff(stream_bytes) = BW_mul_inf * stream_bytes / (BW_mul_inf * tau_mul_start + stream_bytes)`
- `T_mul = tau_mul + stream_bytes / BW_mul_eff(stream_bytes)`

这里：

- `ana_calib_total_us = T_mul`
- `ana_calib_mem_us = stream_bytes / BW_mul_eff(stream_bytes)`
- `ana_calib_compute_us = 0`
- `ana_calib_overhead_us = tau_mul`

### `Sigmoid`

`Sigmoid` 现在按 unary nonlinear kernel 建模，显式保留固定 overhead，并在 memory path 与 nonlinear compute path 之间取较慢者：

- `stream_bytes = feat_io_bytes_sum`
- `output_elements = output_size / bytes_per_element`
- `BW_sigmoid_inf = BW_peak * rho_sigmoid_inf`
- `BW_sigmoid_eff(stream_bytes) = BW_sigmoid_inf * stream_bytes / (BW_sigmoid_inf * tau_sigmoid_start + stream_bytes)`
- `PeakSigmoidEff(T) = PeakAdd(T) * rho_sigmoid_compute`
- `T_sigmoid = tau_sigmoid + max(stream_bytes / BW_sigmoid_eff(stream_bytes), output_elements / PeakSigmoidEff(T))`

这里：

- `ana_calib_total_us = T_sigmoid`
- `ana_calib_mem_us = stream_bytes / BW_sigmoid_eff(stream_bytes)`
- `ana_calib_compute_us = output_elements / PeakSigmoidEff(T)`
- `ana_calib_overhead_us = tau_sigmoid`

### 对相关性结果的影响

因此，在当前版本的数据上：

- `mixed_balanced` 的 `--auto-feature-cols` 通常只会自动选择 `ana_calib_total_us`
- 如果你在 `test_by_op_type.csv` 或 `summary.md` 里看到 `Relu / Add / Mul / Sigmoid` 的误差，这些误差已经来自各自独立的 analytical 子模型，而不是旧的统一 `generic_mixed`
- `mixed_balanced` 的组级相关性结果，本质上是在验证“按 `op_type` 分流后的 `ana_calib_total_us` 能否作为一个稳定的 heterogeneous group proxy”


## 实现逻辑

## 1. 解析命令行参数

脚本入口在 `main()`，首先通过 `parse_args()` 解析参数，支持两种运行模式：

- 单个 `model_group` 分析
- 多个 `model_group` 的 suite 分析

相关入口逻辑见：

- [analyze_analytical_feature_correlation.py:502-548](analyze_analytical_feature_correlation.py#L502-L548)

## 2. 决定分析哪些模型组

- 如果传入 `--all-model-groups`，就使用固定的 5 组默认分桶。
- 如果传入 `--model-groups`，支持空格分隔或逗号分隔。
- 如果两者都不传，则走单组模式，使用 `--model-group`。

相关实现：

- [analyze_analytical_feature_correlation.py:104-115](analyze_analytical_feature_correlation.py#L104-L115)

## 3. 解析要分析的 analytical 特征列

脚本不会盲目相信命令行里传入的列名，而是先读取每个组目录下的 `feature_columns.json`，再确认特征是否真的存在于该组数据中。

逻辑分两种：

- 默认模式：从请求的列里筛出当前数据集确实存在的列。
- `--auto-feature-cols`：自动选择 manifest 中所有以 `ana_calib_` 开头的数值特征。

如果一个组里完全没有可用的 analytical 特征，会直接报错，避免产出误导性结果。

相关实现：

- [analyze_analytical_feature_correlation.py:194-216](analyze_analytical_feature_correlation.py#L194-L216)

## 4. 读取 train / val / test 三个 split

`load_split_frames()` 会读取当前组目录下的：

- `train.csv`
- `val.csv`
- `test.csv`

并且：

- 给每一行补一个 `split` 字段
- 合并成一个总表
- 将 `row_uid` 统一转成字符串，便于后续稳定处理

相关实现：

- [analyze_analytical_feature_correlation.py:180-191](analyze_analytical_feature_correlation.py#L180-L191)

## 5. 针对每个特征计算统计指标

核心计算在 `compute_stats()`。

对每个 `feature_col` 与目标列 `target_col`，脚本会先：

- 把值安全转换为数值
- 将 `inf/-inf` 转成 `NaN`
- 丢弃任一列为空的样本

随后计算：

- `rows`：有效样本数
- `pearson_r`：线性相关系数
- `spearman_rho`：秩相关系数
- `linear_fit_slope_y_on_x`
- `linear_fit_intercept`
- `mean_actual_us`
- `mean_feature_us`
- `median_actual_us`
- `median_feature_us`
- `mape_vs_actual`
- `dwre_vs_actual`

其中：

- `MAPE` 反映平均相对误差
- `DWRE` 反映按总真实时延加权后的相对误差

为了避免异常输入破坏统计，脚本还做了几个保护：

- 样本数不足时返回 `NaN`
- 当特征或标签是常数列时，相关系数返回 `NaN`
- 分母最小被裁剪到 `1e-9`，避免除零

相关实现：

- [analyze_analytical_feature_correlation.py:117-177](analyze_analytical_feature_correlation.py#L117-L177)

## 6. 生成 split 级别汇总

`split_summary()` 会分别对：

- `train`
- `val`
- `test`
- `all`

这 4 个范围计算每个 analytical 特征的统计结果，输出一个汇总表。

相关实现：

- [analyze_analytical_feature_correlation.py:219-231](analyze_analytical_feature_correlation.py#L219-L231)

## 7. 生成分组 breakdown 汇总

`grouped_summary()` 支持按照指定列继续做细分统计。

单组分析里默认会产出两类 breakdown：

- 全量数据按 `op_type` 分组
- test 数据按 `op_type`、`case_id`、`combo` 分组

如果某个分组列在当前数据里不存在，函数会直接返回空表，不会报错。

相关实现：

- [analyze_analytical_feature_correlation.py:234-248](analyze_analytical_feature_correlation.py#L234-L248)
- [analyze_analytical_feature_correlation.py:307-370](analyze_analytical_feature_correlation.py#L307-L370)

## 8. 输出 CSV、JSON 和 Markdown 摘要

单组模式下，`analyze()` 会输出：

- `split_summary.csv`
- `all_by_<group_col>.csv`
- `test_by_<group_col>.csv`
- `summary.json`
- `summary.md`

其中：

- `summary.json` 记录输入参数与产物路径
- `summary.md` 会把关键统计表渲染成便于直接查看的 Markdown

相关实现：

- [analyze_analytical_feature_correlation.py:251-304](analyze_analytical_feature_correlation.py#L251-L304)
- [analyze_analytical_feature_correlation.py:307-370](analyze_analytical_feature_correlation.py#L307-L370)

## 9. 支持 suite 模式做跨组汇总

当传入多个模型组后，`analyze_suite()` 会对每个组分别执行单组分析，然后再合并生成：

- `suite_split_summary.csv`
- `suite_test_summary.csv`
- `suite_test_best_feature_summary.csv`
- `suite_summary.json`
- `suite_summary.md`

其中：

- `suite_test_summary.csv` 聚焦每个组在 `test` split 上的表现
- `suite_test_best_feature_summary.csv` 会按 `mape_vs_actual`、`dwre_vs_actual` 和特征名排序，选出每个组 test 表现最好的 analytical 特征

相关实现：

- [analyze_analytical_feature_correlation.py:373-499](analyze_analytical_feature_correlation.py#L373-L499)

## 如何使用

以下命令都需要在 `ORT/single_op_stage1_mlp/classed_op_mlp` 目录或其上层可访问该脚本的位置执行。

## 1. 分析单个模型组

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/analyze_analytical_feature_correlation.py \
  --model-group gather
```

默认输出目录为：

- `/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/classed_op_mlp_test_analytical_5_200_iter/analysis/analytical_feature_correlation/gather`

## 2. 分析全部默认 5 个模型组

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/analyze_analytical_feature_correlation.py \
  --all-model-groups
```

默认输出目录为：

- `/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/classed_op_mlp_test_analytical_5_200_iter/analysis/analytical_feature_correlation_suite`

## 3. 指定多个模型组

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/analyze_analytical_feature_correlation.py \
  --model-groups gather compute_dominant mixed_balanced
```

也支持逗号分隔写法：

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/analyze_analytical_feature_correlation.py \
  --model-groups gather,compute_dominant,mixed_balanced
```

## 4. 自动选择当前组里所有 `ana_calib_*` 特征

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/analyze_analytical_feature_correlation.py \
  --model-group mixed_balanced \
  --auto-feature-cols
```

这个模式适合不同组的 analytical 特征列不完全一致时使用。

## 5. 手动指定特征列与输出目录

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/analyze_analytical_feature_correlation.py \
  --model-group compute_dominant \
  --feature-cols ana_calib_compute_us ana_calib_total_us \
  --output-dir /tmp/classed_op_corr_compute
```

## 6. 自定义 breakdown 维度

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/analyze_analytical_feature_correlation.py \
  --model-group gather \
  --all-breakdown-cols op_type combo \
  --test-breakdown-cols op_type case_id combo
```

## 输出结果如何解读

优先看这些文件：

- `split_summary.csv`：单组在 `train/val/test/all` 上的总体统计
- `test_by_op_type.csv`：看某个组里不同 `op_type` 上 analytical proxy 是否稳定
- `test_by_case_id.csv`：看不同 case 上是否泛化一致
- `test_by_combo.csv`：看不同配置组合上的误差分布
- `summary.md`：适合快速人工浏览
- `suite_test_best_feature_summary.csv`：适合跨组挑选最有效的 analytical proxy

一般来说：

- `Pearson r` 越接近 1，说明线性关系越强
- `Spearman rho` 越接近 1，说明排序一致性越强
- `MAPE / DWRE` 越低越好，说明 analytical proxy 更接近真实时延

## 适用场景

这个脚本适合在以下场景使用：

- 判断某个组是否值得保留 `ana_calib_*` 特征
- 比较 `ana_calib_mem_us`、`ana_calib_compute_us`、`ana_calib_total_us` 哪个更有效
- 在正式训练前先做低成本特征筛查
- 跨 5 个模型组比较 analytical proxy 的可迁移性和稳定性
