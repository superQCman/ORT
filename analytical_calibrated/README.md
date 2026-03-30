# Analytical Calibrated

这个目录是三分类单算子方案里的可解释校准 analytical 子流水线。

输入固定为：

- [dataset_full.csv](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/dataset_all_no_trace/dataset_full.csv)

它做两件事：

1. 为全量 `dataset_all_no_trace` 导出 `ana_calib_*` 行级特征。
2. 对 analytical model 做泛化验证，包括：
   - `leave_one_case_out`
   - `leave_one_combo_out`

## 覆盖范围

精细公式 heavy family：

- `Gather`
- `ReduceSum`
- `Gemm`
- `MatMul`
- `Transpose`
- `Concat`

通用 proxy：

- `generic_memory`
  - `Reshape`
  - `Shape`
  - `Unsqueeze`
  - `Flatten`
- `generic_mixed`
  - `Relu`
  - `Add`
  - `Mul`
  - `Sigmoid`

## 输出特征含义

| 特征 | 含义 |
| --- | --- |
| `ana_calib_total_us` | 校准 analytical model 预测的总时延。 |
| `ana_calib_mem_us` | 校准 analytical model 中访存主项对应的时延。 |
| `ana_calib_compute_us` | 校准 analytical model 中计算主项对应的时延。 |
| `ana_calib_overhead_us` | dispatch、启动、微核等结构性开销。 |
| `ana_calib_family` | 使用的 analytical family 名称。 |
| `op_class` | 三分类标签：`memory_pure` / `mixed_balanced` / `compute_dominant`。 |

## 运行

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated/run_pipeline.py
```

如果只是想快速产出 `ana_calib_*`，不跑很慢的 held-out 泛化验证，可以直接用：

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated/run_pipeline.py \
  --skip-generalization \
  --passes 1
```

如果只想做较轻量的泛化检查，也可以只跑 case 级：

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated/run_pipeline.py \
  --schemes leave_one_case_out \
  --passes 1
```

默认输出目录：

```text
/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/analytical_calibrated/
├── analytical_features_full.csv
├── full_data_parameters.json
├── manifest.json
├── heavy_fold_metrics.csv
├── light_proxy_fold_metrics.csv
├── fold_parameters.csv
├── generalization_summary.json
├── generalization_summary.md
└── pipeline_summary.json
```

## 相关性分析

如果需要排查 analytical model 为什么误差偏大，可以单独运行：

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated/analyze_correlation.py
```

默认会在下面目录生成按特征、按 `op_type`、按 `op_class`、按 `ana_calib_family` 的误差相关性汇总：

```text
/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/analytical_calibrated/correlation_analysis/
├── raw_feature_target_correlation.csv
├── raw_feature_abs_error_correlation.csv
├── raw_feature_ape_correlation.csv
├── raw_feature_signed_relative_error_correlation.csv
├── analytical_component_target_correlation.csv
├── analytical_component_abs_error_correlation.csv
├── analytical_component_ape_correlation.csv
├── analytical_component_signed_relative_error_correlation.csv
├── op_type_error_summary.csv
├── op_class_error_summary.csv
├── analytical_family_error_summary.csv
├── correlation_summary.json
└── correlation_summary.md
```

说明：

- `raw_feature_*` 只分析原始输入特征和误差之间的相关性，不混入 `target_us/pred_us` 这类派生列。
- `analytical_component_*` 用来观察 `ana_calib_mem_us`、`ana_calib_compute_us`、`ana_calib_overhead_us` 和误差的关系。
- `correlation_summary.md` 会把最关键的相关性和最差 `op_type/family` 直接汇总出来，便于快速定位问题。

## 说明

- heavy family 的参数语义保持与 [ANALYTICAL_MODEL_V3_CALIBRATED_VS_PURE.md](/data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V3_CALIBRATED_VS_PURE.md) 一致。
- `Gather` 相关 analytical 特征现在优先使用节点自身 `indices` tensor shape 推导出的真实 `request_rows`；只有在 shape 缺失时才回退到旧的全局配置近似。
- 这里导出的 `ana_calib_*` 才是三分类 MLP 使用的 analytical proxy。
