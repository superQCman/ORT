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

## 说明

- heavy family 的参数语义保持与 [ANALYTICAL_MODEL_V3_CALIBRATED_VS_PURE.md](/data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V3_CALIBRATED_VS_PURE.md) 一致。
- 这里导出的 `ana_calib_*` 才是三分类 MLP 使用的 analytical proxy。
- 旧主流水线里的 `ana_cache_fit_level`、`ana_expected_latency_ns`、`ana_base_us` 仍然保留在原数据里做参考，但不会直接进入新的分类 MLP。
