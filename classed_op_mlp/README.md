# Classed Op MLP

这个目录实现三分类单算子 MLP 流水线。

输入固定为：

- [dataset_all_no_trace](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/dataset_all_no_trace)

依赖的 analytical proxy 来自：

- [analytical_calibrated](/data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated)

## 静态分桶

- `memory_pure`
  - `Gather`
  - `Transpose`
  - `Concat`
  - `Reshape`
  - `Shape`
  - `Unsqueeze`
  - `Flatten`
- `mixed_balanced`
  - `ReduceSum`
  - `Sigmoid`
  - `Relu`
  - `Add`
  - `Mul`
- `compute_dominant`
  - `Gemm`
  - `MatMul`

未知 `op_type` 默认路由到 `mixed_balanced`。

## 共享类别特征

| 特征 | 含义 |
| --- | --- |
| `op_type` | ONNX 算子类型，用于静态路由和类内建模。 |
| `arch_embedding_size` | DLRM embedding 维度配置。 |
| `arch_mlp_bot` | bottom MLP 结构配置。 |
| `arch_mlp_top` | top MLP 结构配置。 |

不使用：

- `node_scope`
- `node_name_normalized`

## 数值特征定义

### `memory_pure`

| 特征 | 含义 |
| --- | --- |
| `num_threads` | intra-op 线程数。 |
| `feat_io_bytes_sum` | 总 I/O 字节量，近似总访存工作集。 |
| `feat_output_input_bytes_ratio` | 输出相对输入的膨胀/压缩比例。 |
| `feat_lookup_count` | Gather 的 lookup 次数。 |
| `feat_output_elements_per_lookup` | 每次 Gather lookup 产生的元素规模。 |
| `ana_calib_total_us` | 校准 analytical 总时延。 |
| `ana_calib_mem_us` | 校准 analytical 访存主项时延。 |

### `mixed_balanced`

| 特征 | 含义 |
| --- | --- |
| `num_threads` | intra-op 线程数。 |
| `feat_io_bytes_sum` | 总 I/O 字节量。 |
| `feat_output_elements_per_batch` | 每个 batch 的输出元素规模。 |
| `feat_output_input_bytes_ratio` | 输出相对输入的比例。 |
| `feat_reduction_work_items` | 归约/聚合类核心工作量。 |
| `feat_reduction_axes_product` | 被归约维度乘积。 |
| `ana_calib_total_us` | 校准 analytical 总时延。 |
| `ana_calib_mem_us` | 校准 analytical 访存项时延。 |
| `ana_calib_compute_us` | 校准 analytical 计算项时延。 |

### `compute_dominant`

| 特征 | 含义 |
| --- | --- |
| `num_threads` | intra-op 线程数。 |
| `feat_gemm_m` | Gemm/MatMul 的 `M` 维。 |
| `feat_gemm_n` | Gemm/MatMul 的 `N` 维。 |
| `feat_gemm_k` | Gemm/MatMul 的 `K` 维。 |
| `feat_gemm_mac_count` | 矩阵乘总 MAC 数。 |
| `feat_gemm_bytes_per_mac` | 每 MAC 对应的字节开销。 |
| `ana_calib_total_us` | 校准 analytical 总时延。 |
| `ana_calib_compute_us` | 校准 analytical 计算主项时延。 |

## 运行

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/run_pipeline.py
```

默认输出目录：

```text
/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/classed_op_mlp/
├── classed_dataset_full.csv
├── dataset_summary.json
├── datasets/
│   ├── memory_pure/
│   ├── mixed_balanced/
│   └── compute_dominant/
└── models/
    ├── memory_pure/
    ├── mixed_balanced/
    ├── compute_dominant/
    ├── combined/
    └── comparison/
```

## 约束

- 不使用 trace 特征。
- 不使用旧主 MLP 的 `hw_ratio_*`、`local_ctx_*`、`comp_feat_*`、旧 `ana_*` 作为输入。
- 最终对比对象固定为 [model_all_no_trace](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/model_all_no_trace)。
