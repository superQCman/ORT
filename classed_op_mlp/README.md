# Classed Op MLP

这个目录实现三分类单算子 MLP 流水线，并支持两条可并行维护的特征分支：

- `with_analytical`
  - 保留 `ana_calib_*` analytical proxy
  - 对应原始三分类方案
- `no_analytical`
  - 完全移除 `ana_calib_*`
  - 只使用 `dataset_all_no_trace` 里的原始软件/形状特征与派生 `feat_gemm_*`
  - 目的就是排除 analytical model 的影响，直接观察“纯分类 MLP”是否比单 MLP baseline 更好
  - 其中 `memory_pure` 会再细拆为 `gather` / `layout_move` / `view_meta`

输入固定为：

- [dataset_all_no_trace](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/dataset_all_no_trace)

`with_analytical` 分支依赖的 analytical proxy 来自：

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

`with_analytical` 保留：

| 特征 | 含义 |
| --- | --- |
| `op_type` | ONNX 算子类型，用于静态路由和类内建模。 |
| `arch_embedding_size` | DLRM embedding 维度配置。 |
| `arch_mlp_bot` | bottom MLP 结构配置。 |
| `arch_mlp_top` | top MLP 结构配置。 |

`with_analytical` 仍然不使用：

- `node_scope`
- `node_name_normalized`

`no_analytical` 在上面 4 个共享类别特征的基础上，单独回加这 2 个高基数类别特征做受控对照：

| 特征 | 含义 |
| --- | --- |
| `node_scope` | 节点所属的图内作用域或模块上下文，近似反映节点位置和调用背景。 |
| `node_name_normalized` | 归一化后的节点名，近似反映节点身份和固定 kernel/dispatch 模式。 |

这次回加只针对 `no_analytical` 分支，目的就是单独观察这两个节点身份特征能否把 `gather / layout_move / view_meta` 拉回到更接近 baseline 的误差水平。

## 数值特征定义

### `with_analytical`

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

### `no_analytical`

这个分支只从 [dataset_all_no_trace/feature_columns.json](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/dataset_all_no_trace/feature_columns.json) 里吸收更贴近机理的原始特征，不使用：

- `hw_ratio_*`
- `local_ctx_*`
- `comp_feat_*`
- `ana_cache_fit_level`
- `ana_expected_latency_ns`

但会额外保留：

- `node_scope`
- `node_name_normalized`

这个分支的训练组是：

- `gather`
  - `Gather`
- `layout_move`
  - `Concat`
  - `Transpose`
- `view_meta`
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

`gather`：

| 特征 | 含义 |
| --- | --- |
| `batch_size` | DLRM batch size。 |
| `num_indices_per_lookup` | 每次 lookup 的索引数配置。 |
| `num_threads` | intra-op 线程数。 |
| `output_size` | 输出张量字节量。 |
| `activation_size` | 输入激活张量字节量。 |
| `parameter_size` | 参数张量字节量。 |
| `feat_io_bytes_sum` | 总 I/O 字节量。 |
| `feat_output_input_bytes_ratio` | 输出相对输入的比例。 |
| `feat_lookup_count` | lookup 次数。 |
| `feat_output_elements_per_lookup` | 每次 lookup 的输出元素规模。 |
| `feat_output_elements_per_batch` | 每个 batch 的输出元素规模。 |

`layout_move`：

| 特征 | 含义 |
| --- | --- |
| `batch_size` | DLRM batch size。 |
| `num_threads` | intra-op 线程数。 |
| `output_size` | 输出张量字节量。 |
| `activation_size` | 输入激活张量字节量。 |
| `feat_io_bytes_sum` | 总 I/O 字节量。 |
| `feat_output_input_bytes_ratio` | 输出相对输入的比例。 |
| `feat_output_elements_per_batch` | 每个 batch 的输出元素规模。 |

`view_meta`：

| 特征 | 含义 |
| --- | --- |
| `batch_size` | DLRM batch size。 |
| `num_threads` | intra-op 线程数。 |
| `output_size` | 输出张量字节量。 |
| `activation_size` | 输入激活张量字节量。 |
| `feat_output_input_bytes_ratio` | 输出相对输入的比例。 |
| `feat_output_elements_per_batch` | 每个 batch 的输出元素规模。 |

`mixed_balanced`：

| 特征 | 含义 |
| --- | --- |
| `batch_size` | DLRM batch size。 |
| `num_threads` | intra-op 线程数。 |
| `output_size` | 输出张量字节量。 |
| `activation_size` | 输入激活张量字节量。 |
| `feat_io_bytes_sum` | 总 I/O 字节量。 |
| `feat_output_input_bytes_ratio` | 输出相对输入的比例。 |
| `feat_output_elements_per_batch` | 每个 batch 的输出元素规模。 |
| `feat_activation_elements_per_batch` | 每个 batch 的输入激活元素规模。 |
| `feat_reduction_axes_count` | 被归约轴数量。 |
| `feat_reduction_axes_product` | 被归约维度乘积。 |
| `feat_reduction_input_rank` | 归约前张量 rank。 |
| `feat_reduction_output_rank` | 归约后张量 rank。 |
| `feat_reduction_work_items` | 归约核心工作量。 |

`compute_dominant`：

| 特征 | 含义 |
| --- | --- |
| `batch_size` | DLRM batch size。 |
| `num_threads` | intra-op 线程数。 |
| `output_size` | 输出张量字节量。 |
| `activation_size` | 输入激活张量字节量。 |
| `parameter_size` | 参数张量字节量。 |
| `feat_io_bytes_sum` | 总 I/O 字节量。 |
| `feat_output_input_bytes_ratio` | 输出相对输入的比例。 |
| `feat_gemm_m` | Gemm/MatMul 的 `M` 维。 |
| `feat_gemm_n` | Gemm/MatMul 的 `N` 维。 |
| `feat_gemm_k` | Gemm/MatMul 的 `K` 维。 |
| `feat_gemm_mac_count` | 总 MAC 数。 |
| `feat_gemm_bytes_per_mac` | 每 MAC 字节开销。 |

## 运行

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/run_pipeline.py
```

如果要跑“排除 Analytical 影响”的分类版，推荐直接用：

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/run_pipeline.py \
  --feature-branch no_analytical
```

这个模式下：

- 输入仍然固定对齐到 [dataset_all_no_trace](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/dataset_all_no_trace)
- 输出默认落到 [classed_op_mlp/no_analytical](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/classed_op_mlp/no_analytical)
- 会跳过 `analytical_calibrated` 的构建和泛化评估
- 会把原来的 `memory_pure` 再细拆成 `gather / layout_move / view_meta`
- 最终仍然和 [model_all_no_trace](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/model_all_no_trace) 做同口径对比

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
- `no_analytical` 分支不使用 `ana_calib_*`、旧 `ana_*`、`hw_ratio_*`、`local_ctx_*`、`comp_feat_*`。
- `with_analytical` 分支仍然不使用旧主 MLP 的 `hw_ratio_*`、`local_ctx_*`、`comp_feat_*`、旧 `ana_*`。
- 最终对比对象固定为 [model_all_no_trace](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/model_all_no_trace)。
