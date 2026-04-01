# Classed Op MLP

这个目录实现按算子机理分组的单算子 MLP 流水线，并支持两条可并行维护的特征分支：

- `with_analytical`
  - 保留 `ana_calib_*` analytical proxy
  - 现在与 `classed_op_mlp_test` 对齐为同一套 5-way 分桶和同一套非 Analytical 特征
  - 只是在对应训练组上追加 `ana_calib_*`
- `no_analytical`
  - 完全移除 `ana_calib_*`
  - 只使用 `dataset_all_no_trace` 里的原始软件/形状特征与派生 `feat_gemm_*`
  - 目的就是排除 analytical model 的影响，直接观察“纯分类 MLP”是否比单 MLP baseline 更好
  - 分组固定为 `gather / layout_move / view_meta / mixed_balanced / compute_dominant`

输入固定为：

- [dataset_all_no_trace](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/dataset_all_no_trace)

`with_analytical` 分支依赖的 analytical proxy 来自：

- [analytical_calibrated](/data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated)

## 静态分桶

- `gather`
  - `Gather`
- `layout_move`
  - `Transpose`
  - `Concat`
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

未知 `op_type` 默认路由到 `mixed_balanced`。

## 共享类别特征

两个分支现在都保留：

| 特征 | 含义 |
| --- | --- |
| `op_type` | ONNX 算子类型，用于静态路由和类内建模。 |
| `node_scope` | 节点所属的图内作用域或模块上下文，近似反映节点位置和调用背景。 |
| `node_name_normalized` | 归一化后的节点名，近似反映节点身份和固定 kernel/dispatch 模式。 |
| `arch_embedding_size` | DLRM embedding 维度配置。 |
| `arch_mlp_bot` | bottom MLP 结构配置。 |
| `arch_mlp_top` | top MLP 结构配置。 |

## 数值特征定义

说明：

- 下表描述的是各组模型真正用于训练的 `numeric_features`。
- 导出的 grouped dataset CSV 里仍可能保留额外的 `ana_calib_*` 列，供分析、路由和结果检查使用；这些列不等于都会进入该组模型训练。

### `with_analytical`

这个分支与 `classed_op_mlp_test` 保持相同的 5-way 分桶和相同的 raw 特征，但只保留已经通过 proxy 质量验证的 analytical 输入列。

其中 `mixed_balanced` 使用的是一个异构组统一的 `ana_calib_total_us` proxy：

- `ReduceSum` 继续走原有 calibrated reduction 公式
- `Relu / Add / Mul / Sigmoid` 改为各自的 op-aware analytical 子模型
- 因此这个组不再强求统一的 `mem_us / compute_us` 两分法，而是直接用更稳定的总时延 proxy

### `gather`

| 特征 | 含义 |
| --- | --- |
| `batch_size` | DLRM batch size。 |
| `num_indices_per_lookup` | 每次 lookup 的索引数配置。 |
| `num_threads` | intra-op 线程数。 |
| `inter_threads` | inter-op 线程数这种静态并发配置。优先从 `logs/<combo>/build_ops.log` 的 `default_inter_threads` 恢复，缺失时回退到 `case_*_run_*_*.sh` 里的 `INTER_THREADS`。 |
| `output_size` | 输出张量字节量。 |
| `activation_size` | 输入激活张量字节量。 |
| `parameter_size` | 参数张量字节量。 |
| `feat_io_bytes_sum` | 总 I/O 字节量。（output_size + activation_size + parameter_size） |
| `feat_output_input_bytes_ratio`（移除） | 输出相对输入的比例。 |
| `feat_lookup_count` | Gather 的真实请求元素数。优先由节点自身 `indices` tensor shape 推导；只有 shape 缺失时才回退到旧的全局配置近似。 |
| `feat_output_elements_per_lookup`（移除） | 每个真实 Gather request 对应的平均输出元素规模。 |
| `feat_output_elements_per_batch`（移除） | 每个 batch 的输出元素规模。 |
| `ana_calib_total_us`（移除） | 校准 analytical 总时延。 |
| `ana_calib_mem_us` | 校准 analytical 访存主项时延。 |

### `layout_move`

| 特征 | 含义 |
| --- | --- |
| `batch_size` | DLRM batch size。 |
| `num_threads` | intra-op 线程数。 |
| `output_size` | 输出张量字节量。 |
| `activation_size` | 输入激活张量字节量。 |
| `feat_io_bytes_sum` | 总 I/O 字节量。（output_size + activation_size + parameter_size） |
| `feat_output_input_bytes_ratio`（移除） | 输出相对输入的比例。 |
| `feat_output_elements_per_batch` | 每个 batch 的输出元素规模。 |
| `ana_calib_total_us` | 校准 analytical 总时延。当前 `layout_move` 组使用它作为统一 proxy，以同时覆盖 `Concat` 的 dispatch/overhead 和 tuned `Transpose`。 |
| `ana_calib_mem_us`（移除） | 校准 analytical 访存主项时延。 |

### `view_meta`

| 特征 | 含义 |
| --- | --- |
| `batch_size` | DLRM batch size。 |
| `num_threads` | intra-op 线程数。 |
| `output_size` | 输出张量字节量。 |
| `activation_size` | 输入激活张量字节量。 |
| `feat_output_input_bytes_ratio`（移除） | 输出相对输入的比例。 |
| `feat_output_elements_per_batch`（移除） | 每个 batch 的输出元素规模。 |
| `ana_calib_total_us`（移除） | 校准 analytical 总时延。 |
| `ana_calib_mem_us`（移除） | 校准 analytical 访存主项时延。 |

### `mixed_balanced`

| 特征 | 含义 |
| --- | --- |
| `batch_size` | DLRM batch size。 |
| `num_threads` | intra-op 线程数。 |
| `inter_threads` | inter-op 线程数这种静态并发配置。优先从 `logs/<combo>/build_ops.log` 的 `default_inter_threads` 恢复，缺失时回退到 `case_*_run_*_*.sh` 里的 `INTER_THREADS`。 |
| `output_size` | 输出张量字节量。 |
| `activation_size` | 输入激活张量字节量。 |
| `feat_io_bytes_sum` | 总 I/O 字节量。 |
| `feat_output_elements_per_batch` | 每个 batch 的输出元素规模。 |
| `feat_output_input_bytes_ratio` | 输出相对输入的比例。 |
| `feat_activation_elements_per_batch`（移除） | 每个 batch 的输入激活元素规模。 |
| `feat_reduction_axes_count` | 被归约轴数量。 |
| `feat_reduction_work_items` | 归约/聚合类核心工作量。 |
| `feat_reduction_axes_product` | 被归约维度乘积。 |
| `feat_reduction_input_rank` | 归约前张量 rank。 |
| `feat_reduction_output_rank` | 归约后张量 rank。 |
| `ana_calib_total_us` | 校准 analytical 总时延。当前 mixed_balanced 组使用它作为统一 proxy，以同时覆盖 `ReduceSum` 和 `Relu / Add / Mul / Sigmoid` 的 op-aware 子模型。 |
| `ana_calib_mem_us`（移除） | mixed_balanced 是异构组；虽然导出 CSV 仍保留该列做分析，但它不再作为统一训练输入。 |
| `ana_calib_compute_us`（移除） | mixed_balanced 是异构组；虽然导出 CSV 仍保留该列做分析，但它不再作为统一训练输入。 |

### `compute_dominant`

| 特征 | 含义 |
| --- | --- |
| `batch_size` | DLRM batch size。 |
| `num_threads` | intra-op 线程数。 |
| `output_size` | 输出张量字节量。 |
| `activation_size` | 输入激活张量字节量。 |
| `parameter_size` | 参数张量字节量。 |
| `feat_io_bytes_sum` | 总 I/O 字节量。（output_size + activation_size + parameter_size） |
| `feat_output_input_bytes_ratio` （移除）| 输出相对输入的比例。 |
| `feat_gemm_m` | Gemm/MatMul 的 `M` 维。 |
| `feat_gemm_n` | Gemm/MatMul 的 `N` 维。 |
| `feat_gemm_k` | Gemm/MatMul 的 `K` 维。 |
| `feat_gemm_mac_count` | 矩阵乘总 MAC 数。 |
| `feat_gemm_bytes_per_mac` | 每 MAC 对应的字节开销。 |
| `ana_calib_total_us`（移除） | 校准 analytical 总时延。 |
| `ana_calib_compute_us` | 校准 analytical 计算主项时延。 |

### `no_analytical`

这个分支只从 [dataset_all_no_trace/feature_columns.json](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/dataset_all_no_trace/feature_columns.json) 里吸收更贴近机理的原始特征，不使用：

- `hw_ratio_*`
- `local_ctx_*`
- `comp_feat_*`
- `ana_cache_fit_level`
- `ana_expected_latency_ns`

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
| `inter_threads` | inter-op 线程数这种静态并发配置。优先从 `logs/<combo>/build_ops.log` 的 `default_inter_threads` 恢复，缺失时回退到 `case_*_run_*_*.sh` 里的 `INTER_THREADS`。 |
| `output_size` | 输出张量字节量。 |
| `activation_size` | 输入激活张量字节量。 |
| `parameter_size` | 参数张量字节量。 |
| `feat_io_bytes_sum` | 总 I/O 字节量。 |
| `feat_output_input_bytes_ratio` | 输出相对输入的比例。 |
| `feat_lookup_count` | Gather 的真实请求元素数。优先由节点自身 `indices` tensor shape 推导；只有 shape 缺失时才回退到旧的全局配置近似。 |
| `feat_output_elements_per_lookup` | 每个真实 Gather request 对应的平均输出元素规模。 |
| `feat_output_elements_per_batch` | 每个 batch 的输出元素规模。 |

`layout_move`：

| 特征 | 含义 |
| --- | --- |
| `batch_size` | DLRM batch size。 |
| `num_threads` | intra-op 线程数。 |
| `inter_threads` | inter-op 线程数这种静态并发配置。优先从 `logs/<combo>/build_ops.log` 的 `default_inter_threads` 恢复，缺失时回退到 `case_*_run_*_*.sh` 里的 `INTER_THREADS`。 |
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

如果你现在是想快速验证 `with_analytical` 分支，不想被 analytical generalization 拖慢，推荐直接用：

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/run_pipeline.py \
  --feature-branch with_analytical \
  --skip-analytical-generalization \
  --passes 1
```

如果已经提前产出了 [analytical_calibrated](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/analytical_calibrated) 里的 `analytical_features_full.csv`，可以进一步复用，不再重建：

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/run_pipeline.py \
  --feature-branch with_analytical \
  --analytical-dir /data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/analytical_calibrated \
  --reuse-analytical-features \
  --skip-analytical-generalization
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

`with_analytical` 分支的快速参数：

- `--skip-analytical-generalization`
  - 只构建 `ana_calib_*`，跳过最慢的 held-out generalization
- `--reuse-analytical-features`
  - 直接复用已有 `analytical_features_full.csv`
- `--analytical-schemes leave_one_case_out`
  - 只跑较轻量的 case 级泛化

active analytical input 的单 CSV 验证命令：

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/validate_active_analytical_inputs.py \
  --data-root /data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/classed_op_mlp_test_5_analytical_5_200_iter \
  --output-csv /tmp/classed_active_analytical_validation.csv
```

默认输出目录：

```text
/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/classed_op_mlp/
├── classed_dataset_full.csv
├── dataset_summary.json
├── datasets/
│   ├── gather/
│   ├── layout_move/
│   ├── view_meta/
│   ├── mixed_balanced/
│   └── compute_dominant/
└── models/
    ├── gather/
    ├── layout_move/
    ├── view_meta/
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
