# Single-Op Stage-1 MLP

这个目录是一个独立的小流水线，直接从 `ORT/features_extensible_case_*/*.csv` 和对应的 `ORT/sweep_runs_extensible_case_*/op_shapes/*.csv` 提取全量 case 数据，不依赖其他目录下的脚本。

功能包括：

- 抽取全部 16 个 case 的单算子样本
- 按 stage-1 风格恢复特征列
- 生成完整的 `dataset_full.csv`
- 按 `sample_group=combo` 做 7:2:1 的 train/val/test 切分
- 使用 PyTorch MLP 回归模型训练单算子耗时 `label_operator_actual_dur_us`
- 导出 ONNX，并支持通过 ONNX Runtime 部署到 NPU 推理加速
- 支持手动挑选 case，便于只跑指定 case 子集
- 默认剔除每个 profile 的第一个 batch，并按剩余 batch 的波动过滤不稳定单算子样本

## 特征口径

这里采用的是 E2E/competition 里 stage-1 baseline 的输入风格：

- 类别特征

| 特征 | 含义 |
| --- | --- |
| `op_type` | ONNX 算子类型，例如 `Gemm`、`Gather`、`ReduceSum`。 |
| `node_scope` | 节点名最外层 scope，用来粗粒度表示算子属于哪个模块或子图。 |
| `node_name_normalized` | 去掉 trace 后缀后的标准化节点名，用来区分具体算子位置。 |
| `arch_embedding_size` | DLRM 的 embedding size 配置。 |
| `arch_mlp_bot` | bottom MLP 结构配置字符串。 |
| `arch_mlp_top` | top MLP 结构配置字符串。 |

- 数值特征

| 特征 | 含义 |
| --- | --- |
| `batch_size` | 当前样本对应的 batch size。 |
| `num_indices_per_lookup` | 每次 embedding lookup 使用的 index 个数。 |
| `num_threads` | 该算子执行时使用到的线程数。 |
| `output_size` | 算子输出张量总字节数。 |
| `activation_size` | 算子 activation 相关内存规模，通常可理解为输入/中间激活占用的总字节数。 |
| `parameter_size` | 算子参数或权重占用的总字节数。 |
| `load_store_ratio` | load 指令与 store 指令的比例，用来描述访存读写倾向。 |
| `feat_io_bytes_sum` | `output_size + activation_size + parameter_size`，表示整体 I/O/工作集规模。 |
| `feat_output_input_bytes_ratio` | 输出规模与输入规模的比值，近似描述该算子的“扩张/压缩”程度。 |
| `feat_memops_per_inst` | 每条指令对应的平均内存访问次数，约等于 `(loads + stores) / instructions`。 |
| `feat_insts_per_thread` | 平均每个线程承担的指令数，约等于 `instructions / threads`。 |
| `feat_lookup_count` | `batch_size * num_indices_per_lookup`，只对 `Gather` 类算子非零。 |
| `feat_output_elements_per_lookup` | 平均每次 lookup 产生多少输出元素，只对 `Gather` 类算子非零。 |
| `feat_output_elements_per_batch` | 平均每个 batch 样本对应的输出元素数。 |
| `feat_activation_elements_per_batch` | 平均每个 batch 样本对应的 activation 元素数。 |
| `feat_reduction_axes_count` | `ReduceSum` 等归约算子一共归约了多少个轴。 |
| `feat_reduction_axes_product` | 被归约维度尺寸的乘积，反映归约规模。 |
| `feat_reduction_input_rank` | 归约前输入张量的 rank。 |
| `feat_reduction_output_rank` | 归约后输出张量的 rank。 |
| `feat_reduction_work_items` | 估算的归约工作量，近似表示需要被合并的元素规模。 |
| `reuse_time_mean` | 平均复用时间，反映同一数据再次被访问前隔了多久。 |
| `reuse_distance_mean` | 平均复用距离，反映两次复用之间跨过了多少访问。 |
| `reuse_distance_unique_cache_lines_per_k_accesses` | 每千次访问涉及的唯一 cache line 数量，反映局部性强弱。 |
| `opc_branch_ratio` | branch 类指令占比。 |
| `opc_fp_math_ratio` | 浮点数学指令占比。 |
| `opc_load_ratio` | load 类指令占比。 |
| `opc_math_ratio` | 数学运算类指令占比。 |
| `opc_simd_ratio` | SIMD/向量指令占比。 |
| `opc_store_ratio` | store 类指令占比。 |

目标列：

| 列名 | 含义 |
| --- | --- |
| `label_operator_actual_dur_us` | 单算子的真实耗时标签，单位是微秒。 |

补充说明：

- `output_size`、`activation_size`、`parameter_size` 当前都按字节理解。
- `feat_*elements*` 这类特征是用字节数除以 `4` 近似得到元素个数，因此默认按 `float32` 大小估算。
- `feat_reduction_*` 只对 `ReduceSum` 类算子有意义，其他算子会补 `0`。
- `dataset_full.csv` 里还会保留一些 metadata，例如 `row_uid`、`case_id`、`combo`、`op_idx`、`input_type_shape`、`output_type_shape`，这些列主要用于追溯样本，不参与训练。
- 当前数值特征列表已经按 [numeric_feature_target_correlation.csv](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/correlation_analysis/numeric_feature_target_correlation.csv) 做过一次裁剪，去掉了和目标绝对相关性小于 `0.1` 的数值列。
- 标签默认不是 3 个 batch 的原始均值，而是先丢掉最早的 profile batch，再对剩余 batch 的单算子 `dur_us` 取均值。
- 样本波动过滤默认使用 `last2_range_ratio = abs(batch2 - batch3) / mean(batch2, batch3)`。
- 当前推荐阈值是 `0.20`。这个值和 E2E 稳定性审计里常见的 `cv > 0.20` 不稳定口径相近，而且在当前全 case 数据上，去掉第一个 batch 后大约会过滤掉 `17%` 的单算子样本。

## 一键运行

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/run_pipeline.py
```

默认输出：

```text
/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/
├── dataset/
│   ├── dataset_full.csv
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── feature_columns.json
│   └── dataset_summary.json
└── model/
    ├── mlp_model.pt
    ├── mlp_model.onnx
    ├── preprocessor_state.json
    ├── metrics.json
    ├── predictions_train.csv
    ├── predictions_val.csv
    └── predictions_test.csv
```

依赖说明：

- 训练需要安装 `torch`。
- 如果要在 Ascend NPU 上训练，还需要安装 `torch_npu`；训练脚本会像 `dlrm_msprof.py` 一样显式调用 `torch_npu.npu.set_compile_mode(jit_compile=False)` 并设置 NPU device。
- ONNX/NPU 推理需要安装 `onnxruntime`，如果要跑 NPU，还需要对应 provider 版本。
- 当前仓库里与 Ascend NPU 对齐的是 `CANNExecutionProvider`；如果环境里只有 `OpenVINOExecutionProvider`，脚本也会优先尝试它。

## 单独构建数据

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py \
  --output-dir /tmp/single_op_stage1_dataset
```

默认会：

- 丢掉每个 combo 的最早 profile batch
- 用剩余 batch 的均值作为 `label_operator_actual_dur_us`
- 对 `last2_range_ratio > 0.20` 的样本做剔除

手动指定 case：

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py \
  --output-dir /tmp/single_op_stage1_dataset_selected \
  --selected-cases case_1_1_1 case_2_4_4
```

也可以从文件读取：

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py \
  --output-dir /tmp/single_op_stage1_dataset_selected \
  --selected-cases-file /tmp/cases.txt
```

`/tmp/cases.txt` 可以是逗号/空格/换行分隔的文本，也可以是 JSON：

```json
["case_1_1_1", "case_2_4_4"]
```

调整波动过滤阈值：

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py \
  --output-dir /tmp/single_op_stage1_dataset \
  --profile-instability-threshold 0.25
```

切换成 `cv` 口径，或暂时关闭波动过滤：

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py \
  --output-dir /tmp/single_op_stage1_dataset \
  --profile-instability-metric last2_cv
```

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py \
  --output-dir /tmp/single_op_stage1_dataset \
  --disable-profile-stability-filter
```

如果你后面想保留第一个 batch 做实验，也可以显式关闭这步：

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py \
  --output-dir /tmp/single_op_stage1_dataset \
  --drop-first-profile-batch false
```

## 单独训练 MLP

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/train_mlp.py \
  --data-dir /tmp/single_op_stage1_dataset \
  --output-dir /tmp/single_op_stage1_model
```

常用可选参数：

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/train_mlp.py \
  --data-dir /tmp/single_op_stage1_dataset \
  --output-dir /tmp/single_op_stage1_model \
  --train-device npu \
  --npu-device-id 0 \
  --hidden-layers 256,128,64 \
  --max-iter 200 \
  --early-stopping-patience 20
```

说明：

- `--train-device` 支持 `auto`、`cpu`、`cuda`、`npu`。如果选择 `npu`，脚本会显式导入 `torch_npu`，并按 Ascend 口径初始化设备。
- `--npu-device-id` 用于指定训练时使用的 Ascend 卡号。
- 默认会额外导出 `mlp_model.onnx`。如果你只想训练 PyTorch 模型，不导出 ONNX，可以加 `--disable-onnx-export`。
- 训练前会把类别特征做 one-hot、数值特征做 median impute + standardize，并把这套状态保存到 `preprocessor_state.json`，供部署推理复用。

## Smoke 示例

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/run_pipeline.py \
  --case-pattern 'features_extensible_case_1_1_1' \
  --max-files-per-case 4 \
  --output-root /tmp/single_op_stage1_smoke
```

## 手动挑选 case 跑全链路

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/run_pipeline.py \
  --selected-cases case_1_1_1 case_2_4_4 \
  --max-files-per-case 4 \
  --profile-instability-threshold 0.20 \
  --output-root /tmp/single_op_stage1_selected
```

## ONNX / NPU 推理

导出后的模型可以用下面这个脚本做部署推理：

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/infer_mlp_onnx.py \
  --model-dir /tmp/single_op_stage1_model \
  --input-csv /tmp/single_op_stage1_dataset/test.csv \
  --output-csv /tmp/single_op_stage1_model/test_predictions_onnx.csv \
  --provider auto \
  --metrics-json /tmp/single_op_stage1_model/test_predictions_onnx.metrics.json
```

provider 选择规则：

- `--provider auto`：优先 `CANNExecutionProvider`，其次 `OpenVINOExecutionProvider`，最后回退 `CPUExecutionProvider`。
- `--provider cann`：强制要求 CANN NPU provider。
- `--provider openvino`：强制要求 OpenVINO NPU provider。
- `--provider cpu`：只在 CPU 上跑。

如果是 Ascend NPU 环境，可以显式指定：

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/infer_mlp_onnx.py \
  --model-dir /tmp/single_op_stage1_model \
  --input-csv /tmp/single_op_stage1_dataset/test.csv \
  --output-csv /tmp/single_op_stage1_model/test_predictions_cann.csv \
  --provider cann \
  --device-id 0
```

补充说明：

- ONNX 图里只包含 PyTorch MLP 本体；表格特征的数值化和 one-hot 预处理仍然在 Python 侧完成，并和训练阶段严格使用同一份 `preprocessor_state.json`。
- 如果训练时开启了对目标做 `log1p`，推理脚本会自动对 ONNX 输出做 `expm1` 还原成微秒。
- `infer_mlp_onnx.py` 这条部署链路走的是 ONNX Runtime provider，不会调用 `torch_npu`；`torch_npu` 只用于 PyTorch 训练侧。
