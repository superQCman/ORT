# NPU Separation Modeling

这个子项目用于 Ascend 910B3 分离架构下的 NPU 单算子建模。

v1 目标很窄：

- 只处理 `ORT/features_extensible_case_10_4_4_cann` 与对应的 `ORT/sweep_runs_extensible_case_10_4_4_cann/onnx_profiles`
- 只解析 `args.provider == "CANNExecutionProvider"` 的 `Node` 事件
- 把 `MemcpyFromHost` / `MemcpyToHost` 作为独立 `transfer` lane，不并入 `cube` / `vector`
- 先做“解析模型 + 小参数校准”，不做残差学习

## 关键说明

- 这里的 `op_name` 指 CANN provider 侧的算子类型，例如 `MatMul`、`Add`、`Relu`、`Transpose`、`MemcpyFromHost`、`MemcpyToHost`
- 这里的 `op_type` 保留的是 source catalog CSV 里的原始 ONNX 节点类型，用于回溯和补充元数据
- 当前目录树会自动解析真实存在的 profile combo，不硬编码 24 个 combo；以当前数据来看，脚本会统计出 30 个 combo 和 360 条聚合行

## 计划中的产物

- `build_npu_dataset.py`
  从 `case_10_4_4_cann` 构造 `dataset_full.csv`、`train.csv`、`val.csv`、`test.csv`、`dataset_summary.json`、`feature_columns.json`
- `hardware_probe.py`
  读取本机 `npu-smi` / `ascend-dmi` / 微基准信息，导出 `hardware_profile_910b3.json`
- `fit_sep_analytical_model.py`
  在 train split 上拟合解析模型校准参数，导出 `calibration.json` 和 `metrics_summary.json`
- `evaluate_sep_analytical_model.py`
  在 val/test 上比较未校准与已校准的误差

生成结果会落在你指定的 `--output-dir` 下，常见文件包括：

- `dataset_full.csv`
- `train.csv`
- `val.csv`
- `test.csv`
- `dataset_summary.json`
- `feature_columns.json`
- `hardware_profile_910b3.json`
- `calibration.json`
- `metrics_summary.json`
- `comparison_report.md`
- `comparison_summary.json`

## 数据口径

- 真值源：`ORT/sweep_runs_extensible_case_10_4_4_cann/onnx_profiles/bs*_nip*/ort_cann_profile_*.json`
- 只保留 `CANNExecutionProvider` 的 `Node` 事件
- 聚合键：`(combo, node_index, op_name, node_name)`
- 默认丢弃每个节点的第一次调用，再对剩余调用取 `mean/min/max/std/count`
- 标签列：`label_npu_dur_us = mean(after_drop_first)`
- 行级 `npu_lane`：
  - `cube` = `MatMul`
  - `vector` = `Transpose` / `Add` / `Relu`
  - `transfer` = `MemcpyFromHost` / `MemcpyToHost`

## 建模口径

- `cube` 基线：
  - `MatMul` 用 `2 * M * K * N / cube_peak_eff_gflops`
  - 再和 `input/output/activation/parameter` 字节搬运时间取 `max`
- `vector` 基线：
  - `Transpose` / `Add` / `Relu` 用 `vector_elem_count / vector_peak_eff_gflops`
  - 再和 `input/output/activation` 字节搬运时间取 `max`
- `transfer` 基线：
  - `MemcpyFromHost` / `MemcpyToHost` 用 `bytes / h2d_or_d2h_bw`
- 校准策略：
  - 先算无参 baseline
  - 再按 `op_name` 和 `npu_lane` 拟合少量 `scale + bias_us`
  - 评估时比较 baseline、`op_name` 校准、`lane` 校准和 `global` 校准

## 硬件探测

- `hardware_probe.py` 先读 `npu-smi info` 和 `npu-smi info -t common`
- 当前环境里没有 `ascend-dmi`
- 当前环境里也没有可直接跑这套 microbench 的 `onnx` / `onnxruntime` 依赖，所以峰值字段会先以 `null` 记录，`fit_sep_analytical_model.py` 会回退到内置默认值

## 运行入口

后续脚本会固定支持以下入口风格：

```bash
python3 /data/qc/dlrm/ORT/npu_sep_modeling/build_npu_dataset.py \
  --case-id case_10_4_4_cann \
  --output-dir /tmp/npu_sep_modeling_dataset \
  --drop-first-call true

python3 /data/qc/dlrm/ORT/npu_sep_modeling/hardware_probe.py \
  --output-dir /tmp/npu_sep_modeling_hw

python3 /data/qc/dlrm/ORT/npu_sep_modeling/fit_sep_analytical_model.py \
  --data-dir /tmp/npu_sep_modeling_dataset \
  --hardware-profile /tmp/npu_sep_modeling_hw/hardware_profile_910b3.json \
  --output-dir /tmp/npu_sep_modeling_fit

python3 /data/qc/dlrm/ORT/npu_sep_modeling/evaluate_sep_analytical_model.py \
  --data-dir /tmp/npu_sep_modeling_dataset \
  --hardware-profile /tmp/npu_sep_modeling_hw/hardware_profile_910b3.json \
  --calibration /tmp/npu_sep_modeling_fit/calibration.json \
  --output-dir /tmp/npu_sep_modeling_eval
```

## 备注

- 这个子项目保持自包含，不依赖其他目录的核心流水线脚本。
- 当前目录树里可解析的 combo 数会由脚本按真实 profile 文件自动统计，并写入 `dataset_summary.json`。
