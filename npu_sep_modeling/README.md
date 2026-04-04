# NPU Separation Modeling

这个子项目用于 Ascend 910B3 分离架构下的 NPU 单算子建模。

v1 目标很窄：

- 只处理 `ORT/features_extensible_case_10_4_4_cann` 与对应的 `ORT/sweep_runs_extensible_case_10_4_4_cann/onnx_profiles`
- 只解析 `args.provider == "CANNExecutionProvider"` 的 `Node` 事件
- 把 `MemcpyFromHost` / `MemcpyToHost` 作为独立 `transfer` lane，不并入 `cube` / `vector`
- 先做“解析模型 + 小参数校准”，不做残差学习

## 计划中的产物

- `build_npu_dataset.py`
  从 `case_10_4_4_cann` 构造 `dataset_full.csv`、`train.csv`、`val.csv`、`test.csv`、`dataset_summary.json`、`feature_columns.json`
- `hardware_probe.py`
  读取本机 `npu-smi` / `ascend-dmi` / 微基准信息，导出 `hardware_profile_910b3.json`
- `fit_sep_analytical_model.py`
  在 train split 上拟合解析模型校准参数，导出 `calibration.json` 和 `metrics_summary.json`
- `evaluate_sep_analytical_model.py`
  在 val/test 上比较未校准与已校准的误差

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
