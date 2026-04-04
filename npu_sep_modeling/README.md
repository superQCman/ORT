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

这套 NPU analytical model 先做结构化 roofline baseline，再做物理参数化校准：

1. 先按 `npu_lane` 把算子分成 `cube`、`vector`、`transfer` 三类。
2. 再根据算子类型与 shape/size 特征构造每一类的 roofline 下界。
3. 再把 `cpu_main_Wait_avg` 与 `cpu_main_DistributionEnqueue_avg` 合成 `queue_proxy_us`，作为可观测的 host-side queueing proxy。
4. 最后不再做抽象的 `scale + bias_us`，而是拟合一组有明确物理含义的有效参数。

当前 v2 的预测公式是：

- `cube`:
  - `pred_us = launch_runtime_us[op_name] + queueing_scale * queue_proxy_us + max(compute_us, hbm_mem_us)`
- `vector`:
  - `pred_us = launch_runtime_us[op_name] + queueing_scale * queue_proxy_us + max(compute_us, hbm_mem_us)`
- `transfer`:
  - `pred_us = launch_runtime_us[op_name] + queueing_scale * queue_proxy_us + transfer_us`

其中：

- `queue_proxy_us = (cpu_main_Wait_avg + cpu_main_DistributionEnqueue_avg) / 1000`
- `compute_us` 仍然由 lane 对应的算子规模推导
- `hbm_mem_us` 由数据搬运字节数除以有效带宽得到
- `transfer_us` 由显式 H2D / D2H 带宽得到

v2 不要求每个参数都对应单一硬件寄存器，而要求它们对应到可观测、可解释、可复现实验的有效物理量。

硬件输入参数现在明确保留为：

- `ai_core_count`
- `cube_count`
- `vector_count`
- `cube_peak_eff_gflops`
- `vector_peak_eff_gflops`
- `memory_bw_gbps`
- `h2d_bw_gbps`
- `d2h_bw_gbps`

其中 `cube_count` / `vector_count` 来自 910B3 分离架构的 AI Core 布局，本机 `npu-smi info -t common` 显示 `Aicore Count = 20`，再结合官方文档对 AIC/AIV 的定义，可将其解释为 `20 Cube + 40 Vector` 的输入基线。

- `cube` 基线：
  - `MatMul` 用 `2 * M * K * N / cube_peak_eff_gflops`
  - 再和 `input/output/activation/parameter` 字节搬运时间取 `max`
- `vector` 基线：
  - `Transpose` / `Add` / `Relu` 用 `vector_elem_count / vector_peak_eff_gflops`
  - 再和 `input/output/activation` 字节搬运时间取 `max`
- `transfer` 基线：
  - `MemcpyFromHost` / `MemcpyToHost` 用 `bytes / h2d_or_d2h_bw`

更具体地说，v2 把参数分成下面几类：

| 参数 | 物理含义 | 当前状态 |
| --- | --- | --- |
| `launch_runtime_us[op_name]` | 单个算子类型的固定启动与框架开销 | 已拟合 |
| `queueing_scale` | host-side wait / enqueue proxy 对设备可见排队时间的缩放系数 | 已拟合 |
| `cube_memory_bw_gbps` | `cube` lane 的有效 HBM / memory 带宽 | 当前数据上通常不可辨识，若无增益则并入 `launch_runtime_us` |
| `vector_memory_bw_gbps` | `vector` lane 的有效 memory 带宽 | 已拟合 |
| `transfer_h2d_bw_gbps` | Host->Device 显式拷贝带宽 | 已拟合 |
| `transfer_d2h_bw_gbps` | Device->Host 显式拷贝带宽 | 已拟合 |

对当前数据来说，`cube` 的 memory 项往往不如 `launch/runtime` 那么可辨识，因此脚本会自动尝试两种模型：

- `full_physical`：显式拟合 `cube_memory_bw_gbps`
- `reduced_physical`：把 `cube_memory_us` 合并进 `launch_runtime_us`

如果你只看最终导出的 `calibration.json`，就能直接看到这次选择的是哪一种。

如果某个带宽参数被拟合到远高于 `IDENTIFIABILITY_BW_THRESHOLD_GBPS` 的量级，脚本会把对应项标记为 `merged_terms`，这表示它在当前数据上已经退化成“几乎不贡献时延”的合并项，而不是一个可独立解释的硬件参数。

校准策略仍然是“小参数”，但现在参数的语义更物理：

- 先计算无参 roofline baseline
- 再按 `op_name` 拟合固定的 `launch_runtime_us`
- 再拟合少量有效带宽参数和 `queueing_scale`
- 评估时只比较 baseline 和物理校准结果

## 泛化性说明

这套校准**可以**只用少量训练样本来拟合，而且更推荐这么做，但前提是要把“拟合”和“验证”分开。

当前实现里，`fit_sep_analytical_model.py` 支持 `--calibration-fit-fraction < 1.0`，它会：

- 按 `op_name` 做分层抽样
- 只用抽到的那部分训练样本拟合物理参数
- 把没参与拟合的训练样本留作内部 holdout
- 在 `metrics_summary.json` 和评估报告里显式报告 holdout 误差

这里不能给出“对所有未来数据都绝对成立”的无条件证明，但可以给出一个严格的条件性结论：

> 对每个校准桶 `g`，我们拟合的是一个受非负约束的低维物理模型，其中参数个数远少于样本数。
> 如果同一桶内的样本可视为独立同分布，且 `baseline_pred_us` 与真实标签都被限制在有界区间内，那么标准的线性回归/经验风险最小化泛化理论保证：
> 经验风险与期望风险之间的差距会随样本数 `n_g` 以 `O(1/sqrt(n_g))` 的速度收敛。
> 也就是说，校准参数越少、每个桶里的样本越多，泛化上界就越紧。这里的“少”指 `launch_runtime_us + queueing_scale + 少量 bandwidth`，而不是高容量黑盒。

这条结论的实际含义是：

- 我们没有引入高容量黑盒模型
- 每个桶只学 2 个参数
- 只要每个桶保留足够的拟合样本，泛化风险就可以用持出集直接检验，而不是靠主观感觉

更稳妥的使用方式是：

1. 用少量、但分层覆盖的 train 样本拟合校准
2. 用 train holdout 验证泛化
3. 再看 val/test 的最终误差

## 硬件探测

- `hardware_probe.py` 先读 `npu-smi info`、`npu-smi info -t board` 和 `npu-smi info -t common`
- `board_name` 不再作为输出字段保留，避免把板级字符串混进模型输入
- `cube_count` 和 `vector_count` 会写入硬件 profile，当前按 910B3 分离架构固定为 `20` 与 `40`
- 当前环境里没有 `ascend-dmi`
- 当前环境里也没有可直接跑这套 microbench 的 `onnx` / `onnxruntime` 依赖，所以峰值字段会先以 `null` 记录，`fit_sep_analytical_model.py` 会回退到内置默认值
- 当前数据里 `cpu_main_Wait_avg` / `cpu_main_DistributionEnqueue_avg` 只对一部分 `transfer` 节点可见，所以 `queueing_us` 目前更像一个 host-side queueing proxy，而不是严格拆出来的 device queue depth

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
  --hardware-profile /data/qc/dlrm/ORT/npu_sep_modeling/hardware_profile_910b3.json \
  --calibration-fit-fraction 0.3 \
  --calibration-seed 42 \
  --output-dir /tmp/npu_sep_modeling_fit

python3 /data/qc/dlrm/ORT/npu_sep_modeling/evaluate_sep_analytical_model.py \
  --data-dir /tmp/npu_sep_modeling_dataset \
  --hardware-profile /data/qc/dlrm/ORT/npu_sep_modeling/hardware_profile_910b3.json \
  --calibration /tmp/npu_sep_modeling_fit/calibration.json \
  --output-dir /tmp/npu_sep_modeling_eval
```

- `--calibration` 要填的是前一步 `fit_sep_analytical_model.py` 生成的 `calibration.json` 路径，它保存了这次训练出来的物理参数集合。
- 如果你用的是项目里这份 `hardware_profile_910b3.json`，那就表示评估时采用这份固定的 910B3 有效硬件输入，而不是重新做一次硬件探测。

## 备注

- 这个子项目保持自包含，不依赖其他目录的核心流水线脚本。
- 当前目录树里可解析的 combo 数会由脚本按真实 profile 文件自动统计，并写入 `dataset_summary.json`。
