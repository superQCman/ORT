# Analytical Model V4: 泛化性测试与训练内校准结果

## 1. 目标

这份文档回答上一个版本里尚未解决的问题：

> 可解释校准版 analytical model 的参数如果只在训练样本上校准，再拿去预测没参与校准的 held-out 样本，它的误差会是多少？

这里特别强调两点：

- 本文所有 `MAPE` 都是 **held-out test fold** 上计算的，不是 in-sample 误差。
- 参数校准只使用训练 fold，不使用对应测试 fold 的任何样本。
- 除 `MAPE` 外，本文还额外报告按真实耗时加权的相对误差：
  `duration-weighted relative error = sum(|pred-actual|) / sum(actual)`。
  这个指标会自然提高长时算子的权重，因此更接近“对整体 wall-time 影响有多大”的视角。

## 2. 可复现实验入口

实验脚本：

- [evaluate_analytical_generalization.py](/data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py)

默认输入：

- [dataset_full.csv](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/dataset_all_no_trace/dataset_full.csv)

本轮产物：

- [summary.md](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/analytical_generalization/summary.md)
- [summary.json](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/analytical_generalization/summary.json)
- [fold_parameters.csv](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/analytical_generalization/fold_parameters.csv)
- [fold_family_metrics.csv](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/analytical_generalization/fold_family_metrics.csv)
- [heavy_op_eval_slice.csv](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/analytical_generalization/heavy_op_eval_slice.csv)

运行命令：

```bash
python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py
```

## 3. 评估切片

评估范围与 V3 保持一致：

- `case_9_4_4`
- `case_10_2_1`
- `case_10_4_4`
- `combo in {bs1024_nip1500, bs1440_nip1700, bs1888_nip1800}`

heavy-op 选择规则：

| family | 规则 | rows |
| --- | --- | ---: |
| `Gather` | `op_type == Gather and output_size > 1e8` | 59 |
| `ReduceSum` | `op_type == ReduceSum and activation_size > 1e8` | 59 |
| `Gemm` | `op_type == Gemm and dur_us > 1000` | 35 |
| `MatMul` | `node_name == /MatMul` | 8 |
| `Transpose` | `node_name == /Transpose` | 8 |
| `Concat` | `node_name == /Concat` | 9 |

总样本数：`178`

## 4. 校准协议

### 4.1 参数形式

脚本校准的是 V3 文档同一套机制，但把 copy-like family 的半饱和写法显式改写成：

`BW_eff = BW_inf * s / (BW_inf * tau_start + s)`

也就是说，不再把 `B50_*` 作为主参数暴露，而是直接校准对应的 `tau_start_*`。脚本当前校准的参数为：

- `rho_copy_inf`
- `tau_copy_start`
- `tau_dispatch`
- `kappa_reduce`
- `tau_reduce_start`
- `rho_gather_inf`
- `tau_gather_row_start`
- `m_gather`
- `rho_fma_inf`
- `M50 / N50 / K50`
- `occ_ref`
- `rho_tiny_inf`
- `K50_tiny`
- `tau_micro`
- `m_stride`
- `eta_stride`

这些参数仍然维持原来的物理语义：

- `rho_*` 表示持续效率比例
- `tau_*` 表示固定启动成本
- `m_*` 表示有效并发深度
- `eta_stride` 表示线程扩展指数

其中：

- `tau_copy_start` 是 copy chunk 进入 steady-state 前的等效固定启动时间
- `tau_reduce_start` 是 reduction 内层流式阶段的固定启动时间
- `tau_gather_row_start` 是每个 gather row 的固定启动时间

如果需要，仍可以通过 `B50 = BW_inf * tau_start` 折算回半饱和尺度，但不再把 `B50` 当作主建模参数。

### 4.2 拟合方式

为了保证校准可复现且不引入黑盒优化器，本轮使用的是**有界坐标下降**：

1. 对每个参数给出固定候选网格
2. 在训练 fold 上，以 MAPE 为目标逐个坐标搜索
3. 共享 copy 参数先按 `Concat + ReduceSum + Transpose` 的 macro MAPE 联合校准
4. 其他参数再按各自 family 的训练 MAPE 单独校准

这不是自由回归器，而是对少量可解释参数做离散校准。

### 4.3 两种 held-out 协议

本轮做了两种泛化测试：

1. `leave-one-case-out`
   每次留出一个 `case_id` 做测试，其余 case 做训练校准。
2. `leave-one-combo-out`
   每次留出一个 `combo` 做测试，其余 combo 做训练校准。

### 4.4 指标说明

本文同时报告三类聚合指标：

- `mean macro MAPE`
  - 先按 family 算 `MAPE`，再做不加权平均
- `row-count-weighted family MAPE`
  - 按 family 样本数对 `MAPE` 加权
- `duration-weighted relative error`
  - 直接按真实 `actual_us` 给每条样本加权
  - 等价于 `sum(|pred-actual|) / sum(actual)`
  - 长时算子的误差会被赋予更高权重

## 5. Held-Out 结果

### 5.1 Leave-One-Case-Out

聚合结果：

- test fold mean macro `MAPE = 20.89%`
- worst-fold macro `MAPE = 22.16%`
- row-count-weighted family `MAPE = 16.70%`
- duration-weighted relative error `= 14.44%`

各 family 在测试 fold 上的平均误差：

| family | mean test MAPE | mean duration-weighted RE | median test MAPE | max fold MAPE |
| --- | ---: | ---: | ---: | ---: |
| `Gather` | `9.82%` | `10.39%` | `7.51%` | `14.47%` |
| `ReduceSum` | `19.51%` | `18.99%` | `19.59%` | `22.92%` |
| `Gemm` | `16.60%` | `16.26%` | `17.60%` | `25.04%` |
| `MatMul` | `28.77%` | `30.01%` | `33.02%` | `42.43%` |
| `Transpose` | `32.31%` | `31.53%` | `39.81%` | `49.10%` |
| `Concat` | `18.35%` | `18.49%` | `11.58%` | `38.89%` |

按 held-out case 看：

| held-out case | macro test MAPE | duration-weighted RE | rows |
| --- | ---: | ---: | ---: |
| `case_10_2_1` | `18.79%` | `18.07%` | 65 |
| `case_10_4_4` | `22.16%` | `10.44%` | 55 |
| `case_9_4_4` | `21.73%` | `14.00%` | 58 |

### 5.2 Leave-One-Combo-Out

聚合结果：

- test fold mean macro `MAPE = 15.79%`
- worst-fold macro `MAPE = 16.80%`
- row-count-weighted family `MAPE = 11.73%`
- duration-weighted relative error `= 10.31%`

各 family 在测试 fold 上的平均误差：

| family | mean test MAPE | mean duration-weighted RE | median test MAPE | max fold MAPE |
| --- | ---: | ---: | ---: | ---: |
| `Gather` | `8.10%` | `8.14%` | `7.45%` | `9.49%` |
| `ReduceSum` | `11.52%` | `12.83%` | `10.81%` | `13.71%` |
| `Gemm` | `11.19%` | `9.93%` | `11.43%` | `16.17%` |
| `MatMul` | `14.96%` | `16.45%` | `12.32%` | `20.70%` |
| `Transpose` | `34.35%` | `35.58%` | `34.66%` | `39.68%` |
| `Concat` | `14.65%` | `21.09%` | `14.96%` | `15.68%` |

按 held-out combo 看：

| held-out combo | macro test MAPE | duration-weighted RE | rows |
| --- | ---: | ---: | ---: |
| `bs1024_nip1500` | `14.91%` | `9.05%` | 54 |
| `bs1440_nip1700` | `15.66%` | `10.25%` | 66 |
| `bs1888_nip1800` | `16.80%` | `10.88%` | 58 |

## 6. 与 In-Sample 结果对比

V3 文档中的 in-sample 结果是：

- family macro `MAPE = 14.99%`
- weighted overall `MAPE = 11.78%`

而这轮真正的 held-out 结果是：

| protocol | mean macro MAPE | row-count-weighted family MAPE | duration-weighted RE |
| --- | ---: | ---: | ---: |
| in-sample prototype | `14.99%` | `11.78%` | `N/A` |
| leave-one-case-out test | `20.89%` | `16.70%` | `14.44%` |
| leave-one-combo-out test | `15.79%` | `11.73%` | `10.31%` |

这个对比说明：

1. `leave-one-combo-out` 与 in-sample 非常接近，说明模型对 batch size / `nip` 变化的外推是稳的。
2. `leave-one-case-out` 有明显升高，但仍保持在可用区间，说明跨 case 的执行上下文差异确实会带来额外误差。
3. 按真实耗时加权后，整体误差进一步下降，说明高耗时算子上的拟合质量整体好于单纯样本平均所给出的印象。

## 7. 参数稳定性分析

从 [fold_parameters.csv](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/analytical_generalization/fold_parameters.csv) 可以看到三类现象。

### 7.1 很稳定的参数

这些参数在不同 fold 之间几乎不动：

- `rho_gather_inf = 0.12`
- `kappa_reduce = 0.5`
- `occ_ref = 16`
- `rho_tiny_inf = 0.3`
- `tau_micro = 0`
- `eta_stride = 0`

这说明这些参数更多反映的是**稳定 kernel 机制**，而不是某个 fold 的偶然数值。

### 7.2 中等稳定的参数

这些参数在不同 fold 之间小幅摆动，但仍集中在少数几个取值：

- `rho_copy_inf in {0.15, 0.18}`
- `tau_copy_start in {0, 0.032}`
- `m_gather in {4, 6}`
- `rho_fma_inf in {0.50, 0.55}`
- `M50 / N50 / K50` 在少数网格点间切换

这类摆动是合理的，因为它们本来就是“渐近效率/半饱和尺度”。

### 7.3 最不稳定的 family

最不稳定的不是 `Gather` 或 `Gemm`，而是 `Transpose / MatMul` 的相关配置：

- `Transpose` 的最优值几乎总是 `m_stride = 1, eta_stride = 0`
- 这意味着当前公式最终把它解释成“stride penalty 几乎不能靠线程数摊薄”
- `MatMul` 的 `K50_tiny` 在不同 fold 之间会切到 `0` 或 `64`

这和 held-out 误差最大的 family 恰好一致，说明剩余误差不是简单调网格能完全解决的，而是模型结构本身还缺一层信息。

## 8. 结论

本轮泛化测试可以得出四个直接结论。

### 8.1 当前可解释校准模型确实有泛化能力

它不是单纯的 in-sample 过拟合。最明显的证据是：

- `leave-one-combo-out` 的 mean macro MAPE 只有 `15.79%`
- `leave-one-combo-out` 的 duration-weighted relative error 只有 `10.31%`
- `Gather / ReduceSum / Gemm / Concat` 在两种 held-out 方案上都维持得比较稳

此外，把 copy-like family 的表达从 `B50_*` 改写成显式 `tau_start_*` 后，held-out 结果与上一版几乎等价，说明这次改写主要提升的是参数解释性，而不是靠重新定义参数偷换指标口径。

### 8.2 跨 case 外推比跨 combo 外推更难

`combo` 变化主要改变 problem size；而 `case` 变化还隐含改变了：

- 并发上下文
- branch/task 竞争环境
- 某些 kernel 的实际 wall-time 扩展性

因此 `leave-one-case-out` 比 `leave-one-combo-out` 更接近“真正的 deployment 外推”。

从耗时加权视角看，这个结论仍然成立：

- `leave-one-case-out`: `14.44%`
- `leave-one-combo-out`: `10.31%`

### 8.3 当前 hardest families 仍然是 `Transpose` 和 `MatMul`

它们的 held-out 误差最高：

- `Transpose`
  - `32.31%` on leave-one-case-out
  - `34.35%` on leave-one-combo-out
- `MatMul`
  - `28.77%` on leave-one-case-out

其中 `Transpose` 已经明显不是纯 bandwidth/stride 两项就能完全解释的对象。

### 8.4 下一步最值得加的不是更多裸参数，而是显式 contention 项

从切片数据本身可以看到，`case_10_2_1` 基本是串行上下文，而 `case_9_4_4 / case_10_4_4` 往往伴随较高的 `combo_task_parallel_fraction`。这正好对应 `Transpose / MatMul` 在 held-out case 上的误差升高。

因此下一步最值得尝试的是：

- 为 `Transpose / MatMul` 增加一个显式、可解释的 `C_contention` 乘子
- 把 wall-time 膨胀与 kernel base time 分开建模

## 9. 最终判断

如果问题是：

> 这套可解释校准 analytical model 在真正 held-out 测试上还能不能站得住？

当前答案是：

- `可以`
- 但要分 family 看

更具体地说：

- `Gather / ReduceSum / Gemm / Concat` 已经具备比较稳的外推能力
- `MatMul` 勉强达标，但还有优化空间
- `Transpose` 仍然是最大的泛化短板

这也意味着：V3 的模型方向是对的，但如果下一阶段要追更稳的跨 case 泛化，就需要把“并发竞争导致的 wall-time inflation”显式建进去，而不是继续只调 kernel base time 参数。
