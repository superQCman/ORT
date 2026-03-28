# Analytical Model V4: 泛化性测试与训练内校准结果

## 1. 目标

这份文档回答上一个版本里尚未解决的问题：

> 可解释校准版 analytical model 的参数如果只在训练样本上校准，再拿去预测没参与校准的 held-out 样本，它的误差会是多少？

这里特别强调两点：

- 本文所有 `MAPE` 都是 **held-out test fold** 上计算的，不是 in-sample 误差。
- 参数校准只使用训练 fold，不使用对应测试 fold 的任何样本。

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

脚本校准的是 V3 文档里定义的同一组可解释参数：

- `rho_copy_inf`
- `B50_copy`
- `tau_dispatch`
- `kappa_reduce`
- `B50_reduce`
- `rho_gather_inf`
- `B50_gather_row`
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
- `B50_*` 表示半饱和尺度
- `m_*` 表示有效并发深度
- `tau_*` 表示固定启动成本
- `eta_stride` 表示线程扩展指数

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

## 5. Held-Out 结果

### 5.1 Leave-One-Case-Out

聚合结果：

- test fold mean macro `MAPE = 21.20%`
- worst-fold macro `MAPE = 23.22%`
- row-count-weighted family `MAPE = 16.96%`

各 family 在测试 fold 上的平均 MAPE：

| family | mean test MAPE | median test MAPE | max fold MAPE |
| --- | ---: | ---: | ---: |
| `Gather` | `12.02%` | `10.39%` | `16.45%` |
| `ReduceSum` | `19.10%` | `19.59%` | `21.68%` |
| `Gemm` | `16.60%` | `17.60%` | `25.04%` |
| `MatMul` | `28.77%` | `33.02%` | `42.43%` |
| `Transpose` | `32.31%` | `39.81%` | `49.10%` |
| `Concat` | `18.38%` | `11.77%` | `38.80%` |

按 held-out case 看：

| held-out case | macro test MAPE | rows |
| --- | ---: | ---: |
| `case_10_2_1` | `17.89%` | 65 |
| `case_10_4_4` | `22.48%` | 55 |
| `case_9_4_4` | `23.22%` | 58 |

### 5.2 Leave-One-Combo-Out

聚合结果：

- test fold mean macro `MAPE = 15.75%`
- worst-fold macro `MAPE = 16.81%`
- row-count-weighted family `MAPE = 11.66%`

各 family 在测试 fold 上的平均 MAPE：

| family | mean test MAPE | median test MAPE | max fold MAPE |
| --- | ---: | ---: | ---: |
| `Gather` | `8.03%` | `7.35%` | `9.49%` |
| `ReduceSum` | `11.36%` | `10.19%` | `13.81%` |
| `Gemm` | `11.19%` | `11.43%` | `16.17%` |
| `MatMul` | `14.96%` | `12.32%` | `20.70%` |
| `Transpose` | `34.35%` | `34.66%` | `39.68%` |
| `Concat` | `14.61%` | `14.98%` | `15.63%` |

按 held-out combo 看：

| held-out combo | macro test MAPE | rows |
| --- | ---: | ---: |
| `bs1024_nip1500` | `14.77%` | 54 |
| `bs1440_nip1700` | `15.66%` | 66 |
| `bs1888_nip1800` | `16.81%` | 58 |

## 6. 与 In-Sample 结果对比

V3 文档中的 in-sample 结果是：

- family macro `MAPE = 14.99%`
- weighted overall `MAPE = 11.78%`

而这轮真正的 held-out 结果是：

| protocol | mean macro MAPE | weighted family MAPE |
| --- | ---: | ---: |
| in-sample prototype | `14.99%` | `11.78%` |
| leave-one-case-out test | `21.20%` | `16.96%` |
| leave-one-combo-out test | `15.75%` | `11.66%` |

这个对比说明：

1. `leave-one-combo-out` 与 in-sample 非常接近，说明模型对 batch size / `nip` 变化的外推是稳的。
2. `leave-one-case-out` 有明显升高，但仍保持在可用区间，说明跨 case 的执行上下文差异确实会带来额外误差。

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
- `B50_copy in {0, 4096}`
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

- `leave-one-combo-out` 的 mean macro MAPE 只有 `15.75%`
- `Gather / ReduceSum / Gemm / Concat` 在两种 held-out 方案上都维持得比较稳

### 8.2 跨 case 外推比跨 combo 外推更难

`combo` 变化主要改变 problem size；而 `case` 变化还隐含改变了：

- 并发上下文
- branch/task 竞争环境
- 某些 kernel 的实际 wall-time 扩展性

因此 `leave-one-case-out` 比 `leave-one-combo-out` 更接近“真正的 deployment 外推”。

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
