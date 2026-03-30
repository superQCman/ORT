# Analytical Model V3: 可解释校准模型与纯 Analytical 上限分析

## 摘要

这份文档记录本轮 `single_op_stage1_mlp` analytical feature 设计的两条并行路线：

1. 一组“有限校准但强可解释”的 analytical model。它允许少量参数拟合，但每个参数都必须具有明确物理语义，例如持续带宽比例、memory-level parallelism、微核启动开销、tile 饱和尺度，而不能直接引入无语义常数。
2. 一组“纯 Analytical、无拟合”的 analytical model。它只允许使用 shape、dtype、硬件 profile 和算子执行语义，目的是估计在完全不做校准时，这类模型在当前 ORT DLRM heavy-op 子集上的准确度上限。

评估范围固定为：

- `case_9_4_4`
- `case_10_2_1`
- `case_10_4_4`
- `combo in {bs1024_nip1500, bs1440_nip1700, bs1888_nip1800}`
- heavy-op family:
  - `Gather`
  - `ReduceSum`
  - `Gemm`
  - `MatMul`
  - `Transpose`
  - `Concat`

在这个切片上：

- 可解释校准模型的 family 宏平均 `MAPE = 14.99%`
- 可解释校准模型的全样本加权 `MAPE = 11.78%`
- 6 个算子族都压到了 `30%` 以内
- 纯 Analytical 上限组的 family 宏平均 `MAPE = 66.62%`
- 纯 Analytical 上限组的全样本加权 `MAPE = 61.62%`

这说明两点：

1. “纯公式 + 零校准”在当前 heavy-op DLRM CPU kernel 上远不足以提供可用的绝对时延 proxy。
2. 只要把校准限制在少量、可解释、可迁移的效率参数上，analytical model 就可以同时保留机制解释力和工程可用精度。

## 1. 研究问题与评估设定

### 1.1 研究问题

本轮分析回答两个问题：

1. 在不把 analytical model 变成黑盒回归器的前提下，能否把 heavy-op 的平均相对误差控制到 `30%` 左右？
2. 如果严格禁止拟合，只使用硬件 profile 和张量代数，纯 analytical model 的准确度上限大致在哪里？

核心约束是：我们要的不是单纯“拟合得好”，而是“误差低且参数可解释”。

### 1.2 数据切片

数据源为：

- [dataset_full.csv](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/dataset_all_no_trace/dataset_full.csv)

评估切片规则如下：

| family | 选择规则 | 样本数 |
| --- | --- | ---: |
| `Gather` | `op_type == Gather and output_size > 1e8` | 59 |
| `ReduceSum` | `op_type == ReduceSum and activation_size > 1e8` | 59 |
| `Gemm` | `op_type == Gemm and label_operator_actual_dur_us > 1000` | 35 |
| `MatMul` | `node_name == /MatMul` | 8 |
| `Transpose` | `node_name == /Transpose` | 8 |
| `Concat` | `node_name == /Concat` | 9 |

总样本数为 `178`。

### 1.3 heavy-op 形状语义

这个切片里的 heavy-op 结构非常集中，因此适合做机制分析：

- `Gather`
  - 主要是 embedding table gather
  - shape 近似为 `[51200, 200/400]` 按大批量 `int64` index 拉取成 `[B * nip, 200/400]`
- `ReduceSum`
  - 主要是 `[B, nip, 200/400] -> [B, 200/400]`
  - `nip in {1500, 1700, 1800}`
- `Gemm`
  - 主要是 DLRM bottom/top MLP 中的大矩阵乘
- `MatMul`
  - 主要是 `[{B, 9, 200/400}] x [{B, 200/400, 9}] -> [{B, 9, 9}]`
  - 它不是“大 GEMM”，而是大量 tiny batched matmul
- `Transpose`
  - 主要是 `[{B, 9, 200/400}] -> [{B, 200/400, 9}]`
  - 明显属于 stride-heavy transpose
- `Concat`
  - 主要是 9 路 `[B, 200/400]` 沿末轴拼接到 `[B, 1800/3600]`
  - 本质是多个大块 copy + dispatch

### 1.4 误差指标

统一使用：

`MAPE = mean(abs(pred_us - actual_us) / actual_us)`

同时报告两种聚合方式：

- `family macro MAPE`：6 个算子族 MAPE 的简单平均
- `weighted overall MAPE`：按样本数加权后的总体 MAPE

前者衡量“每个 family 是否都做得足够稳”，后者衡量“整个 heavy-op slice 的整体误差水平”。

## 2. 共享硬件常量与符号

硬件 profile 来自：

- [kunpeng920_single_numa.yaml](/data/qc/dlrm/ORT/single_op_stage1_mlp/hardware_profile/kunpeng920_single_numa.yaml)

其中最重要的共享常量如下。

| 符号 | 定义 | 数值/来源 |
| --- | --- | --- |
| `T` | `min(num_threads, total_cores)` | 每条样本动态决定 |
| `BW_peak` | `bandwidth_gb_s_total * 1e3` bytes/us | `100000 bytes/us` |
| `f_cpu` | `2.6 GHz` | profile |
| `cacheline` | `64 bytes` | profile |
| `lanes_fp32` | `simd_width_bits / 32` | `4` |
| `lat_L1` | `1 cycle / 2.6GHz` | `0.385 ns` |
| `lat_L2` | `10 cycles / 2.6GHz` | `3.846 ns` |
| `lat_L3` | `20 cycles / 2.6GHz` | `7.692 ns` |
| `lat_MEM` | `local_mem_delay_ns` | `90 ns` |
| `PeakFMA(T)` | `Th_fma * lanes_fp32 * 2 * f_cpu * 1e3 * T` | `41600 * T ops/us` |
| `PeakAdd(T)` | `Th_add * lanes_fp32 * f_cpu * 1e3 * T` | `20800 * T ops/us` |
| `IssueSlots(T)` | `pipeline_width * f_cpu * 1e3 * T` | `10400 * T slots/us` |

说明：
- `FMA` 全称是Fused Multiply-Add，表示一次同时包含乘法和加法的复合算子。它的峰值吞吐量通常是单纯加法或乘法的两倍，因此 `2 * f_cpu`。
- `PeakFMA(T)` 用于 `Gemm/MatMul`
- `PeakAdd(T)` 用于 `ReduceSum`
- `IssueSlots(T)` 只作为 copy-like kernel 的粗粒度 issue ceiling

## 3. 建模原则

### 3.1 为什么需要“有限校准”

如果一个模型完全使用 `BW_peak`、`PeakFMA(T)`、`lat_MEM` 这类理想上界，那么它默认：

- copy kernel 可以接近 DRAM 峰值
- GEMM kernel 可以接近 SIMD/FMA 峰值
- random/stride-heavy miss 可以按线程数近似线性摊薄
- tiny matmul 没有明显启动开销和 occupancy 损失

这些假设对 roofline 分析有价值，但对 ORT 上真实 kernel 的绝对时间预测通常过于乐观。

因此，本轮允许的拟合只限于如下五类“有物理语义”的参数：

| 参数类 | 物理含义 |
| --- | --- |
| `rho_bw` | 持续有效带宽占峰值带宽的比例 |
| `tau_start` | kernel 进入 steady-state 前的固定启动时间 |
| `m_mlp` | 可并发隐藏的 miss 数，或有效 memory-level parallelism |
| `rho_fma` | 计算核的持续 FMA 利用率 |
| `tau_launch` | 每个微核、chunk 或 dispatch 的固定启动开销 |

其中 copy-like family 仍可定义等价量：

`B50 = BW_inf * tau_start`

也就是说，`B50` 不是主参数，而是把固定启动时间折算成“等效字节数”后的中间解释量。这样既保留半饱和直觉，又避免把 `B50` 当成神秘阈值直接拟合。

这些参数都能解释为“理想上界与真实执行之间的机制缺口”，而不是神秘常数。

### 3.2 为什么还要保留纯 Analytical 组

纯 Analytical 组的目的不是实际部署，而是回答：

> 如果完全不允许做任何效率校准，那么 analytical model 的准确度上限在哪里？

这有两个价值：

1. 判断“误差来自公式结构”还是“误差来自缺少校准”
2. 为后续参数预算提供下界与上界

## 4. 可解释校准 Analytical Model

### 4.1 参数总表

本轮最终保留的可解释参数如下。

| 参数 | 含义 | 取值 | 解释 |
| --- | --- | ---: | --- |
| `rho_copy_inf` | 大块 copy 的渐近持续带宽比例 | `0.18` | ORT copy path 只能拿到约 `18%` 的 DRAM 峰值 |
| `tau_copy_start` | 每个 copy chunk 的固定启动时间 | `0 us` | 当前 heavy-op chunk 已足够大，steady-state copy 直接主导 |
| `tau_dispatch` | 每个输入块的 dispatch/loop 启动开销 | `20 us` | `Concat` 小而固定的 per-input 管理成本 |
| `kappa_reduce` | reduction 相对纯 copy 的带宽折损 | `0.5` | 连续 reduce 只有大约一半 copy 效率 |
| `tau_reduce_start` | reduction 流式阶段的固定启动时间 | `0 us` | 当前重负载已处于渐近带宽区间 |
| `rho_gather_inf` | 大 row gather 的渐近有效带宽比例 | `0.12` | random-row copy 比顺序 copy 更慢 |
| `tau_gather_row_start` | 每个 gather row 的固定启动时间 | `0.0107 us` | 在当前 `BW_row_inf` 下约折算为 `128 B` 的等效 row 大小 |
| `m_gather` | gather source miss 的有效并发深度 | `4` | 每线程组约能隐藏 4 个 miss |
| `rho_fma_inf` | 大 shape GEMM 的渐近 FMA 利用率 | `0.55` | MLAS 相对理论峰值的持续效率 |
| `M50` | `M` 方向半饱和尺度 | `32` | `M` 太小时 tile 填充不足 |
| `N50` | `N` 方向半饱和尺度 | `0` | 当前重负载 `N` 已不构成额外惩罚 |
| `K50` | `K` 方向半饱和尺度 | `64` | `K` 太小时 packing/compute 摊销差 |
| `occ_ref` | tiny matmul 饱和 occupancy 参考尺度 | `16` | `M/N` 小于 16 时 occupancy 明显下降 |
| `rho_tiny_inf` | tiny GEMM 的渐近持续利用率 | `0.3` | tiny batched matmul 远低于大 GEMM |
| `K50_tiny` | tiny matmul 的 `K` 半饱和尺度 | `0` | 当前 `K=200/400` 已非主瓶颈 |
| `tau_micro` | tiny microkernel 启动成本 | `0 us` | 当前切片中主要由低 occupancy，而非显著固定启动成本主导 |
| `m_stride` | stride-heavy transpose 的有效并发深度 | `8` | 表征可同时容忍的 stride miss 数量 |
| `eta_stride` | stride penalty 的线程扩展指数 | `0` | 该惩罚几乎未随线程数有效摊薄 |

这些参数不是任意自由度，而是被限制在明确语义的物理槽位中：

- `rho_*` 必须在 `(0, 1]` 内
- `tau_start_*` 必须是时间量纲
- `m_*` 必须是无量纲并发深度
- `tau_*` 必须是时间量纲
- `eta_*` 必须解释线程扩展性质

### 4.2 `Concat`: 大块 copy + dispatch

#### 4.2.1 内核语义

`/Concat` 在这个切片里是 9 路大块输入沿末轴拼接。它的主执行过程不是浮点计算，而是：

1. 为每路输入计算输出偏移
2. 调度一次 strided/contiguous copy
3. 顺序把各路块写入输出张量

因此一阶瓶颈来自：

- 持续 copy 带宽
- 每路输入的 dispatch/loop 启动开销

#### 4.2.2 公式

定义：

- `stream_bytes = input_bytes_sum + output_size`
- `chunk_mean = input_bytes_sum / input_count`

有效带宽模型：

`BW_copy_inf = BW_peak * rho_copy_inf`

`BW_copy_eff(chunk_mean) = BW_copy_inf * chunk_mean / (BW_copy_inf * tau_copy_start + chunk_mean)`

时延模型：

`T_concat = stream_bytes / BW_copy_eff(chunk_mean) + input_count * tau_dispatch`

#### 4.2.3 解释

- `rho_copy_inf`
  - 表示大块 copy 的渐近效率
  - 反映 ORT copy path、NUMA 局部内存、cache hierarchy、预取和软件循环开销综合后的稳定持续带宽
- `tau_copy_start`
  - 表示每个 copy chunk 进入 steady-state 前的固定启动时间
  - 若需要保留半饱和直觉，可等价写成 `B50_copy = BW_copy_inf * tau_copy_start`
- `tau_dispatch`
  - 不是任意偏置项
  - 它表示每个输入块都要付出的 loop setup、offset maintenance、dispatch 与边界处理成本

当前重负载下，9 路 chunk 都已经足够大，因此 `tau_copy_start=0` 合理地表示“steady-state copy 已直接主导，总时延几乎不再受块内起步损失控制”。这不是过拟合，而是说明这批 heavy concat 不再处于小块 regime。

### 4.3 `ReduceSum`: 流式读写上的 reduction 惩罚

#### 4.3.1 内核语义

本切片中的 `ReduceSum` 基本都是：

`[B, nip, K] -> [B, K]`

这类 kernel 的本质是：

- 输入读几乎是连续的
- 输出 accumulator 能在较小工作集内反复复用
- 真正拖慢它的不是随机 miss，而是“在 copy 基线之上的 reduction 惩罚”

### 4.2.4 Shared Hardware Submodel

为了让各 family 的公式显式绑定到底层硬件，本版本保留一组共享硬件子模型，而不是把这些硬件量全部藏进隐式 helper：

- `fit(bytes)`
  - 根据 active working-set bytes 与 `L1/L2/L3 active bytes` 的比较来判断当前数据更接近 `L1/L2/L3/MEM` 哪一层
- `lat(level)`
  - 根据 `L1/L2/L3 response latency cycles`、`local_mem_delay_ns` 与 `f_cpu` 折算成 `us`
- `PeakAdd(T)`
  - 由 `vector_sp_add throughput/cycle`、`SIMD width bits`、`CPU 频率` 和活跃线程数 `T` 显式构成
- `PeakFMA(T)`
  - 由 `vector_sp_fma throughput/cycle`、`SIMD width bits`、`CPU 频率` 和活跃线程数 `T` 显式构成
- `IssueSlots(T)`
  - 由 `pipeline_width * f_cpu * T` 显式构成

因此：

- `Gather / ReduceSum / Transpose` 至少显式使用 `L1/L2/L3 size` 与 `latency`
- `ReduceSum / Gemm / MatMul` 至少显式使用 `SIMD width`、`指令 throughput/latency` 和 `CPU 频率`
- `Concat / ReduceSum / Transpose` 至少显式使用 `pipeline width`

这样做的目的不是让每个算子吃掉所有硬件参数，而是让每个 family 只显式使用与自身瓶颈机制匹配的那部分硬件量。

#### 4.3.2 公式

定义：

- `stream_bytes = activation_size + output_size`
- `add_ops = feat_reduction_work_items`
- `inner = feat_reduction_axes_product`

有效带宽模型：

`BW_reduce_inf = BW_peak * rho_copy_inf * kappa_reduce`

`BW_reduce_eff(inner) = BW_reduce_inf * inner / (BW_reduce_inf * tau_reduce_start + inner)`

时延模型：

`T_reduce = max(stream_bytes / BW_reduce_eff(inner), add_ops / PeakAdd(T))`

#### 4.3.3 解释

- `kappa_reduce`
  - 表示“连续 reduce”相对“纯 copy”的额外折损
  - 它不是一个神秘常数，而是在 copy 基线上乘上的 reduction penalty
- `tau_reduce_start`
  - 表示 reduction 流式阶段的固定启动时间
  - 若需要半饱和尺度，可等价写成 `B50_reduce = BW_reduce_inf * tau_reduce_start`

在当前 heavy reduce 里，`nip >= 1500`，已经远大于典型小 reduction 区间，因此 `tau_reduce_start=0` 仍然合理。

### 4.4 `Gather`: source miss 与目标 copy 的双机制模型

#### 4.4.1 内核语义

embedding gather 不能被视为单纯的 `bytes / BW`。它同时包含：

1. source row 随机访问导致的 cache/L3/DRAM miss
2. 取回 row 之后写入 output 的 copy 成本

因此 `Gather` 至少需要一个二元机制模型：

- 一个源侧 latency 路径
- 一个目的侧 bandwidth 路径

#### 4.4.2 公式

定义：

- `request_rows = feat_lookup_count`
- `row_bytes = output_size / request_rows`
- `cachelines_per_row = ceil(row_bytes / cacheline)`
- `stream_bytes = 2 * output_size + 8 * request_rows`

共享硬件子模型在 `Gather` 中的显式展开为：

- `src_fit = fit(src_working_set_bytes)`
- `lat_src_us = lat(src_fit)`

行粒度有效带宽：

`BW_gather_inf = BW_peak * rho_gather_inf`

`BW_gather_eff(row_bytes) = BW_gather_inf * row_bytes / (BW_gather_inf * tau_gather_row_start + row_bytes)`

源侧 miss 路径：

`T_src = request_rows * cachelines_per_row * lat_src_us / (T * m_gather)`

整体时延：

`T_gather = max(stream_bytes / BW_gather_eff(row_bytes), T_src)`

其中 `lat_src_us` 由 source working set 所在 `L1/L2/L3/MEM` tier 显式决定。

本版本最终采纳的 source working set 仍然是保守的请求规模近似，而不是唯一行修正：

- 采纳：`src_working_set_bytes ~= request_rows * row_bytes`
- 不采纳：`unique_rows_est = table_rows * (1 - exp(-request_rows / table_rows))`

原因不是唯一行公式不合理，而是当前 heavy-op DLRM 切片中：

- `request_rows / table_rows` 已经非常大
- `unique_rows_est / table_rows ~= 1`
- source tier 几乎全部仍落在 `MEM`

因此唯一行修正不会实质降低 `src_fit`，只会改变 source row 计数；在 held-out case 上，这个修正并没有带来更好的泛化误差。

#### 4.4.3 解释

- `rho_gather_inf`
  - 对应“大 row、充足流水化”条件下 gather 所能获得的渐近持续带宽
  - 它自然应低于 `rho_copy_inf`，因为 source 地址不连续
- `tau_gather_row_start`
  - 对应“row 太小时，单次寻址、边界处理和 cacheline 粒度浪费还没被 row payload 覆盖”的固定启动时间
  - 在当前校准点下可等价折算成 `B50_gather_row = BW_gather_inf * tau_gather_row_start ≈ 128 B`
- `m_gather`
  - 对应“random source miss 能被并发隐藏的程度”
  - 这本质上是 memory-level parallelism，不是任意常数
- `fit(src_working_set_bytes)` 与 `lat(src_fit)`
  - 负责把 `L1/L2/L3 size` 和 `L1/L2/L3/MEM latency` 显式融入 `Gather` 的 source miss 路径
  - 它们决定的不是 row-copy 带宽，而是“每次 source cacheline 访问更像 L2/L3/DRAM 中的哪一层”

这个参数化方式比直接写一个 `8000 bytes/us` 更强，因为它清楚说明了常数来自哪种机制。

#### 4.4.4 候选修订：真实 `request_rows` 与 level-aware fixed overhead

下面这版是 2026-03-30 做的离线候选修订，目的不是立刻改代码，而是先验证 `Gather` 的纯 analytical 结构还有没有继续降 MAPE 的空间。

当前实现里：

- `request_rows = feat_lookup_count = batch_size * num_indices_per_lookup`

这个定义对典型 embedding gather 是合理近似，但对一批非 embedding 的 tiny gather 明显失真。最典型的几类样本是：

- `input_type_shape = [{int64:[2]}, {int64:[]}]`, `output_type_shape = [{int64:[]}]`
- `input_type_shape = [{int64:[3]}, {int64:[1]}]`, `output_type_shape = [{int64:[1]}]`
- `input_type_shape = [{float32:[81,1056]}, {int64:[36]}]`, `output_type_shape = [{float32:[36,1056]}]`

这些样本的真实 index 请求元素数分别更接近 `1 / 1 / 36`，但当前 `feat_lookup_count` 会统一落到百万级，导致：

- `row_bytes = output_size / request_rows` 被压到极小
- `stream_bytes = 2 * output_size + 8 * request_rows` 被严重放大
- `T_gather` 对 tiny gather 出现系统性结构误判

因此候选修订的第一步不是分类，而是先把 `Gather` 自己的请求规模定义修正为节点真实 shape 所决定的值：

- `request_rows_true = num_elements(indices_shape)`
- 其中 `indices_shape` 来自 `input_type_shape` 的第二个输入

然后保留当前双机制结构不变，只把其中的 `request_rows` 全部替换为 `request_rows_true`：

- `row_bytes = output_size / max(request_rows_true, 1)`
- `cachelines_per_row = ceil(row_bytes / cacheline)`
- `stream_bytes = 2 * output_size + 8 * request_rows_true`
- `T_src = request_rows_true * cachelines_per_row * lat_src_us / (T * m_gather)`
- `T_bw = stream_bytes / BW_gather_eff(row_bytes)`

离线验证表明，这一步本身已经能大幅修正 tiny gather 的异常 MAPE：

| 版本 | all MAPE | all DWRE | test MAPE | test DWRE |
|---|---:|---:|---:|---:|
| 当前公式 | 2674.08% | 30.70% | 2642.89% | 30.59% |
| 仅替换为真实 `request_rows` | 57.73% | 30.63% | 57.08% | 30.53% |

这说明 `Gather` 当前最关键的问题之一，不是“大 embedding gather 的主公式完全错误”，而是很多 tiny gather 被错误地套用了全局 batch 级 `lookup_count`。

但只修正 `request_rows` 之后，tiny gather 又会出现另一类系统性低估：`T_bw` 和 `T_src` 都会变得接近 `0 us`，而真实时延通常仍有 `8-30 us` 左右的固定运行时开销。因此离线验证又测试了第二步：在保留当前 `max(T_bw, T_src)` 主结构的同时，补一个可解释的 fixed-overhead 下界。

第一种候选是常数下界：

- `T_gather_candidate = max(T_bw, T_src, tau_floor)`
- 其中 `tau_floor = 8 us`

结果是：

| 版本 | all MAPE | all DWRE | test MAPE | test DWRE |
|---|---:|---:|---:|---:|
| 真实 `request_rows` + 常数 `8 us` floor | 34.89% | 30.63% | 34.45% | 30.53% |

常数下界已经非常有效，但它还不够“层级感知”。因此又进一步验证了一个更可解释的 level-aware floor：

- `lat_L1_us = hw_cache_l1d_response_latency_cycles / cpu_clock / 1000`
- `tau_floor(row) = 8 us * (lat_src_us / lat_L1_us)^0.75`
- `T_gather_candidate = max(T_bw, T_src, tau_floor(row))`

这个项的意义是：

- 当 source hit 更接近 `L1` 时，floor 接近一个较小的 L1 级开销
- 当 source working set 更像 `L2/L3/DRAM` 访问时，floor 会随 `lat_src_us` 自适应放大
- 仍然只依赖显式的 cache-latency 硬件量，不引入额外黑盒回归特征

这版离线结果是当前最好的纯 analytical 候选：

| 版本 | all MAPE | all DWRE | test MAPE | test DWRE | test MedianAPE |
|---|---:|---:|---:|---:|---:|
| 真实 `request_rows` + level-aware floor | 33.39% | 30.63% | 32.91% | 30.53% | 33.28% |

这说明：

1. `Gather` 是否需要先按 `output_size` 分类，其实不是第一优先级
2. 更高优先级的问题，是把 `request_rows` 改成节点真实 indices shape 所决定的值
3. 在此基础上，再补一个与 cache latency level 绑定的 fixed-overhead 项，就已经能把 `test MAPE` 从 `2642.89%` 降到 `32.91%`

额外做过的小范围离线搜索还验证了：

- 再把 `tau_floor` 乘一个弱的 `cachelines_per_row^beta` 项并没有继续变好
- 当前最优点仍然是：
  - `request_rows = num_elements(indices_shape)`
  - `tau_floor(row) = 8 us * (lat_src_us / lat_L1_us)^0.75`
  - 即 `beta = 0`

注意：本节只是记录截至 2026-03-30 的离线候选修订及其效果，尚未同步到 `analytical_calibrated` 的正式代码实现中。

### 4.5 `Gemm`: tile 饱和不足下的持续 FMA 利用率

#### 4.5.1 内核语义

大 `Gemm` 的理想 roofline 通常写成：

`T = max(flops / PeakFMA, bytes / BW_peak)`

但这在真实 MLAS kernel 上会系统性低估时延，因为它忽略了：

- tile 填充不足
- packing 成本摊销不足
- 边界 tile
- 线程分块和 cache reuse 的非理想性

#### 4.5.2 公式

定义：

- `flops = 2MNK`
- `mem_bytes = input_bytes + weight_bytes + output_bytes`

先定义大尺寸、边界效应已消退时的基础计算时间：

`T_base = flops / (PeakFMA(T) * rho_fma_inf)`

然后把 `M/N/K` 不足饱和带来的额外损失写成对基础时间的相对放大：

`T_shape = T_base * (M50 / M + N50 / N + K50 / K)`

于是总计算时间可以近似写成：

`T_compute = T_base + T_shape`

当把上式整理为乘法形式时，可得到：

`T_compute ≈ T_base * (1 + M50 / M) * (1 + N50 / N) * (1 + K50 / K)`

再把它改写成“等效持续利用率”的形式，就得到：

`rho_fma_eff = rho_fma_inf * M / (M + M50) * N / (N + N50) * K / (K + K50)`

时延模型：

`T_gemm = max(flops / (PeakFMA(T) * rho_fma_eff), mem_bytes / BW_peak)`

#### 4.5.3 解释

- `rho_fma_inf`
  - 表示“大尺寸、边界效应已消退”时 MLAS 相对理论峰值的持续利用率
- `M50/N50/K50`
  - 不是直接从硬件手册读取的原生常数
  - 而是把 `M/N/K` 三个方向的形状惩罚压缩成“等效半饱和尺度”后的结果
  - 可以理解为：当某一维度等于对应的 `50` 尺度时，仅该维度一项就会把该方向的有效利用率压到大约一半
  - 因而：
    - `M50` 更偏向解释 `M` 方向 tile 填充与并行度不足
    - `N50` 更偏向解释输出列方向的利用率不足
    - `K50` 更偏向解释 packing、流水线预热和计算摊销不足

这套写法的优点是，它不需要显式暴露更细的 kernel 内部特征；即使当前数据里没有 tile shape、packing path 等信息，也仍可以用少量可解释参数把“维度太小会额外变慢”这件事编码进去。

在当前切片中，`N` 的变化没有额外引入明显惩罚，因此 `N50=0` 的解释是“在这批 heavy `Gemm` 上，`N` 方向已不再构成额外折损”，而不是强行把参数压到零。

### 4.6 `MatMul`: tiny batched kernel 的 occupancy 模型

#### 4.6.1 内核语义

本切片里的 `/MatMul` 不是通用大矩阵乘，而是：

- `batch_count = B`
- `M = 9`
- `N = 9`
- `K = 200 or 400`

也就是说，它是大量 tiny batched matmul。其核心问题不是“算力不够”，而是：

- `M/N` 太小
- occupancy 极低
- packing 与调度成本难以完全摊销

#### 4.6.2 公式

定义：

- `flops = 2 * batch_count * M * N * K`

有效 tiny-kernel 利用率：

`rho_tiny_eff = rho_tiny_inf * min(M / occ_ref, 1) * min(N / occ_ref, 1) * K / (K + K50_tiny)`

时延模型：

`T_matmul = flops / (PeakFMA(T) * rho_tiny_eff) + ceil(batch_count / T) * tau_micro`

#### 4.6.3 解释

- `occ_ref`
  - 表示 tiny GEMM 在 `M/N` 方向接近“足够填满 micro-kernel”时的参考尺度
- `rho_tiny_inf`
  - 表示 tiny-kernel 在该 regime 下相对理论峰值的渐近利用率
- `tau_micro`
  - 表示每个 micro-batch 的固定启动成本

当前切片里 `tau_micro=0` 的结论很有意义：说明这个子集上主要误差不是固定启动时间，而是 tiny occupancy 本身。

### 4.7 `Transpose`: 流式搬运叠加 stride penalty

#### 4.7.1 内核语义

当前 `/Transpose` 主要是：

`[B, 9, K] -> [B, K, 9]`

这类 transpose 的问题不是总字节数，而是：

- 虽然总流量近似是 `2 * out_bytes`
- 但由于访问顺序变成 stride-heavy，存在额外的地址跳跃与 locality 惩罚

#### 4.7.2 公式

定义：

- `out_bytes = output_size`
- `prefix_blocks = product(prefix_dims_before_contiguous_suffix)`

时延模型：

`T_transpose = 2 * out_bytes / (BW_peak * rho_copy_inf) + prefix_blocks * lat_src_us / (T ^ eta_stride * m_stride)`

#### 4.7.3 解释

- 第一项是大块 copy 主体
- 第二项是 stride-heavy 额外 penalty
- `m_stride`
  - 表示 stride miss 可以被隐藏的程度
- `eta_stride`
  - 表示这部分 penalty 对线程数的扩展性

当前切片里 `eta_stride=0`，说明这类 transpose 的额外 penalty 基本没有被线程数有效摊薄。这是一个可解释的体系结论：它指向的是线程扩展失效，而不是任意常数。

## 5. 可解释校准模型结果

### 5.1 各算子族 MAPE

| family | 样本数 | MAPE |
| --- | ---: | ---: |
| `Gather` | 59 | `7.89%` |
| `ReduceSum` | 59 | `11.82%` |
| `Gemm` | 35 | `13.22%` |
| `MatMul` | 8 | `15.29%` |
| `Transpose` | 8 | `27.09%` |
| `Concat` | 9 | `14.66%` |

聚合结果：

- `family macro MAPE = 14.99%`
- `weighted overall MAPE = 11.78%`

### 5.2 结果解读

- `Gather`
  - 误差显著下降，说明“row 粒度有效带宽 + source miss 并发深度”比纯 `bytes/BW` 更接近真实机制
- `ReduceSum`
  - 误差已经很低，说明这一 family 主要缺口确实来自“copy 基线上的 reduction penalty”
- `Gemm`
  - 结果稳定，说明 MLAS sustained utilization 可以被少量饱和参数有效描述
- `MatMul`
  - 即使是 tiny batched regime，也已经明显低于 `30%`
  - 这证明单独建 tiny occupancy 模型是必要的
- `Transpose`
  - 它仍然是 6 个 family 中最难的一个
  - 但 `27.09%` 已经满足目标
  - 主要剩余误差来自 stride penalty 与并发 wall-time 竞争项的混合
- `Concat`
  - 单用 `stream_bytes/BW_peak` 明显过于乐观
  - 共享 copy baseline 与 dispatch 项后可以稳定压低误差

## 6. 纯 Analytical Model 组：无拟合准确度上限

### 6.1 纯 Analytical 的严格约束

本节中的“纯 Analytical”必须同时满足：

1. 不使用任何从 `dur_us` 回推得到的效率参数
2. 不引入 `rho_*`、`tau_*`、`m_*` 这类校准量
3. 只允许使用：
   - tensor shape
   - dtype bytes
   - `BW_peak`
   - `lat_L1/L2/L3/MEM`
   - `PeakFMA(T)` / `PeakAdd(T)` / `IssueSlots(T)`
   - ORT kernel 的结构性执行语义

为了探索“无拟合模型的上限”，我们允许每个 family 预定义若干个**物理上合理但零拟合**的结构分支，然后报告该 family 在这些纯结构分支中的最低 MAPE。这个过程不是拟合参数，而是在无拟合约束下搜索“哪类机制结构最像真实 kernel”。

### 6.2 各算子族的最佳纯公式

#### 6.2.1 `Gather`

最佳纯分支：`req_lat_t`

公式：

- `T_bw = stream_bytes / BW_peak`
- `T_req = request_rows * lat(src_fit) / T`
- `T_issue = stream_bytes / (cacheline * IssueSlots(T))`
- `T_gather_pure = max(T_bw, T_req, T_issue)`

解释：

- 不允许引入 `rho_gather_inf` 或 `m_gather`
- 因此 source miss 只能被粗略看成“每次 request 付一次 tier latency，再除以线程数”

#### 6.2.2 `ReduceSum`

最佳纯分支：`stream_plus_blocks_lat_mem`

公式：

- `T_stream = stream_bytes / BW_peak`
- `T_block = row_blocks * lat_MEM`
- `T_reduce_pure = max(T_stream + T_block, add_ops / PeakAdd(T))`

解释：

- `row_blocks` 表示独立输出块数量
- 这个结构已经尽量利用了“连续流 + 外层块切换延迟”的信息
- 但仍无法表达 reduction 相对 copy 的持续带宽折损

#### 6.2.3 `Gemm`

最佳纯公式：

- `T_comp = flops / PeakFMA(T)`
- `T_mem = mem_bytes / BW_peak`
- `T_dep = (K / lanes_fp32) * lat_fma_us`
- `T_gemm_pure = max(T_comp, T_mem, T_dep)`

解释：

- 它已经是最标准、最公平的 roofline + dependency floor
- 但没有 `rho_fma_inf` 与 `M50/K50`，所以会把 MLAS 看得过于理想化

#### 6.2.4 `MatMul`

最佳纯分支：`dep_batch_over_t`

公式：

- `T_comp = flops / PeakFMA(T)`
- `T_mem = mem_bytes / BW_peak`
- `T_dep = ceil(batch_count / T) * (K / lanes_fp32) * lat_fma_us`
- `T_matmul_pure = max(T_comp, T_mem, T_dep)`

解释：

- 对 tiny batched kernel，dependency floor 已经比单纯 roofline 更合理
- 但它仍然不能表达 occupancy 低下和微核饱和不足

#### 6.2.5 `Transpose`

最佳纯分支：`prefix_lat_outfit`

公式：

- `T_stream = 2 * out_bytes / BW_peak`
- `T_stride = prefix_blocks * lat(fit(out_bytes / T))`
- `T_transpose_pure = T_stream + T_stride`

解释：

- 这个公式已经把“总流量 + stride penalty”拆开了
- 但它无法表达真实 transpose copy path 的非理想带宽与线程扩展失效

#### 6.2.6 `Concat`

最佳纯分支：`memory_plus_chunk_lat`

公式：

- `T_stream = stream_bytes / BW_peak`
- `T_chunk = input_count * chunk_mean / BW_peak`
- `T_concat_pure = T_stream + T_chunk`

解释：

- 这是对“大块 copy + per-input chunk 额外管理成本”的零拟合近似
- 但它没有显式 `tau_dispatch`，只能把 dispatch 近似成额外的 chunk 流量

### 6.3 纯 Analytical 上限结果

| family | 最优纯分支 | MAPE |
| --- | --- | ---: |
| `Gather` | `req_lat_t` | `85.88%` |
| `ReduceSum` | `stream_plus_blocks_lat_mem` | `34.97%` |
| `Gemm` | `roofline+dependency` | `56.63%` |
| `MatMul` | `dep_batch_over_t` | `91.15%` |
| `Transpose` | `prefix_lat_outfit` | `52.23%` |
| `Concat` | `memory_plus_chunk_lat` | `78.83%` |

聚合结果：

- `family macro MAPE = 66.62%`
- `weighted overall MAPE = 61.62%`

### 6.4 纯 Analytical 结果的含义

这个结果非常关键，因为它回答了“纯 Analytical 能做到多好”：

- `ReduceSum` 是最接近可用的一个 family，纯模型也能做到约 `35%`
- `Gather`、`Concat`、`MatMul` 都远高于 `70%`
- `Gemm` 即使是最标准的 roofline family，也有 `56.63%`
- `Transpose` 在 stride-heavy regime 下纯公式仍然无法进入 `30%`

换句话说，当前这类 CPU kernel 的主要误差并不是“公式完全写错了”，而是“理想上界与持续执行效率之间存在系统性缺口”。这正是 `rho_bw / rho_fma / m_mlp / tau_launch` 这类可解释参数存在的必要性。

## 7. 校准组与纯组的直接对比

| family | 纯 Analytical MAPE | 可解释校准 MAPE | 误差下降 |
| --- | ---: | ---: | ---: |
| `Gather` | `85.88%` | `7.89%` | `77.99 pt` |
| `ReduceSum` | `34.97%` | `11.82%` | `23.15 pt` |
| `Gemm` | `56.63%` | `13.22%` | `43.41 pt` |
| `MatMul` | `91.15%` | `15.29%` | `75.86 pt` |
| `Transpose` | `52.23%` | `27.09%` | `25.14 pt` |
| `Concat` | `78.83%` | `14.66%` | `64.17 pt` |

这个对比表说明：

1. 校准最有价值的地方不是“把公式从 20% 提到 10%”，而是把本来根本不可用的 family 拉回到可部署区间。
2. 最值得校准的 family 依次是：
   - `MatMul`
   - `Gather`
   - `Concat`
   - `Gemm`
3. `ReduceSum` 的纯模型已经不算差，这说明它的执行机制最接近“纯流式 analytical 可描述”的 regime。

## 8. 结论

本轮分析可以形成三个明确结论。

### 8.1 关于“可解释校准 analytical model”

只要把校准限制在少数物理可解释参数上：

- `rho_bw`
- `tau_start`
- `m_mlp`
- `rho_fma`
- `tau_launch`
- `eta_stride`

那么 heavy-op analytical model 完全可以兼顾：

- 机制解释性
- 参数可迁移性
- 绝对时延精度

在当前 heavy-op 切片上，这组模型已经实现：

- 全 family `MAPE < 30%`
- 宏平均 `14.99%`

### 8.2 关于“纯 Analytical model 的准确度上限”

如果严格禁止拟合，那么在相同切片上：

- 宏平均 MAPE 仍高达 `66.62%`
- 加权总体 MAPE 仍高达 `61.62%`

因此，纯 Analytical 更适合作为：

- 机制分解工具
- bottleneck 分类器
- 结构化 feature generator

而不适合作为直接的绝对时延 estimator。

### 8.3 对后续落代码的启示

后续如果把这套设计真正落入 [feature_engineering.py](/data/qc/dlrm/ORT/single_op_stage1_mlp/feature_engineering.py)，建议遵循以下顺序：

1. 先落共享硬件中间量与 family dispatcher
2. 再落 `Concat / ReduceSum / Gather / Gemm`
3. 最后单独处理 `MatMul` 的 tiny regime 和 `Transpose` 的 stride regime

原因是：

- `Concat / ReduceSum / Gather / Gemm` 的公式与参数语义已经比较稳定
- `MatMul / Transpose` 仍然更依赖 regime 分支与上下文竞争项

## 9. 结语

这份文档的核心立场是：

> 我们不需要在“纯解析”和“纯拟合”之间二选一。

真正可用的 analytical feature 应该位于两者之间：结构上由 kernel 机制决定，参数上只允许少量、可解释、可迁移的效率校准。当前 heavy-op 结果表明，这条路线不仅可行，而且已经足以把所有目标 family 的误差压到 `30%` 以内。
