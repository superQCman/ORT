# 可解释校准 Analytical Model 设计说明

## 摘要

这份文档只保留 `single_op_stage1_mlp` 中可解释校准 analytical model 的设计内容，用来说明这套模型为什么存在、依赖哪些共享硬件量、每个 heavy-op family 的机制假设是什么，以及这些公式应该如何理解。

这里不再讨论“纯 Analytical Model 组”，也不保留任何量化误差结果。重点不是对比实验结论，而是把当前这套可解释校准方案整理成一份结构更清晰、逻辑更连贯的设计文档。

当前覆盖的 heavy-op family 为：

- `Gather`
- `ReduceSum`
- `Gemm`
- `MatMul`
- `Transpose`
- `Concat`

这些 family 都服务于 [analytical_calibrated](README.md) 子流水线，用于生成 `ana_calib_*` analytical proxy，并作为后续 classed-op MLP 的输入特征之一。

## 1. 设计目标与适用范围

### 1.1 设计目标

这套 analytical model 的目标不是构造一个纯解析上界模型，也不是把 analytical feature 做成黑盒回归器，而是在两者之间保留一条可解释的中间路线：

- 公式结构必须来自算子执行机制
- 允许少量参数校准
- 每个参数都必须有明确物理语义
- 参数应该能够解释为理想上界与真实执行之间的机制缺口

因此，这里的校准并不是随意加常数，而是只允许出现在下列物理槽位中：

- 持续有效带宽比例
- steady-state 之前的固定启动时间
- 可并发隐藏的 miss 数
- 持续 FMA 利用率
- 微核、dispatch 或 chunk 级固定结构性开销
- 线程扩展失效的指数或并发深度

### 1.2 适用范围

当前文档聚焦的对象是 ORT DLRM 中机制相对稳定、且值得单独建模的 heavy-op family。

这些算子之所以单独建模，是因为它们的主瓶颈比较明确，但又不能用单一 `bytes / BW_peak` 或 `flops / PeakFMA` 近似覆盖：

- `Concat` 更接近大块 copy 加 dispatch
- `ReduceSum` 更接近连续流式读写上的 reduction 惩罚
- `Gather` 同时包含 source miss 与 output copy
- `Gemm` 受 tile 饱和和 packing 摊销影响明显
- `MatMul` 是 tiny batched matmul，不适合用大 GEMM 思路直接套
- `Transpose` 是典型 stride-heavy memory movement

## 2. 数据切片与 heavy-op 语义

### 2.1 数据来源

数据来自：

- [dataset_full.csv](../artifacts/latest/dataset_all_no_trace/dataset_full.csv)

### 2.2 当前 heavy-op 切片

本轮设计针对的 heavy-op 主要集中在以下 case 与 combo 切片中：

- `case_9_4_4`
- `case_10_2_1`
- `case_10_4_4`
- `combo in {bs1024_nip1500, bs1440_nip1700, bs1888_nip1800}`

之所以单独强调这个切片，是因为这些样本的形状结构和执行行为足够集中，适合先把 family-level 机制公式收敛稳定。

### 2.3 family 语义概览

- `Gather`
  - 主要对应 embedding table gather
  - 典型形状接近从大表中按大量 `int64` index 拉取大 row
- `ReduceSum`
  - 主要是 `[B, nip, K] -> [B, K]`
  - 核心是流式累加而不是随机访问
- `Gemm`
  - 主要是 DLRM bottom/top MLP 中的大矩阵乘
- `MatMul`
  - 主要是批量 tiny matmul，而不是常规大 GEMM
- `Transpose`
  - 主要是 stride-heavy transpose
- `Concat`
  - 主要是多路大块输入沿末轴拼接

这类集中结构意味着 family 机制分析比通用 one-size-fits-all 公式更合适。

## 3. 共享硬件常量与中间子模型

硬件 profile 来自：

- [kunpeng920_single_numa.yaml](../hardware_profile/kunpeng920_single_numa.yaml)

为了让 family 公式显式绑定到底层硬件，而不是把硬件量藏进经验常数，这里保留一组共享硬件符号和子模型。

### 3.1 共享符号

| 符号 | 定义 | 说明 |
| --- | --- | --- |
| `T` | `min(num_threads, total_cores)` | 活跃线程数上限。 |
| `BW_peak` | `bandwidth_gb_s_total * 1e3` bytes/us | 理想峰值内存带宽。 |
| `f_cpu` | `cpu frequency` | CPU 主频。 |
| `cacheline` | `64 bytes` | cacheline 大小。 |
| `lanes_fp32` | `simd_width_bits / 32` | 每条 SIMD 指令可处理的 fp32 lane 数。 |
| `lat_L1` / `lat_L2` / `lat_L3` / `lat_MEM` | 各层访问延迟 | 由 profile 与频率共同决定。 |
| `PeakFMA(T)` | FMA 理论峰值吞吐 | 主要用于 `Gemm/MatMul`。 |
| `PeakAdd(T)` | 加法理论峰值吞吐 | 主要用于 `ReduceSum`。 |
| `IssueSlots(T)` | issue ceiling | 作为 copy-like kernel 的粗粒度 issue 上界。 |

### 3.2 共享硬件子模型

为了避免不同 family 重复定义硬件逻辑，设计上保留以下共享子模型：

- `fit(bytes)`
  - 根据 working set 与 `L1/L2/L3 active bytes` 的关系判断数据更接近 `L1/L2/L3/MEM` 哪一层
- `lat(level)`
  - 将 `L1/L2/L3 response latency cycles` 和 `local_mem_delay_ns` 折算成统一延迟量纲
- `PeakAdd(T)`
  - 显式绑定 SIMD 宽度、加法吞吐、频率和线程数
- `PeakFMA(T)`
  - 显式绑定 SIMD 宽度、FMA 吞吐、频率和线程数
- `IssueSlots(T)`
  - 显式绑定 pipeline width、频率和线程数

这样每个 family 只使用与自身瓶颈最相关的硬件量：

- `Gather / ReduceSum / Transpose` 显式依赖 cache tier 与 latency
- `ReduceSum / Gemm / MatMul` 显式依赖 SIMD 与算术吞吐
- `Concat / ReduceSum / Transpose` 可以显式依赖 issue/copy 相关上界

## 4. 校准参数的语义边界

当前设计只保留少量可解释参数，它们不是自由偏置项，而是机制缺口的压缩表达。

| 参数类 | 物理含义 |
| --- | --- |
| `rho_bw` | 持续有效带宽占峰值带宽的比例 |
| `tau_start` | kernel 进入 steady-state 前的固定启动时间 |
| `m_mlp` | 可并发隐藏的 miss 数，或有效 memory-level parallelism |
| `rho_fma` | 计算核的持续 FMA 利用率 |
| `tau_launch` | 微核、chunk 或 dispatch 的固定启动开销 |
| `eta_stride` | stride penalty 的线程扩展性质 |

对于 copy-like family，还可以引入一个等价解释量：

`B50 = BW_inf * tau_start`

这里的 `B50` 不是独立参数，而是把固定启动时间换算成“等效半饱和字节数”的解释形式，用于帮助理解 chunk 太小时为什么不能直接达到渐近带宽。

## 5. 各 family 的可解释校准模型

### 5.1 `Concat`: 大块 copy + dispatch

#### 机制假设

当前 `/Concat` 更像多路大块输入的顺序拼接，而不是计算密集型 kernel。主要成本来自：

1. 数据搬运的持续带宽
2. 每一路输入都要支付的 dispatch / loop 管理开销

#### 定义

- `stream_bytes = input_bytes_sum + output_size`
- `chunk_mean = input_bytes_sum / input_count`
- `BW_copy_inf = BW_peak * rho_copy_inf`
- `BW_copy_eff(chunk_mean) = BW_copy_inf * chunk_mean / (BW_copy_inf * tau_copy_start + chunk_mean)`

#### 时延模型

`T_concat = stream_bytes / BW_copy_eff(chunk_mean) + input_count * tau_dispatch`

#### 解释

- `rho_copy_inf` 表示 ORT copy path 在大块 steady-state 下的有效持续带宽比例
- `tau_copy_start` 表示单个 chunk 进入 steady-state 前的固定起步成本
- `tau_dispatch` 表示每路输入的 offset 维护、dispatch 与边界处理成本

因此，`Concat` 的慢并不来自算术，而来自 copy efficiency 与 per-input 管理开销。

### 5.2 `ReduceSum`: 流式读写上的 reduction 惩罚

#### 机制假设

当前切片中的 `ReduceSum` 基本是：

`[B, nip, K] -> [B, K]`

它的主要特征是：

- 输入读近似连续
- 输出 accumulator 可复用
- 真实瓶颈不在随机 miss，而在 reduction 相对 copy 基线的效率折损

#### 定义

- `stream_bytes = activation_size + output_size`
- `add_ops = feat_reduction_work_items`
- `inner = feat_reduction_axes_product`
- `BW_reduce_inf = BW_peak * rho_copy_inf`
- `BW_reduce_eff(inner) = BW_reduce_inf * inner / (BW_reduce_inf * tau_reduce_start + inner)`

#### 时延模型

`T_reduce = max(stream_bytes / BW_reduce_eff(inner), add_ops / PeakAdd(T))`

#### 解释

- `feat_reduction_work_items` 对应输出元素总工作量
- `feat_reduction_axes_product` 对应每个输出元素的内循环规模
- `tau_reduce_start` 表示 reduction 流式阶段的起步成本

这个 family 的关键不是单纯“读写多少字节”，而是 reduction 内循环对流式带宽的折损程度。

### 5.3 `Gather`: source miss + output copy 的双机制模型

#### 机制假设

embedding gather 不能近似成单一路径的 `bytes / BW`。它同时包含：

1. source row 随机访问带来的 cache / L3 / DRAM miss
2. row 拉回后写入 output 的 copy 成本

因此需要双机制模型：

- 一个 source latency 路径
- 一个 destination bandwidth 路径

#### 定义

- `request_rows_true = num_elements(indices_shape)`
- `row_bytes = output_size / request_rows_true`
- `cachelines_per_row = ceil(row_bytes / cacheline)`
- `stream_bytes = 2 * output_size + 8 * request_rows_true`
- `src_fit = fit(src_working_set_bytes)`
- `lat_src_us = lat(src_fit)`
- `BW_gather_inf = BW_peak * rho_gather_inf`
- `BW_gather_eff(row_bytes) = BW_gather_inf * row_bytes / (BW_gather_inf * tau_gather_row_start + row_bytes)`

#### 时延模型

- `T_src = request_rows_true * cachelines_per_row * lat_src_us / (T * m_gather)`
- `T_bw = stream_bytes / BW_gather_eff(row_bytes)`
- `T_gather = max(T_bw, T_src, tau_floor)`

#### 解释

- `rho_gather_inf` 表示大 row gather 的渐近持续带宽比例
- `tau_gather_row_start` 表示 row 太小时单次寻址与粒度浪费的固定启动成本
- `m_gather` 表示 source miss 的有效 memory-level parallelism
- `fit(src_working_set_bytes)` 与 `lat(src_fit)` 显式决定 source path 更接近哪一级缓存或内存延迟

这里 `Gather` 的关键不只是“搬了多少字节”，而是 row 粒度、source tier 和 miss 并发隐藏能力三者共同决定 wall time。

### 5.4 `Gemm`: tile 饱和不足下的持续 FMA 利用率

#### 机制假设

大 `Gemm` 不能只用标准 roofline：

`T = max(flops / PeakFMA, bytes / BW_peak)`

因为真实 MLAS kernel 还受下列因素影响：

- tile 填充不足
- packing 成本摊销不足
- 边界 tile
- 并行分块与 cache reuse 的非理想性

#### 定义

- `flops = 2MNK`
- `mem_bytes = input_bytes + weight_bytes + output_bytes`
- `T_base = flops / (PeakFMA(T) * rho_fma_inf)`
- `rho_fma_eff = rho_fma_inf * M / (M + M50) * N / (N + N50) * K / (K + K50)`

#### 时延模型

`T_gemm = max(flops / (PeakFMA(T) * rho_fma_eff), mem_bytes / BW_peak)`

#### 解释

- `rho_fma_inf` 表示大尺寸下 MLAS 相对理论峰值的持续利用率
- `M50/N50/K50` 表示三个维度各自的半饱和尺度
- 当某一维偏小时，对应方向的有效利用率下降

因此，`Gemm` 的误差主要不是 roofline 结构完全错，而是需要显式描述 shape saturation 对 sustained efficiency 的影响。

### 5.5 `MatMul`: tiny batched kernel 的 occupancy 模型

#### 机制假设

当前 `/MatMul` 更接近大量 tiny batched matmul，而不是通用大矩阵乘。主要问题不是算力上限，而是：

- `M/N` 太小
- occupancy 很低
- packing 与调度成本难以充分摊销

#### 定义

- `flops = 2 * batch_count * M * N * K`
- `rho_tiny_eff = rho_tiny_inf * min(M / occ_ref, 1) * min(N / occ_ref, 1) * K / (K + K50_tiny)`

#### 时延模型

`T_matmul = flops / (PeakFMA(T) * rho_tiny_eff) + ceil(batch_count / T) * tau_micro`

#### 解释

- `occ_ref` 表示 tiny GEMM 接近微核饱和时的参考 occupancy 尺度
- `rho_tiny_inf` 表示 tiny regime 下的渐近持续利用率
- `tau_micro` 表示每个 micro-batch 的固定启动成本

因此，`MatMul` 需要把 tiny regime 单独建模，而不是当作小一号的普通 `Gemm`。

### 5.6 `Transpose`: 流式搬运叠加 stride penalty

#### 机制假设

当前 `/Transpose` 主要是：

`[B, 9, K] -> [B, K, 9]`

它的总字节量虽然接近 copy-like kernel，但真实执行还包含明显 stride penalty：

- 总体上是读写搬运
- 但访问顺序不连续
- locality 与线程扩展性都可能恶化

#### 定义

- `out_bytes = output_size`
- `prefix_blocks = product(prefix_dims_before_contiguous_suffix)`

#### 时延模型

`T_transpose = 2 * out_bytes / (BW_peak * rho_copy_inf) + prefix_blocks * lat_src_us / (T ^ eta_stride * m_stride)`

#### 解释

- 第一项表示 copy 主体
- 第二项表示 stride-heavy 额外 penalty
- `m_stride` 表示 stride miss 的可隐藏并发度
- `eta_stride` 表示这部分 penalty 随线程扩展的有效程度

因此，`Transpose` 的难点不只是数据量，而是 stride 访问对 locality 和扩展性的破坏。

## 6. 参数解释与实现落点

### 6.1 参数不是黑盒自由度

这套设计里最重要的边界是：

- 不直接引入无语义常数
- 不把误差压低作为唯一目标
- 不把 analytical model 偷偷变成回归器

所有保留的参数都必须能落回到具体机制：

- 为什么带宽达不到峰值
- 为什么小块无法立刻进入 steady-state
- 为什么 random miss 不能完全被线程掩蔽
- 为什么 tiny kernel 占用不足
- 为什么 stride penalty 不随线程线性缩放

### 6.2 对代码落地的意义

如果后续继续把这套设计沉淀到实现中，应优先保持以下结构不变：

1. 共享硬件中间量与 family dispatcher 分离
2. family 公式显式引用共享硬件子模型
3. 校准参数只出现在有物理语义的位置
4. `ana_calib_mem_us`、`ana_calib_compute_us`、`ana_calib_overhead_us` 保持机制可拆解

这样做的目的，是让 `analytical_calibrated` 既能产出可用的 proxy 特征，也能继续承担“机制解释层”的角色。

## 7. 总结

这套可解释校准 analytical model 的核心立场是：

- 不走纯公式上界路线
- 也不走黑盒拟合路线
- 而是让公式结构由算子机制决定，让少量参数只负责表达理想上界与真实执行之间的稳定缺口

对当前 `single_op_stage1_mlp` 而言，这种设计最重要的价值不是写出一套更复杂的公式，而是把 heavy-op family 的 latency proxy 拆成可解释、可复用、可继续演进的 analytical 组件。
