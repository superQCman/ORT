# 可解释校准 Analytical Model 设计说明

## 摘要

这份文档只保留 `single_op_stage1_mlp` 中可解释校准 analytical model 的设计内容，用来说明这套模型为什么存在、依赖哪些共享硬件量、每个 calibrated family 的机制假设是什么，以及这些公式应该如何理解。

这里不再讨论“纯 Analytical Model 组”，也不保留任何量化误差结果。重点不是对比实验结论，而是把当前这套可解释校准方案整理成一份结构更清晰、逻辑更连贯的设计文档。

本文默认对齐当前 `analytical_calibrated/build_analytical_features.py` 导出的 `ana_calib_*` 公式口径；`evaluate_analytical_generalization.py` 中仅用于试验或对比的额外 `variant` 分支，不作为这里的默认设计定义。

当前覆盖的 calibrated family 为：

- `Gather`
- `ReduceSum`
- `Relu`
- `Add`
- `Mul`
- `Sigmoid`
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

当前文档聚焦的对象是 ORT DLRM 中机制相对稳定、且值得单独建模的 analytical family。

这些算子之所以单独建模，是因为它们的主瓶颈比较明确，但又不能用单一 `bytes / BW_peak` 或 `flops / PeakFMA` 近似覆盖：

- `Concat` 更接近大块 copy 加 dispatch
- `ReduceSum` 更接近连续流式读写上的 reduction 惩罚
- `Relu` 更接近 unary elementwise 的 memory-dominant 流式 kernel
- `Add / Mul` 更接近 binary elementwise micro-kernel
- `Sigmoid` 同时包含流式访存和非线性逐元素计算
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

与此同时，`mixed_balanced` 中已经被单独建模的 `Relu / Add / Mul / Sigmoid` 也纳入这份设计说明；它们不属于最初的 heavy-op 切片重点，但已经成为当前 `ana_calib_*` 导出路径的一部分。

### 2.3 family 语义概览

- `Gather`
  - 主要对应 embedding table gather
  - 典型形状接近从大表中按大量 `int64` index 拉取大 row
- `ReduceSum`
  - 主要是 `[B, nip, K] -> [B, K]`
  - 核心是流式累加而不是随机访问
- `Relu`
  - 主要是 unary elementwise activation
  - 当前实现按 memory-dominant 的流式读写 kernel 建模
- `Add`
  - 主要是 binary elementwise 加法
  - 当前实现显式保留小流量带宽效率和固定 kernel overhead
- `Mul`
  - 主要是 binary elementwise 乘法
  - 机制与 `Add` 类似，但保留独立校准参数
- `Sigmoid`
  - 主要是 unary nonlinear activation
  - 需要同时考虑流式访存路径和逐元素非线性 compute 路径
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
| `T` | `min(num_threads, hw_core_total_cores)` | 活跃线程数上限；所有 throughput 型共享子模型都用这个线程裁剪。 |
| `BW_peak` | `hw_memory_bandwidth_gb_s_total * 1e3` bytes/us | 理想峰值内存带宽；代码里会对它做 `>= 1e-6` 裁剪。 |
| `f_cpu` | `hw_core_cpu_clock` GHz | CPU 主频；latency 和 throughput 都显式依赖它。 |
| `cacheline` | `hw_cache_cacheline_bytes`，默认 `64 bytes` | cacheline 大小；当前主要用于 `Concat` 和 `Gather`。 |
| `lanes_fp32` | `hw_instruction_simd_width_bits / 32` | 每条 SIMD 指令可处理的 fp32 lane 数，代码里下界裁到 `1.0`。 |
| `L1_active / L2_active / L3_active` | `hw_cache_l1d_active_bytes / hw_cache_l2_active_bytes / hw_cache_l3_active_bytes` | `fit(bytes)` 的三级 cache 容量阈值。 |
| `lat_L1 / lat_L2 / lat_L3` | `response_latency_cycles / f_cpu / 1000` us | 各级 cache 命中延迟。 |
| `lat_MEM` | `hw_memory_local_mem_delay_ns / 1000` us | 本地内存访问延迟。 |
| `PeakFMA(T)` | `throughput_fma * lanes_fp32 * 2 * f_cpu * 1e3 * T` | FMA 理论峰值吞吐；主要用于 `Gemm/MatMul`。 |
| `PeakAdd(T)` | `throughput_add * lanes_fp32 * f_cpu * 1e3 * T` | 加法理论峰值吞吐；主要用于 `ReduceSum` 与 `Sigmoid` compute path。 |
| `IssueSlots(T)` | `min(fetch, decode, rename, dispatch, issue, commit) * f_cpu * 1e3 * T` | copy-like kernel 的粗粒度前端发射上界。 |

### 3.2 共享硬件子模型

为了避免不同 family 重复定义硬件逻辑，设计上保留以下共享子模型：

- `fit(bytes)`
  - 输入 `working_set_bytes`
  - 先把 working set 裁到 `>= 0`
  - 再依次与 `L1_active / L2_active / L3_active` 比较
  - 返回 `1 / 2 / 3 / 4`，分别对应 `L1 / L2 / L3 / MEM`
- `lat(level)`
  - `lat(1) = hw_cache_l1d_response_latency_cycles / f_cpu / 1000`
  - `lat(2) = hw_cache_l2_response_latency_cycles / f_cpu / 1000`
  - `lat(3) = hw_cache_l3_per_die_response_latency_cycles / f_cpu / 1000`
  - `lat(4) = hw_memory_local_mem_delay_ns / 1000`
  - 统一输出单位是 `us`
- `BW_eff(size; BW_inf, tau_start)`
  - `BW_eff = BW_inf * size / (BW_inf * tau_start + size)`
  - 这是当前所有 bandwidth-saturation 路径共享的半饱和子模型
  - 当前默认被 `Concat / ReduceSum / Gather / Relu / Add / Mul / Sigmoid` 复用
- `PeakAdd(T)`
  - `PeakAdd(T) = throughput_add * lanes_fp32 * f_cpu * 1e3 * T`
  - 其中 `throughput_add = hw_instruction_fp_throughput_per_cycle_vector_sp_add`
- `PeakFMA(T)`
  - `PeakFMA(T) = throughput_fma * lanes_fp32 * 2 * f_cpu * 1e3 * T`
  - 其中额外的 `2` 对应一次 FMA 包含乘和加两类 flop
- `IssueSlots(T)`
  - `IssueSlots(T) = min(fetch, decode, rename, dispatch, issue, commit) * f_cpu * 1e3 * T`
  - 各宽度来自 `hw_pipeline_*_width`
  - 代码里先取这几级前端宽度的最小值，再裁到 `>= 1.0`
  - 当前默认只有 `Concat` 把它作为主公式的一部分

这样每个 family 只使用与自身瓶颈最相关的硬件量：

- `Gather / Transpose` 显式依赖 `fit -> lat` 这条 cache-tier / latency 路径
- `ReduceSum / Gemm / MatMul / Sigmoid` 显式依赖 SIMD 与算术吞吐
- `Concat / ReduceSum / Relu / Add / Mul / Sigmoid / Transpose` 共享 copy/bandwidth baseline
- 当前默认只有 `Concat` 显式依赖 `IssueSlots(T)`，`ReduceSum / Transpose` 不再把 issue ceiling 放进主公式

## 4. 校准参数的语义边界

当前设计里真正参与拟合的参数，全部来自

- `evaluate_analytical_generalization.py` 中的 `PARAM_SEARCH_SPACE`
- `fit_family_parameters()` 的按 family coordinate search

这些参数不是自由偏置项，而是机制缺口的压缩表达。共享硬件量来自 profile，不属于拟合对象；只有下表这些参数会在训练切分或 full-data build 时被搜索。

### 4.1 拟合规则

- 所有参数都按离散网格搜索，而不是连续优化
- 搜索是按 family 分组做 coordinate search
- 当前默认顺序是：
  - `Concat`: `rho_copy_inf`, `tau_copy_start`, `tau_dispatch`
  - `ReduceSum`: `kappa_reduce`, `tau_reduce_start`
  - `Gather`: `rho_gather_inf`, `tau_gather_row_start`, `m_gather`
  - `Gemm`: `rho_fma_inf`, `M50`, `N50`, `K50`
  - `MatMul`: `occ_ref`, `rho_tiny_inf`, `K50_tiny`, `tau_micro`
  - `Transpose`: `m_stride`, `eta_stride`
  - `Relu`: `rho_relu_inf`, `tau_relu_start`
  - `Add`: `rho_add_inf`, `tau_add_start`, `tau_add`
  - `Mul`: `rho_mul_inf`, `tau_mul_start`, `tau_mul`
  - `Sigmoid`: `rho_sigmoid_inf`, `tau_sigmoid_start`, `tau_sigmoid`, `rho_sigmoid_compute`
- `rho_copy_inf` 是共享 copy baseline，所以它虽然通过 `Concat` 目标搜索，但也会被 `ReduceSum` 和 `Transpose` 的默认公式复用

### 4.2 参数清单

| 参数 | 用到的 family | 搜索范围 | 含义 |
| --- | --- | --- | --- |
| `rho_copy_inf` | `Concat` 直接使用；`ReduceSum / Transpose` 间接复用 copy baseline | `[0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30]` | 大流量 copy path 的渐近有效带宽比例，单位为峰值带宽占比。 |
| `tau_copy_start` | `Concat` | `[0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032] us` | concat 单个 chunk 从冷启动到进入 steady-state bandwidth 路径的固定起步成本。 |
| `tau_dispatch` | `Concat` | `[0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0] us` | 每一路输入的 dispatch、offset 维护和边界处理开销。 |
| `kappa_reduce` | `ReduceSum` | `[0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00]` | reduction 流式阶段相对 copy baseline 的持续带宽折损比例。 |
| `tau_reduce_start` | `ReduceSum` | `[0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016] us` | reduction bandwidth path 进入 steady-state 前的固定起步成本。 |
| `rho_gather_inf` | `Gather` | `[0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26]` | 大 row gather 的渐近有效带宽比例。 |
| `tau_gather_row_start` | `Gather` | `[0.0, 0.002, 0.004, 0.008, 0.012, 0.016, 0.024, 0.032] us` | row 很小时，单次寻址与粒度浪费带来的固定 row-start 成本。 |
| `m_gather` | `Gather` | `[1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]` | source miss 的有效 memory-level parallelism，可理解为可并发隐藏的 miss 数。 |
| `rho_fma_inf` | `Gemm` | `[0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80]` | 大尺寸 GEMM 在 tile 饱和后相对 `PeakFMA(T)` 的持续利用率。 |
| `M50` | `Gemm` | `[0.0, 8.0, 16.0, 32.0, 64.0, 128.0]` | `M / (M + M50)` 饱和模型的半饱和尺度，单位是矩阵维度大小。 |
| `N50` | `Gemm` | `[0.0, 8.0, 16.0, 32.0, 64.0]` | `N / (N + N50)` 饱和模型的半饱和尺度，单位是矩阵维度大小。 |
| `K50` | `Gemm` | `[0.0, 16.0, 32.0, 64.0, 128.0, 256.0]` | `K / (K + K50)` 饱和模型的半饱和尺度，单位是矩阵维度大小。 |
| `occ_ref` | `MatMul` | `[4.0, 8.0, 12.0, 16.0, 24.0, 32.0]` | tiny matmul 接近微核 occupancy 饱和时的参考维度尺度。 |
| `rho_tiny_inf` | `MatMul` | `[0.08, 0.12, 0.18, 0.24, 0.30, 0.40, 0.50, 0.60]` | tiny regime 下相对 `PeakFMA(T)` 的渐近持续利用率。 |
| `K50_tiny` | `MatMul` | `[0.0, 16.0, 32.0, 64.0, 128.0, 256.0]` | tiny matmul 中 `K / (K + K50_tiny)` 的半饱和尺度。 |
| `tau_micro` | `MatMul` | `[0.0, 0.25, 0.5, 1.0, 2.0, 4.0] us` | 每个 micro-batch 的固定启动成本。 |
| `m_stride` | `Transpose` | `[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.50, 0.75, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]` | stride penalty 的可隐藏并发度；越大表示越多 stride-latency 可以被并发吞掉。 |
| `eta_stride` | `Transpose` | `[0.0, 0.25, 0.5, 0.75, 1.0]` | stride penalty 随线程数缩放的指数。 |
| `rho_relu_inf` | `Relu` | `[0.008, 0.010, 0.012, 0.015, 0.020, 0.030, 0.040, 0.060, 0.080, 0.10, 0.12, 0.15, 0.18, 0.20, 0.24, 0.30]` | unary memory path 的渐近有效带宽比例。 |
| `tau_relu_start` | `Relu` | `[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0] us` | `Relu` 流式 bandwidth path 的起步成本。 |
| `rho_add_inf` | `Add` | `[0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18]` | `Add` 大流量 steady-state 的有效带宽比例。 |
| `tau_add_start` | `Add` | `[0.0, 0.5, 1.0, 2.0, 4.0, 8.0] us` | `Add` bandwidth path 的起步成本。 |
| `tau_add` | `Add` | `[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0] us` | `Add` kernel 的固定结构性 overhead。 |
| `rho_mul_inf` | `Mul` | `[0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18]` | `Mul` 大流量 steady-state 的有效带宽比例。 |
| `tau_mul_start` | `Mul` | `[0.0, 0.5, 1.0, 2.0, 4.0, 8.0] us` | `Mul` bandwidth path 的起步成本。 |
| `tau_mul` | `Mul` | `[0.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0] us` | `Mul` kernel 的固定结构性 overhead。 |
| `rho_sigmoid_inf` | `Sigmoid` | `[0.008, 0.010, 0.012, 0.015, 0.020, 0.030, 0.040, 0.060, 0.080]` | `Sigmoid` memory path 的渐近有效带宽比例。 |
| `tau_sigmoid_start` | `Sigmoid` | `[0.0, 0.5, 1.0, 2.0, 4.0, 8.0] us` | `Sigmoid` memory path 的起步成本。 |
| `tau_sigmoid` | `Sigmoid` | `[0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 40.0] us` | `Sigmoid` kernel 的固定结构性 overhead。 |
| `rho_sigmoid_compute` | `Sigmoid` | `[1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 8e-3, 1e-2, 1.5e-2, 2e-2]` | 把 `PeakAdd(T)` 缩放成有效 nonlinear compute ceiling 的比例系数。 |

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
- `T_issue = (stream_bytes / cacheline) / IssueSlots(T)`
  - concat算子引入IssueSlots(T)的主要原因是它的 copy-loop 行为会受到 cacheline 粒度访存发射和前端 issue ceiling 的双重约束，尤其在 chunk 很小时，issue ceiling 可能成为更紧的上界。
  - 每个 cacheline 的处理都依赖前端持续发射 load/store loop

#### 时延模型

`T_concat = max(stream_bytes / BW_copy_eff(chunk_mean), T_issue) + input_count * tau_dispatch`

#### 解释

- `rho_copy_inf` 表示 ORT copy path 在大块 steady-state 下的有效持续带宽比例
- `tau_copy_start` 表示单个 chunk 进入 steady-state 前的固定起步成本
- `T_issue` 表示 copy loop 至少还要受到 cacheline 粒度访存发射与前端 issue ceiling 的约束
- `tau_dispatch` 表示每路输入的 offset 维护、dispatch 与边界处理成本

因此，`Concat` 的慢并不来自算术，而来自 copy efficiency、issue ceiling 与 per-input 管理开销。

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
- `BW_reduce_inf = BW_peak * rho_copy_inf * kappa_reduce`
- `BW_reduce_eff(inner) = BW_reduce_inf * inner / (BW_reduce_inf * tau_reduce_start + inner)`

#### 时延模型

`T_reduce = max(stream_bytes / BW_reduce_eff(inner), add_ops / PeakAdd(T))`

#### 解释

- `feat_reduction_work_items` 对应输出元素总工作量
- `feat_reduction_axes_product` 对应每个输出元素的内循环规模
- `kappa_reduce` 表示 reduction 流式阶段相对纯 copy 基线的持续带宽折损
- `tau_reduce_start` 表示 reduction 流式阶段的起步成本
- 当前导出 `ana_calib_*` 的实现保持这条 baseline `max(mem, compute)` 结构，不再叠加额外的依赖或 issue 修正项

这个 family 的关键不是单纯“读写多少字节”，而是 reduction 内循环对流式带宽的折损程度。

### 5.3 `Relu / Add / Mul / Sigmoid`: mixed_balanced 中的 op-aware elementwise 子模型

#### 机制假设

这 4 个算子都属于逐元素 kernel，但不能再统一塞回旧的 `generic_mixed` fallback。

当前实现显式区分：

- `Relu`
  - unary elementwise
  - 主要受流式读写支配
- `Add / Mul`
  - binary elementwise
  - 除了流式读写，还要显式保留小 kernel 的固定 overhead
- `Sigmoid`
  - unary nonlinear
  - 同时受流式访存和逐元素非线性 compute 限制

因此，这个 family 在设计上不再追求统一的 `mem_us / compute_us` 两分法，而是按 `op_type` 分流后分别输出自己的 `ana_calib_total_us`。

#### `Relu` 定义

- `stream_bytes = feat_io_bytes_sum`
- `BW_relu_inf = BW_peak * rho_relu_inf`
- `BW_relu_eff(stream_bytes) = BW_relu_inf * stream_bytes / (BW_relu_inf * tau_relu_start + stream_bytes)`

#### `Relu` 时延模型

`T_relu = stream_bytes / BW_relu_eff(stream_bytes)`

#### `Relu` 解释

- `Relu` 当前被当成 memory-dominant unary kernel
- `rho_relu_inf` 表示大流量 steady-state 下的有效持续带宽比例
- `tau_relu_start` 表示小流量时进入 steady-state 前的固定启动成本
- 当前导出实现里：
  - `ana_calib_total_us = T_relu`
  - `ana_calib_mem_us = T_relu`
  - `ana_calib_compute_us = 0`
  - `ana_calib_overhead_us = 0`

#### `Add` 定义

- `stream_bytes = feat_io_bytes_sum`
- `BW_add_inf = BW_peak * rho_add_inf`
- `BW_add_eff(stream_bytes) = BW_add_inf * stream_bytes / (BW_add_inf * tau_add_start + stream_bytes)`

#### `Add` 时延模型

`T_add = tau_add + stream_bytes / BW_add_eff(stream_bytes)`

#### `Add` 解释

- `Add` 按 binary elementwise micro-kernel 建模
- `rho_add_inf` 表示大流量下加法 kernel 的渐近有效带宽比例
- `tau_add_start` 表示小流量带宽路径的起步成本
- `tau_add` 表示 kernel 固定结构性 overhead
- 当前导出实现里：
  - `ana_calib_total_us = T_add`
  - `ana_calib_mem_us = stream_bytes / BW_add_eff(stream_bytes)`
  - `ana_calib_compute_us = 0`
  - `ana_calib_overhead_us = tau_add`

#### `Mul` 定义

- `stream_bytes = feat_io_bytes_sum`
- `BW_mul_inf = BW_peak * rho_mul_inf`
- `BW_mul_eff(stream_bytes) = BW_mul_inf * stream_bytes / (BW_mul_inf * tau_mul_start + stream_bytes)`

#### `Mul` 时延模型

`T_mul = tau_mul + stream_bytes / BW_mul_eff(stream_bytes)`

#### `Mul` 解释

- `Mul` 与 `Add` 的结构类似，但保留独立的 `rho_mul_inf / tau_mul_start / tau_mul`
- 这样可以显式表达乘法 kernel 与加法 kernel 在 steady-state 效率和固定开销上的差异
- 当前导出实现里：
  - `ana_calib_total_us = T_mul`
  - `ana_calib_mem_us = stream_bytes / BW_mul_eff(stream_bytes)`
  - `ana_calib_compute_us = 0`
  - `ana_calib_overhead_us = tau_mul`

#### `Sigmoid` 定义

- `stream_bytes = feat_io_bytes_sum`
- `output_elements = output_size / bytes_per_element`
- `BW_sigmoid_inf = BW_peak * rho_sigmoid_inf`
- `BW_sigmoid_eff(stream_bytes) = BW_sigmoid_inf * stream_bytes / (BW_sigmoid_inf * tau_sigmoid_start + stream_bytes)`
- `PeakSigmoidEff(T) = PeakAdd(T) * rho_sigmoid_compute`

#### `Sigmoid` 时延模型

`T_sigmoid = tau_sigmoid + max(stream_bytes / BW_sigmoid_eff(stream_bytes), output_elements / PeakSigmoidEff(T))`

#### `Sigmoid` 解释

- `Sigmoid` 不能近似成纯 memory path，因为逐元素非线性计算本身也可能成为主项
- `rho_sigmoid_inf` 与 `tau_sigmoid_start` 控制流式访存路径
- `rho_sigmoid_compute` 把 `PeakAdd(T)` 缩放成有效的 nonlinear per-element compute ceiling
- `tau_sigmoid` 表示固定 kernel overhead
- 当前导出实现里：
  - `ana_calib_total_us = T_sigmoid`
  - `ana_calib_mem_us = stream_bytes / BW_sigmoid_eff(stream_bytes)`
  - `ana_calib_compute_us = output_elements / PeakSigmoidEff(T)`
  - `ana_calib_overhead_us = tau_sigmoid`

这一组小算子的关键不是追求一个统一 family 公式，而是保留 op-aware 子模型，让 `mixed_balanced` 最终以按 `op_type` 分流后的 `ana_calib_total_us` 作为稳定 proxy。

### 5.4 `Gather`: source miss + output copy 的双机制模型

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
- `src_read_bytes = output_size`
- `dst_write_bytes = output_size`
- `bytes_per_index = sizeof(int64)`
- `index_read_bytes = bytes_per_index * request_rows_true`
- `stream_bytes = src_read_bytes + dst_write_bytes + index_read_bytes`
- `table_rows = source_tensor_dim0`
- `unique_rows_est = table_rows * (1 - exp(-request_rows_true / table_rows))`
- `src_working_set_bytes = unique_rows_est * row_bytes`
- `src_fit = fit(src_working_set_bytes)`
- `lat_src_us = lat(src_fit)`
- `l1_latency_us = lat(L1)`
- `BW_gather_inf = BW_peak * rho_gather_inf`
- `BW_gather_eff(row_bytes) = BW_gather_inf * row_bytes / (BW_gather_inf * tau_gather_row_start + row_bytes)`
- `tau_floor = 8 * max(lat_src_us / l1_latency_us, 1) ^ 0.75`

#### 时延模型

- `T_src = request_rows_true * cachelines_per_row * lat_src_us / (T * m_gather)`
- `T_bw = stream_bytes / BW_gather_eff(row_bytes)`
- `T_gather = max(T_bw, T_src, tau_floor)`

#### 解释

- `src_read_bytes` 表示从 source tensor/table 读取 payload 的字节量
- `dst_write_bytes` 表示把 gather 结果写入 output tensor 的字节量
- `bytes_per_index` 表示单个 index 元素的字节宽度；当前这里默认对应 `int64`
- `index_read_bytes` 表示读取 index tensor 本身的开销
- `stream_bytes` 因此被显式拆成“源数据读 + 目标数据写 + index 读”三部分，而不是用没有语义的常数直接相加
- `unique_rows_est` 用 occupancy 形式近似 source table 中真正触达的不同 row 数，只用于估计 source working set 落在哪一级 cache / memory
- 实际实现里 `unique_rows_est` 还会被裁到 `[0, min(request_rows_true, table_rows)]`，而在缺少 `table_rows` 时回退为 `request_rows_true`
- `rho_gather_inf` 表示大 row gather 的渐近持续带宽比例
- `tau_gather_row_start` 表示 row 太小时单次寻址与粒度浪费的固定启动成本
- `m_gather` 表示 source miss 的有效 memory-level parallelism
- `fit(src_working_set_bytes)` 与 `lat(src_fit)` 显式决定 source path 更接近哪一级缓存或内存延迟；当前代码里 miss 次数项仍保持 `request_rows_true`
- `tau_floor` 是一个随 source tier 变大的 level-aware 下界，用来避免超小 row 在模型里被压得不合理地过低

这里 `Gather` 的关键不只是“搬了多少字节”，而是 row 粒度、source tier 和 miss 并发隐藏能力三者共同决定 wall time。

### 5.5 `Gemm`: tile 饱和不足下的持续 FMA 利用率

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

### 5.6 `MatMul`: tiny batched kernel 的 occupancy 模型

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

### 5.7 `Transpose`: 流式搬运叠加 stride penalty

#### 机制假设

当前 `/Transpose` 主要是：

`[B, 9, K] -> [B, K, 9]`

它的总字节量虽然接近 copy-like kernel，但真实执行还包含明显 stride penalty：

- 总体上是读写搬运
- 但访问顺序不连续
- locality 与线程扩展性都可能恶化

#### 定义

- `out_bytes = output_size`
- `prefix_blocks = product(output_dims[:-1])`
- `suffix_block_bytes = output_dims[-1] * bytes_per_element`
- `stride_fit = fit(suffix_block_bytes)`
- `lat_stride_us = lat(stride_fit)`

#### 时延模型
`copy_us = 2 * out_bytes / (BW_peak * rho_copy_inf)`
`stride_us = prefix_blocks * lat_stride_us / (T ^ eta_stride * m_stride)`

`T_transpose = copy_us + stride_us = 2 * out_bytes / (BW_peak * rho_copy_inf) + prefix_blocks * lat_stride_us / (T ^ eta_stride * m_stride)`

#### 解释

- 第一项表示 copy 主体
- 第二项表示每个 prefix block 都要支付一次 contiguous suffix block 的 stride-heavy 额外 penalty
- `lat_stride_us` 当前按 suffix block working set 的 `fit -> lat` 路径决定，而不是旧版按整体 per-thread working set 估计
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
