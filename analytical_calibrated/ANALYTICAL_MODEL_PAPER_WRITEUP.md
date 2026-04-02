# Analytical Model 论文写作版

## 1. 解析模型设计动机

算子性能同时受到软件特征与硬件特征的共同影响。其中，软件特征通常可通过调整 DLRM 模型结构参数、张量形状、批大小以及线程配置等方式快速获得；相比之下，硬件特征往往依赖处理器微架构、缓存层次、访存带宽与执行流水线等底层配置，若希望通过直接更换硬件平台获取充分样本，往往需要付出较高的实验成本和较长的采样周期。因此，在硬件配置多样性有限的条件下，若仅依赖原始硬件字段训练 MLP 模型，模型往往难以充分学习不同硬件配置对算子时延的真实作用机制。

为缓解上述问题，本文引入具有物理可解释性的解析模型，将硬件特征与软件特征在算子执行机理层面进行耦合，构造软硬件融合的 analytical proxy 作为 MLP 的输入特征。该代理特征并不直接替代数据驱动模型，而是作为连接硬件机理与统计学习模型的中间表征：一方面，它显式编码了缓存命中层级、有效带宽、算术吞吐、线程扩展性以及固定调度开销等硬件约束；另一方面，它又能够随张量规模、访存模式、归约深度、矩阵维度等软件特征的变化而连续变化。通过这种方式，即使训练集中可观测的硬件配置较少，MLP 仍能够借助解析模型学习到“硬件条件变化如何通过执行机制影响时延”的规律。

本文解析模型的输出包括理论总耗时、理论访存耗时、理论计算耗时以及理论结构性开销。需要指出的是，不同算子的执行逻辑存在显著差异，无法使用统一公式进行准确刻画。例如，`Gather` 同时包含随机访存与结果回写，`Transpose` 受到明显的 stride penalty 影响，`Gemm/MatMul` 则主要受算术吞吐与微核占用率控制。因此，本文按照算子机理分别构建解析模型，并在统一硬件子模型的基础上，为每一类算子设计具有明确物理意义的时延表达式。

## 2. 统一符号与共享硬件子模型

为保证不同算子解析模型之间具有一致的物理语义，本文先定义统一的共享符号。设活跃线程数为

`T = min(num_threads, hw_core_total_cores)`。

内存峰值带宽记为

`BW_peak = hw_memory_bandwidth_gb_s_total * 1e3` bytes/us。

CPU 主频记为 `f_cpu`，cacheline 大小记为 `cacheline`。向量加法峰值吞吐与 FMA 峰值吞吐分别定义为

`PeakAdd(T) = throughput_add * lanes_fp32 * f_cpu * 1e3 * T`

`PeakFMA(T) = throughput_fma * lanes_fp32 * 2 * f_cpu * 1e3 * T`

其中 `lanes_fp32` 表示单条 SIMD 指令可处理的 fp32 lane 数。进一步地，考虑到部分算子并非直接受算术吞吐限制，而是受访存发射能力或前端流水线宽度限制，本文还定义 copy-like 算子的粗粒度发射上界

`IssueSlots(T) = min(fetch, decode, rename, dispatch, issue, commit) * f_cpu * 1e3 * T`。

在缓存层次建模方面，本文定义 `fit(bytes)` 用于根据工作集大小判断数据更可能落入 `L1/L2/L3/DRAM` 中的哪一级；随后利用 `lat(level)` 将不同层级映射为相应访问延迟。该机制使得解析模型能够根据工作集大小自动切换延迟主项，而不必为不同缓存层级分别手工设计公式。

在带宽建模方面，本文统一采用半饱和带宽模型

`BW_eff(size; BW_inf, tau_start) = BW_inf * size / (BW_inf * tau_start + size)`，

其中 `BW_inf` 表示大规模 steady-state 下的渐近有效带宽，`tau_start` 表示从启动阶段进入稳态带宽阶段所需付出的固定时间。该表达式能够同时反映两类现象：当数据块较大时，算子趋近于稳定带宽受限；当数据块较小时，固定启动成本会显著拉低等效带宽。后续多个 memory-dominant 算子均复用这一共享形式。

## 3. 统一输出定义

为便于后续与 MLP 模型进行特征融合，本文将解析模型统一拆解为以下四个分量：

- `T_total`：理论总耗时。
- `T_mem`：理论访存主项耗时。
- `T_compute`：理论计算主项耗时。
- `T_overhead`：理论结构性开销，如 dispatch、微批启动、固定 kernel 管理成本等。

其中，

`T_total = f(T_mem, T_compute, T_overhead)`，

函数形式由具体算子机理决定。对于纯访存主导算子，`T_total` 通常近似为 `T_mem + T_overhead`；对于同时存在访存与算术竞争的算子，则采用 `max(T_mem, T_compute)` 或 `T_overhead + max(T_mem, T_compute)` 的形式。统一拆解的目的是使解析特征既能表达总时延，也能保留“误差来自访存还是来自计算”的可解释性。

## 4. 各类算子的解析模型分析与论证

### 4.1 Gather 算子

#### 4.1.1 执行机理分析

`Gather` 的性能瓶颈不能简单归结为“总字节数除以峰值带宽”。其执行过程至少包含两条性质不同的路径：其一，按照索引从 embedding table 中读取源行，这一过程存在明显的随机访问与 cache miss；其二，将读取结果顺序写入输出张量，这一过程更接近带宽受限的 copy。由于前者主要表现为延迟受限，后者主要表现为带宽受限，因此必须采用双机制联合建模。

#### 4.1.2 模型构建

设真实请求行数为 `R`，每行字节数为 `S_row`，则总流量可写为

`S_stream = S_src_read + S_dst_write + S_index_read`，

其中 `S_src_read = output_size`，`S_dst_write = output_size`，`S_index_read = 8 * R`。为估计源表访问所在的缓存层级，本文使用不同访问行数的 occupancy 近似，得到有效工作集 `S_ws`，并据此计算源访存延迟 `lat_src = lat(fit(S_ws))`。进一步地，若每行平均涉及 `CL_row` 个 cacheline，则源路径耗时表示为

`T_src = R * CL_row * lat_src / (T * m_gather)`，

其中 `m_gather` 表示可并发隐藏的 miss 数。对于结果写回及顺序流量路径，定义

`BW_gather_eff = BW_eff(S_row; BW_peak * rho_gather_inf, tau_gather_row_start)`，

则带宽路径耗时为

`T_bw = S_stream / BW_gather_eff`。

最终，`Gather` 的理论耗时定义为

`T_gather = max(T_src, T_bw, tau_floor)`，

其中 `tau_floor` 是随缓存层级增长的最小时延下界，用于避免超小行长样本被低估。

#### 4.1.3 合理性论证

上述模型的合理性在于同时保留了 `Gather` 的“随机读”与“顺序写”两种本质。若只保留带宽项，则会忽略索引驱动的随机访问延迟；若只保留延迟项，则会低估大输出张量回写时的持续带宽消耗。`m_gather` 的引入则刻画了多线程和 memory-level parallelism 对随机 miss 的隐藏能力，因此该模型能够更真实地反映不同线程数和不同表规模下 `Gather` 的性能变化趋势。

### 4.2 ReduceSum 算子

#### 4.2.1 执行机理分析

当前 DLRM 场景中的 `ReduceSum` 主要对应 `[B, nip, K] -> [B, K]` 的归约模式。该算子访问模式总体上是连续流式的，其主要困难不在于随机访存，而在于归约内循环会降低持续带宽效率，并引入逐元素加法操作。因此，`ReduceSum` 需要同时考虑带宽受限路径和加法吞吐路径。

#### 4.2.2 模型构建

设总流量为

`S_stream = activation_size + output_size`，

归约工作量为 `N_add = feat_reduction_work_items`，归约内层规模为 `I = feat_reduction_axes_product`。考虑到归约过程相较纯 copy 会产生额外的数据相关与 accumulator 维护成本，本文定义

`BW_reduce_eff = BW_eff(I; BW_peak * rho_copy_inf * kappa_reduce, tau_reduce_start)`，

其中 `kappa_reduce` 表示归约相对于 copy 基线的带宽折损比例。于是访存路径和计算路径分别为

`T_mem = S_stream / BW_reduce_eff`

`T_compute = N_add / PeakAdd(T)`。

最终理论总耗时定义为

`T_reduce = max(T_mem, T_compute)`。

#### 4.2.3 合理性论证

该模型保留了 `ReduceSum` 的两个核心约束。首先，归约并非纯 copy，因此需要使用 `kappa_reduce` 明确表示 steady-state 带宽退化；其次，随着归约轴乘积增大，逐元素加法数量也同步上升，因此必须引入 `PeakAdd(T)` 对算术吞吐进行约束。使用 `max(T_mem, T_compute)` 可以自然区分“访存瓶颈主导”和“加法吞吐主导”两种场景，符合该类算子的实际执行机理。

### 4.3 逐元素算子

#### 4.3.1 共同机理

`Relu`、`Add`、`Mul` 与 `Sigmoid` 都属于逐元素算子，整体上共享“流式读写 + 轻量逐元素计算”的基本形态。其中，`Relu` 主要表现为内存主导的单输入算子，`Add` 与 `Mul` 属于二元逐元素算子，除流量搬运外还存在固定结构性开销，而 `Sigmoid` 则进一步引入不可忽略的非线性计算路径。因此，这四类算子适合在统一框架下分别建模，而不必重复引入相同的访存解释。

#### 4.3.2 `Relu`

设总流量为 `S_stream = feat_io_bytes_sum`，则

`BW_relu_eff = BW_eff(S_stream; BW_peak * rho_relu_inf, tau_relu_start)`，

`T_relu = S_stream / BW_relu_eff`。

由于 `Relu` 的逐元素逻辑仅包含简单比较与选择，因此可直接视为流式访存主导算子，此时 `T_total = T_mem = T_relu`，`T_compute = 0`，`T_overhead = 0`。`rho_relu_inf` 描述稳态带宽上限，`tau_relu_start` 描述小张量或短流量样本的启动损失。

#### 4.3.3 `Add` 与 `Mul`

`Add` 与 `Mul` 均属于二元逐元素算子。设总流量仍记为 `S_stream = feat_io_bytes_sum`，则

`BW_add_eff = BW_eff(S_stream; BW_peak * rho_add_inf, tau_add_start)`，

`T_add = S_stream / BW_add_eff + tau_add`。

同理，

`BW_mul_eff = BW_eff(S_stream; BW_peak * rho_mul_inf, tau_mul_start)`，

`T_mul = S_stream / BW_mul_eff + tau_mul`。

其中，`rho_add_inf` 与 `rho_mul_inf` 分别表示二者在大流量稳态下的渐近有效带宽系数，`tau_add_start` 与 `tau_mul_start` 表示带宽路径的启动成本，`tau_add` 与 `tau_mul` 则用于吸收各自固定的结构性开销。与 `Relu` 相比，`Add` 和 `Mul` 除了数据搬运，还需要显式保留多输入合并、边界处理和调度管理带来的额外代价。

#### 4.3.4 `Sigmoid`

`Sigmoid` 同样是逐元素算子，但其非线性计算路径不可忽略。设总流量为 `S_stream = feat_io_bytes_sum`，输出元素个数为 `N_out`，则

`BW_sigmoid_eff = BW_eff(S_stream; BW_peak * rho_sigmoid_inf, tau_sigmoid_start)`，

`T_mem = S_stream / BW_sigmoid_eff`，

`PeakSigmoidEff(T) = PeakAdd(T) * rho_sigmoid_compute`，

`T_compute = N_out / PeakSigmoidEff(T)`，

`T_sigmoid = tau_sigmoid + max(T_mem, T_compute)`。

其中，`rho_sigmoid_inf` 用于描述稳态访存带宽折减，`tau_sigmoid_start` 用于描述带宽路径的启动损耗，`rho_sigmoid_compute` 用于将向量加法峰值吞吐缩放为有效非线性计算上限，`tau_sigmoid` 则用于刻画固定 kernel 开销。与 `Relu`、`Add`、`Mul` 不同，`Sigmoid` 的主导瓶颈可能在访存侧，也可能在计算侧，因此需要采用 `max(T_mem, T_compute)` 的联合刻画方式。

### 4.7 Gemm 算子

#### 4.7.1 执行机理分析

`Gemm` 是 DLRM 中典型的计算密集型算子，其主导成本通常来自矩阵乘加运算。然而，直接使用理想 roofline 模型会忽略一个关键事实：当矩阵维度不足以支撑 kernel tile 饱和时，实际持续 FMA 利用率会显著低于理论峰值。此外，packing、边界 tile 和并行分块不理想等因素也会进一步降低有效算术吞吐。

#### 4.7.2 模型构建

设矩阵维度为 `M`、`N`、`K`，则总浮点操作数为

`F = 2MNK`。

在带宽路径上，令输入、权重与输出总字节量为 `S_mem`，则

`T_mem = S_mem / BW_peak`。

在计算路径上，本文定义维度相关的有效利用率

`rho_fma_eff = rho_fma_inf * M / (M + M50) * N / (N + N50) * K / (K + K50)`，

从而得到

`T_compute = F / (PeakFMA(T) * rho_fma_eff)`。

最终，

`T_gemm = max(T_mem, T_compute)`。

#### 4.7.3 合理性论证

`M50`、`N50` 与 `K50` 的引入，本质上是在刻画不同维度接近“半饱和”时的效率退化规律。当某一维过小，kernel 难以充分填充微核或 tile，持续 FMA 利用率便会下降。该模型因此不是单纯拟合常数，而是在保留 roofline 主结构的基础上，补充了 shape saturation 对实际吞吐的影响。

### 4.8 MatMul 算子

#### 4.8.1 执行机理分析

当前 `/MatMul` 更接近大量 tiny batched matmul，而非标准大矩阵乘。其性能瓶颈并不完全由理论 FLOPs 决定，而更多取决于微核占用率不足、批量切分下的启动代价以及小维度导致的 sustained utilization 下降。因此，`MatMul` 不能简单视为缩小版 `Gemm`。

#### 4.8.2 模型构建

设 batch 数为 `B_mat`，矩阵维度为 `M`、`N`、`K`，则总浮点操作数为

`F = 2 * B_mat * M * N * K`。

本文定义 tiny-regime 下的有效利用率为

`rho_tiny_eff = rho_tiny_inf * min(M / occ_ref, 1) * min(N / occ_ref, 1) * K / (K + K50_tiny)`，

则计算主项为

`T_compute = F / (PeakFMA(T) * rho_tiny_eff)`。

考虑到每轮微批调度都需要额外的固定启动成本，定义

`T_overhead = ceil(B_mat / T) * tau_micro`。

最终总耗时为

`T_matmul = T_compute + T_overhead`。

#### 4.8.3 合理性论证

该模型的核心在于显式刻画 tiny matmul 的 occupancy 问题。`occ_ref` 对应微核接近饱和时的参考尺度，当 `M` 或 `N` 远小于该尺度时，实际利用率会快速下降。与此同时，批次级微核启动开销会在小 batch 或线程数较低时被放大，因此需要通过 `tau_micro` 单独建模。该处理能够更真实地反映 tiny batched matmul 与常规 `Gemm` 的机制差异。

### 4.9 Transpose 算子

#### 4.9.1 执行机理分析

`Transpose` 在总体数据量上接近 copy-like kernel，但其性能并不完全由总字节量决定。根本原因在于转置会改变数据访问顺序，使得读写路径中出现明显的 stride 访问，从而破坏局部性并降低线程扩展效率。因此，`Transpose` 的模型必须在“流式搬运主项”之外，额外引入 stride penalty。

#### 4.9.2 模型构建

设输出字节量为 `S_out`，前缀块数量为 `N_prefix`，单个连续后缀块的访存延迟记为 `lat_stride`。则 copy 主项写为

`T_copy = 2 * S_out / (BW_peak * rho_copy_inf)`。

同时定义 stride penalty 为

`T_stride = N_prefix * lat_stride / (T ^ eta_stride * m_stride)`。

于是 `Transpose` 的理论总耗时为

`T_transpose = T_copy + T_stride`。

其中，`lat_stride` 由后缀块工作集的 `fit -> lat` 路径得到，`m_stride` 用于描述可并发隐藏的 stride miss 数，`eta_stride` 则描述 stride penalty 随线程数增长时的有效缩放程度。

#### 4.9.3 合理性论证

若只保留 copy 项，则模型无法解释不同张量布局和不同线程数下的显著时延波动；而若只保留 stride 项，则又会忽略大规模读写搬运的主体成本。将二者相加可以自然对应 `Transpose` 的双重本质：既要完成完整的数据搬运，又要为非连续访问模式支付额外延迟代价。因此，该模型能更准确地反映转置算子在 locality 和并发扩展性方面的损失。

### 4.10 Concat 算子

#### 4.10.1 执行机理分析

`Concat` 本质上是多路输入沿目标维度顺序拼接的过程，其主要代价并非算术计算，而是多块数据的连续搬运以及每路输入对应的 dispatch、offset 更新和边界处理。特别是在单个输入块较小时，前端发射能力和 cacheline 粒度的 copy loop 效率可能成为更紧的约束。

#### 4.10.2 模型构建

设总流量为

`S_stream = input_bytes_sum + output_size`，

平均块大小为 `S_chunk`，输入路数为 `N_in`。定义

`BW_copy_eff = BW_eff(S_chunk; BW_peak * rho_copy_inf, tau_copy_start)`，

则纯流量项为

`T_stream = S_stream / BW_copy_eff`。

进一步考虑 cacheline 粒度的发射上界，可得

`T_issue = (S_stream / cacheline) / IssueSlots(T)`。

最终，`Concat` 的理论总耗时写为

`T_concat = max(T_stream, T_issue) + N_in * tau_dispatch`。

#### 4.10.3 合理性论证

该模型同时考虑了 `Concat` 的三类主要成本。首先，`T_stream` 刻画了大块 copy 的持续带宽需求；其次，`T_issue` 用于约束小块场景下 cacheline 粒度的发射能力；最后，`N_in * tau_dispatch` 则显式表达多输入拼接带来的结构性管理开销。三者结合后，模型能够区分“数据搬运慢”和“输入路数多导致调度慢”这两种常见情况，因此更适合作为 `layout_move` 类算子的解析表达。

## 5. 本节小结

综上，本文并未采用单一经验公式统一描述所有算子，而是在共享硬件子模型的基础上，针对不同算子的主导执行机制分别建立了解析模型。对于 `Gather`、`Transpose` 与 `Concat` 等访存模式复杂的算子，模型重点刻画缓存层级、带宽饱和与 stride 或 dispatch 等结构性惩罚；对于 `ReduceSum` 与 `Sigmoid` 等 mixed-balanced 算子，模型同时保留访存路径与计算路径；对于 `Gemm` 与 `MatMul` 等计算密集型算子，模型则强调维度饱和、微核占用率和启动开销的影响。由此得到的 `T_total`、`T_mem`、`T_compute` 与 `T_overhead` 不仅能够作为 MLP 的软硬件融合输入特征，也为后续误差分析和机理解释提供了统一、可解释的分析框架。
