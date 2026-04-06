# 第四章 CPU 实验与结果分析

## 4.1 实验平台与数据采集方法

本章实验围绕 ORT 上的 DLRM CPU 推理路径展开，目标是验证第三章提出的两级建模思路，即先对单算子时延进行静态预测，再将节点级预测结果通过静态流水线模型聚合为整图时延。所有第四章脚本统一维护在 `ORT/static_pipeline_eval/chapter4_experiments/`，最终产物写入 `ORT/static_pipeline_eval/artifacts/latest/chapter4_cpu/`。

### 4.1.1 实验平台

实验服务器为 Kunpeng-920 主机。整机从 `lscpu` 可观测到 4 路 CPU、192 个物理核心、单核单线程；但本章建模与运行口径采用单 NUMA 执行域，对应 24 个核心、24 MiB 共享 L3 以及 4 个 DDR4-2933 内存通道。单核缓存参数采用 `64 KiB L1I + 64 KiB L1D + 512 KiB L2`，本地 NUMA 理论带宽按 `100 GB/s` 建模，操作系统为 `Huawei Cloud EulerOS 2.0 (aarch64)`，编译器为 `GCC 10.3.1`。实验软件环境来自 `ort` conda 环境：`Python 3.11.14`、`PyTorch 2.9.0`、`ONNX 1.19.1`、`ONNX Runtime CANN 1.23.2`、`NumPy 1.26.4`、`pandas 3.0.1`。单算子标签采集依赖 DynamoRIO 侧的 profiling 结果，整图真值来自 ORT branch-parallel 运行的时间线记录。为减小系统噪声，实验固定 `intra-op` 与 `inter-op` 线程配置，并按单 NUMA 口径组织样本和静态流水线排程。

### 4.1.2 DLRM 负载与算子样本生成

单算子样本来自 ORT 导出的 DLRM ONNX 图及其配套 `op_shapes`/profile 数据。统一数据集共包含 195,900 条样本、9 个 case、3,265 个 `case_id-combo` 配置，覆盖 `1024-2048` 的 batch size、`1000-2000` 的每次 lookup 索引数、`1-5` 的 intra-op 线程和 `1/3/4/5/6` 的 inter-op 线程。算子类型共 14 种，分别映射为五类：索引访存类、布局搬移类、视图/元数据类、轻计算-访存混合类和计算主导类。每个 ONNX 节点都保留输入/输出 shape、线程配置、I/O 字节规模、归约规模、Gemm 的 M/N/K 与 analytical proxy 列，用于后续单算子建模。

### 4.1.3 运行与标注流程

单算子标签来自 ORT 执行后的 per-op profiling 结果。每个样本先丢弃最早的一个 profile batch，再对剩余 batch 的算子时延求均值，形成 `label_operator_actual_dur_us`。为了恢复整图真值，本文使用 branch-parallel runner 导出的 `branch_parallel_op_timeline.csv`，对每个 batch 计算整图 span，并采用与单算子一致的策略丢弃首个 batch 后再求均值。静态整图模型并不是简单求和，而是根据 `op_shapes` 重建 DAG，将 8 条 embedding 分支折叠为 branch task，再依据 `inter_threads` 控制的并行槽位进行离线排程，从而生成预测时间线、关键路径和整图时延。

### 4.1.4 数据集划分与评价指标

单算子数据按 `sample_group=combo` 做 `7:2:1` 划分，实际样本数为 train/val/test = `137,160 / 38,880 / 19,860`。这种切分方式保证同一 `case_id-combo` 下的节点不会同时出现在训练集与测试集，避免 shape 级信息泄漏。除随机划分外，本文还构造了未见 shape 与未见线程数两种外推测试：前者将 batch size `1856|1920|1984|2016|2048` 作为保留配置，后者仅用 `num_threads=3` 做测试。单算子评价指标采用 `MAE`、`MAPE`、`RMSE` 和 `R^2`；整图评价指标采用 `MAE`、`MAPE`、`P50/P90` 相对误差，并按 batch size 和 branch parallelism 分组统计。

## 4.2 算子级性能建模实验

本节首先给出 fair single MLP 在测试集上的总体精度，再按五类算子统计误差，并对 Gather、ReduceSum、Transpose/Concat 以及 Gemm/MatMul 的代表性行为做可视化分析。整体作图风格参考 Concorde 中“真实值-预测值关系、误差分布、典型 case 解释”的组织方式，但在本组实验里不再引入 grouped MLP 对照。

### 4.2.1 单算子总体预测精度

表 4-3 给出了两种单算子模型口径的总体结果：纯解析模型，以及同数据同特征池重跑的 fair single MLP。当前 single MLP 使用与 grouped 特征池并集一致的 30 个输入特征，在测试集上的 `MAPE` 为 0.1190，`R^2` 为 0.9706。纯解析模型按 analysis 套件一致的口径，在每个算子组上选取最佳 active analytical proxy 后，在 13,902 条可覆盖测试样本上得到 `MAPE = 0.2364`。此前若把 `ana_calib_total_us` 直接施加到全部测试样本，会把缺少 active analytical feature 的 `view_meta` 组也纳入比例误差，从而显著夸大 pure analytical 的总体误差；当前这部分覆盖率约为 70.0%。图 4-3 的散点结果显示，统一 single MLP 的大部分样本仍围绕 `y=x` 参考线分布，说明其作为后续整图聚合输入是稳定可用的。

### 4.2.2 分类别预测精度

表 4-4 和图 4-4 展示了五类算子的误差差异。总体上，视图/元数据类和布局搬移类更容易预测，因为其执行路径较稳定；索引访存类和轻计算-访存混合类误差相对更大，主要原因是它们更容易受到随机访存、线程调度和小张量固定开销的影响。这种类别差异与第三章中对 ORT CPU kernel 机制的分析是一致的，也说明即使统一 single MLP 使用同一特征池，不同算子机理仍然会在误差分布上留下明显结构。

### 4.2.3 典型算子预测结果分析

图 4-5 至图 4-8 分别给出了 Gather、ReduceSum、Transpose/Concat 以及 Gemm/MatMul 的代表性结果。Gather 的误差主要受到随机表访存影响，线程数变化会显著改变尾部误差；ReduceSum 的误差随归约工作量增长而逐步收敛，说明解析代理对归约规模具有较好刻画能力；Transpose 与 Concat 的误差主要受数据搬移规模影响；Gemm/MatMul 在 MAC 数较小时误差增大，反映出小维度下微核利用率不足的问题。整体来看，这些现象与第三章关于 cache fit、数据搬移和 kernel 饱和度的解释是相互印证的。

### 4.2.4 跨规模泛化实验

图 4-9 和图 4-10 分别给出未见 shape 与未见线程数的测试结果。未见 shape 配置的 `MAPE` 为 0.1172，未见线程数配置的 `MAPE` 为 0.1035。这说明解析代理特征在 shape 外推上提供了稳定支撑，而线程数外推的难度更高，因为线程切分会同时影响并行粒度、调度开销和实际 cache 行为。

## 4.3 整图性能聚合实验

整图实验验证第三章 3.3 中的静态流水线聚合模型。与单纯求和不同，该模型显式考虑 bottom、8 条 embedding branch 和 tail 的依赖关系，以及 `inter_threads` 决定的 branch 并行槽位约束，因此能够用节点级预测结果恢复整图执行的关键路径。

### 4.3.1 整图预测总体精度

表 4-5 汇总了整图预测精度。静态流水线模型在完整图样本上的 `MAPE` 为 0.0967，`P50` 和 `P90` 相对误差分别为 0.0893 和 0.1788。图 4-11 的散点结果表明，绝大多数整图配置都能被稳定地落在参考线附近，说明节点级误差在流水线聚合后没有被系统性放大。

### 4.3.2 不同 batch size 下的整图精度

图 4-12 给出了不同 batch size 下的整图 `MAPE`。整体趋势较平滑，说明模型在小 batch、中 batch 和大 batch 区间都保持了较稳定的误差水平。这一点非常重要，因为 DLRM 的 embedding lookup 和 top MLP 都会随 batch size 改变张量规模与工作集，若聚合模型缺乏稳定性，误差会在 batch 变化时明显抖动。

### 4.3.3 不同分支并行度下的整图精度

图 4-13 按 `inter_threads` 展示了真实整图时延与预测整图时延的变化。可以看到，随着可用 branch 槽位增加，整图时延整体下降，但收益逐渐递减；预测曲线能够较好跟随这一趋势。这说明第三章提出的 `kappa` 槽位近似虽然是静态模型，但已经能够反映并行度增加后边际收益递减的核心行为。

### 4.3.4 典型时间线案例分析

图 4-14 和图 4-15 展示了 3 个典型配置的真实/预测时间线与关键路径分解。通过这些案例可以看到，模型不仅预测了整图 makespan，还较准确地恢复了 bottom、embedding pool 和 tail 之间的先后关系。误差主要集中在分支完成时刻附近的同步边界和少数高波动节点上，这为后续误差分析提供了直接证据。

## 4.4 消融实验与误差分析

这一节保留与主实验一致的逐步加组件消融思路，但在新构造的 single-only 版本里去掉 grouped MLP，仅比较四个变体：`Analytical model + Simple add`、`Analytical model + pipeline`、`single MLP + Simple add` 和 `single MLP + pipeline`。这样的设计主要回答三个问题：解析模型本身能做多好、仅将整图聚合从 simple add 换成 pipeline 能带来多少收益，以及统一 single MLP 接入静态流水线后能把整图误差进一步压到什么水平。

### 4.4.1 公平对比消融结果

表 4-6、图 4-16、图 4-17 和图 4-18 共同展示了四个变体的差异。对纯解析模型而言，仅把整图聚合从 simple add 改为 pipeline 后，整图 `MAPE` 由 2.0450 下降到 0.1389，说明静态调度本身就能纠正简单求和对并行重叠的系统性高估。进一步引入统一 single MLP 后，在本组 single-only 实验里，`single MLP + simple add` 的整图 `MAPE` 为 1.8644，而 `single MLP + pipeline` 下降到 0.0967。这说明在不引入 grouped MLP 的前提下，静态流水线聚合仍然是整图精度提升的关键来源。

### 4.4.2 误差来源分析

表 4-7 和图 4-19 汇总了四类代表性误差样本。第一类是大表随机访存下的 Gather，误差主要来自实际内存访问时延波动；第二类是小张量视图/元数据算子，固定框架开销占比过高导致样本噪声明显；第三类是小维度 Gemm/MatMul，微核未充分饱和时更难被解析代理拟合；第四类则是整图级同步场景，单个 branch 的偏差在 barrier 处被放大。这些现象说明，当前模型的主要剩余误差并不来自统计偶然，而是来自 ORT CPU 执行中仍难以静态精确恢复的动态效应。

## 4.5 本章小结

本章首先在 195,900 条单算子样本和 331 个完整图配置上完成了 single-only 版本实验。结果表明，统一 single MLP 在随机切分单算子测试上的 `MAPE` 为 0.1190`，显著优于纯解析模型；进一步把 same-split single MLP 接入同一静态流水线后，其整图 `MAPE` 为 0.0967`，相较 `single MLP + simple add` 的 1.8644` 进一步下降。新构造的 single-only 实验说明：即使完全去掉 grouped MLP，解析代理、统一 single MLP 与静态流水线聚合这条两级建模链路依然成立，并且整图层面的主要收益仍来自 pipeline 对并行重叠和同步边界的显式建模。

本章共生成 19 张图，全部由对应 Chapter 4 runner 自动复现，并写入 `/data/qc/dlrm/ORT/static_pipeline_eval/chapter4_cpu_single_only_experiments_draft.md`。
