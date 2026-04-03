# 面向 ORT DLRM Branch-Parallel 执行的静态流水线排程方法

## 摘要

本文给出一种面向 ORT DLRM branch-parallel 执行的静态流水线排程方法，用于将已有单算子时延预测结果提升为整图级 end-to-end 时延估计。该方法复用 `single_op_stage1_mlp` 已输出的 per-op 预测值，从 `op_shapes` 重建 combo 级 DAG，将 8 条 embedding branch 折叠为复合任务，并引入由 `inter_threads` 控制的 FIFO 槽位调度机制，进而构建整图静态时间线。随后，方法从 `branch_parallel_op_timeline.csv` 中提取真实整图 span，并与预测结果进行误差对比。本文系统描述了问题定义、图重建策略、branch 折叠规则、排程算法、真值提取口径以及误差热点，为后续黑盒校准模型提供结构化基础。

## 1. 引言

单算子性能建模是整图时延预测的基础，但并不足以直接给出高质量的 end-to-end 估计。即使单个算子的预测误差较小，整图误差仍然可能主要来自以下因素：

- branch 启动顺序
- 并发槽位竞争
- barrier 同步行为
- join 之后的短尾段累计误差

在 ORT DLRM 的 branch-parallel 执行中，embedding 部分尤其关键。其根本原因在于：8 条 embedding lookup branch 并不是无约束地同时发起，而是受到 `inter_threads` 控制的并发上限约束。因此，一个整图级预测器不能仅对所有 op 的预测时长做简单求和，而必须显式建模图拓扑与 branch 排队规则。

基于此，本文提出一个一阶静态排程器 `v1`。该方法强调三个特点：

1. 结构上可解释
2. 规则上可复现
3. 与 branch-parallel runner 的实际执行语义保持一致

## 2. 问题定义

对于任意一个 combo 样本 `c`，假设已知如下输入：

1. 单算子预测集合  
   \[
   P_c = \{(v_i, \hat{t}_i)\}
   \]
   其中 `v_i` 表示图中的算子节点，`\hat{t}_i` 表示对应的预测时长。

2. 来自 `op_shapes` 的图结构信息

3. combo 对应的静态变量  
   \[
   x_c = (\text{batch\_size}, \text{num\_indices\_per\_lookup}, \text{inter\_threads}, \text{case\_id})
   \]

4. 真实执行时间线  
   \[
   T_c
   \]

目标是构造一个整图级静态排程函数：

\[
\hat{M}_c = f(P_c, G_c, x_c)
\]

其中 `G_c` 是由 `op_shapes` 重建得到的 combo 级 DAG，`\hat{M}_c` 是预测 makespan。进一步，将其与真实整图 span `M_c` 比较，并计算误差：

\[
\mathrm{AE}_c = |\hat{M}_c - M_c|
\]

\[
\mathrm{APE}_c = \frac{|\hat{M}_c - M_c|}{M_c}
\]

## 3. 数据来源与图结构恢复

### 3.1 单算子预测结果

静态排程器并不重新训练模型，而是直接复用 `single_op_stage1_mlp` 的已有输出。默认使用的文件为：

- `models/combined/combined_predictions_test.csv`

该文件中每一行给出一个 op 的：

- `case_id`
- `combo`
- `op_idx`
- `pred_us`
- `target_us`

因此，静态排程器的重点不是重新估计单个算子时长，而是如何把这些时长排进一条符合执行语义的整图时间线。

### 3.2 从 `op_shapes` 重建 DAG

`op_shapes_*.csv` 不是显式的节点表，而是按 tensor edge 展开的结构化描述表。为获得可调度的图结构，本文采用如下恢复流程：

1. 按以下字段去重得到节点集合：
   - `node_idx`
   - `node_name`
   - `op_type`
2. 遍历每个输入 tensor 行，读取其 `producer_node`
3. 由 `producer_node -> consumer_node` 构建前驱边
4. 在调度图中忽略以下非计算型来源：
   - `initializer`
   - `graph_input`
   - `Constant`

最终得到一个非 Constant 的有向无环图：

\[
G_c = (V_c, E_c)
\]

该图用于表示整图的静态依赖关系。

### 3.3 整图真值提取

真实整图时延从以下文件中提取：

- `branch_parallel_op_timeline.csv`

给定某个节点集合 `S`，定义 batch `b` 上的图 span 为：

\[
\mathrm{span}(b, S) =
\max_{v \in S} \mathrm{end}(b, v) -
\min_{v \in S} \mathrm{start}(b, v)
\]

为了与 `single_op_stage1_mlp` 的标签策略保持一致，本文采用以下真值协议：

1. 对每个 batch 计算图 span
2. 丢弃最早的一个 batch
3. 对剩余 batch 的 span 求均值

记保留后的 batch 集合为 `B_c'`，则：

\[
M_c(S) = \frac{1}{|B_c'|} \sum_{b \in B_c'} \mathrm{span}(b, S)
\]

## 4. 静态排程假设

`v1` 排程器依赖以下已从真实 branch-parallel timeline 中验证的执行假设。

### 4.1 Embedding 启动顺序固定

8 条 embedding branch 的启动顺序满足固定 FIFO 规律：

\[
0 \rightarrow 1 \rightarrow 2 \rightarrow 3 \rightarrow 4 \rightarrow 5 \rightarrow 6 \rightarrow 7
\]

即，branch 不会在启动阶段发生乱序。

### 4.2 `inter_threads` 决定 branch 并发槽位数

`inter_threads` 不是普通附加特征，而是 branch 级排程规则中的核心变量。它等价于 embedding 分支的最大并发槽位数：

\[
K = \mathrm{inter\_threads}
\]

当 `K=4` 时，表示最多允许 4 条 embedding branch 同时处于活动状态。

### 4.3 槽位占用覆盖整个 branch 生命周期

对于第 `j` 条 embedding branch，其槽位占用起点为：

- `/emb_lj/Gather`

终点为：

- `/emb_lj/ReduceSum`

这意味着下列 3 个算子必须被视为一个整体调度单元：

- `Gather`
- `Reshape`
- `ReduceSum`

### 4.4 Tail 受 barrier 约束

`tail` 不能与 embedding 分支自由交叠。本文将其建模为一个 barrier 之后的独立阶段，即：

- `bottom` 完成
- 所有 embedding branch 完成

上述两个条件同时满足后，`tail` 才能启动。`tail` 内部仍保持 `op_shapes` 中恢复出的真实 DAG 依赖。

## 5. Embedding Branch 折叠策略

对每条 embedding branch `j`，定义其复合任务：

\[
B_j = \{\mathrm{Gather}_j, \mathrm{Reshape}_j, \mathrm{ReduceSum}_j\}
\]

其预测时长定义为：

\[
\hat{t}(B_j) = \sum_{v \in B_j} \hat{t}_v
\]

进行 branch 折叠有两个直接好处：

1. 排程粒度从 node-level 提升到 branch-level，更贴近真实执行语义。
2. 槽位竞争可以直接建模为 branch 竞争，而不是用多个微小算子间接逼近。

除 embedding 分支外，其余节点继续保留为 node-level 任务。

## 6. 静态流水线排程算法

### 6.1 任务集合定义

对一个 combo 样本，最终调度任务集合定义为：

\[
\mathcal{T}_c =
\mathcal{T}_{bottom} \cup
\mathcal{T}_{branch} \cup
\mathcal{T}_{tail} \cup
\{\mathrm{barrier}_{tail}\}
\]

其中：

- `bottom` 为 node-level 任务
- `branch` 为 8 个 embedding 复合 branch 任务
- `tail` 为 node-level 任务
- `barrier_tail` 用于连接 `bottom + branch pool` 与 `tail`

### 6.2 非 branch 任务的开始时间

对任意任务 `u`，定义其依赖满足时间为：

\[
r(u) = \max_{p \in \mathrm{pred}(u)} \mathrm{end}(p)
\]

对于 `bottom` 与 `tail` 中的普通 node-level 任务，开始时间直接取：

\[
\mathrm{start}(u) = r(u)
\]

### 6.3 Branch 任务的开始时间

对于 embedding branch 任务 `B_j`，除依赖满足外，还必须等待槽位释放。记当前最早可用槽位释放时间为 `s_{\min}`，则：

\[
\mathrm{start}(B_j) = \max(r(B_j), s_{\min})
\]

所有 branch 按固定 FIFO 顺序 `0 -> 7` 被送入调度器。当可用槽位数少于 `K` 时，新 branch 可以立即进入；否则必须等待最早释放的那个槽位。

### 6.4 结束时间与 makespan

每个任务的结束时间定义为：

\[
\mathrm{end}(u) = \mathrm{start}(u) + \hat{t}(u)
\]

整图预测 makespan 则为：

\[
\hat{M}_c = \max_{u \in \mathcal{T}_c} \mathrm{end}(u)
\]

### 6.5 伪代码

```text
输入: DAG G_c, 任务预测时长, inter_threads = K
输出: 预测 makespan \hat{M}_c

1. 将每条 embedding branch 折叠为一个复合任务 B_j
2. 构建任务图:
   - bottom 任务
   - branch 任务 B_0 ... B_7
   - barrier_tail
   - tail 任务
3. 对任务图做拓扑遍历
4. 对非 branch 任务:
   start = 所有前驱 end 的最大值
5. 对 branch 任务, 按 FIFO 顺序处理:
   若当前活动槽位数 < K:
       start = ready_time
   否则:
       start = max(ready_time, 最早槽位释放时间)
   并保持该槽位直到 branch 的 ReduceSum 完成
6. 当 bottom 和全部 branch 完成后, 释放 barrier_tail
7. 按 DAG ready time 调度 tail
8. 返回所有任务 end 的最大值
```

## 7. 覆盖率定义

为了避免将不同数据质量条件下的样本混在一起，本文区分两种 coverage regime。

### 7.1 Full Graph

若某个 combo 的 60 个已建模非 Constant 节点全部存在于预测结果中，则将其定义为 full-graph combo。

### 7.2 Partial Graph

若原始 sweep/profile 存在，但部分节点因为上游标签稳定性过滤而从预测结果中缺失，则该 combo 被定义为 partial combo。

在这种情况下，评估器只对已观测子图做诊断性误差分析，而不将其作为整图 E2E 指标纳入主统计。

因此，最终报告必须分为：

- full-graph E2E 指标
- partial observed-subgraph 指标

二者不能混合汇总。

## 8. 误差结构分析

`v1` 的目标是先把结构骨架建对，而不是一次性解决所有高阶误差。因此，其残差本身提供了后续黑盒校准的方向。

### 8.1 Branch 槽位交接空隙

真实 timeline 显示，排队 branch 的 `Gather` 启动时间通常晚于前一条 branch 的 `ReduceSum` 结束时间，即存在稳定但非零的 handoff gap。该 gap 不天然属于某个单独 op，因此适合作为 branch 释放开销的加性校准项。

### 8.2 Embedding 复合 Branch 残差

最差样本的主误差往往集中在 `Gather + ReduceSum`，说明 embedding branch 更适合作为复合任务进行校准，而不是把误差拆散到多个微小算子上。

### 8.3 Join 后微尾段 Bundle

`Shape_1/Gather_9/.../Concat_4` 这一段由多个极短算子组成，单个 op 的误差看起来不大，但它们在图级时间线中会形成 bundle 型残差，因此适合做 bundle correction。

### 8.4 Top MLP 尾段波动

在启用标签稳定性过滤的 artifact 中，`top_l` 末段容易被删点。这说明该区域不仅存在建模误差，也存在更强的标签波动问题。

## 9. 实验观察

已有两个 artifact 可以说明过滤策略对调度评估的影响。

### 9.1 含过滤的 Artifact

对于 `classed_op_mlp_test_78910_analytical_5_200_iter_quick`：

- total test combos: `331`
- full combos: `49`
- partial combos: `282`
- full-graph MAPE: `0.041985`

### 9.2 不做 Drop 的 Artifact

对于 `classed_op_mlp_test_78910_analytical_5_300_iter_quick_nodrop`：

- total test combos: `331`
- full combos: `331`
- partial combos: `0`
- full-graph MAPE: `0.063821`

这一对比说明：过滤异常数据会提高剩余 full-graph 样本上的表观精度，而不做 drop 则能暴露更完整但更困难的整图预测问题。

## 10. 方法局限性

当前 `v1` 排程器尚未显式建模以下行为：

- runtime 乱序调度
- work stealing
- cache 干扰的动态状态
- 带宽竞争的时间变化过程
- branch release gap 的学习型修正

因此，`v1` 更适合作为结构化基线方法，而不是最终版的 fully calibrated end-to-end predictor。

## 11. 结论

本文提出了一种面向 ORT DLRM branch-parallel 执行的静态流水线排程方法。该方法的核心思想是：不再把 embedding lookup 看作分散的 node-level 小算子，而是提升为受 `inter_threads` 限制的 branch-level 复合任务，并通过固定 FIFO + 槽位竞争规则来构造整图静态时间线。该方法在保留结构可解释性的同时，将单算子预测自然提升为整图级时延估计，并为后续黑盒校准提供了清晰、可定位的误差分解基础。
