# 第四章实验结果

## 4.1 实验平台与数据采集方法

- 单算子统一实验目录：`/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/classed_op_mlp_test_78910_analytical_5_300_iter_quick_nodrop`
- 整图静态聚合目录：`/data/qc/dlrm/ORT/static_pipeline_eval/artifacts/latest/v1_300_iter_quick_nodrop`
- 基线目录：`/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/model_all_no_trace`
- 统一输出目录：`/data/qc/dlrm/ORT/static_pipeline_eval/artifacts/latest/chapter4_cpu`

表 4-1 记录了平台与数据规模概览，图 4-1 对应统一统计视图。

## 4.2 算子级性能建模实验

- 表 4-2 汇总五个模型组的测试指标与训练口径。
- 表 4-3 展示代表算子 `Gather / ReduceSum / Transpose / Concat / Gemm` 的测试误差统计。
- 图 4-2 到图 4-8 展示模型组误差、baseline 对比、代表算子图、散点与训练曲线。

- 章节内单算子核心结果摘要：group mean MAPE = 0.082556, worst group MAPE = 0.116650.

### 4.2.1 OOD 泛化

- 章节 OOD 规则采用 batch holdout `1856|1920|1984|2016|2048` 与 `num_threads=3`。
- OOD 切片摘要写入单独的 CSV/JSON 文件，图 4-9 到图 4-10 展示 slice 与 generalization 参考结果。
- 当前已写入 2 条 OOD slice 记录。

## 4.3 整图性能聚合实验

- 表 4-5 汇总静态调度器在 `v1_300_iter_quick_nodrop` 上的主指标。
- 图 4-11 到图 4-13 给出预测-真实散点、batch/inter_threads 分组热力图与误差分布。
- 图 4-14 与图 4-15 来自典型时间线与关键路径导出。

- E2E 结果摘要：full_graph MAPE = 0.063821, worst combo = case_8_1_1 / bs2048_nip2000.

## 4.4 消融实验与误差分析

- 表 4-6 汇总简单求和基线与静态调度器的差异。
- 表 4-7 汇总单算子特征消融实验结果。
- 图 4-16、图 4-17、图 4-18、图 4-19 分别对应特征消融热图、增量误差、求和基线与最佳消融结果。

- 总体消融样本数：34。

## 4.5 本章小结

- 关键时间线案例数：3。
- 统一 figure catalog 共收录 19 张图。
- 本章草稿由 `chapter4_cpu_experiments_draft.md` 自动生成，并可由总入口重复刷新。
