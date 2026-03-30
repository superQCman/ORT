# Feature Ablation

这个目录提供一个独立、可复用的特征消融入口，用来在现有数据集和现有 MLP 训练器之上快速验证：

- 某个类别模型依赖哪些 numeric feature
- 删掉某个特征后验证/测试集会退化多少
- 多个候选特征联合删除后，整体模型是否明显变差

当前实现优先面向 `classed_op_mlp` 的每个 `model_group`，但接口本身也支持直接传任意包含：

- `train.csv`
- `val.csv`
- `test.csv`
- `feature_columns.json`

的数据目录。

## 默认行为

当传入多个 `--ablation-feature` 时，脚本默认会自动生成：

- `baseline`
- `drop_<feature>`
- `drop_all_selected`

也就是：

- 原始模型
- 每次只删一个候选特征
- 一次性删除全部候选特征

如果后续你要研究别的类别模型或别的特征，只需要换：

- `--source-experiment-root`
- `--model-group`
- `--ablation-feature`

如果默认变体不够，还可以通过 `--variant name=feat_a,feat_b` 明确指定自定义组合。

## 推荐用法

使用现有 `classed_op_mlp` 产物作为输入，直接对某个组做消融：

```bash
conda run -n ort python /data/qc/dlrm/ORT/single_op_stage1_mlp/feature_ablation/run_feature_ablation.py \
  --source-experiment-root /data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/classed_op_mlp_test_2_analytical_5_200_iter \
  --model-group gather \
  --ablation-feature feat_output_elements_per_batch \
  --ablation-feature feat_output_elements_per_lookup \
  --ablation-feature feat_output_input_bytes_ratio \
  --train-device auto
```

如果你已经有某个数据目录，不想走 `classed_op_mlp` 的 root 解析，也可以直接传：

```bash
conda run -n ort python /data/qc/dlrm/ORT/single_op_stage1_mlp/feature_ablation/run_feature_ablation.py \
  --data-dir /path/to/dataset_dir \
  --baseline-model-dir /path/to/baseline_model_dir \
  --model-group gather \
  --ablation-feature feat_a \
  --ablation-feature feat_b
```

## 输出内容

默认输出到：

- `/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/feature_ablation/<source>/<model_group>`

主要文件：

- `ablation_summary.csv`
  - 每个 variant 的 val/test MAE、RMSE、MAPE、R2，以及相对 baseline 的 delta
- `ablation_summary.json`
  - 机器可读的完整配置和汇总
- `ablation_summary.md`
  - 方便快速查看的文本摘要
- `variants/<variant>/data`
  - 为该变体生成的 dataset manifest
- `variants/<variant>/model`
  - 训练产物与 prediction CSV

## 结果口径

每个变体除了直接记录 split-level 指标，还会额外和 baseline 做逐行配对比较，输出：

- `mean_abs_error_us_delta`
- `mean_ape_delta`
- `improved_row_fraction`
- `worsened_row_fraction`

其中：

- delta 为 `variant - baseline`
- 所以对误差类指标来说，负值代表删掉该特征后误差变小，正值代表删掉该特征后误差变大
- `improved_row_fraction` 表示删掉该特征后，逐样本绝对误差比 baseline 更小的比例

## 训练配置继承

如果提供的 `baseline-model-dir/metrics.json` 存在，脚本会默认继承其中的：

- `hidden_layers`
- `batch_size`
- `max_iter`
- `alpha`
- `learning_rate_init`
- `seed`
- `log_target_requested`

这样可以尽量对齐已有基线模型。

默认会复用已有 baseline 模型，只训练真正需要消融的变体；如需强制重训 baseline，可加：

```bash
--disable-reuse-source-baseline
```
