# Inter-Threads Eval

这个目录固定用于复现 `classed_op_mlp` 中把静态 `inter_threads` 特征加入分组 MLP 之后的对照实验。

当前默认基准是：

- [classed_op_mlp_test_2_analytical_5_200_iter](/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/classed_op_mlp_test_2_analytical_5_200_iter)

默认测试组是：

- `gather`
- `mixed_balanced`

## 做的事情

脚本会按下面流程自动执行：

1. 读取 baseline experiment 的 `dataset_summary.json`
2. 用当前代码重新构建 grouped dataset
3. 对指定 `model_group` 继承 baseline 的训练超参重新训练
4. 输出 baseline vs new 的指标对比

默认继承的训练配置包括：

- `hidden_layers`
- `batch_size`
- `max_iter`
- `alpha`
- `learning_rate_init`
- `seed`
- `log_target`
- `target_mode`

## 一键复现

```bash
/data/qc/anaconda3/envs/ort/bin/python \
  /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/inter_threads_eval/run_inter_threads_eval.py
```

默认输出目录：

```text
/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/inter_threads_eval/classed_op_mlp_test_2_analytical_5_200_iter
```

## 只跑某一个组

```bash
/data/qc/anaconda3/envs/ort/bin/python \
  /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/inter_threads_eval/run_inter_threads_eval.py \
  --model-group gather
```

或者：

```bash
/data/qc/anaconda3/envs/ort/bin/python \
  /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/inter_threads_eval/run_inter_threads_eval.py \
  --model-group mixed_balanced
```

## 快速 smoke

如果只是验证流程，可以暂时把训练轮数压低：

```bash
/data/qc/anaconda3/envs/ort/bin/python \
  /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/inter_threads_eval/run_inter_threads_eval.py \
  --model-group gather \
  --max-iter-override 2 \
  --disable-onnx-export
```

## 主要输出

- `datasets_rebuilt/`
  - 当前代码重新导出的 grouped datasets
- `models/<model_group>/`
  - 重新训练后的模型与预测
- `comparison/<model_group>_metric_comparison.csv`
  - 各 split / 各指标对比表
- `comparison/<model_group>_summary.json`
  - 单组完整对比摘要
- `suite_summary.json`
  - 整体实验摘要
