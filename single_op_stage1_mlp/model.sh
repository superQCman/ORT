# !/bin/bash

set -e

python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/train_mlp.py --data-dir /data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/dataset_all_clean_8 --output-dir /data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/model_all_clean_8 --train-device npu --npu-device-id 0 --hidden-layers "128,128,128,64,64,64"