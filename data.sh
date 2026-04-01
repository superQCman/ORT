# !/bin/bash

set -e 

python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py --selected-cases case_1_1_1 case_1_2_1 case_1_4_1 case_2_1_1 case_2_2_1 case_2_4_1 case_1_1_4 case_1_2_4 case_1_3_4 case_1_4_4 case_2_1_4 case_2_2_4 case_2_3_4 case_2_4_4 --output-dir=./artifacts/latest/dataset_all_clean_8 --profile-instability-threshold 0.8