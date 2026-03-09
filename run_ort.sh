# !/bin/bash

set -e

source /data/qc/Ascend/ascend-toolkit/set_env.sh

# numactl 将进程的 CPU 执行和内存分配都限制在 node 0（cores 0-23）。
# 背景：liteserver-c005-4 共 8 个 NUMA node，不绑定时 ORT 的 16 个 intra-op 线程
# 会被 OS 调度到 node 0/2/3/4/5/6/7 等跨 node 的核上（已在 profile 中确认），
# 而 embedding 权重内存（~100MB）通常分配在主线程所在的 node 0，
# 导致远端线程（距离 24-25）的内存延迟是本地的 2.5 倍。
# node 0 有 24 个核（0-23），足够 --intra-threads=16 使用，且内存充裕（145GB free）。
# 注意：node 3 当前内存几乎耗尽（338MB），OS 可能把内存溢出分配到那里，需避开。
numactl --cpunodebind=0 --membind=0 \
python run_ort_dlrm.py \
    --onnx-path ../dlrm_onnx/dlrm_s_pytorch.onnx.cann_patched.onnx.loop_to_gather.onnx \
    --batch-size 64 --num-batches 3 --warmup-batches 2 \
    --shape-csv ./op_shapes.csv --enable-profiling --intra-threads=4 --inter-threads=1 \
    --no-replace-loop --num-indices-per-lookup 100
    # --use-cann --device-id 0 \
    # --num-indices-per-lookup 100