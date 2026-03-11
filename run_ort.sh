# !/bin/bash

set -e

source /data/qc/Ascend/ascend-toolkit/set_env.sh

MANUAL_BRANCH_PARALLEL="${MANUAL_BRANCH_PARALLEL:-0}"

# numactl 将进程的 CPU 执行和内存分配都限制在 node 0（cores 0-23）。
# 背景：liteserver-c005-4 共 8 个 NUMA node，不绑定时 ORT 的 16 个 intra-op 线程
# 会被 OS 调度到 node 0/2/3/4/5/6/7 等跨 node 的核上（已在 profile 中确认），
# 而 embedding 权重内存（~100MB）通常分配在主线程所在的 node 0，
# 导致远端线程（距离 24-25）的内存延迟是本地的 2.5 倍。
# node 0 有 24 个核（0-23），足够 --intra-threads=16 使用，且内存充裕（145GB free）。
# 注意：node 3 当前内存几乎耗尽（338MB），OS 可能把内存溢出分配到那里，需避开。
if [ "$MANUAL_BRANCH_PARALLEL" = "1" ]; then
    numactl --cpunodebind=0 --membind=0 \
    /data/qc/anaconda3/envs/ort/bin/python run_ort_dlrm_branch_parallel.py \
        --onnx-path ../dlrm_onnx/dlrm_s_pytorch.onnx \
        --batch-size 6400 --num-batches 1 --warmup-batches 0 \
        --shape-csv ./op_shapes.csv \
        --enable-profiling \
        --profile-dir ./onnx_operator_analysis/branch_parallel \
        --out-dir ./onnx_operator_analysis/branch_parallel \
        --intra-threads 1 --inter-threads 4 \
        --num-indices-per-lookup 1000
else
    numactl --cpunodebind=0 --membind=0 \
    python run_ort_dlrm.py \
        --onnx-path ../dlrm_onnx/dlrm_s_pytorch.onnx \
        --batch-size 6400 --num-batches 1 --warmup-batches 0 \
        --shape-csv ./op_shapes.csv --enable-profiling --intra-threads=1 --inter-threads=4 --disable-graph-optimizations \
        --num-indices-per-lookup 1000
        # --no-replace-loop
        # --use-cann --device-id 0 \
        # --num-indices-per-lookup 100
fi
