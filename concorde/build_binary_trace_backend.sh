#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SRC_DIR="${SCRIPT_DIR}/native"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build/native}"
JOBS="${JOBS:-$(nproc)}"
DYNAMORIO_DIR="${DYNAMORIO_DIR:-/data/qc/simulator/DynamoRIO-AArch64-Linux-11.3.0-1/cmake}"

cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" -DDynamoRIO_DIR="${DYNAMORIO_DIR}"
cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

echo "${BUILD_DIR}/drmemtrace_pseudoview"
