#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DR_ROOT="${DR_ROOT:-/data/qc/simulator/DynamoRIO-AArch64-Linux-11.3.0-1}"
BIN="${BINARY_STREAMER_BIN:-${SCRIPT_DIR}/build/native/drmemtrace_pseudoview}"

if [[ ! -x "${BIN}" ]]; then
  echo "Binary trace streamer not found: ${BIN}" >&2
  echo "Build it first with: bash ORT/concorde/build_binary_trace_backend.sh" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${DR_ROOT}/lib64/release:${DR_ROOT}/ext/lib64/release:${LD_LIBRARY_PATH:-}"
exec "${BIN}" "$@"
