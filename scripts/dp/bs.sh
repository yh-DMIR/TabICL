#!/usr/bin/env bash
set -e

PYTHON=python
SCRIPT=benchmark_tabicl_dp_serial.py

DATA_ROOT=limix
OUT_ROOT=results/v1.1_dp_serial
mkdir -p "${OUT_ROOT}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

CKPT_VERSION=tabicl-classifier-v1.1-0506.ckpt
LOCAL_CKPT_PATH="./tabicl-classifier-v1.1-0506.ckpt"

WORKERS=8
GPUS="0,1,2,3,4,5,6,7"

COMMON_ARGS="
  --workers ${WORKERS}
  --gpus ${GPUS}
  --device cuda:0
  --batch-size 32
  --n-estimators 32
  --norm-methods none,power
  --feat-shuffle latin
  --softmax-temp 0.9
  --checkpoint-version ${CKPT_VERSION}
  --model-path ${LOCAL_CKPT_PATH}
  --verbose
"

echo "===== Running TALENT (DP-8GPU SERIAL per-dataset) ====="
${PYTHON} ${SCRIPT} \
  --root "${DATA_ROOT}/talent_csv" \
  --out-dir "${OUT_ROOT}/talent" \
  --all-out "${OUT_ROOT}/tabicl_talent.ALL.csv" \
  --summary-txt "${OUT_ROOT}/tabicl_talent.summary.txt" \
  ${COMMON_ARGS}

echo "===== Running OpenML-CC18 (DP-8GPU SERIAL per-dataset) ====="
${PYTHON} ${SCRIPT} \
  --root "${DATA_ROOT}/openml_cc18_csv" \
  --out-dir "${OUT_ROOT}/openml_cc18" \
  --all-out "${OUT_ROOT}/tabicl_openml_cc18.ALL.csv" \
  --summary-txt "${OUT_ROOT}/tabicl_openml_cc18.summary.txt" \
  ${COMMON_ARGS}

echo "===== Running TabZilla (DP-8GPU SERIAL per-dataset) ====="
${PYTHON} ${SCRIPT} \
  --root "${DATA_ROOT}/tabzilla_csv" \
  --out-dir "${OUT_ROOT}/tabzilla" \
  --all-out "${OUT_ROOT}/tabicl_tabzilla.ALL.csv" \
  --summary-txt "${OUT_ROOT}/tabicl_tabzilla.summary.txt" \
  ${COMMON_ARGS}

echo "✅ All datasets finished."
echo "Results saved in: ${OUT_ROOT}"
