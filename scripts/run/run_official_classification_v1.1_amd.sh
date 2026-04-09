#!/usr/bin/env bash
set -e

mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHON=${PYTHON:-python}
SCRIPT=${SCRIPT:-benchmark_tabicl_classification_amd.py}
ROOT=${ROOT:-.}
BENCHMARKS=${BENCHMARKS:-openml_cc18_csv=dataset/openml_cc18_72,tabarena_cls=dataset/tabarena/cls,tabzilla_csv=dataset/tabzilla35,talent_cls=dataset/talent_cls}
CHECKPOINT_VERSION=${CHECKPOINT_VERSION:-tabicl-classifier-v1.1-0506.ckpt}
MODEL_PATH=${MODEL_PATH:-tabicl-classifier-v1.1-0506.ckpt}
OUT_DIR=${OUT_DIR:-results/official_classification_v1.1}
WORKERS=${WORKERS:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
BATCH_SIZE=${BATCH_SIZE:-2}

COMMON_ARGS="
  --root ${ROOT}
  --benchmarks ${BENCHMARKS}
  --checkpoint-version ${CHECKPOINT_VERSION}
  --model-path ${MODEL_PATH}
  --out-dir ${OUT_DIR}
  --workers ${WORKERS}
  --gpus ${GPUS}
  --device cuda:0
  --batch-size ${BATCH_SIZE}
  --n-estimators 32
  --norm-methods none,power
  --feat-shuffle latin
  --softmax-temp 0.9
  --test-size 0.2
  --verbose
"

${PYTHON} ${SCRIPT} ${COMMON_ARGS}
