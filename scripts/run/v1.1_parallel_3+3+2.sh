#!/usr/bin/env bash
set -e

# =========================
# 基本配置（对齐 amd.sh）
# =========================
PYTHON=python
SCRIPT=benchmark_tabicl_dynamic.py

DATA_ROOT=limix
OUT_ROOT=results/v1.1_parallel_3+3+2
mkdir -p "${OUT_ROOT}"

# 多进程时建议限制 CPU 线程争用（对齐 amd.sh）
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# TabICL 模型版本（对齐 amd.sh）
CKPT_VERSION=tabicl-classifier-v1.1-0506.ckpt
LOCAL_CKPT_PATH="./tabicl-classifier-v1.1-0506.ckpt"

# =========================
# 通用运行参数（完全对齐 amd.sh COMMON_ARGS）
# 注意：每个子任务内部的 --workers/--gpus 会覆盖这里
# =========================
COMMON_ARGS="
  --device cuda:0
  --batch-size 4
  --n-estimators 32
  --norm-methods none,power
  --feat-shuffle latin
  --softmax-temp 0.9
  --checkpoint-version ${CKPT_VERSION}
  --model-path ${LOCAL_CKPT_PATH}
  --verbose
"

echo "🚀 Launching TALENT(3 GPUs) + TabZilla(3 GPUs) + OpenML-CC18(2 GPUs) in parallel..."

# =========================
# TALENT (占 3 张卡: 0,1,2)
# =========================
(
  export HIP_VISIBLE_DEVICES=0,3,2
  ${PYTHON} ${SCRIPT} \
    --root "${DATA_ROOT}/talent_csv" \
    --out-dir "${OUT_ROOT}/talent" \
    --all-out "${OUT_ROOT}/tabicl_talent.ALL.csv" \
    --summary-txt "${OUT_ROOT}/tabicl_talent.summary.txt" \
    --workers 3 \
    --gpus 0,1,2 \
    ${COMMON_ARGS}
) &

# =========================
# TabZilla (占 3 张卡: 3,4,5)
# 重要：由于 HIP_VISIBLE_DEVICES 做了隔离，这里 --gpus 必须写 0,1,2
# =========================
(
  export HIP_VISIBLE_DEVICES=1,4,7
  ${PYTHON} ${SCRIPT} \
    --root "${DATA_ROOT}/tabzilla_csv" \
    --out-dir "${OUT_ROOT}/tabzilla" \
    --all-out "${OUT_ROOT}/tabicl_tabzilla.ALL.csv" \
    --summary-txt "${OUT_ROOT}/tabicl_tabzilla.summary.txt" \
    --workers 3 \
    --gpus 0,1,2 \
    ${COMMON_ARGS}
) &

# =========================
# OpenML-CC18 (占 2 张卡: 6,7)
# 同理：隔离后 --gpus 写 0,1
# =========================
(
  export HIP_VISIBLE_DEVICES=6,5
  ${PYTHON} ${SCRIPT} \
    --root "${DATA_ROOT}/openml_cc18_csv" \
    --out-dir "${OUT_ROOT}/openml_cc18" \
    --all-out "${OUT_ROOT}/tabicl_openml_cc18.ALL.csv" \
    --summary-txt "${OUT_ROOT}/tabicl_openml_cc18.summary.txt" \
    --workers 2 \
    --gpus 0,1 \
    ${COMMON_ARGS}
) &

wait
echo "✅ All datasets finished."
echo "Results saved in: ${OUT_ROOT}"
