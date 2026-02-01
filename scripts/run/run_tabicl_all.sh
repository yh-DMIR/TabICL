mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER

#!/usr/bin/env bash
set -e

# =========================
# 基本配置
# =========================
export CUDA_VISIBLE_DEVICES=2

PYTHON=python
SCRIPT=benchmark_tabicl.py

DATA_ROOT=limix
OUT_ROOT=results/v1.1
mkdir -p ${OUT_ROOT}

# =========================
# TabICL 模型版本
# 可选：
#   tabicl-classifier-v1.1-0506.ckpt  (最新，默认)
#   tabicl-classifier-v1-0208.ckpt    (论文复现)
# =========================
CKPT_VERSION=tabicl-classifier-v1.1-0506.ckpt

# =========================
# 本地 ckpt 路径（离线机器必填）
# 假设 ckpt 放在项目根目录：./tabicl-classifier-v1.1-0506.ckpt
# =========================
LOCAL_CKPT_PATH="./tabicl-classifier-v1.1-0506.ckpt"

# =========================
# 通用运行参数
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
"

# =========================
# 1️⃣ OpenML-CC18
# =========================
echo "===== Running OpenML-CC18 ====="
${PYTHON} ${SCRIPT} \
  --root ${DATA_ROOT}/openml_cc18_csv \
  --out  ${OUT_ROOT}/tabicl_openml_cc18.csv \
  ${COMMON_ARGS}

# =========================
# 2️⃣ TabZilla
# =========================
echo "===== Running TabZilla ====="
${PYTHON} ${SCRIPT} \
  --root ${DATA_ROOT}/tabzilla_csv \
  --out  ${OUT_ROOT}/tabicl_tabzilla.csv \
  ${COMMON_ARGS}

# =========================
# 3️⃣ TALENT
# =========================
echo "===== Running TALENT ====="
${PYTHON} ${SCRIPT} \
  --root ${DATA_ROOT}/talent_csv \
  --out  ${OUT_ROOT}/tabicl_talent.csv \
  ${COMMON_ARGS}

echo "✅ All datasets finished."
echo "Results saved in: ${OUT_ROOT}"
