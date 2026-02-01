mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER

#!/usr/bin/env bash
set -euo pipefail

# =========================
# 1) 路径与脚本设置（按需改）
# =========================
PYTHON=${PYTHON:-python}

# 使用带 skip 的版本
SCRIPT=${SCRIPT:-benchmark_tabicl_dynamic_skip.py}

# 你的 ckpt 目录（指定 dir4）
CKPT_DIR=${CKPT_DIR:-/vast/users/guangyi.chen/causal_group/zijian.li/LDM/tabicl_new/tabicl/stabe1/checkpoint/dir4}

# 只跑 talent
DATA_ROOT=${DATA_ROOT:-limix}
TALENT_ROOT="${DATA_ROOT}/talent_csv"

# 输出根目录
OUT_ROOT=${OUT_ROOT:-result/ckpt}

# GPU/worker 设置
WORKERS=${WORKERS:-8}
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}

# TabICL checkpoint version（如果你提供 --model-path 本地 ckpt，这个更多是兼容参数）
CKPT_VERSION=${CKPT_VERSION:-tabicl-classifier-v1.1-0506.ckpt}

# 断点续跑：从某个 ckpt 名称开始（例如 step-29400.ckpt）
# 用法示例：
#   START_CKPT_NAME=step-29400.ckpt bash run_dir4_ckpts_talent_resume.sh
# 或者：
#   bash run_dir4_ckpts_talent_resume.sh step-29400.ckpt
START_CKPT_NAME="${START_CKPT_NAME:-${1:-}}"

# 是否跳过已记录的 ckpt（默认跳过）
SKIP_DONE="${SKIP_DONE:-1}"   # 1=跳过已完成，0=不跳过（强制重跑）

# 通用运行参数（保持你之前那套）
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

COMMON_ARGS="
  --workers ${WORKERS}
  --gpus ${GPUS}
  --device cuda:0
  --batch-size 4
  --n-estimators 32
  --norm-methods none,power
  --feat-shuffle latin
  --softmax-temp 0.9
  --checkpoint-version ${CKPT_VERSION}
  --verbose
"

mkdir -p "${OUT_ROOT}"

MASTER_CSV="${OUT_ROOT}/summary_all_ckpts_talent_only.csv"

# =========================
# 2) 检查 CKPT_DIR 是否存在 + 是否有 ckpt
# =========================
if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "❌ CKPT_DIR 不存在: ${CKPT_DIR}"
  exit 1
fi

mapfile -t CKPTS < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name "*.ckpt" | sort)

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "❌ 在目录下没找到 *.ckpt: ${CKPT_DIR}"
  exit 1
fi

echo "✅ CKPT_DIR OK: ${CKPT_DIR}"
echo "✅ Found ckpts: ${#CKPTS[@]}"

# =========================
# 3) master csv：不存在就写表头，存在就续写
# =========================
if [[ ! -f "${MASTER_CSV}" ]]; then
  cat > "${MASTER_CSV}" <<'CSV'
ckpt,ckpt_path,started_at,finished_at,total_wall_seconds,\
talent_avg_acc,talent_wall_seconds,talent_discovered_pairs,talent_processed_pairs,talent_failed_count,talent_missing_test_count
CSV
  echo "✅ Created master CSV with header: ${MASTER_CSV}"
else
  echo "✅ Master CSV exists, will append: ${MASTER_CSV}"
fi

# =========================
# 工具函数：解析 summary txt
# =========================
parse_summary_field () {
  local summary_txt="$1"
  local key="$2"
  awk -F': ' -v k="${key}" '$1==k {print $2; found=1} END{if(!found) print ""}' "${summary_txt}"
}

# 判断 master csv 是否已有 ckpt 记录（按第一列 ckpt 精确匹配）
csv_has_ckpt () {
  local ckpt_name="$1"
  # 跳过表头，从第二行开始
  awk -F',' -v k="${ckpt_name}" 'NR>1 && $1==k {found=1} END{exit(found?0:1)}' "${MASTER_CSV}"
}

# =========================
# 跑 talent 单数据集
# =========================
run_talent () {
  local out_dir="$1"
  local ckpt_path="$2"

  mkdir -p "${out_dir}"

  local all_out="${out_dir}/tabicl_talent.ALL.csv"
  local summary_txt="${out_dir}/tabicl_talent.summary.txt"

  echo "===== Running talent with ckpt: ${ckpt_path} ====="
  ${PYTHON} ${SCRIPT} \
    --root "${TALENT_ROOT}" \
    --out-dir "${out_dir}/talent" \
    --all-out "${all_out}" \
    --summary-txt "${summary_txt}" \
    --model-path "${ckpt_path}" \
    ${COMMON_ARGS}

  echo "${summary_txt}"
}

# =========================
# 4) 计算起始 index：从 START_CKPT_NAME 开始（如果提供）
# =========================
start_idx=0
if [[ -n "${START_CKPT_NAME}" ]]; then
  found=0
  for i in "${!CKPTS[@]}"; do
    if [[ "$(basename "${CKPTS[$i]}")" == "${START_CKPT_NAME}" ]]; then
      start_idx="$i"
      found=1
      break
    fi
  done
  if [[ "${found}" -eq 0 ]]; then
    echo "❌ 没在 CKPT_DIR 里找到 START_CKPT_NAME=${START_CKPT_NAME}"
    echo "   你可以用：ls ${CKPT_DIR} | grep ckpt | head 来确认名字是否一致"
    exit 1
  fi
  echo "✅ Resume from ckpt: ${START_CKPT_NAME} (index=${start_idx})"
else
  echo "✅ No START_CKPT_NAME provided, run from beginning."
fi

# =========================
# 5) 主循环：从 start_idx 开始遍历
# =========================
for (( idx=start_idx; idx<${#CKPTS[@]}; idx++ )); do
  ckpt_abs="${CKPTS[$idx]}"
  ckpt_base="$(basename "${ckpt_abs}")"
  ckpt_stem="${ckpt_base%.ckpt}"
  ckpt_out="${OUT_ROOT}/${ckpt_stem}"

  # 用 summary 文件作为“已完成”的辅助判断
  summary_done_file="${ckpt_out}/tabicl_talent.summary.txt"

  if [[ "${SKIP_DONE}" == "1" ]]; then
    # 如果 master csv 已经记录过该 ckpt 或者 summary 文件存在，则跳过
    if csv_has_ckpt "${ckpt_base}"; then
      echo "⏭️  Skip (already in master CSV): ${ckpt_base}"
      continue
    fi
    if [[ -f "${summary_done_file}" ]]; then
      echo "⏭️  Skip (summary exists): ${ckpt_base} -> ${summary_done_file}"
      continue
    fi
  fi

  mkdir -p "${ckpt_out}"

  echo
  echo "#################################################################"
  echo "### CKPT: ${ckpt_base}"
  echo "### PATH: ${ckpt_abs}"
  echo "### OUT : ${ckpt_out}"
  echo "#################################################################"

  started_at="$(date '+%Y-%m-%d %H:%M:%S')"

  # 只跑 talent
  talent_summary="$(run_talent "${ckpt_out}" "${ckpt_abs}")"

  finished_at="$(date '+%Y-%m-%d %H:%M:%S')"

  # 取 talent summary 字段
  talent_avg_acc="$(parse_summary_field "${talent_summary}" "avg_accuracy_ok")"
  talent_wall_seconds="$(parse_summary_field "${talent_summary}" "wall_seconds")"
  talent_discovered_pairs="$(parse_summary_field "${talent_summary}" "discovered_pairs")"
  talent_processed_pairs="$(parse_summary_field "${talent_summary}" "processed_pairs")"
  talent_failed_count="$(parse_summary_field "${talent_summary}" "failed_count")"
  talent_missing_test_count="$(parse_summary_field "${talent_summary}" "missing_test_count")"

  total_wall_seconds="$(python - <<PY
def f(x):
    try:
        return float(x)
    except:
        return 0.0
print(f"{f('${talent_wall_seconds}'):.6f}")
PY
)"

  echo "${ckpt_base},${ckpt_abs},${started_at},${finished_at},${total_wall_seconds},\
${talent_avg_acc},${talent_wall_seconds},${talent_discovered_pairs},${talent_processed_pairs},${talent_failed_count},${talent_missing_test_count}" \
  >> "${MASTER_CSV}"

  echo "✅ Done ckpt: ${ckpt_base}"
  echo "   - outputs: ${ckpt_out}"
  echo "   - master : ${MASTER_CSV}"
done

echo
echo "🎉 Finished (talent only)."
echo "Master CSV: ${MASTER_CSV}"
