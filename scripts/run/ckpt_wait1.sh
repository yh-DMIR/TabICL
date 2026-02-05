mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER

#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Config
# ============================================================
PYTHON=${PYTHON:-python}

# 你的 benchmark 脚本（建议用绝对路径更稳）
SCRIPT=${SCRIPT:-benchmark_tabicl_dynamic_skip.py}

# ckpt 目录
CKPT_DIR=${CKPT_DIR:-/vast/users/guangyi.chen/causal_group/zijian.li/LDM/tabicl_new/tabicl/stabe1/checkpoint/dir1}

# 只跑 talent
DATA_ROOT=${DATA_ROOT:-limix}
TALENT_ROOT="${DATA_ROOT}/talent_csv"

# 输出目录
OUT_ROOT=${OUT_ROOT:-result/ckpt_dir1}

# 断点：只处理 step >= MIN_STEP
MIN_STEP="${MIN_STEP:-0}"

# 是否跳过已完成
SKIP_DONE="${SKIP_DONE:-1}"   # 1=跳过已完成，0=强制重跑

# GPU/worker
WORKERS=${WORKERS:-8}
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}

# checkpoint version（兼容参数）
CKPT_VERSION=${CKPT_VERSION:-tabicl-classifier-v1.1-0506.ckpt}

# 队列空时等待策略
SLEEP_SECS="${SLEEP_SECS:-60}"

# “写完判定”：文件大小连续 STABLE_ROUNDS 次不变
STABLE_ROUNDS="${STABLE_ROUNDS:-3}"
STABLE_INTERVAL_SECS="${STABLE_INTERVAL_SECS:-5}"

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
LOCK_FILE="${MASTER_CSV}.lock"

# ============================================================
# Checks
# ============================================================
if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "❌ CKPT_DIR 不存在: ${CKPT_DIR}"
  exit 1
fi
echo "✅ CKPT_DIR OK: ${CKPT_DIR}"

# ============================================================
# Master CSV header
# ============================================================
if [[ ! -f "${MASTER_CSV}" ]]; then
  cat > "${MASTER_CSV}" <<'CSV'
ckpt,ckpt_path,started_at,finished_at,total_wall_seconds,talent_avg_acc,talent_wall_seconds,talent_discovered_pairs,talent_processed_pairs,talent_failed_count,talent_missing_test_count
CSV
  echo "✅ Created master CSV: ${MASTER_CSV}"
else
  echo "✅ Master CSV exists, will append: ${MASTER_CSV}"
fi

# ============================================================
# In-memory state (FIX BUG)
# ============================================================
# 已完成/已记录（ckpt_base -> 1）
declare -A PROCESSED=()

# 已忽略（ckpt_abs -> 1）：用于“skip 后不再入队”
declare -A IGNORED=()

load_processed_from_master_csv () {
  PROCESSED=()
  # 只读第 1 列（ckpt 名），跳过 header
  # 兼容空文件/读失败
  if [[ -f "${MASTER_CSV}" ]]; then
    while IFS=, read -r ckpt_name _rest; do
      [[ "${ckpt_name}" == "ckpt" ]] && continue
      [[ -n "${ckpt_name}" ]] && PROCESSED["${ckpt_name}"]=1
    done < "${MASTER_CSV}" || true
  fi
}

# 初始加载一次
load_processed_from_master_csv

# ============================================================
# Helpers
# ============================================================
parse_summary_field () {
  local summary_txt="$1"
  local key="$2"
  awk -F': ' -v k="${key}" '$1==k {print $2; found=1} END{if(!found) print ""}' "${summary_txt}"
}

# 用内存 PROCESSED 判断（不再频繁 awk，且不会出现“明明写了 CSV 但判断不到”的怪现象）
csv_has_ckpt () {
  local ckpt_name="$1"
  [[ -n "${PROCESSED[${ckpt_name}]+x}" ]]
}

append_csv_row_locked () {
  {
    flock 200
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$@"
  } 200>"${LOCK_FILE}" >> "${MASTER_CSV}"

  # 立即同步到内存集合，避免下一轮 refill 又把它加入队列
  local ckpt_name="$1"
  PROCESSED["${ckpt_name}"]=1
}

# 从文件名里提取 step：step-19450.ckpt -> 19450
extract_step () {
  local base="$1"
  if [[ "${base}" =~ step-([0-9]+)\.ckpt$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "-1"
  fi
}

# 判断 ckpt 是否“写完”：文件大小连续多次不变
wait_for_ckpt_ready () {
  local f="$1"
  local rounds="${STABLE_ROUNDS}"
  local interval="${STABLE_INTERVAL_SECS}"

  local last_size="-1"
  local stable=0

  while true; do
    if [[ ! -f "${f}" ]]; then
      stable=0
      last_size="-1"
      sleep "${interval}"
      continue
    fi

    local sz
    if ! sz="$(stat -c %s "${f}" 2>/dev/null)"; then
      sleep "${interval}"
      continue
    fi

    if [[ "${sz}" == "${last_size}" ]]; then
      stable=$((stable + 1))
    else
      stable=0
      last_size="${sz}"
    fi

    if (( stable >= rounds )); then
      return 0
    fi

    sleep "${interval}"
  done
}

# ============================================================
# Run talent
# ============================================================
run_talent () {
  local out_dir="$1"
  local ckpt_path="$2"

  mkdir -p "${out_dir}"

  local all_out="${out_dir}/tabicl_talent.ALL.csv"
  local summary_txt="${out_dir}/tabicl_talent.summary.txt"
  local run_log="${out_dir}/tabicl_talent.run.log"

  echo "===== Running talent with ckpt: ${ckpt_path} =====" >&2
  echo "      log -> ${run_log}" >&2

  ${PYTHON} ${SCRIPT} \
    --root "${TALENT_ROOT}" \
    --out-dir "${out_dir}/talent" \
    --all-out "${all_out}" \
    --summary-txt "${summary_txt}" \
    --model-path "${ckpt_path}" \
    ${COMMON_ARGS} \
    > "${run_log}" 2>&1

  echo "${summary_txt}"
}

# ============================================================
# Pending queue logic
# ============================================================
declare -a PENDING=()
declare -A IN_PENDING=()

refill_pending_from_dir () {
  local tmp
  tmp="$(mktemp)"

  # 每次 refill 前，保证 PROCESSED 是最新（如果你担心别的进程也在写 CSV）
  # 这行开销很小，但能避免“多进程并发”时的漏判
  load_processed_from_master_csv

  while IFS= read -r f; do
    # 已忽略（skip 过）就不再考虑
    if [[ -n "${IGNORED[${f}]+x}" ]]; then
      continue
    fi

    local base step
    base="$(basename "${f}")"
    step="$(extract_step "${base}")"

    # 不符合命名规则就跳过
    if (( step < 0 )); then
      continue
    fi

    # 断点：step 必须 >= MIN_STEP
    if (( step < MIN_STEP )); then
      continue
    fi

    # 已处理就不入队
    if [[ "${SKIP_DONE}" == "1" ]]; then
      local ckpt_stem="${base%.ckpt}"
      local ckpt_out="${OUT_ROOT}/${ckpt_stem}"
      local summary_done_file="${ckpt_out}/tabicl_talent.summary.txt"

      if csv_has_ckpt "${base}"; then
        # 直接加入 IGNORED，彻底避免重复扫描再入队
        IGNORED["${f}"]=1
        continue
      fi
      if [[ -f "${summary_done_file}" ]]; then
        IGNORED["${f}"]=1
        continue
      fi
    fi

    # 已在 pending 就不重复入队
    if [[ -n "${IN_PENDING[${f}]+x}" ]]; then
      continue
    fi

    printf '%s\t%s\n' "${step}" "${f}" >> "${tmp}"
  done < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name "*.ckpt" 2>/dev/null)

  if [[ -s "${tmp}" ]]; then
    while IFS=$'\t' read -r _step f; do
      PENDING+=("${f}")
      IN_PENDING["${f}"]=1
    done < <(sort -n -k1,1 "${tmp}")
  fi

  rm -f "${tmp}"
}

pop_pending () {
  local f="${PENDING[0]:-}"
  if [[ -z "${f}" ]]; then
    echo ""
    return 0
  fi
  PENDING=("${PENDING[@]:1}")
  unset "IN_PENDING[${f}]"
  echo "${f}"
}

wait_until_dir_changes () {
  if command -v inotifywait >/dev/null 2>&1; then
    inotifywait -q -e create -e close_write -e moved_to "${CKPT_DIR}" >/dev/null 2>&1 || true
  else
    sleep "${SLEEP_SECS}"
  fi
}

# ============================================================
# Main (online consumer)
# ============================================================
echo "✅ Online ckpt consumer started."
echo "   - MIN_STEP=${MIN_STEP}"
echo "   - SKIP_DONE=${SKIP_DONE}"
echo "   - If queue empty: inotifywait(if exists) else sleep ${SLEEP_SECS}s"
echo "   - Ready check: size stable ${STABLE_ROUNDS} rounds, interval ${STABLE_INTERVAL_SECS}s"

while true; do
  refill_pending_from_dir

  if [[ ${#PENDING[@]} -eq 0 ]]; then
    echo "⏳ Queue empty. Waiting for new ckpt in ${CKPT_DIR} ..."
    wait_until_dir_changes
    continue
  fi

  ckpt_abs="$(pop_pending)"
  ckpt_base="$(basename "${ckpt_abs}")"
  ckpt_stem="${ckpt_base%.ckpt}"
  ckpt_out="${OUT_ROOT}/${ckpt_stem}"
  summary_done_file="${ckpt_out}/tabicl_talent.summary.txt"

  # 再次 double-check：如果已处理，直接忽略（并写入 IGNORED 防止再次入队）
  if [[ "${SKIP_DONE}" == "1" ]]; then
    if csv_has_ckpt "${ckpt_base}"; then
      echo "⏭️  Skip (already in master CSV): ${ckpt_base}"
      IGNORED["${ckpt_abs}"]=1
      continue
    fi
    if [[ -f "${summary_done_file}" ]]; then
      echo "⏭️  Skip (summary exists): ${ckpt_base} -> ${summary_done_file}"
      IGNORED["${ckpt_abs}"]=1
      continue
    fi
  fi

  echo
  echo "#################################################################"
  echo "### NEXT CKPT (by step order): ${ckpt_base}"
  echo "### PATH: ${ckpt_abs}"
  echo "### OUT : ${ckpt_out}"
  echo "#################################################################"

  echo "⏱️  Waiting ckpt ready (size stable): ${ckpt_abs}"
  wait_for_ckpt_ready "${ckpt_abs}"
  echo "✅ CKPT ready: ${ckpt_abs}"

  mkdir -p "${ckpt_out}"
  started_at="$(date '+%Y-%m-%d %H:%M:%S')"

  talent_summary="$(run_talent "${ckpt_out}" "${ckpt_abs}")"
  finished_at="$(date '+%Y-%m-%d %H:%M:%S')"

  talent_avg_acc="$(parse_summary_field "${talent_summary}" "avg_accuracy_ok")"
  talent_wall_seconds="$(parse_summary_field "${talent_summary}" "wall_seconds")"
  talent_discovered_pairs="$(parse_summary_field "${talent_summary}" "discovered_pairs")"
  talent_processed_pairs="$(parse_summary_field "${talent_summary}" "processed_pairs")"
  talent_failed_count="$(parse_summary_field "${talent_summary}" "failed_count")"
  talent_missing_test_count="$(parse_summary_field "${talent_summary}" "missing_test_count")"

  total_wall_seconds="$(${PYTHON} - <<PY
def f(x):
    try:
        return float(x)
    except:
        return 0.0
print(f"{f('${talent_wall_seconds}'):.6f}")
PY
)"

  append_csv_row_locked \
    "${ckpt_base}" \
    "${ckpt_abs}" \
    "${started_at}" \
    "${finished_at}" \
    "${total_wall_seconds}" \
    "${talent_avg_acc}" \
    "${talent_wall_seconds}" \
    "${talent_discovered_pairs}" \
    "${talent_processed_pairs}" \
    "${talent_failed_count}" \
    "${talent_missing_test_count}"

  # 跑完也加入 IGNORED（防止再次被入队）
  IGNORED["${ckpt_abs}"]=1

  echo "✅ Done ckpt: ${ckpt_base}"
  echo "   - outputs: ${ckpt_out}"
  echo "   - summary: ${talent_summary}"
  echo "   - master : ${MASTER_CSV}"
done
