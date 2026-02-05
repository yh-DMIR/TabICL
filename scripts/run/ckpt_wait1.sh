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
SCRIPT=${SCRIPT:-benchmark_tabicl_dynamic_skip.py}

CKPT_DIR=${CKPT_DIR:-/vast/users/guangyi.chen/causal_group/zijian.li/LDM/tabicl_new/tabicl/stabe1/checkpoint/dir1}

DATA_ROOT=${DATA_ROOT:-limix}
TALENT_ROOT="${DATA_ROOT}/talent_csv"

OUT_ROOT=${OUT_ROOT:-result/ckpt_dir1}

MIN_STEP="${MIN_STEP:-0}"
SKIP_DONE="${SKIP_DONE:-1}"

WORKERS=${WORKERS:-8}
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
CKPT_VERSION=${CKPT_VERSION:-tabicl-classifier-v1.1-0506.ckpt}

SLEEP_SECS="${SLEEP_SECS:-60}"
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

if [[ ! -f "${MASTER_CSV}" ]]; then
  cat > "${MASTER_CSV}" <<'CSV'
ckpt,ckpt_path,started_at,finished_at,total_wall_seconds,talent_avg_acc,talent_wall_seconds,talent_discovered_pairs,talent_processed_pairs,talent_failed_count,talent_missing_test_count
CSV
  echo "✅ Created master CSV: ${MASTER_CSV}"
else
  echo "✅ Master CSV exists, will append: ${MASTER_CSV}"
fi

# ============================================================
# In-memory state (FIX)
# ============================================================
declare -A PROCESSED=()   # ckpt_base -> 1
declare -A IGNORED=()     # ckpt_abs  -> 1   (skip/done 后永久忽略)

load_processed_from_master_csv () {
  PROCESSED=()
  if [[ -f "${MASTER_CSV}" ]]; then
    while IFS=, read -r ckpt_name _rest; do
      [[ "${ckpt_name}" == "ckpt" ]] && continue
      [[ -n "${ckpt_name}" ]] && PROCESSED["${ckpt_name}"]=1
    done < "${MASTER_CSV}" || true
  fi
}

csv_has_ckpt () {
  local ckpt_name="$1"
  [[ -n "${PROCESSED[${ckpt_name}]+x}" ]]
}

append_csv_row_locked () {
  {
    flock 200
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$@"
  } 200>"${LOCK_FILE}" >> "${MASTER_CSV}"

  # 同步到内存，防止下一轮 refill 又入队
  PROCESSED["$1"]=1
}

# ============================================================
# Helpers
# ============================================================
parse_summary_field () {
  local summary_txt="$1"
  local key="$2"
  awk -F': ' -v k="${key}" '$1==k {print $2; found=1} END{if(!found) print ""}' "${summary_txt}"
}

extract_step () {
  local base="$1"
  if [[ "${base}" =~ step-([0-9]+)\.ckpt$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "-1"
  fi
}

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
# Pending queue
# ============================================================
declare -a PENDING=()
declare -A IN_PENDING=()

refill_pending_from_dir () {
  local tmp
  tmp="$(mktemp)"

  # 每轮 reload 一次，兼容并发写 master CSV
  load_processed_from_master_csv

  while IFS= read -r f; do
    # ✅ 关键：已忽略的永不再入队（修复刷屏）
    if [[ -n "${IGNORED[${f}]+x}" ]]; then
      continue
    fi

    local base step
    base="$(basename "${f}")"
    step="$(extract_step "${base}")"
    (( step < 0 )) && continue
    (( step < MIN_STEP )) && continue

    if [[ "${SKIP_DONE}" == "1" ]]; then
      local ckpt_stem="${base%.ckpt}"
      local ckpt_out="${OUT_ROOT}/${ckpt_stem}"
      local summary_done_file="${ckpt_out}/tabicl_talent.summary.txt"

      if csv_has_ckpt "${base}"; then
        IGNORED["${f}"]=1
        continue
      fi
      if [[ -f "${summary_done_file}" ]]; then
        IGNORED["${f}"]=1
        continue
      fi
    fi

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
  [[ -z "${f}" ]] && { echo ""; return 0; }
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
# Main
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

  # double-check：若已完成/已存在，加入 IGNORED，避免下一轮 refill 又入队
  if [[ "${SKIP_DONE}" == "1" ]]; then
    load_processed_from_master_csv
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

  # ✅ 跑完也加入 IGNORED，避免再次入队
  IGNORED["${ckpt_abs}"]=1

  echo "✅ Done ckpt: ${ckpt_base}"
  echo "   - outputs: ${ckpt_out}"
  echo "   - summary: ${talent_summary}"
  echo "   - master : ${MASTER_CSV}"
done
