#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Temp settings (keep your original)
# ============================================================
mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER

# ============================================================
# Config (keep original defaults)
# ============================================================
PYTHON=${PYTHON:-python}

# 使用带 skip + priority 的版本（你之前下载的那个）
SCRIPT=${SCRIPT:-benchmark_tabicl_dynamic_skip.py}

# ckpt 目录（训练过程会不断往这里写 ckpt）
CKPT_DIR=${CKPT_DIR:-/vast/users/guangyi.chen/causal_group/zijian.li/LDM/tabicl_new/tabicl/stabe1/checkpoint/dir1}

# 只跑 talent
DATA_ROOT=${DATA_ROOT:-limix}
TALENT_ROOT="${DATA_ROOT}/talent_csv"

# 输出
OUT_ROOT=${OUT_ROOT:-result/ckpt_dir1}

# 断点续跑：支持环境变量或第一个参数
# 用法：
#   bash ckpt.sh                      # 从头跑
#   bash ckpt.sh step-29400.ckpt      # 从这个 ckpt 开始跑（包含它）
#   START_CKPT_NAME=step-29400.ckpt bash ckpt.sh
START_CKPT_NAME="${START_CKPT_NAME:-${1:-}}"

# 是否跳过已完成（默认跳过）
SKIP_DONE="${SKIP_DONE:-1}"   # 1=跳过已完成，0=强制重跑

# GPU/worker 设置
WORKERS=${WORKERS:-8}
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}

# checkpoint version（兼容参数；实际会用 --model-path 本地 ckpt）
CKPT_VERSION=${CKPT_VERSION:-tabicl-classifier-v1.1-0506.ckpt}

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
# NEW: Dynamic watch options
# ============================================================
# 1=持续监控 CKPT_DIR 新增 ckpt；0=只扫描一次（原行为类似）
WATCH="${WATCH:-1}"

# 没有新 ckpt 时的等待秒数
SLEEP_SEC="${SLEEP_SEC:-120}"

# find 搜索深度：1=只看 CKPT_DIR 一层（原来就是 maxdepth 1）
# 0=递归所有子目录
CKPT_MAXDEPTH="${CKPT_MAXDEPTH:-1}"

# 当 START_CKPT_NAME 指定但文件暂不存在时：
# 1=等待直到出现；0=直接报错退出（原行为）
WAIT_FOR_START_CKPT="${WAIT_FOR_START_CKPT:-1}"

# 如果你希望在“空闲 N 次轮询后退出”，可以设置：
# IDLE_EXIT_CYCLES=0 表示永不退出（默认）
IDLE_EXIT_CYCLES="${IDLE_EXIT_CYCLES:-0}"

# ============================================================
# Checks: CKPT_DIR
# ============================================================
if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "❌ CKPT_DIR 不存在: ${CKPT_DIR}"
  exit 1
fi

echo "✅ CKPT_DIR OK: ${CKPT_DIR}"
echo "✅ WATCH=${WATCH}, SLEEP_SEC=${SLEEP_SEC}, CKPT_MAXDEPTH=${CKPT_MAXDEPTH}"

# ============================================================
# Master CSV header (write once)
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
# Helpers
# ============================================================
parse_summary_field () {
  local summary_txt="$1"
  local key="$2"
  awk -F': ' -v k="${key}" '$1==k {print $2; found=1} END{if(!found) print ""}' "${summary_txt}"
}

csv_has_ckpt () {
  local ckpt_name="$1"
  awk -F',' -v k="${ckpt_name}" 'NR>1 && $1==k {found=1} END{exit(found?0:1)}' "${MASTER_CSV}"
}

append_csv_row_locked () {
  # 参数按列顺序传入
  # 使用 flock 防止并发写坏 CSV
  {
    flock 200
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$@"
  } 200>"${LOCK_FILE}" >> "${MASTER_CSV}"
}

# ------------------------------------------------------------
# NEW: scan ckpts (supports maxdepth or recursive)
# ------------------------------------------------------------
scan_ckpts () {
  local dir="$1"
  local maxdepth="$2"

  if [[ "${maxdepth}" == "0" ]]; then
    # recursive
    find "${dir}" -type f -name "*.ckpt" 2>/dev/null | sort
  else
    find "${dir}" -maxdepth "${maxdepth}" -type f -name "*.ckpt" 2>/dev/null | sort
  fi
}

# ============================================================
# Run talent (stdout only returns summary path; logs go to file)
# ============================================================
run_talent () {
  local out_dir="$1"
  local ckpt_path="$2"

  mkdir -p "${out_dir}"

  local all_out="${out_dir}/tabicl_talent.ALL.csv"
  local summary_txt="${out_dir}/tabicl_talent.summary.txt"
  local run_log="${out_dir}/tabicl_talent.run.log"

  # ✅ 所有提示信息都打到 stderr，避免污染 stdout（stdout 只留给 summary path）
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

  # ✅ stdout 只输出路径
  echo "${summary_txt}"
}

# ============================================================
# NEW: wait for a given start ckpt to appear (optional)
# ============================================================
wait_until_start_ckpt_exists () {
  local start_name="$1"
  [[ -z "${start_name}" ]] && return 0

  while true; do
    # refresh list
    mapfile -t _ckpts < <(scan_ckpts "${CKPT_DIR}" "${CKPT_MAXDEPTH}")
    for p in "${_ckpts[@]}"; do
      if [[ "$(basename "$p")" == "${start_name}" ]]; then
        echo "✅ Found START_CKPT_NAME now exists: ${start_name}"
        return 0
      fi
    done

    if [[ "${WAIT_FOR_START_CKPT}" == "1" ]]; then
      echo "⏳ START_CKPT_NAME=${start_name} not found yet. Sleep ${SLEEP_SEC}s and retry..."
      sleep "${SLEEP_SEC}"
    else
      echo "❌ 没在 CKPT_DIR 里找到 START_CKPT_NAME=${start_name}"
      echo "   你可以用：ls ${CKPT_DIR} | grep ckpt | head 来确认名字是否一致"
      exit 1
    fi
  done
}

# ============================================================
# NEW: resolve start_idx dynamically each scan
# ============================================================
calc_start_idx () {
  local start_name="$1"
  local -n _arr_ref="$2"   # nameref to array
  local start_idx_out=0

  if [[ -n "${start_name}" ]]; then
    for i in "${!_arr_ref[@]}"; do
      if [[ "$(basename "${_arr_ref[$i]}")" == "${start_name}" ]]; then
        start_idx_out="$i"
        echo "${start_idx_out}"
        return 0
      fi
    done
    # not found
    echo "-1"
    return 0
  fi

  echo "0"
  return 0
}

# ============================================================
# Main loop (dynamic)
# ============================================================
idle_cycles=0

# If user specifies a START_CKPT_NAME, optionally wait for it to show up.
wait_until_start_ckpt_exists "${START_CKPT_NAME}"

while true; do
  # scan ckpts each round
  mapfile -t CKPTS < <(scan_ckpts "${CKPT_DIR}" "${CKPT_MAXDEPTH}")

  if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "⚠️  当前未找到任何 *.ckpt in ${CKPT_DIR}"
    if [[ "${WATCH}" == "1" ]]; then
      echo "⏳ Sleep ${SLEEP_SEC}s and retry..."
      sleep "${SLEEP_SEC}"
      continue
    else
      exit 1
    fi
  fi

  echo "🔎 Found ckpts: ${#CKPTS[@]} (scan at $(date '+%Y-%m-%d %H:%M:%S'))"

  # resume start index (dynamic)
  start_idx="$(calc_start_idx "${START_CKPT_NAME}" CKPTS)"
  if [[ "${start_idx}" == "-1" ]]; then
    # Should not happen if we waited, but keep safe
    if [[ -n "${START_CKPT_NAME}" ]]; then
      echo "⚠️  START_CKPT_NAME=${START_CKPT_NAME} still not found in this scan."
      if [[ "${WATCH}" == "1" ]]; then
        echo "⏳ Sleep ${SLEEP_SEC}s and retry..."
        sleep "${SLEEP_SEC}"
        continue
      else
        exit 1
      fi
    fi
    start_idx=0
  fi

  did_any=0

  # process all unprocessed ckpts starting from start_idx
  for (( idx=start_idx; idx<${#CKPTS[@]}; idx++ )); do
    ckpt_abs="${CKPTS[$idx]}"
    ckpt_base="$(basename "${ckpt_abs}")"
    ckpt_stem="${ckpt_base%.ckpt}"
    ckpt_out="${OUT_ROOT}/${ckpt_stem}"
    summary_done_file="${ckpt_out}/tabicl_talent.summary.txt"

    if [[ "${SKIP_DONE}" == "1" ]]; then
      if csv_has_ckpt "${ckpt_base}"; then
        # already recorded
        continue
      fi
      if [[ -f "${summary_done_file}" ]]; then
        # summary exists but not in CSV (rare), still skip to avoid rerun
        continue
      fi
    fi

    did_any=1

    mkdir -p "${ckpt_out}"

    echo
    echo "#################################################################"
    echo "### CKPT: ${ckpt_base}"
    echo "### PATH: ${ckpt_abs}"
    echo "### OUT : ${ckpt_out}"
    echo "#################################################################"

    started_at="$(date '+%Y-%m-%d %H:%M:%S')"

    # run
    talent_summary="$(run_talent "${ckpt_out}" "${ckpt_abs}")"

    finished_at="$(date '+%Y-%m-%d %H:%M:%S')"

    # parse summary fields
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

    # append to master csv (locked)
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

    echo "✅ Done ckpt: ${ckpt_base}"
    echo "   - outputs: ${ckpt_out}"
    echo "   - summary: ${talent_summary}"
    echo "   - master : ${MASTER_CSV}"
  done

  if [[ "${WATCH}" != "1" ]]; then
    echo
    echo "🎉 Finished (single scan, talent only)."
    echo "Master CSV: ${MASTER_CSV}"
    exit 0
  fi

  if [[ "${did_any}" == "0" ]]; then
    idle_cycles=$((idle_cycles + 1))
    echo "😴 No new ckpt to process. idle_cycles=${idle_cycles}. Sleep ${SLEEP_SEC}s..."
    if [[ "${IDLE_EXIT_CYCLES}" != "0" && "${idle_cycles}" -ge "${IDLE_EXIT_CYCLES}" ]]; then
      echo "🛑 Reached IDLE_EXIT_CYCLES=${IDLE_EXIT_CYCLES}, exit."
      exit 0
    fi
    sleep "${SLEEP_SEC}"
  else
    idle_cycles=0
    # 立刻进入下一轮扫描，尽快捕获“又新增的 ckpt”
    echo "🔁 Scan again immediately to catch newly added ckpts..."
  fi
done
