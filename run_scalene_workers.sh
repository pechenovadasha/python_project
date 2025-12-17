#!/usr/bin/env bash
set -euo pipefail

# Запускает scalene для reading_files.py с разным числом потоков (workers)
# и складывает результаты в отдельные подпапки внутри profile/full_optim.
# Пример: ./run_scalene_workers.sh 8 3 --folders dataset other_folder
# где 8 — максимум потоков, 3 — число повторов для каждого количества потоков.

# максимум потоков (по умолчанию CPU count)
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  MAX_WORKERS="$1"
  shift
else
  MAX_WORKERS="$(python3 - <<'PY'
import os
print(os.cpu_count() or 1)
PY
)"
fi

# число повторов на каждое значение workers (по умолчанию 1)
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  RUNS_PER_WORKER="$1"
  shift
else
  RUNS_PER_WORKER=1
fi

OUT_ROOT="profile/full_optim"
TARGET="./reading_files.py"
EXTRA_ARGS=("$@")

mkdir -p "${OUT_ROOT}"

run_with_timeout() {
  # Runs a command with a 10s timeout using the best available tool; returns
  # 124 on timeout to mirror GNU timeout.
  local -a cmd=("$@")
  local status
  if command -v timeout >/dev/null 2>&1; then
    if timeout 10s "${cmd[@]}"; then
      status=0
    else
      status=$?
    fi
  elif command -v gtimeout >/dev/null 2>&1; then
    if gtimeout 10s "${cmd[@]}"; then
      status=0
    else
      status=$?
    fi
  else
    if python3 - "$@" <<'PY'
import subprocess, sys

cmd = sys.argv[1:]
try:
    completed = subprocess.run(cmd, timeout=10)
    sys.exit(completed.returncode)
except subprocess.TimeoutExpired:
    sys.exit(124)
PY
    then
      status=0
    else
      status=$?
    fi
  fi
  return "${status}"
}

for ((w = 1; w <= MAX_WORKERS; w++)); do
  for ((r = 1; r <= RUNS_PER_WORKER; r++)); do
    out_dir="${OUT_ROOT}/w${w}/run${r}"
    mkdir -p "${out_dir}"
    out_file="${out_dir}/profile.html"
    echo "Running scalene with ${w} worker(s), run ${r} -> ${out_file}"
    if run_with_timeout python -m scalene --outfile "${out_file}" "${TARGET}" --workers "${w}" "${EXTRA_ARGS[@]}"; then
      continue
    fi

    exit_code=$?
    if [[ ${exit_code} -eq 124 ]]; then
      echo "Skipped w${w} run ${r}: exceeded 10 seconds, moving on."
    else
      echo "Skipped w${w} run ${r}: scalene failed with exit ${exit_code}, moving on."
    fi
    rm -f "${out_file}"
  done
done

echo "Done. Profiles saved under ${OUT_ROOT}/w*/run*/profile.html"
