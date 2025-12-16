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

for ((w = 1; w <= MAX_WORKERS; w++)); do
  for ((r = 1; r <= RUNS_PER_WORKER; r++)); do
    out_dir="${OUT_ROOT}/w${w}/run${r}"
    mkdir -p "${out_dir}"
    out_file="${out_dir}/profile.html"
    echo "Running scalene with ${w} worker(s), run ${r} -> ${out_file}"
    python -m scalene --outfile "${out_file}" "${TARGET}" --workers "${w}" "${EXTRA_ARGS[@]}"
  done
done

echo "Done. Profiles saved under ${OUT_ROOT}/w*/run*/profile.html"
