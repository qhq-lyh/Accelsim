#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$ROOT_DIR/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/gpgpusim.config"
ARCHIVE_ROOT="$ROOT_DIR/sim_run_12.8-signle-4G"
RUN_DIR="$ROOT_DIR/sim_run_12.8"
LOG_ROOT="$ROOT_DIR/cluster_sweep_logs"
JOB_NAME="Accelwattch_Test"

START_CLUSTER="${START_CLUSTER:-11}"
END_CLUSTER="${END_CLUSTER:-79}"
FORCE="${FORCE:-0}"

cd "$ROOT_DIR"
mkdir -p "$ARCHIVE_ROOT" "$LOG_ROOT"

backup="$LOG_ROOT/gpgpusim.config.backup.$(date +%Y%m%d-%H%M%S)"
cp "$CONFIG_FILE" "$backup"
echo "Backed up config to $backup"

set_cluster_count() {
  local clusters="$1"
  python3 - "$CONFIG_FILE" "$clusters" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
clusters = sys.argv[2]
text = path.read_text()
new_text, count = re.subn(
    r"(?m)^(-gpgpu_n_clusters\s+)\d+(\s*)$",
    rf"\g<1>{clusters}\2",
    text,
    count=1,
)
if count != 1:
    raise SystemExit(f"Expected exactly one -gpgpu_n_clusters line, found {count}")
path.write_text(new_text)
PY
}

clean_job_logs() {
  local log_dir="$ROOT_DIR/util/job_launching/logfiles"
  local procman_dir="$ROOT_DIR/util/job_launching/procman"

  mkdir -p "$log_dir" "$procman_dir"
  find "$log_dir" -mindepth 1 -maxdepth 1 -type f -delete
  find "$procman_dir" -mindepth 1 -maxdepth 1 -type f -delete
}

for clusters in $(seq "$START_CLUSTER" "$END_CLUSTER"); do
  echo
  echo "========== gpgpu_n_clusters=$clusters =========="

  dest="$ARCHIVE_ROOT/sim_run_12.8-$clusters"
  iter_log_dir="$LOG_ROOT/clusters-$clusters"
  mkdir -p "$iter_log_dir"

  if [[ -e "$dest" ]]; then
    if [[ "$FORCE" == "1" ]]; then
      echo "Removing existing archive: $dest"
      rm -rf "$dest"
    else
      echo "Archive already exists, skipping gpgpu_n_clusters=$clusters: $dest"
      continue
    fi
  fi

  if [[ -e "$RUN_DIR" ]]; then
    echo "ERROR: existing run directory found before starting this iteration: $RUN_DIR"
    echo "Move or delete it before rerunning the sweep."
    exit 1
  fi

  set_cluster_count "$clusters"
  grep -n "^-gpgpu_n_clusters" "$CONFIG_FILE" | tee "$iter_log_dir/config_check.log"

  echo "Building simulator..."
  (
    set +u
    export OPENCL_REMOTE_GPU_HOST="${OPENCL_REMOTE_GPU_HOST:-}"
    if [[ -n "${CUDA_INSTALL_PATH:-}" ]]; then
      export PATH="$CUDA_INSTALL_PATH/bin:$PATH"
    fi
    source ./gpu-simulator/setup_environment.sh
    set -u
    make -j -C ./gpu-simulator
  ) 2>&1 | tee "$iter_log_dir/build.log"

  echo "Launching simulations at $(date '+%F %T')..."
  ./util/job_launching/run_simulations.py \
    -B rodinia_2.0-ft \
    -C QV100-Accelwattch_SASS_SIM \
    -T ./hw_run/rodinia_2.0-ft/9.1 \
    -N "$JOB_NAME" 2>&1 | tee "$iter_log_dir/run_simulations.log"
  echo "run_simulations.py finished at $(date '+%F %T'); starting monitor."

  echo "Monitoring simulations..."
  ./util/job_launching/monitor_func_test.py -v -N "$JOB_NAME" \
    2>&1 | tee "$iter_log_dir/monitor.log"

  if ! grep -q "Congratulations! All Tests Pass!" "$iter_log_dir/monitor.log"; then
    echo "ERROR: monitor did not report success for gpgpu_n_clusters=$clusters"
    exit 1
  fi

  clean_job_logs

  if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: expected run directory does not exist after successful monitor: $RUN_DIR"
    exit 1
  fi

  mv "$RUN_DIR" "$dest"
  echo "Archived run directory to $dest"
done

echo
echo "All cluster sweep runs completed successfully."
echo "Generating IPC lookup tables and plots..."
python3 "$SCRIPT_DIR/plot_cluster_ipc.py" --root "$ARCHIVE_ROOT"
