#!/usr/bin/env bash
# Arguments: mixed trace directory, MHz, stream:SM-count specification, fresh output directory.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# != 4 ]]; then
  echo "Usage: bash $0 TRACE_DIR FREQ_MHZ '1:43,2:9,3:28' OUT_DIR" >&2
  exit 2
fi
TRACE_DIR="$(realpath "$1")"
FREQ_MHZ="$2"
SM_COUNTS="$3"
OUT_DIR="$(realpath -m "$4")"
test -f "$TRACE_DIR/kernelslist.g"
if [[ -e "$OUT_DIR" ]]; then
  echo "Output already exists; use a fresh directory: $OUT_DIR" >&2
  exit 2
fi
# This example deliberately keeps all 80 physical SMs in the GV100 thermal model.
python3 - "$FREQ_MHZ" "$SM_COUNTS" <<'PY'
import math, sys
f = float(sys.argv[1])
assert math.isfinite(f) and 0 < f <= 4294, 'frequency must be positive and <=4294 MHz (current get_core_freq returns unsigned Hz)'
entries = [entry.split(':') for entry in sys.argv[2].split(',')]
ids = [int(s) for s, _ in entries]
counts = [int(n) for _, n in entries]
assert len(set(ids)) == len(ids) and all(s > 0 for s in ids), 'stream IDs must be distinct and positive'
assert all(n > 0 for n in counts) and sum(counts) == 80, 'positive counts must sum to 80'
PY
mkdir -p "$OUT_DIR"
BASE_CONFIG="$ROOT_DIR/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100"
cp "$BASE_CONFIG/gpgpusim.config" "$OUT_DIR/gpgpusim.config"
cp "$BASE_CONFIG"/*.xml "$BASE_CONFIG"/*.icnt "$OUT_DIR/"
cp "$ROOT_DIR/gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config" "$OUT_DIR/trace.config"
printf '%s\n' "$SM_COUNTS" > "$OUT_DIR/stream_sm_alloc.txt"
cat > "$OUT_DIR/experiment.config" <<EOF
-gpgpu_clock_domains $FREQ_MHZ:$FREQ_MHZ:$FREQ_MHZ:850.0
-gpgpu_n_clusters 80
-gpgpu_n_cores_per_cluster 1
-gpgpu_concurrent_kernel_sm 1
-gpgpu_max_concurrent_kernel 128
-gpgpu_concurrent_stream_sm_config_file ./stream_sm_alloc.txt
-gpgpu_concurrent_stream_sm_reload_interval 100
-power_simulation_enabled 1
-power_simulation_mode 0
-accelwattch_xml_file ./accelwattch_sass_sim.xml
-dvfs_enabled 0
-gpgpu_runtime_stat 100:0
-power_per_cycle_dump 0
EOF
cd "$OUT_DIR"
"$ROOT_DIR/gpu-simulator/bin/release/accel-sim.out" \
  -config ./gpgpusim.config -config ./trace.config -config ./experiment.config \
  -trace "$TRACE_DIR/kernelslist.g" > sim.log 2>&1
test -s lyhong_power_report.log
test -s lyhong_SM_power_report.log
# Per-SM mode emits the native Lyhong reports; the standard aggregate report is optional.
rg -q '\*\*\* exit detected \*\*\*' sim.log
cp lyhong_power_report.log power.ptrace
python3 "$ROOT_DIR/manual-workflow/summarize.py" "$OUT_DIR" "$FREQ_MHZ"
"$ROOT_DIR/hotspot-link/hotspot" -c "$ROOT_DIR/hotspot-link/GV100-hotspot.config" -p "$OUT_DIR/power.ptrace" \
  -grid_layer_file "$ROOT_DIR/hotspot-link/flp/gv100.lcf" -model_type grid \
  -steady_file "$OUT_DIR/temp.steady" \
  -grid_steady_file "$OUT_DIR/temp.grid.steady" > "$OUT_DIR/hotspot.log" 2>&1
echo "Completed performance, power and steady thermal round: $OUT_DIR"
