#!/bin/bash
set -e

# === 1. 基础路径 ===
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
# 默认沿用原来的 sim_run_11.8；可传入其他目录，如 bash scripts/run_plots.sh sim_run_12.8。
SIM_ROOT_BASE="${1:-$ROOT_DIR/sim_run_11.8}"
# 只执行原 Lyhong 绘图流程，避免选中 scripts 中其他 plot_*.py。
PLOT_SCRIPTS=(
    plot_core_change.py
    plot_log.py
    plot_power_trace.py
    plot_sm_log.py
    plot_staticP_static_P_peractive.py
)

# === 2. 遍历 sim_run_11.8 下的所有一级子目录 ===
for SIM_ROOT in "$SIM_ROOT_BASE"/*; do
    # 只处理目录
    [ -d "$SIM_ROOT" ] || continue

    BASENAME=$(basename "$SIM_ROOT")

    # === 2.1 跳过不需要的目录 ===
    if [[ "$BASENAME" == "gpgpu-sim-builds" ]]; then
        echo "Skipping $SIM_ROOT"
        continue
    fi

    echo "========================================"
    echo "Processing workload directory: $SIM_ROOT"
    echo "========================================"

    # === 3. 找到第一层唯一子目录 ===
    LEVEL1_DIR=$(find "$SIM_ROOT" -mindepth 1 -maxdepth 1 -type d | head -n 1)

    if [ -z "$LEVEL1_DIR" ]; then
        echo "Warning: no subdirectory found in $SIM_ROOT, skipping."
        continue
    fi

    # === 4. 找到第二层唯一子目录 ===
    LEVEL2_DIR=$(find "$LEVEL1_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)

    if [ -z "$LEVEL2_DIR" ]; then
        echo "Warning: no subdirectory found in $LEVEL1_DIR, skipping."
        continue
    fi

    echo "Target directory: $LEVEL2_DIR"

    # === 5. 复制原流程使用的绘图脚本 ===
    for py in "${PLOT_SCRIPTS[@]}"; do
        cp "$SCRIPT_DIR/$py" "$LEVEL2_DIR/"
    done

    # === 6. 进入目标目录并运行 ===
    (
        cd "$LEVEL2_DIR"
        for py in "${PLOT_SCRIPTS[@]}"; do
            echo "Running $py ..."
            python3 "$py"
        done
    )

    echo "Finished $SIM_ROOT"
    echo
done

echo "All plot scripts finished successfully."
