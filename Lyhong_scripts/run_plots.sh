#!/bin/bash
set -e

# === 1. 基础路径 ===
ROOT_DIR="$(pwd)"
SCRIPT_DIR="$ROOT_DIR/Lyhong_scripts"
SIM_ROOT_BASE="$ROOT_DIR/sim_run_11.8"

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

    # === 5. 复制 plot_*.py ===
    cp "$SCRIPT_DIR"/plot_*.py "$LEVEL2_DIR"/

    # === 6. 进入目标目录并运行 ===
    (
        cd "$LEVEL2_DIR"
        for py in plot_*.py; do
            echo "Running $py ..."
            python3 "$py"
        done
    )

    echo "Finished $SIM_ROOT"
    echo
done

echo "All plot scripts finished successfully."
