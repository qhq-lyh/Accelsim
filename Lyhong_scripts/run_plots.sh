#!/bin/bash
set -e

# === 1. 获取当前工作目录（运行脚本的位置）===
ROOT_DIR="$(pwd)"

SCRIPT_DIR="$ROOT_DIR/Lyhong_scripts"
SIM_ROOT="$ROOT_DIR/sim_run_11.8/bfs-rodinia-2.0-ft"

# === 2. 找到第一层唯一子目录 ===
LEVEL1_DIR=$(find "$SIM_ROOT" -mindepth 1 -maxdepth 1 -type d | head -n 1)

if [ -z "$LEVEL1_DIR" ]; then
    echo "Error: no subdirectory found in $SIM_ROOT"
    exit 1
fi

# === 3. 找到第二层唯一子目录 ===
LEVEL2_DIR=$(find "$LEVEL1_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)

if [ -z "$LEVEL2_DIR" ]; then
    echo "Error: no subdirectory found in $LEVEL1_DIR"
    exit 1
fi

echo "Target directory: $LEVEL2_DIR"

# === 4. 复制 plot_*.py ===
cp "$SCRIPT_DIR"/plot_*.py "$LEVEL2_DIR"/

# === 5. 进入目标目录并运行 ===
cd "$LEVEL2_DIR"

for py in plot_*.py; do
    echo "Running $py ..."
    python3 "$py"
done

echo "All plot scripts finished successfully."
