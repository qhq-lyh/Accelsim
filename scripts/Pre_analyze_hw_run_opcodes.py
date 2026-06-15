#!/usr/bin/env python3
# 作用：扫描 hw_run 中已有 benchmark trace，统计每个 benchmark 的 SASS opcode 组成。
# benchmark 名称只保留 “-rodinia” 前的部分，例如 bfs-rodinia-2.0-ft 输出为 bfs。
#
# 默认输入：
#   /home/lyhong/Downloads/accel-sim-framework/hw_run
#
# 默认输出：
#   /home/lyhong/Downloads/accel-sim-framework/trace_semantic/opcode_mix.csv
#   /home/lyhong/Downloads/accel-sim-framework/trace_semantic/opcode_distribution.png
#
# 用法：
#   ./scripts/analyze_hw_run_opcodes.py
#   ./scripts/analyze_hw_run_opcodes.py --hw-run ./hw_run --out-dir ./trace_semantic

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/home/lyhong/Downloads/accel-sim-framework")
UTIL_DIR = ROOT / "util" / "job_launching"
if str(UTIL_DIR) not in sys.path:
    sys.path.insert(0, str(UTIL_DIR))

import analyze_trace_features

DEFAULT_HW_RUN = ROOT / "hw_run"
DEFAULT_OUT_DIR = ROOT / "trace_semantic"
DEFAULT_ORDER_CSV = ROOT / "trace_semantic_instruction_pressure_plots_nomix" / "semantic_opcode_order.csv"


def short_benchmark_name(name):
    name = name.strip()
    if "-rodinia" in name:
        return name.split("-rodinia", 1)[0]
    return name


def is_mixed_benchmark(short_name):
    return "-" in short_name


def benchmark_from_trace_root(trace_root, hw_run):
    rel = trace_root.resolve().relative_to(hw_run.resolve())
    parts = rel.parts
    if len(parts) < 4:
        return short_benchmark_name(trace_root.parent.parent.name)
    return short_benchmark_name(parts[2])


def find_trace_roots(hw_run):
    for path in sorted(hw_run.rglob("kernelslist.g")):
        yield path.parent


def load_opcode_order(path):
    order = []
    if not path.is_file():
        return order
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            opcode = row.get("opcode", "").strip()
            if opcode:
                order.append(opcode)
    return order


def opcode_sort_key(opcode, order_index):
    return (order_index.get(opcode, 10**9), opcode)


def collect_opcode_histograms(hw_run):
    hist_by_benchmark = defaultdict(Counter)
    trace_count = Counter()
    skipped_mixed = []
    for trace_root in find_trace_roots(hw_run):
        benchmark = benchmark_from_trace_root(trace_root, hw_run)
        if is_mixed_benchmark(benchmark):
            skipped_mixed.append(str(trace_root))
            continue
        _stats, hist = analyze_trace_features.analyze_benchmark(str(trace_root))
        hist_by_benchmark[benchmark].update(hist)
        trace_count[benchmark] += 1
    return hist_by_benchmark, trace_count, skipped_mixed


def write_mix_csv(path, hist_by_benchmark, order):
    order_index = {opcode: i for i, opcode in enumerate(order)}
    rows = []
    for benchmark in sorted(hist_by_benchmark):
        hist = hist_by_benchmark[benchmark]
        total = sum(hist.values())
        for opcode in sorted(hist, key=lambda op: opcode_sort_key(op, order_index)):
            count = hist[opcode]
            percent = (count * 100.0 / total) if total else 0.0
            rows.append(
                {
                    "benchmark": benchmark,
                    "opcode": opcode,
                    "percent": f"{percent:.5f}",
                    "count": count,
                    "total": total,
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames = ["benchmark", "opcode", "percent", "count", "total"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_distribution(path, rows, order):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot")
        return None

    if not rows:
        return None

    benchmarks = sorted({row["benchmark"] for row in rows})
    seen_opcodes = {row["opcode"] for row in rows}
    ordered_opcodes = [opcode for opcode in order if opcode in seen_opcodes]
    ordered_opcodes.extend(sorted(seen_opcodes - set(ordered_opcodes)))

    lookup = {(row["benchmark"], row["opcode"]): float(row["percent"]) for row in rows}
    matrix = [[lookup.get((benchmark, opcode), 0.0) for opcode in ordered_opcodes] for benchmark in benchmarks]

    fig_w = max(10, len(ordered_opcodes) * 0.24)
    fig_h = max(4, len(benchmarks) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ordered_opcodes)))
    ax.set_xticklabels(ordered_opcodes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(benchmarks)))
    ax.set_yticklabels(benchmarks)
    ax.set_title("Opcode distribution by benchmark")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("percent (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Analyze opcode distribution for traces under hw_run.")
    parser.add_argument("--hw-run", type=Path, default=DEFAULT_HW_RUN, help="hw_run root directory")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--opcode-order", type=Path, default=DEFAULT_ORDER_CSV, help="Optional opcode order CSV")
    args = parser.parse_args()

    hw_run = args.hw_run.resolve()
    out_dir = args.out_dir.resolve()
    if not hw_run.is_dir():
        raise SystemExit(f"hw_run directory does not exist: {hw_run}")

    order = load_opcode_order(args.opcode_order.resolve())
    hist_by_benchmark, trace_count, skipped_mixed = collect_opcode_histograms(hw_run)
    if not hist_by_benchmark:
        raise SystemExit(f"No kernelslist.g found under {hw_run}")

    csv_path = out_dir / "opcode_mix.csv"
    png_path = out_dir / "opcode_distribution.png"
    rows = write_mix_csv(csv_path, hist_by_benchmark, order)
    plot_path = plot_distribution(png_path, rows, order)

    print(f"Analyzed {len(hist_by_benchmark)} benchmark(s) under {hw_run}")
    if skipped_mixed:
        print(f"Skipped {len(skipped_mixed)} mixed benchmark trace root(s)")
    print(f"Wrote {csv_path}")
    if plot_path:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
