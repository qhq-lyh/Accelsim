#!/usr/bin/env python3
# 作用：分析一次 Accel-Sim 运行将要使用的 trace 中各 benchmark/component 的 SASS opcode 组成。
# 对单 benchmark，输出该 benchmark 自己的 opcode 统计；对 mixed benchmark，
# 按 kernel-<benchmark>-*.traceg 的命名把 bfs、hotspot 等 component 分开统计。
#
# 默认输出：
#   IPC-sm/opcode_analysis/<benchmark>__<args>/
#   其中包含 opcode_counts.csv、opcode_percentages.csv、trace_component_summary.csv，
#   以及各 benchmark 的 opcode 数量图、opcode 百分比图和总 heatmap。
#
# 用法：
#   ./scripts/analyze_run_opcode_mix.py --trace-root <traces目录> --benchmark bfs --args-name input1
#   ./scripts/analyze_run_opcode_mix.py --trace-root <mixed_traces目录> --benchmark bfs-hotspot-rodinia-2.0-ft --args-name bfs10_hotspot40

import argparse
import csv
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UTIL_DIR = ROOT / "util" / "job_launching"
if str(UTIL_DIR) not in sys.path:
    sys.path.insert(0, str(UTIL_DIR))

import analyze_trace_features

DEFAULT_OUT_DIR = ROOT / "IPC-sm" / "opcode_analysis"
DEFAULT_ORDER_CSV = ROOT / "trace_semantic_instruction_pressure_plots_nomix" / "semantic_opcode_order.csv"


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "unnamed"


def load_opcode_order(path):
    order = []
    module = {}
    if not path.is_file():
        return order, module
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            opcode = row.get("opcode", "").strip()
            if not opcode:
                continue
            order.append(opcode)
            module[opcode] = row.get("primary_module", "unknown").strip() or "unknown"
    return order, module


def opcode_sort_key(opcode, index):
    return (index.get(opcode, 10**9), opcode)


def semantic_module(opcode, module_by_opcode):
    if opcode in module_by_opcode:
        return module_by_opcode[opcode]
    return analyze_trace_features.classify_opcode(opcode)


def component_rows(opcode_rows, module_by_opcode, order):
    index = {opcode: idx for idx, opcode in enumerate(order)}
    count_rows = []
    percent_rows = []

    for benchmark, component, args_name, hist in opcode_rows:
        total = sum(hist.values())
        for opcode in sorted(hist, key=lambda op: opcode_sort_key(op, index)):
            count = hist[opcode]
            percent = (count * 100.0 / total) if total else 0.0
            module = semantic_module(opcode, module_by_opcode)
            row = {
                "benchmark": benchmark,
                "component": component,
                "args": args_name,
                "opcode": opcode,
                "primary_module": module,
                "count": count,
                "percent": percent,
            }
            count_rows.append({k: row[k] for k in ["benchmark", "component", "args", "opcode", "primary_module", "count"]})
            percent_rows.append(row)

    return count_rows, percent_rows


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(out_dir, label, rows, percent_rows, order):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skip opcode plots")
        return []

    written = []
    components = sorted({(r["benchmark"], r["component"], r["args"]) for r in percent_rows})

    def save(fig, filename):
        path = out_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    for benchmark, component, args_name in components:
        comp_percent = [r for r in percent_rows if r["benchmark"] == benchmark and r["component"] == component and r["args"] == args_name]
        comp_counts = [r for r in rows if r["benchmark"] == benchmark and r["component"] == component and r["args"] == args_name]
        comp_percent = sorted(comp_percent, key=lambda r: r["percent"], reverse=True)[:30]
        comp_counts = sorted(comp_counts, key=lambda r: int(r["count"]), reverse=True)[:30]
        name = safe_name(f"{label}_{component}")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(comp_percent)), [r["percent"] for r in comp_percent])
        ax.set_ylabel("opcode percent (%)")
        ax.set_xticks(range(len(comp_percent)))
        ax.set_xticklabels([r["opcode"] for r in comp_percent], rotation=45, ha="right")
        ax.set_title(f"Opcode Percent: {component}")
        save(fig, f"{name}_opcode_percent.png")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(comp_counts)), [int(r["count"]) for r in comp_counts])
        ax.set_ylabel("opcode count")
        ax.set_xticks(range(len(comp_counts)))
        ax.set_xticklabels([r["opcode"] for r in comp_counts], rotation=45, ha="right")
        ax.set_title(f"Opcode Counts: {component}")
        save(fig, f"{name}_opcode_counts.png")

    heatmap_opcodes = order[:]
    seen = {r["opcode"] for r in percent_rows}
    heatmap_opcodes.extend(sorted(seen - set(heatmap_opcodes)))
    heatmap_opcodes = [op for op in heatmap_opcodes if op in seen]
    if heatmap_opcodes and components:
        matrix = []
        labels = []
        lookup = {(r["component"], r["opcode"]): r["percent"] for r in percent_rows}
        for _, component, _ in components:
            labels.append(component)
            matrix.append([lookup.get((component, op), 0.0) for op in heatmap_opcodes])
        fig, ax = plt.subplots(figsize=(max(10, len(heatmap_opcodes) * 0.22), max(3, len(labels) * 0.55)))
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(heatmap_opcodes)))
        ax.set_xticklabels(heatmap_opcodes, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title("Opcode Percent Heatmap")
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("percent (%)")
        save(fig, f"{safe_name(label)}_opcode_percent_heatmap.png")

    return written


def main():
    parser = argparse.ArgumentParser(description="Analyze opcode composition for one selected Accel-Sim run trace.")
    parser.add_argument("--trace-root", required=True, type=Path, help="Selected traces directory containing kernelslist.g")
    parser.add_argument("--benchmark", required=True, help="Benchmark name from run_simulations.py")
    parser.add_argument("--args-name", required=True, help="Argument/input folder name from run_simulations.py")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output root under IPC-sm")
    parser.add_argument("--opcode-order", type=Path, default=DEFAULT_ORDER_CSV, help="Optional semantic opcode order CSV")
    args = parser.parse_args()

    if not (args.trace_root / "kernelslist.g").is_file():
        raise SystemExit(f"missing kernelslist.g under trace root: {args.trace_root}")

    label = f"{safe_name(args.benchmark)}__{safe_name(args.args_name)}"
    out_dir = args.out_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, opcode_rows = analyze_trace_features.analyze_trace_root_by_component(
        str(args.trace_root), benchmark=args.benchmark, app_args=args.args_name
    )
    order, module_by_opcode = load_opcode_order(args.opcode_order)
    count_rows, percent_rows = component_rows(opcode_rows, module_by_opcode, order)

    write_csv(
        out_dir / "trace_component_summary.csv",
        rows,
        [
            "benchmark", "component", "args", "trace_root", "kernel_count", "instructions",
            "global_mem_inst", "global_load_inst", "global_store_inst", "atomic_inst",
            "shared_mem_inst", "local_mem_inst", "const_mem_inst", "floating_point_inst",
            "integer_inst", "control_inst", "other_inst", "active_lanes", "global_mem_bytes",
            "unique_128b_sectors", "global_mem_address_samples", "memcpy_total_bytes",
            "memcpy_htod_bytes", "memcpy_dtoh_bytes", "memcpy_dtod_bytes", "missing_trace_files",
            "dram_dependency_score",
        ],
    )
    write_csv(out_dir / "opcode_counts.csv", count_rows, ["benchmark", "component", "args", "opcode", "primary_module", "count"])
    write_csv(out_dir / "opcode_percentages.csv", percent_rows, ["benchmark", "component", "args", "opcode", "primary_module", "count", "percent"])
    plots = make_plots(out_dir, label, count_rows, percent_rows, order)

    print(f"Analyzed trace root: {args.trace_root}")
    print(f"Wrote opcode analysis to {out_dir}")
    for path in plots:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
