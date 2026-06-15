#!/usr/bin/env python3
# 作用：扫描 hw_run 中已有 benchmark trace，统计每个 benchmark 的每个 kernel 的 SASS opcode 组成。
# kernel 名按 kernelslist.g 中出现顺序输出为 benchmark-k1、benchmark-k2 ...
# benchmark 名只保留 “-rodinia” 前的部分，例如 backprop-rodinia-2.0-ft 输出为 backprop。
#
# 默认输入：
#   /home/lyhong/Downloads/accel-sim-framework/hw_run
#
# 默认输出：
#   /home/lyhong/Downloads/accel-sim-framework/IPC-sm/training_data/opcode_mix.csv
#   CSV 列为：benchmark-kernel,opcode,percent,count,total
#
# 用法：
#   ./scripts/kernel_analysis/analyze_hw_run_kernel_opcodes.py
#   ./scripts/kernel_analysis/analyze_hw_run_kernel_opcodes.py --hw-run ./hw_run
#   ./scripts/kernel_analysis/analyze_hw_run_kernel_opcodes.py --hw-run <hw_run目录> --out-dir <输出目录>

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/home/lyhong/Downloads/accel-sim-framework")
DEFAULT_HW_RUN = ROOT / "hw_run"
DEFAULT_OUT_DIR = ROOT / "IPC-sm" / "training_data"
DEFAULT_ORDER_CSV = ROOT / "trace_semantic_instruction_pressure_plots_nomix" / "semantic_opcode_order.csv"
KERNEL_NAME_RE = re.compile(r"^(.*)-k(\d+)$")


def short_benchmark_name(name):
    name = name.strip()
    if "-rodinia" in name:
        return name.split("-rodinia", 1)[0]
    return name


def benchmark_from_trace_root(trace_root, hw_run):
    rel = trace_root.resolve().relative_to(hw_run.resolve())
    parts = rel.parts
    if len(parts) >= 3:
        return short_benchmark_name(parts[2])
    return short_benchmark_name(trace_root.parent.parent.name)


def find_trace_roots(hw_run):
    for path in sorted(hw_run.rglob("kernelslist.g")):
        yield path.parent


def kernel_trace_files(trace_root):
    kernelslist = trace_root / "kernelslist.g"
    files = []
    for line in kernelslist.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("kernel") and stripped.endswith(".traceg"):
            trace_file = trace_root / stripped
            if trace_file.is_file():
                files.append(trace_file)
    return files


def parse_opcode_from_trace_line(line):
    parts = line.split()
    if len(parts) < 8:
        return None
    if not parts[0].lstrip("-").isdigit():
        return None
    if not parts[1].lstrip("-").isdigit() or not parts[2].lstrip("-").isdigit():
        return None

    # trace format:
    # tb_x tb_y tb_z warpid_tb PC mask dest_num reg_dests opcode src_num ...
    # opcode position depends on dest_num, so fixed-column parsing is wrong.
    try:
        dest_num = int(parts[6])
    except ValueError:
        return None
    opcode_index = 7 + dest_num
    if opcode_index >= len(parts):
        return None
    return parts[opcode_index]


def count_kernel_opcodes(trace_file):
    hist = Counter()
    with trace_file.open("r", errors="ignore") as f:
        for line in f:
            opcode = parse_opcode_from_trace_line(line)
            if opcode:
                hist[opcode] += 1
    return hist


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


def benchmark_kernel_sort_key(name):
    match = KERNEL_NAME_RE.match(name)
    if match:
        return (match.group(1), int(match.group(2)))
    return (name, 10**9)


def collect_kernel_histograms(hw_run):
    hist_by_kernel = defaultdict(Counter)
    missing = []
    for trace_root in find_trace_roots(hw_run):
        benchmark = benchmark_from_trace_root(trace_root, hw_run)
        traces = kernel_trace_files(trace_root)
        if not traces:
            missing.append(str(trace_root))
            continue
        for kernel_idx, trace_file in enumerate(traces, 1):
            hist = count_kernel_opcodes(trace_file)
            if hist:
                hist_by_kernel[f"{benchmark}-k{kernel_idx}"].update(hist)
    return hist_by_kernel, missing


def write_opcode_mix(path, hist_by_kernel, order):
    order_index = {opcode: i for i, opcode in enumerate(order)}
    rows = []
    for benchmark_kernel in sorted(hist_by_kernel, key=benchmark_kernel_sort_key):
        hist = hist_by_kernel[benchmark_kernel]
        total = sum(hist.values())
        for opcode in sorted(hist, key=lambda op: opcode_sort_key(op, order_index)):
            count = hist[opcode]
            percent = (count * 100.0 / total) if total else 0.0
            rows.append(
                {
                    "benchmark-kernel": benchmark_kernel,
                    "opcode": opcode,
                    "percent": f"{percent:.5f}",
                    "count": count,
                    "total": total,
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["benchmark-kernel", "opcode", "percent", "count", "total"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_opcode_heatmap(path, rows, order):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib is not installed; skip opcode heatmap")
        return False

    if not rows:
        return False

    kernels = sorted({row["benchmark-kernel"] for row in rows}, key=benchmark_kernel_sort_key)
    opcodes_seen = {row["opcode"] for row in rows}
    ordered_opcodes = [opcode for opcode in order if opcode in opcodes_seen]
    ordered_opcodes.extend(sorted(opcodes_seen - set(ordered_opcodes)))
    value_by_key = {(row["benchmark-kernel"], row["opcode"]): float(row["percent"]) for row in rows}
    matrix = [[value_by_key.get((kernel, opcode), 0.0) for opcode in ordered_opcodes] for kernel in kernels]

    fig_width = max(12.0, len(ordered_opcodes) * 0.24)
    fig_height = max(6.0, len(kernels) * 0.16)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title("Raw SASS opcode percent by benchmark-kernel")
    ax.set_xlabel("opcode")
    ax.set_ylabel("benchmark-kernel")

    ax.set_xticks(range(len(ordered_opcodes)))
    ax.set_xticklabels(ordered_opcodes, rotation=90, fontsize=6)
    ax.set_yticks(range(len(kernels)))
    ax.set_yticklabels(kernels, fontsize=5)
    fig.colorbar(image, ax=ax, label="percent (%)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Analyze per-kernel opcode distribution for traces under hw_run.")
    parser.add_argument("--hw-run", type=Path, default=DEFAULT_HW_RUN, help="hw_run root directory")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--opcode-order", type=Path, default=DEFAULT_ORDER_CSV, help="Optional opcode order CSV")
    args = parser.parse_args()

    hw_run = args.hw_run.resolve()
    out_dir = args.out_dir.resolve()
    if not hw_run.is_dir():
        raise SystemExit(f"hw_run directory does not exist: {hw_run}")

    order = load_opcode_order(args.opcode_order.resolve())
    hist_by_kernel, missing = collect_kernel_histograms(hw_run)
    if not hist_by_kernel:
        raise SystemExit(f"No kernel trace opcode data found under {hw_run}")

    out_csv = out_dir / "opcode_mix.csv"
    rows = write_opcode_mix(out_csv, hist_by_kernel, order)
    heatmap_png = out_dir / "all_components_raw_sass_opcode_percent_heatmap.png"
    heatmap_written = plot_opcode_heatmap(heatmap_png, rows, order)

    if missing:
        with (out_dir / "missing_kernel_traces.txt").open("w") as f:
            for path in missing:
                f.write(path + "\n")

    print(f"Analyzed {len(hist_by_kernel)} benchmark-kernel item(s)")
    print(f"Wrote {out_csv}")
    if heatmap_written:
        print(f"Wrote {heatmap_png}")
    if missing:
        print(f"WARNING: {len(missing)} trace root(s) had no kernel trace files; see missing_kernel_traces.txt")


if __name__ == "__main__":
    main()
