#!/usr/bin/env python3

import argparse
import csv
import json
import os
from collections import Counter, defaultdict


GLOBAL_LOAD_PREFIXES = ("LDG", "LD.E", "LD.")
GLOBAL_STORE_PREFIXES = ("STG", "ST.E", "ST.")
LOCAL_PREFIXES = ("LDL", "STL")
SHARED_PREFIXES = ("LDS", "STS")
CONST_PREFIXES = ("LDC", "ULDC")
ATOMIC_PREFIXES = ("ATOM", "ATOMG", "RED")
CONTROL_PREFIXES = ("BRA", "BRX", "CALL", "RET", "EXIT", "SSY", "SYNC", "BAR")


def is_trace_line(tokens):
    if len(tokens) < 8:
        return False
    return all(is_int(tokens[i]) for i in (0, 1, 2, 3, 6))


def is_int(value):
    try:
        int(value, 0)
        return True
    except ValueError:
        return False


def active_lanes(mask):
    try:
        return bin(int(mask, 16)).count("1")
    except ValueError:
        return 0


def classify_opcode(opcode):
    op = opcode.upper()
    if op.startswith(GLOBAL_LOAD_PREFIXES):
        return "global_load"
    if op.startswith(GLOBAL_STORE_PREFIXES):
        return "global_store"
    if op.startswith(ATOMIC_PREFIXES):
        return "atomic"
    if op.startswith(SHARED_PREFIXES):
        return "shared_mem"
    if op.startswith(LOCAL_PREFIXES):
        return "local_mem"
    if op.startswith(CONST_PREFIXES):
        return "const_mem"
    if op.startswith(CONTROL_PREFIXES):
        return "control"
    if op.startswith(("F", "D", "H")):
        return "floating_point"
    if op.startswith(("I", "U", "V", "P", "LOP", "SHF", "SHL", "SHR", "LEA")):
        return "integer"
    return "other"


def parse_inst(tokens):
    dest_num = int(tokens[6])
    opcode_idx = 7 + dest_num
    if opcode_idx >= len(tokens):
        return None

    opcode = tokens[opcode_idx]
    src_idx = opcode_idx + 1
    if src_idx >= len(tokens) or not is_int(tokens[src_idx]):
        return opcode, 0, []

    src_num = int(tokens[src_idx])
    mem_width_idx = src_idx + 1 + src_num
    if mem_width_idx >= len(tokens) or not is_int(tokens[mem_width_idx]):
        return opcode, 0, []

    mem_width = int(tokens[mem_width_idx], 0)
    addresses = [int(tok, 16) for tok in tokens[mem_width_idx + 1 :] if tok.startswith("0x")]
    return opcode, mem_width, addresses


def parse_kernelslist(path):
    memcpy = Counter()
    kernels = []
    with open(path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.endswith(".traceg"):
                kernels.append(line)
                continue
            parts = line.split(",")
            if len(parts) == 3 and parts[0].startswith("Memcpy") and is_int(parts[2]):
                memcpy[parts[0]] += int(parts[2], 0)
    return kernels, memcpy


def kernel_component_name(kernel_name, fallback):
    parts = os.path.basename(kernel_name).split("-")
    if len(parts) >= 3 and parts[0] == "kernel":
        return parts[1]
    return fallback


def parse_kernelslist_events(path):
    pending_memcpy = Counter()
    with open(path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.endswith(".traceg"):
                yield line, pending_memcpy
                pending_memcpy = Counter()
                continue
            parts = line.split(",")
            if len(parts) == 3 and parts[0].startswith("Memcpy") and is_int(parts[2]):
                pending_memcpy[parts[0]] += int(parts[2], 0)



def analyze_trace(path):
    stats = Counter()
    opcode_hist = Counter()
    sectors_128b = set()

    with open(path, "r", errors="ignore") as f:
        for raw in f:
            if not raw or raw[0] in "#-\n":
                continue
            tokens = raw.split()
            if not is_trace_line(tokens):
                continue
            parsed = parse_inst(tokens)
            if parsed is None:
                continue

            opcode, mem_width, addresses = parsed
            category = classify_opcode(opcode)
            lanes = active_lanes(tokens[5])
            opcode_hist[opcode] += 1
            stats["instructions"] += 1
            stats[category + "_inst"] += 1
            stats["active_lanes"] += lanes

            if category in ("global_load", "global_store", "atomic"):
                stats["global_mem_inst"] += 1
                stats["global_mem_bytes"] += mem_width * max(1, len(addresses))
                stats["global_mem_address_samples"] += len(addresses)
                for addr in addresses:
                    sectors_128b.add(addr // 128)

    stats["unique_128b_sectors"] = len(sectors_128b)
    return stats, opcode_hist


def score_dram_dependency(row):
    mem_ratio = row["global_mem_inst"] / row["instructions"] if row["instructions"] else 0.0
    bytes_per_inst = row["global_mem_bytes"] / row["instructions"] if row["instructions"] else 0.0
    sector_pressure = row["unique_128b_sectors"] / row["global_mem_inst"] if row["global_mem_inst"] else 0.0

    score = 0.0
    score += min(mem_ratio / 0.35, 1.0) * 45.0
    score += min(bytes_per_inst / 8.0, 1.0) * 35.0
    score += min(sector_pressure / 4.0, 1.0) * 20.0
    return round(score, 2)


def analyze_benchmark(trace_root):
    kernel_list = os.path.join(trace_root, "kernelslist.g")
    kernels, memcpy = parse_kernelslist(kernel_list)
    total = Counter()
    opcode_hist = Counter()

    for kernel in kernels:
        trace_path = os.path.join(trace_root, kernel)
        if not os.path.isfile(trace_path):
            total["missing_trace_files"] += 1
            continue
        stats, hist = analyze_trace(trace_path)
        total.update(stats)
        opcode_hist.update(hist)

    total["kernel_count"] = len(kernels)
    total["memcpy_htod_bytes"] = memcpy["MemcpyHtoD"]
    total["memcpy_dtoh_bytes"] = memcpy["MemcpyDtoH"]
    total["memcpy_dtod_bytes"] = memcpy["MemcpyDtoD"]
    total["memcpy_total_bytes"] = sum(memcpy.values())
    total["dram_dependency_score"] = score_dram_dependency(total)
    return total, opcode_hist


def analyze_benchmark_components(trace_root, benchmark):
    kernel_list = os.path.join(trace_root, "kernelslist.g")
    totals = defaultdict(Counter)
    opcode_hists = defaultdict(Counter)

    for kernel, memcpy in parse_kernelslist_events(kernel_list):
        component = kernel_component_name(kernel, benchmark)
        total = totals[component]
        total["kernel_count"] += 1
        total["memcpy_htod_bytes"] += memcpy["MemcpyHtoD"]
        total["memcpy_dtoh_bytes"] += memcpy["MemcpyDtoH"]
        total["memcpy_dtod_bytes"] += memcpy["MemcpyDtoD"]
        total["memcpy_total_bytes"] += sum(memcpy.values())

        trace_path = os.path.join(trace_root, kernel)
        if not os.path.isfile(trace_path):
            total["missing_trace_files"] += 1
            continue
        stats, hist = analyze_trace(trace_path)
        total.update(stats)
        opcode_hists[component].update(hist)

    for total in totals.values():
        total["dram_dependency_score"] = score_dram_dependency(total)
    return totals, opcode_hists



def resolve_trace_labels(trace_root, input_root=None, benchmark=None, app_args=None):
    if benchmark is None or app_args is None:
        if input_root is None:
            benchmark = benchmark or os.path.basename(os.path.dirname(trace_root))
            app_args = app_args or "NO_ARGS"
        else:
            rel_benchmark, rel_args = rel_parts(trace_root, input_root)
            benchmark = benchmark or rel_benchmark
            app_args = app_args or rel_args
    return benchmark, app_args


def analyze_trace_root(trace_root, input_root=None, benchmark=None, app_args=None):
    benchmark, app_args = resolve_trace_labels(trace_root, input_root, benchmark, app_args)
    stats, hist = analyze_benchmark(trace_root)
    row = defaultdict(int)
    row.update(stats)
    row["benchmark"] = benchmark
    row["component"] = benchmark
    row["args"] = app_args
    row["trace_root"] = trace_root
    return dict(row), hist


def analyze_trace_root_by_component(trace_root, input_root=None, benchmark=None, app_args=None):
    benchmark, app_args = resolve_trace_labels(trace_root, input_root, benchmark, app_args)
    totals, hists = analyze_benchmark_components(trace_root, benchmark)
    rows = []
    opcode_rows = []
    for component in sorted(totals):
        row = defaultdict(int)
        row.update(totals[component])
        row["benchmark"] = benchmark
        row["component"] = component
        row["args"] = app_args
        row["trace_root"] = trace_root
        rows.append(dict(row))
        opcode_rows.append((benchmark, component, app_args, hists[component]))
    return rows, opcode_rows


def find_trace_roots(root):
    for dirpath, _, filenames in os.walk(root):
        if "kernelslist.g" in filenames:
            yield dirpath


def rel_parts(trace_root, input_root):
    rel = os.path.relpath(os.path.dirname(trace_root), input_root)
    parts = rel.split(os.sep)
    benchmark = parts[0] if parts else rel
    args = os.path.join(*parts[1:]) if len(parts) > 1 else "NO_ARGS"
    return benchmark, args


def write_outputs(rows, opcode_rows, out_prefix):
    csv_path = out_prefix + ".csv"
    json_path = out_prefix + ".json"
    opcode_path = out_prefix + ".opcodes.csv"

    fieldnames = [
        "benchmark",
        "component",
        "args",
        "trace_root",
        "kernel_count",
        "instructions",
        "global_mem_inst",
        "global_load_inst",
        "global_store_inst",
        "atomic_inst",
        "shared_mem_inst",
        "local_mem_inst",
        "const_mem_inst",
        "floating_point_inst",
        "integer_inst",
        "control_inst",
        "other_inst",
        "active_lanes",
        "global_mem_bytes",
        "unique_128b_sectors",
        "global_mem_address_samples",
        "memcpy_total_bytes",
        "memcpy_htod_bytes",
        "memcpy_dtoh_bytes",
        "memcpy_dtod_bytes",
        "missing_trace_files",
        "dram_dependency_score",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, 0) for key in fieldnames})

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)

    with open(opcode_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark", "component", "args", "opcode", "count"])
        for bench, component, args, hist in opcode_rows:
            for opcode, count in hist.most_common():
                writer.writerow([bench, component, args, opcode, count])

    return csv_path, json_path, opcode_path


def re_safe_filename(name):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def write_plots(rows, opcode_rows, out_prefix):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping trace feature plots")
        return []

    if not rows:
        return []

    plot_dir = out_prefix + ".plots"
    os.makedirs(plot_dir, exist_ok=True)
    labels = ["{0}\n{1}\n{2}".format(row.get("benchmark", ""), row.get("component", ""), row.get("args", "")) for row in rows]
    x = list(range(len(rows)))
    written = []

    def save(fig, name):
        path = os.path.join(plot_dir, name)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.8), 4.5))
    metrics = [
        ("instructions", "instructions"),
        ("global_mem_inst", "global mem inst"),
        ("global_mem_bytes", "global mem bytes"),
        ("unique_128b_sectors", "unique 128B sectors"),
        ("memcpy_total_bytes", "memcpy bytes"),
    ]
    width = 0.8 / len(metrics)
    for idx, (key, label) in enumerate(metrics):
        values = [float(row.get(key, 0)) for row in rows]
        offset = (idx - (len(metrics) - 1) / 2.0) * width
        ax.bar([pos + offset for pos in x], values, width=width, label=label)
    ax.set_yscale("symlog")
    ax.set_ylabel("raw count / bytes (symlog)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(fontsize=8)
    ax.set_title("Raw Trace Feature Counts")
    save(fig, "raw_feature_counts.png")

    mix_keys = [
        ("global_load_inst", "global load"),
        ("global_store_inst", "global store"),
        ("atomic_inst", "atomic/reduction"),
        ("shared_mem_inst", "shared mem"),
        ("local_mem_inst", "local mem"),
        ("const_mem_inst", "const mem"),
        ("floating_point_inst", "floating point"),
        ("integer_inst", "integer"),
        ("control_inst", "control"),
        ("other_inst", "other"),
    ]
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.6), 4.8))
    bottoms = [0.0] * len(rows)
    for key, label in mix_keys:
        values = [float(row.get(key, 0)) for row in rows]
        ax.bar(x, values, bottom=bottoms, label=label)
        bottoms = [bottoms[i] + values[i] for i in range(len(rows))]
    ax.set_ylabel("instruction count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Raw Instruction Category Mix")
    save(fig, "instruction_category_mix.png")

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.6), 4.5))
    memory_metrics = [
        ("global_load_inst", "global load inst"),
        ("global_store_inst", "global store inst"),
        ("atomic_inst", "atomic inst"),
        ("global_mem_address_samples", "address samples"),
        ("unique_128b_sectors", "unique 128B sectors"),
        ("memcpy_total_bytes", "memcpy bytes"),
    ]
    width = 0.8 / len(memory_metrics)
    for idx, (key, label) in enumerate(memory_metrics):
        values = [float(row.get(key, 0)) for row in rows]
        offset = (idx - (len(memory_metrics) - 1) / 2.0) * width
        ax.bar([pos + offset for pos in x], values, width=width, label=label)
    ax.set_yscale("symlog")
    ax.set_ylabel("raw count / bytes (symlog)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(fontsize=8)
    ax.set_title("Raw Memory Trace Counts")
    save(fig, "raw_memory_counts.png")

    for bench, component, app_args, hist in opcode_rows:
        if not hist:
            continue
        top = hist.most_common(20)
        fig, ax = plt.subplots(figsize=(9, 5))
        names = [item[0] for item in top]
        values = [item[1] for item in top]
        ax.bar(range(len(top)), values)
        ax.set_ylabel("count")
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_title("Top Opcodes: {0} {1} {2}".format(bench, component, app_args))
        save(fig, re_safe_filename("{0}_{1}_{2}_top_opcodes.png".format(bench, component, app_args)))

    return written


def main():
    parser = argparse.ArgumentParser(
        description="Pre-analyze Accel-Sim SASS trace instruction and memory features."
    )
    parser.add_argument("-T", "--trace-dir", required=True, help="Root directory passed to run_simulations.py -T")
    parser.add_argument(
        "-o",
        "--out-prefix",
        default="trace_feature_summary",
        help="Output prefix for .csv, .json, and .opcodes.csv files.",
    )
    parser.add_argument(
        "--single-trace-root",
        action="store_true",
        help="Treat --trace-dir as one selected traces directory instead of scanning below it.",
    )
    parser.add_argument("--benchmark", default=None, help="Benchmark name for --single-trace-root output.")
    parser.add_argument("--args-name", default=None, help="Argument/input name for --single-trace-root output.")
    parser.add_argument("--plot", action="store_true", help="Also write PNG plots under OUT_PREFIX.plots.")
    args = parser.parse_args()

    rows = []
    opcode_rows = []
    if args.single_trace_root:
        new_rows, new_opcode_rows = analyze_trace_root_by_component(
            args.trace_dir, benchmark=args.benchmark, app_args=args.args_name
        )
        rows.extend(new_rows)
        opcode_rows.extend(new_opcode_rows)
    else:
        for trace_root in sorted(find_trace_roots(args.trace_dir)):
            new_rows, new_opcode_rows = analyze_trace_root_by_component(trace_root, input_root=args.trace_dir)
            rows.extend(new_rows)
            opcode_rows.extend(new_opcode_rows)

    csv_path, json_path, opcode_path = write_outputs(rows, opcode_rows, args.out_prefix)
    plot_paths = write_plots(rows, opcode_rows, args.out_prefix) if args.plot else []
    print("Analyzed {0} component rows".format(len(rows)))
    print("Wrote {0}".format(csv_path))
    print("Wrote {0}".format(json_path))
    print("Wrote {0}".format(opcode_path))
    for path in plot_paths:
        print("Wrote {0}".format(path))


if __name__ == "__main__":
    main()
