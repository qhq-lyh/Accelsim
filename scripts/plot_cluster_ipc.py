#!/usr/bin/env python3
# 作用：从不同 SM 数量 sweep 的 Accel-Sim 仿真输出中提取每个 benchmark 的 IPC 曲线。
# 脚本会扫描 sim_run_12.8-<sm> 目录下的 .o* 输出文件，读取最后一次出现的
# gpu_tot_ipc，将同一 benchmark、同一 SM 数量的结果求平均，并生成 lookup 表和图片。
#
# 默认输入：
#   /home/lyhong/Downloads/accel-sim-framework/sim_run_12.8-signle-4G
#   目录结构应类似：sim_run_12.8-2、sim_run_12.8-3、...，每个目录内包含仿真 .o* 输出。
#
# 默认输出：
#   <root>/png/IPC/ipc_lookup.csv   ：Alg1 使用的 benchmark,sm,ipc,sample_count 表。
#   <root>/png/IPC/ipc_lookup.json  ：同一 lookup 的 JSON 版本。
#   <root>/png/IPC/ipc_detail.csv   ：每个 .o* 文件提取到的原始 IPC 记录。
#   <root>/png/IPC/*.png            ：每个 benchmark 和所有 benchmark 的 IPC-SM 曲线图。
#
# 用法：
#   ./scripts/plot_cluster_ipc.py
#   ./scripts/plot_cluster_ipc.py --root ./sim_run_12.8-signle-4G
#   ./scripts/plot_cluster_ipc.py --root <sweep结果目录> --out-dir <输出目录>
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IPC_RE = re.compile(r"\bgpu_tot_ipc\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
RUN_DIR_RE = re.compile(r"^sim_run_12\.8-(\d+)$")
O_FILE_RE = re.compile(r"\.o\d*$")


def last_gpu_tot_ipc(path):
    last = None
    with path.open("r", errors="ignore") as f:
        for line in f:
            match = IPC_RE.search(line)
            if match:
                last = float(match.group(1))
    return last


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "unnamed"


def collect_records(root):
    records = []
    missing = []
    for run_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if not run_dir.is_dir():
            continue
        run_match = RUN_DIR_RE.match(run_dir.name)
        if not run_match:
            continue
        sm = int(run_match.group(1))
        for o_file in sorted(run_dir.rglob("*.o*")):
            if not O_FILE_RE.search(o_file.name):
                continue
            rel = o_file.relative_to(run_dir)
            if len(rel.parts) < 4:
                continue
            benchmark = rel.parts[0]
            args = rel.parts[1]
            ipc = last_gpu_tot_ipc(o_file)
            if ipc is None:
                missing.append(str(o_file))
                continue
            records.append(
                {
                    "sm": sm,
                    "benchmark": benchmark,
                    "args": args,
                    "ipc": ipc,
                    "o_file": str(o_file),
                }
            )
    return records, missing


def aggregate(records):
    grouped = defaultdict(list)
    for rec in records:
        grouped[(rec["benchmark"], rec["sm"])].append(rec["ipc"])

    rows = []
    for (benchmark, sm), values in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        rows.append(
            {
                "benchmark": benchmark,
                "sm": sm,
                "ipc": sum(values) / len(values),
                "sample_count": len(values),
            }
        )
    return rows


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_lookup_json(path, rows):
    lookup = defaultdict(dict)
    for row in rows:
        lookup[row["benchmark"]][str(row["sm"])] = {
            "ipc": row["ipc"],
            "sample_count": row["sample_count"],
        }
    with path.open("w") as f:
        json.dump(lookup, f, indent=2, sort_keys=True)
        f.write("\n")


def plot_per_benchmark(out_dir, rows):
    by_bench = defaultdict(list)
    for row in rows:
        by_bench[row["benchmark"]].append(row)

    for benchmark, bench_rows in sorted(by_bench.items()):
        bench_rows = sorted(bench_rows, key=lambda r: r["sm"])
        xs = [r["sm"] for r in bench_rows]
        ys = [r["ipc"] for r in bench_rows]
        plt.figure(figsize=(9, 5))
        plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=3)
        plt.xlabel("SM")
        plt.ylabel("IPC")
        plt.title(f"{benchmark}: last gpu_tot_ipc vs SM")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{safe_name(benchmark)}_ipc.png", dpi=160)
        plt.close()


def plot_combined(out_dir, rows):
    by_bench = defaultdict(list)
    for row in rows:
        by_bench[row["benchmark"]].append(row)

    plt.figure(figsize=(12, 7))
    for benchmark, bench_rows in sorted(by_bench.items()):
        bench_rows = sorted(bench_rows, key=lambda r: r["sm"])
        xs = [r["sm"] for r in bench_rows]
        ys = [r["ipc"] for r in bench_rows]
        plt.plot(xs, ys, marker="o", linewidth=1.4, markersize=2.5, label=benchmark)
    plt.xlabel("SM")
    plt.ylabel("IPC")
    plt.title("Last gpu_tot_ipc vs SM")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "all_benchmarks_ipc.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract the last gpu_tot_ipc from Accel-Sim .o files and plot IPC vs SM."
    )
    parser.add_argument(
        "--root",
        default="/home/lyhong/Downloads/accel-sim-framework/sim_run_12.8-signle-4G",
        help="Directory containing sim_run_12.8-<sm> result folders.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Defaults to <root>/png/IPC.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Result root does not exist: {root}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else root / "png" / "IPC"
    out_dir.mkdir(parents=True, exist_ok=True)

    records, missing = collect_records(root)
    if not records:
        raise SystemExit(f"No .o files with gpu_tot_ipc found under {root}")

    rows = aggregate(records)

    write_csv(
        out_dir / "ipc_detail.csv",
        ["sm", "benchmark", "args", "ipc", "o_file"],
        sorted(records, key=lambda r: (r["benchmark"], r["sm"], r["args"], r["o_file"])),
    )
    write_csv(
        out_dir / "ipc_lookup.csv",
        ["benchmark", "sm", "ipc", "sample_count"],
        rows,
    )
    write_lookup_json(out_dir / "ipc_lookup.json", rows)

    if missing:
        with (out_dir / "missing_gpu_tot_ipc.txt").open("w") as f:
            for path in missing:
                f.write(path + "\n")

    plot_per_benchmark(out_dir, rows)
    plot_combined(out_dir, rows)

    print(f"Read {len(records)} output files with gpu_tot_ipc")
    print(f"Wrote lookup tables and PNG plots to {out_dir}")
    if missing:
        print(f"WARNING: {len(missing)} .o files had no gpu_tot_ipc; see missing_gpu_tot_ipc.txt")


if __name__ == "__main__":
    main()
