#!/usr/bin/env python3
# 作用：从不同 SM 数量 sweep 的 Accel-Sim .o 输出中提取每个 kernel 的 IPC 曲线。
# 脚本扫描 sim_run_12.8-<sm> 目录，读取 .o 文件中每次出现的 gpu_ipc，
# 按出现顺序输出为 benchmark-k1、benchmark-k2 ... 的 kernel 级 IPC lookup。
#
# 默认输入：
#   /home/lyhong/Downloads/accel-sim-framework/sim_run_12.8-signle-4G
#
# 默认输出：
#   /home/lyhong/Downloads/accel-sim-framework/IPC-sm/training_data/ipc_lookup.csv
#   CSV 列为：benchmark-kernel,sm,ipc
#
# 用法：
#   ./scripts/kernel_analysis/plot_cluster_ipc_kernel.py
#   ./scripts/kernel_analysis/plot_cluster_ipc_kernel.py --root ./sim_run_12.8-signle-4G
#   ./scripts/kernel_analysis/plot_cluster_ipc_kernel.py --root <sweep结果目录> --out-dir <输出目录>

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/lyhong/Downloads/accel-sim-framework")
DEFAULT_ROOT = ROOT / "sim_run_12.8-signle-4G"
DEFAULT_OUT_DIR = ROOT / "IPC-sm" / "training_data"

GPU_IPC_RE = re.compile(r"\bgpu_ipc\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
RUN_DIR_RE = re.compile(r"^sim_run_12\.8-(\d+)$")
O_FILE_RE = re.compile(r"\.o\d*$")
KERNEL_NAME_RE = re.compile(r"^(.*)-k(\d+)$")


def short_benchmark_name(name):
    name = name.strip()
    if "-rodinia" in name:
        return name.split("-rodinia", 1)[0]
    return name


def extract_gpu_ipcs(path):
    values = []
    with path.open("r", errors="ignore") as f:
        for line in f:
            match = GPU_IPC_RE.search(line)
            if match:
                values.append(float(match.group(1)))
    return values


def benchmark_kernel_sort_key(name):
    match = KERNEL_NAME_RE.match(name)
    if match:
        return (match.group(1), int(match.group(2)))
    return (name, 10**9)


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
            benchmark = short_benchmark_name(rel.parts[0])
            ipcs = extract_gpu_ipcs(o_file)
            if not ipcs:
                missing.append(str(o_file))
                continue
            for kernel_idx, ipc in enumerate(ipcs, 1):
                records.append(
                    {
                        "benchmark_kernel": f"{benchmark}-k{kernel_idx}",
                        "sm": sm,
                        "ipc": ipc,
                        "o_file": str(o_file),
                    }
                )
    return records, missing


def aggregate(records):
    grouped = defaultdict(list)
    for rec in records:
        grouped[(rec["benchmark_kernel"], rec["sm"])].append(rec["ipc"])

    rows = []
    for (benchmark_kernel, sm), values in sorted(
        grouped.items(),
        key=lambda item: (*benchmark_kernel_sort_key(item[0][0]), item[0][1]),
    ):
        rows.append(
            {
                "benchmark-kernel": benchmark_kernel,
                "sm": sm,
                "ipc": f"{(sum(values) / len(values)):.4f}",
            }
        )
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["benchmark-kernel", "sm", "ipc"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Extract per-kernel gpu_ipc from Accel-Sim .o files.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Directory containing sim_run_12.8-<sm> folders.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Result root does not exist: {root}")

    records, missing = collect_records(root)
    if not records:
        raise SystemExit(f"No .o files with gpu_ipc found under {root}")

    rows = aggregate(records)
    out_csv = out_dir / "ipc_lookup.csv"
    write_csv(out_csv, rows)

    if missing:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "missing_gpu_ipc.txt").open("w") as f:
            for path in missing:
                f.write(path + "\n")

    print(f"Read {len(records)} kernel gpu_ipc record(s)")
    print(f"Wrote {out_csv}")
    if missing:
        print(f"WARNING: {len(missing)} .o file(s) had no gpu_ipc; see missing_gpu_ipc.txt")


if __name__ == "__main__":
    main()
