#!/usr/bin/env python3
# 作用：根据 IPC-sm/ipc_lookup.csv 中各 benchmark 的 IPC-SM 曲线，
# 按“短期 1 个 SM 增益 + 剩余 SM 全部分给该 benchmark 的观察增益”的贪心策略，
# 计算多 benchmark 同时仿真时的 SM 分配方案。
#
# 默认输入：
#   1. IPC-sm/ipc_lookup.csv；缺失时使用 scripts/models/qv100_ipc_lookup.json。
#      均包含 benchmark,sm,ipc,sample_count 等字段，参考曲线适用于对应输入规模。
#   2. SM7_QV100/gpgpusim.config，从 -gpgpu_n_clusters 读取总 SM 数。
#
# 默认输出：
#   IPC-sm/sm_allocation.csv，包含 benchmark,sm,ipc 三列。
#
# 用法：
#   ./scripts/Alg1_compute_sm_allocation.py
#   ./scripts/Alg1_compute_sm_allocation.py --benchmarks backprop,bfs,nn
#   ./scripts/Alg1_compute_sm_allocation.py --total-sm 80 --alpha 0.5 --out-csv /tmp/sm_allocation.csv
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# 优先使用用户已有 CSV；干净 checkout 则使用随算法提供的参考 IPC 曲线。
DEFAULT_IPC_CSV = ROOT/'IPC-sm/ipc_lookup.csv'
if not DEFAULT_IPC_CSV.is_file():
    DEFAULT_IPC_CSV = ROOT/'scripts/models/qv100_ipc_lookup.json'
DEFAULT_CONFIG = ROOT/'gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/gpgpusim.config'
DEFAULT_OUT_CSV = ROOT/'IPC-sm/sm_allocation.csv'
CLUSTER_RE = re.compile(r"^\s*-gpgpu_n_clusters\s+(\d+)\b")


def parse_benchmark_list(values):
    if not values:
        return None
    benchmarks = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                benchmarks.append(item)
    return benchmarks or None


def load_ipc_table(path):
    grouped = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    with path.open(newline="") as f:
        if path.suffix.lower() == '.json':
            reader = json.load(f)
            fieldnames = set.intersection(*(set(row) for row in reader)) if reader else set()
        else:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
        required = {"benchmark", "sm", "ipc"}
        missing = required - fieldnames
        if missing:
            raise SystemExit(f"{path} missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            benchmark = row["benchmark"].strip()
            if not benchmark:
                continue
            sm = int(row["sm"])
            ipc = float(row["ipc"])
            sample_count = float(row.get("sample_count") or 1.0)
            if sample_count <= 0:
                sample_count = 1.0
            grouped[benchmark][sm][0] += ipc * sample_count
            grouped[benchmark][sm][1] += sample_count

    table = {}
    for benchmark, by_sm in grouped.items():
        table[benchmark] = {
            sm: weighted_sum / weight
            for sm, (weighted_sum, weight) in by_sm.items()
            if weight > 0
        }
    return table


def read_total_sm(config_path):
    total_sm = None
    with config_path.open() as f:
        for line in f:
            match = CLUSTER_RE.match(line)
            if match:
                total_sm = int(match.group(1))
    if total_sm is None:
        raise SystemExit(f"Could not find -gpgpu_n_clusters in config: {config_path}")
    return total_sm


def weighted_gain(current_ipc, next_ipc, lookahead_ipc, alpha):
    short_delta = next_ipc - current_ipc
    if short_delta < 0:
        short_delta = 0.0
    lookahead_delta = lookahead_ipc - current_ipc

    if current_ipc == 0:
        weighted_delta = alpha * short_delta + (1.0 - alpha) * lookahead_delta
        if weighted_delta > 0:
            return math.inf
        if weighted_delta == 0:
            return 0.0
        return -math.inf
    return (alpha * short_delta + (1.0 - alpha) * lookahead_delta) / current_ipc


def best_available_sm(ipc_by_sm, target_sm, minimum_sm):
    candidates = [sm for sm in ipc_by_sm if minimum_sm <= sm <= target_sm]
    if not candidates:
        return None
    return max(candidates)


def allocate_sms(ipc_table, selected_benchmarks, total_sm, alpha=0.5):
    unknown = [b for b in selected_benchmarks if b not in ipc_table]
    if unknown:
        available = ", ".join(sorted(ipc_table))
        raise SystemExit(f"Unknown benchmark(s): {', '.join(unknown)}. Available: {available}")

    allocation = {}
    for benchmark in selected_benchmarks:
        sms = sorted(ipc_table[benchmark])
        if not sms:
            raise SystemExit(f"No IPC points for benchmark: {benchmark}")
        allocation[benchmark] = sms[0]

    initial_total = sum(allocation.values())
    if total_sm < initial_total:
        raise SystemExit(
            f"total SM {total_sm} is smaller than the minimum required {initial_total} "
            f"for {len(selected_benchmarks)} benchmark(s)."
        )

    while sum(allocation.values()) < total_sm:
        allocated_total = sum(allocation.values())
        remaining_sm = total_sm - allocated_total
        candidates = []
        for benchmark in selected_benchmarks:
            current_sm = allocation[benchmark]
            next_sm = current_sm + 1
            ipc_by_sm = ipc_table[benchmark]
            if next_sm not in ipc_by_sm:
                continue
            lookahead_sm = best_available_sm(ipc_by_sm, current_sm + remaining_sm, next_sm)
            if lookahead_sm is None:
                lookahead_sm = next_sm

            current_ipc = ipc_by_sm[current_sm]
            next_ipc = ipc_by_sm[next_sm]
            lookahead_ipc = ipc_by_sm[lookahead_sm]
            short_delta = max(0.0, next_ipc - current_ipc)
            lookahead_delta = lookahead_ipc - current_ipc
            gain = weighted_gain(current_ipc, next_ipc, lookahead_ipc, alpha)
            abs_gain = alpha * short_delta + (1.0 - alpha) * lookahead_delta
            candidates.append((gain, abs_gain, lookahead_delta, benchmark, next_sm))

        if not candidates:
            current_total = sum(allocation.values())
            raise SystemExit(
                f"Cannot allocate {total_sm} SMs: all selected benchmarks reached their "
                f"largest available IPC point at total {current_total} SMs."
            )

        _gain, _abs_gain, _lookahead_delta, benchmark, new_sm = max(
            candidates, key=lambda x: (x[0], x[1], x[2], x[3])
        )
        allocation[benchmark] = new_sm

    return allocation


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["benchmark", "sm", "ipc"])
        writer.writeheader()
        writer.writerows(rows)


def allocation_rows(allocation, ipc_table):
    rows = []
    for benchmark in sorted(allocation):
        sm = allocation[benchmark]
        rows.append({"benchmark": benchmark, "sm": sm, "ipc": ipc_table[benchmark][sm]})
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute a greedy SM allocation from benchmark IPC-vs-SM lookup data. "
            "Each step combines immediate one-SM gain and remaining-SM lookahead gain."
        )
    )
    parser.add_argument("--ipc-csv", "--ipc-lookup", default=DEFAULT_IPC_CSV, help="Input IPC lookup (CSV or JSON); defaults to local CSV or bundled reference.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="gpgpusim.config path used to read -gpgpu_n_clusters as the total SM budget.",
    )
    parser.add_argument(
        "--total-sm",
        type=int,
        default=None,
        help="Optional total SM override. By default this is read from --config.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for immediate one-SM IPC gain. The remaining weight is used for lookahead gain.",
    )
    parser.add_argument(
        "-B",
        "--benchmarks",
        action="append",
        help="Benchmark subset, comma-separated or repeated. Defaults to all benchmarks in the CSV.",
    )
    parser.add_argument(
        "--out-csv",
        default=DEFAULT_OUT_CSV,
        help="Output CSV path. Defaults to IPC-sm/sm_allocation.csv.",
    )
    args = parser.parse_args()

    ipc_csv = Path(args.ipc_csv).resolve()
    if not ipc_csv.is_file():
        raise SystemExit(f"IPC lookup CSV does not exist: {ipc_csv}")

    config = Path(args.config).resolve()
    if args.total_sm is None and not config.is_file():
        raise SystemExit(f"Config file does not exist: {config}")
    total_sm = args.total_sm if args.total_sm is not None else read_total_sm(config)

    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit(f"--alpha must be between 0 and 1, got {args.alpha}")

    ipc_table = load_ipc_table(ipc_csv)
    selected = parse_benchmark_list(args.benchmarks) or sorted(ipc_table)
    allocation = allocate_sms(ipc_table, selected, total_sm, alpha=args.alpha)
    rows = allocation_rows(allocation, ipc_table)

    out_csv = Path(args.out_csv).resolve()
    write_csv(out_csv, rows)

    total_ipc = sum(row["ipc"] for row in rows)
    print(f"Input IPC lookup: {ipc_csv}")
    if args.total_sm is None:
        print(f"Total SM read from config: {config}")
    print(f"Selected benchmarks: {', '.join(selected)}")
    print(f"Total SM: {sum(row['sm'] for row in rows)} / {total_sm}")
    print(f"Alpha: {args.alpha:.3f}")
    print(f"Total IPC: {total_ipc:.6f}")
    for row in rows:
        print(f"{row['benchmark']}: sm={row['sm']} ipc={row['ipc']:.6f}")
    print(f"Wrote allocation CSV to {out_csv}")


if __name__ == "__main__":
    main()
