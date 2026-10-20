#!/usr/bin/env python3
# 作用：在 Alg1 已经计算出每个 benchmark 需要多少个 SM 后，继续计算每个
# benchmark 具体放到哪些物理 SM 上，并画出 benchmark 在 GV100 floorplan 上的放置图。
#
# 输入：
#   1. IPC-sm/sm_allocation.csv，Alg1 输出，包含 benchmark,sm,ipc 三列；
#   2. hotspot-link/GV100.flp，包含 SM_0、SM_1... 的物理 floorplan 信息。
#
# 算法：
#   先计算每个 benchmark 的平均 SM IPC：ipc_per_sm = ipc / sm；
#   按 ipc_per_sm 从高到低排序；然后每轮取当前剩余 SM 数量大于 0 的
#   最高 ipc_per_sm benchmark 和最低 ipc_per_sm benchmark 成对交错放置，
#   例如先放 high,low，再放 low,high，使高 IPC/SM 的 benchmark 尽量分散。
#
# 输出：
#   1. IPC-sm/sm_map.csv：每个 benchmark 分到的 SM 列表；
#   2. IPC-sm/sm_placement_flp.png：按 benchmark 着色的 SM floorplan 放置图。
#
# 用法：
#   ./scripts/Alg2_compute_sm_placement.py
#   ./scripts/Alg2_compute_sm_placement.py --allocation-csv IPC-sm/sm_allocation.csv
#   ./scripts/Alg2_compute_sm_placement.py --flp hotspot-link/GV100.flp --out-dir IPC-sm

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALLOCATION_CSV = ROOT / "IPC-sm" / "sm_allocation.csv"
DEFAULT_FLP = ROOT / "hotspot-link" / "flp" / "gv100_new.flp"
DEFAULT_OUT_DIR = ROOT / "IPC-sm"
SM_RE = re.compile(r"^SM_(\d+)$")


def load_allocation(path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"benchmark", "sm", "ipc"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            benchmark = row["benchmark"].strip()
            if not benchmark:
                continue
            sm_count = int(row["sm"])
            ipc = float(row["ipc"])
            if sm_count <= 0:
                raise SystemExit(f"benchmark {benchmark} has non-positive sm count: {sm_count}")
            rows.append(
                {
                    "benchmark": benchmark,
                    "sm_count": sm_count,
                    "ipc": ipc,
                    "ipc_per_sm": ipc / sm_count,
                }
            )

    if not rows:
        raise SystemExit(f"No allocation rows found in {path}")
    return rows


def load_flp_blocks(path):
    blocks = []
    with path.open() as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 5:
                raise SystemExit(f"{path}:{lineno}: expected 5 columns, got {len(parts)}")
            name = parts[0]
            width, height, left_x, bottom_y = (float(part) for part in parts[1:])
            match = SM_RE.match(name)
            blocks.append(
                {
                    "name": name,
                    "sm_id": int(match.group(1)) if match else None,
                    "width": width,
                    "height": height,
                    "left_x": left_x,
                    "bottom_y": bottom_y,
                }
            )
    if not blocks:
        raise SystemExit(f"Floorplan is empty: {path}")
    return blocks


def load_flp_sms(path):
    sms = []
    for block in load_flp_blocks(path):
        if block["sm_id"] is not None:
            sms.append(
                {
                    "sm_id": block["sm_id"],
                    "unit": block["name"],
                    "width": block["width"],
                    "height": block["height"],
                    "left_x": block["left_x"],
                    "bottom_y": block["bottom_y"],
                }
            )
    if not sms:
        raise SystemExit(f"No SM_* entries found in floorplan: {path}")
    return sorted(sms, key=lambda row: row["sm_id"])


def rank_allocations(allocations):
    return sorted(allocations, key=lambda row: (-row["ipc_per_sm"], row["benchmark"]))


def choose_active(sorted_rows, remaining):
    return [row for row in sorted_rows if remaining[row["benchmark"]] > 0]


def build_placement_sequence(allocations):
    sorted_rows = rank_allocations(allocations)
    remaining = {row["benchmark"]: row["sm_count"] for row in sorted_rows}
    sequence = []
    pair_index = 0

    while sum(remaining.values()) > 0:
        active = choose_active(sorted_rows, remaining)
        if len(active) == 1:
            benchmark = active[0]["benchmark"]
            sequence.extend([benchmark] * remaining[benchmark])
            remaining[benchmark] = 0
            break

        high = active[0]["benchmark"]
        low = active[-1]["benchmark"]
        pair = [high, low] if pair_index % 2 == 0 else [low, high]
        for benchmark in pair:
            if remaining[benchmark] > 0:
                sequence.append(benchmark)
                remaining[benchmark] -= 1
        pair_index += 1

    return sequence, sorted_rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_placement(flp_sms, sequence, allocation_by_name):
    if len(sequence) > len(flp_sms):
        raise SystemExit(
            f"Allocated {len(sequence)} SMs, but floorplan only has {len(flp_sms)} SM entries"
        )

    placement_rows = []
    sm_map = defaultdict(list)
    for index, benchmark in enumerate(sequence):
        sm = flp_sms[index]
        alloc = allocation_by_name[benchmark]
        sm_map[benchmark].append(sm["sm_id"])
        placement_rows.append(
            {
                "placement_index": index,
                "sm_id": sm["sm_id"],
                "unit": sm["unit"],
                "benchmark": benchmark,
                "benchmark_sm_count": alloc["sm_count"],
                "benchmark_ipc": alloc["ipc"],
                "benchmark_ipc_per_sm": alloc["ipc_per_sm"],
                "width": sm["width"],
                "height": sm["height"],
                "left_x": sm["left_x"],
                "bottom_y": sm["bottom_y"],
            }
        )
    return placement_rows, sm_map


def write_sm_map(out_dir, sm_map, allocation_by_name, sorted_rows):
    rank_by_benchmark = {row["benchmark"]: rank for rank, row in enumerate(sorted_rows, start=1)}
    rows = []
    for benchmark, sm_ids in sorted(sm_map.items(), key=lambda item: rank_by_benchmark[item[0]]):
        alloc = allocation_by_name[benchmark]
        rows.append(
            {
                "rank_by_ipc_per_sm": rank_by_benchmark[benchmark],
                "benchmark": benchmark,
                "sm_count": alloc["sm_count"],
                "ipc": alloc["ipc"],
                "ipc_per_sm": alloc["ipc_per_sm"],
                "sm_ids": " ".join(str(sm_id) for sm_id in sm_ids),
            }
        )

    path = out_dir / "sm_map.csv"
    write_csv(path, rows, ["rank_by_ipc_per_sm", "benchmark", "sm_count", "ipc", "ipc_per_sm", "sm_ids"])
    return path, rows


def plot_flp(blocks, placement_rows, sorted_rows, out_png, annotate):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib is not installed; skipping FLP placement plot")
        return None

    assigned = {row["sm_id"]: row["benchmark"] for row in placement_rows}
    benchmark_order = [row["benchmark"] for row in sorted_rows if row["benchmark"] in {p["benchmark"] for p in placement_rows}]
    cmap = plt.get_cmap("Blues")
    if len(benchmark_order) == 1:
        shades = [0.85]
    else:
        shades = [0.90 - 0.52 * i / (len(benchmark_order) - 1) for i in range(len(benchmark_order))]
    colors = {benchmark: cmap(shade) for benchmark, shade in zip(benchmark_order, shades)}
    unassigned_color = "#d9d9d9"
    other_color = "#f0f0f0"

    xmin = min(block["left_x"] for block in blocks)
    ymin = min(block["bottom_y"] for block in blocks)
    xmax = max(block["left_x"] + block["width"] for block in blocks)
    ymax = max(block["bottom_y"] + block["height"] for block in blocks)

    fig_w = 12
    fig_h = max(4, fig_w * (ymax - ymin) / (xmax - xmin))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for block in blocks:
        sm_id = block["sm_id"]
        benchmark = assigned.get(sm_id) if sm_id is not None else None
        if benchmark is not None:
            facecolor = colors[benchmark]
            alpha = 0.82
        elif sm_id is not None:
            facecolor = unassigned_color
            alpha = 0.55
        else:
            facecolor = other_color
            alpha = 0.35

        rect = Rectangle(
            (block["left_x"] * 1000, block["bottom_y"] * 1000),
            block["width"] * 1000,
            block["height"] * 1000,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=0.35,
            alpha=alpha,
        )
        ax.add_patch(rect)

        if annotate and sm_id is not None:
            label = f"SM{sm_id}\n{benchmark}" if benchmark else f"SM{sm_id}"
            ax.text(
                (block["left_x"] + block["width"] / 2) * 1000,
                (block["bottom_y"] + block["height"] / 2) * 1000,
                label,
                ha="center",
                va="center",
                fontsize=5,
            )

    ipc_by_benchmark = {row["benchmark"]: row["ipc_per_sm"] for row in sorted_rows}
    handles = [Rectangle((0, 0), 1, 1, facecolor=colors[b], edgecolor="black", alpha=0.82) for b in benchmark_order]
    labels = [
        f"{b} ({sum(1 for row in placement_rows if row['benchmark'] == b)}, ipc/sm={ipc_by_benchmark[b]:.3f})"
        for b in benchmark_order
    ]
    if any(block["sm_id"] is not None and block["sm_id"] not in assigned for block in blocks):
        handles.append(Rectangle((0, 0), 1, 1, facecolor=unassigned_color, edgecolor="black", alpha=0.55))
        labels.append("unassigned SM")
    ax.legend(handles, labels, loc="upper right", fontsize=9)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin * 1000, xmax * 1000)
    ax.set_ylim(ymin * 1000, ymax * 1000)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("GV100 SM benchmark placement")
    ax.grid(True, linewidth=0.2, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return out_png


def main():
    parser = argparse.ArgumentParser(description="Compute physical SM placement from Alg1 benchmark SM allocation.")
    parser.add_argument("--allocation-csv", type=Path, default=DEFAULT_ALLOCATION_CSV)
    parser.add_argument("--flp", type=Path, default=DEFAULT_FLP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--plot", type=Path, default=None, help="Output PNG path. Defaults to OUT_DIR/sm_placement_flp.png.")
    parser.add_argument("--annotate", action="store_true", help="Label each SM on the FLP plot.")
    args = parser.parse_args()

    allocation_csv = args.allocation_csv.resolve()
    flp = args.flp.resolve()
    out_dir = args.out_dir.resolve()
    if not allocation_csv.is_file():
        raise SystemExit(f"Allocation CSV does not exist: {allocation_csv}")
    if not flp.is_file():
        raise SystemExit(f"Floorplan file does not exist: {flp}")

    allocations = load_allocation(allocation_csv)
    allocation_by_name = {row["benchmark"]: row for row in allocations}
    flp_sms = load_flp_sms(flp)
    flp_blocks = load_flp_blocks(flp)
    sequence, sorted_rows = build_placement_sequence(allocations)
    placement_rows, sm_map = build_placement(flp_sms, sequence, allocation_by_name)
    sm_map_csv, summary_rows = write_sm_map(out_dir, sm_map, allocation_by_name, sorted_rows)
    plot_path = (args.plot.resolve() if args.plot else out_dir / "sm_placement_flp.png")
    written_plot = plot_flp(flp_blocks, placement_rows, sorted_rows, plot_path, args.annotate)

    print(f"Input allocation: {allocation_csv}")
    print(f"Input floorplan: {flp}")
    print(f"Placed SMs: {len(sequence)} / {len(flp_sms)}")
    print("Benchmark order by ipc_per_sm:")
    for row in summary_rows:
        print(
            f"  {row['rank_by_ipc_per_sm']}. {row['benchmark']}: "
            f"sm={row['sm_count']} ipc_per_sm={row['ipc_per_sm']:.6f} sm_ids={row['sm_ids']}"
        )
    print(f"Wrote SM map CSV to {sm_map_csv}")
    if written_plot:
        print(f"Wrote FLP placement plot to {written_plot}")


if __name__ == "__main__":
    main()
