#!/usr/bin/env python3
# 作用：把两个已有 Accel-Sim trace 目录合并成一个 mixed trace 目录，
# 并把两个 benchmark 的 kernel trace 分别改写到不同 CUDA stream 上。
# 默认场景是 bfs 使用 stream 1、hotspot 使用 stream 2，输出到
# hw_run/rodinia_2.0-ft/9.1/bfs-hotspot-rodinia-2.0-ft/bfs10_hotspot40/traces。
#
# 默认用法：
#   ./scripts/gen_mix_trace.py
#
# 自定义用法：
#   ./scripts/gen_mix_trace.py --trace-a <A的traces目录> --trace-b <B的traces目录>
#   可同时指定 --prefix-a/--prefix-b、--stream-a/--stream-b 和 --out。
from argparse import ArgumentParser
from pathlib import Path
import re


ROOT = Path("/home/lyhong/Downloads/accel-sim-framework")
DEFAULT_BFS = (
    ROOT
    / "hw_run/rodinia_2.0-ft/9.1/bfs-rodinia-2.0-ft"
    / "__data_graph4096_txt___data_graph4096_result_txt/traces"
)
DEFAULT_HOTSPOT = (
    ROOT
    / "hw_run/rodinia_2.0-ft/9.1/hotspot-rodinia-2.0-ft"
    / "30_6_40___data_result_30_6_40_txt/traces"
)
DEFAULT_OUT = (
    ROOT
    / "hw_run/rodinia_2.0-ft/9.1/bfs-hotspot-rodinia-2.0-ft"
    / "bfs10_hotspot40/traces"
)


def parse_args():
    parser = ArgumentParser(
        description="Build a mixed Accel-Sim trace from two trace directories."
    )
    parser.add_argument("--trace-a", type=Path, default=DEFAULT_BFS)
    parser.add_argument("--trace-b", type=Path, default=DEFAULT_HOTSPOT)
    parser.add_argument("--prefix-a", default="bfs")
    parser.add_argument("--prefix-b", default="hotspot")
    parser.add_argument("--stream-a", type=int, default=1)
    parser.add_argument("--stream-b", type=int, default=2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def clean_output(out, prefixes):
    out.mkdir(parents=True, exist_ok=True)
    for prefix in prefixes:
        for stale in out.glob(f"kernel-{prefix}-*.traceg"):
            stale.unlink()
        for stale in out.glob(f"{prefix}-kernel-*.traceg"):
            stale.unlink()


def copy_traces(srcdir, out, prefix, stream):
    kernelslist = srcdir / "kernelslist.g"
    if not kernelslist.is_file():
        raise FileNotFoundError(f"missing kernelslist.g: {kernelslist}")

    mp = {}
    for src in sorted(srcdir.glob("kernel-*.traceg")):
        dst = f"kernel-{prefix}-{src.name.removeprefix('kernel-')}"
        text = src.read_text()
        text = re.sub(
            r"^-cuda stream id = .*$",
            f"-cuda stream id = {stream}",
            text,
            flags=re.M,
        )
        (out / dst).write_text(text)
        mp[src.name] = dst
    return mp


def chunks(klist, mp):
    pending, result = [], []
    for line in klist.read_text().splitlines():
        if not line:
            continue
        if line.startswith("kernel"):
            result.append(pending + [mp[line]])
            pending = []
        else:
            pending.append(line)
    return result


def main():
    args = parse_args()
    clean_output(args.out, [args.prefix_a, args.prefix_b])

    a_mp = copy_traces(args.trace_a, args.out, args.prefix_a, args.stream_a)
    b_mp = copy_traces(args.trace_b, args.out, args.prefix_b, args.stream_b)
    a_chunks = chunks(args.trace_a / "kernelslist.g", a_mp)
    b_chunks = chunks(args.trace_b / "kernelslist.g", b_mp)

    combined = []
    for i in range(max(len(a_chunks), len(b_chunks))):
        if i < len(a_chunks):
            combined += a_chunks[i]
        if i < len(b_chunks):
            combined += b_chunks[i]

    (args.out / "kernelslist.g").write_text("\n".join(combined) + "\n")
    print(f"Wrote mixed trace: {args.out / 'kernelslist.g'}")


if __name__ == "__main__":
    main()
