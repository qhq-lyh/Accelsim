#!/usr/bin/env python3
# 作用：把 HotSpot grid 模式输出的 grid steady 温度文件绘制成 GV100 PNG 热图。
# 输入：
#   1. grid steady 文件，默认 /home/lyhong/Downloads/hotspot_with_model_extract/gv100.grid.steady
#      文件格式通常为两列：grid_index temperature(K)。
#   2. GV100 floorplan 文件，默认 /home/lyhong/Downloads/accel-sim-framework/hotspot-link/gv100_new.flp
#      用于在热图上叠加 SM/L2/HBM 的模块边框。
# 输出：
#   PNG 热图，默认 /home/lyhong/Downloads/hotspot_with_model_extract/gv100_grid_heatmap_celsius.png
# 用法：
#   ./hotspot -c GV100-hotspot.config -p <ptrace> -o <ttrace> -sampling_intvl <time_intvl> -grid_layer_file gv100.lcf -model_type grid -grid_steady_file gv100.grid.steady
#   cd /home/lyhong/Downloads/hotspot_with_model_extract
#   ./plot_gv100_grid_heatmap.py
#   ./plot_gv100_grid_heatmap.py --grid-steady gv100.grid.steady --out gv100_grid_heatmap_celsius.png

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

DEFAULT_GRID_STEADY = Path("/home/lyhong/Downloads/hotspot_with_model_extract/gv100.grid.steady")
DEFAULT_FLP = Path("/home/lyhong/Downloads/accel-sim-framework/hotspot-link/gv100_new.flp")
DEFAULT_OUT = Path("/home/lyhong/Downloads/hotspot_with_model_extract/gv100_grid_heatmap_celsius.png")
LIMIT_EXTREMA_TO_SM_REGION = True


def load_flp(path: Path):
    blocks = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 5:
                raise ValueError(f"{path}:{lineno}: expected 5 columns, got {len(parts)}")
            name = parts[0]
            width, height, x, y = (float(v) for v in parts[1:])
            blocks.append((name, width, height, x, y))
    if not blocks:
        raise ValueError(f"floorplan is empty: {path}")
    return blocks


def load_grid_layer(path: Path, rows: int, cols: int, layer: int):
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path} should contain at least two columns: index temperature")

    layer_size = rows * cols
    start = layer * layer_size
    end = start + layer_size
    if data.shape[0] < end:
        raise ValueError(
            f"{path} has {data.shape[0]} rows, but layer {layer} requires rows [{start}, {end})"
        )

    kelvin = data[start:end, 1].reshape(rows, cols)
    # HotSpot row zero is at the top; floorplan y is measured from the bottom.
    return np.flipud(kelvin) - 273.15


def flp_extent_mm(blocks):
    xmin = min(x for _, _, _, x, _ in blocks)
    ymin = min(y for _, _, _, _, y in blocks)
    xmax = max(x + width for _, width, _, x, _ in blocks)
    ymax = max(y + height for _, _, height, _, y in blocks)
    return xmin, ymin, xmax, ymax


def sm_region_mask(grid_shape, blocks, extent):
    rows, cols = grid_shape
    xmin, ymin, xmax, ymax = extent
    x_centers = xmin + (np.arange(cols) + 0.5) * (xmax - xmin) / cols
    y_centers = ymin + (np.arange(rows) + 0.5) * (ymax - ymin) / rows
    xs, ys = np.meshgrid(x_centers, y_centers)
    mask = np.zeros(grid_shape, dtype=bool)

    for name, width, height, x, y in blocks:
        if not name.startswith("SM_"):
            continue
        mask |= (xs >= x) & (xs <= x + width) & (ys >= y) & (ys <= y + height)

    return mask


def extrema_indices(grid_c, blocks, extent, limit_to_sm_region: bool):
    if not limit_to_sm_region:
        max_row, max_col = np.unravel_index(np.argmax(grid_c), grid_c.shape)
        min_row, min_col = np.unravel_index(np.argmin(grid_c), grid_c.shape)
        return max_row, max_col, min_row, min_col

    mask = sm_region_mask(grid_c.shape, blocks, extent)
    if not np.any(mask):
        raise ValueError("no grid cell centers fall inside SM_ floorplan blocks")

    sm_grid = np.where(mask, grid_c, np.nan)
    max_row, max_col = np.unravel_index(np.nanargmax(sm_grid), sm_grid.shape)
    min_row, min_col = np.unravel_index(np.nanargmin(sm_grid), sm_grid.shape)
    return max_row, max_col, min_row, min_col


def plot_heatmap(grid_c, blocks, out: Path, title: str, annotate: bool, limit_extrema_to_sm_region: bool):
    xmin, ymin, xmax, ymax = flp_extent_mm(blocks)
    extent = [xmin * 1000, xmax * 1000, ymin * 1000, ymax * 1000]

    fig_w = 12
    fig_h = max(4, fig_w * (ymax - ymin) / (xmax - xmin))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    image = ax.imshow(grid_c, origin="lower", extent=extent, cmap="inferno", aspect="equal")

    rows, cols = grid_c.shape
    max_row, max_col, min_row, min_col = extrema_indices(
        grid_c, blocks, (xmin, ymin, xmax, ymax), limit_extrema_to_sm_region
    )

    x_min_mm = xmin * 1000 + (min_col + 0.5) * (xmax - xmin) * 1000 / cols
    y_min_mm = ymin * 1000 + (min_row + 0.5) * (ymax - ymin) * 1000 / rows
    x_max_mm = xmin * 1000 + (max_col + 0.5) * (xmax - xmin) * 1000 / cols
    y_max_mm = ymin * 1000 + (max_row + 0.5) * (ymax - ymin) * 1000 / rows

    max_temp = grid_c[max_row, max_col]
    min_temp = grid_c[min_row, min_col]

    for name, width, height, x, y in blocks:
        line_width = 0.22 if name.startswith("SM_") else 0.55
        ax.add_patch(
            Rectangle(
                (x * 1000, y * 1000),
                width * 1000,
                height * 1000,
                fill=False,
                edgecolor="white",
                linewidth=line_width,
                alpha=0.85,
            )
        )
        if annotate and not name.startswith("SM_"):
            ax.text(
                (x + width / 2) * 1000,
                (y + height / 2) * 1000,
                name,
                ha="center",
                va="center",
                fontsize=7,
                color="white",
            )

    ax.scatter([x_max_mm], [y_max_mm], marker="x", s=90, linewidths=2.2, color="cyan")
    ax.scatter([x_min_mm], [y_min_mm], marker="o", s=70, facecolors="none", edgecolors="cyan", linewidths=2.0)

    ax.annotate(
        f"Max {max_temp:.2f} C\nGrid({rows - 1 - max_row},{max_col})",
        xy=(x_max_mm, y_max_mm),
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="cyan",
        fontsize=8,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="cyan", alpha=0.65),
    )

    ax.annotate(
        f"Min {min_temp:.2f} C\nGrid({rows - 1 - min_row},{min_col})",
        xy=(x_min_mm, y_min_mm),
        xytext=(-8, -24),
        textcoords="offset points",
        ha="right",
        va="top",
        color="cyan",
        fontsize=8,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="cyan", alpha=0.65),
    )

    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    colorbar.set_label("Temperature (C)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot a GV100 HotSpot grid steady heatmap in Celsius.")
    parser.add_argument("--grid-steady", type=Path, default=DEFAULT_GRID_STEADY, help="Input grid steady temperature file.")
    parser.add_argument("--flp", type=Path, default=DEFAULT_FLP, help="GV100 floorplan used to overlay block borders.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output PNG path.")
    parser.add_argument("--rows", type=int, default=64, help="Grid rows. Must match HotSpot -grid_rows.")
    parser.add_argument("--cols", type=int, default=64, help="Grid cols. Must match HotSpot -grid_cols.")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to plot. Layer 0 is the active silicon layer.")
    parser.add_argument("--annotate", action="store_true", help="Annotate non-SM blocks such as L2/HBM.")
    parser.add_argument("--extrema-region", choices=("all", "sm"), default="sm",
                        help="Limit extrema markers to SMs or include the entire silicon grid.")
    args = parser.parse_args()

    if not args.grid_steady.is_file():
        raise SystemExit(f"grid steady file does not exist: {args.grid_steady}")
    if not args.flp.is_file():
        raise SystemExit(f"floorplan file does not exist: {args.flp}")
    if args.rows <= 0 or args.cols <= 0:
        raise SystemExit("--rows and --cols must be positive")
    if args.layer < 0:
        raise SystemExit("--layer must be non-negative")

    blocks = load_flp(args.flp)
    grid_c = load_grid_layer(args.grid_steady, args.rows, args.cols, args.layer)
    title = f"GV100 grid steady temperature heatmap, layer {args.layer}"
    plot_heatmap(grid_c, blocks, args.out, title, args.annotate, args.extrema_region == "sm")

    print(f"Input grid steady: {args.grid_steady}")
    print(f"Input floorplan: {args.flp}")
    print(f"Grid: {args.rows} x {args.cols}, layer={args.layer}")
    print(f"Temperature range: {grid_c.min():.2f} C to {grid_c.max():.2f} C")
    print(f"Limit min/max markers to SM region: {args.extrema_region == 'sm'}")
    print(f"Wrote PNG: {args.out}")


if __name__ == "__main__":
    main()
