#!/usr/bin/env python3
"""Plot a HotSpot floorplan file.

Expected non-comment line format:
<unit-name> <width> <height> <left-x> <bottom-y>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def read_flp(path: Path) -> list[tuple[str, float, float, float, float]]:
    blocks: list[tuple[str, float, float, float, float]] = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 5:
                raise ValueError(f"{path}:{lineno}: expected 5 columns, got {len(parts)}")
            name = parts[0]
            width, height, x, y = (float(part) for part in parts[1:])
            blocks.append((name, width, height, x, y))
    return blocks


def classify(name: str) -> str:
    if name.startswith("SM_"):
        return "SM"
    if name.startswith("L2"):
        return "L2 cache"
    if name.startswith("HBM"):
        return "HBM2"
    if "hub" in name.lower():
        return "high-speed hub"
    return "other"


def plot_flp(
    blocks: list[tuple[str, float, float, float, float]],
    out: Path,
    annotate: bool,
) -> None:
    if not blocks:
        raise ValueError("floorplan is empty")

    xmin = min(x for _, _, _, x, _ in blocks)
    ymin = min(y for _, _, _, _, y in blocks)
    xmax = max(x + width for _, width, _, x, _ in blocks)
    ymax = max(y + height for _, _, height, _, y in blocks)

    colors = {
        "SM": "#4c78a8",
        "L2 cache": "#59a14f",
        "HBM2": "#f28e2b",
        "high-speed hub": "#b07aa1",
        "other": "#9c755f",
    }

    fig_w = 12
    fig_h = max(4, fig_w * (ymax - ymin) / (xmax - xmin))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    counts: dict[str, int] = {}
    for name, width, height, x, y in blocks:
        kind = classify(name)
        counts[kind] = counts.get(kind, 0) + 1
        rect = Rectangle(
            (x * 1000, y * 1000),
            width * 1000,
            height * 1000,
            facecolor=colors[kind],
            edgecolor="black",
            linewidth=0.35,
            alpha=0.78,
        )
        ax.add_patch(rect)

        if annotate:
            fontsize = 5 if name.startswith("SM_") else 7
            ax.text(
                (x + width / 2) * 1000,
                (y + height / 2) * 1000,
                name,
                ha="center",
                va="center",
                fontsize=fontsize,
            )

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=colors[k], edgecolor="black", alpha=0.78)
        for k in colors
        if k in counts
    ]
    labels = [f"{k} ({counts[k]})" for k in colors if k in counts]
    ax.legend(handles, labels, loc="upper right", fontsize=9)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin * 1000, xmax * 1000)
    ax.set_ylim(ymin * 1000, ymax * 1000)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"{path_label(out)} floorplan ({len(blocks)} active blocks)")
    ax.grid(True, linewidth=0.2, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)

    area_mm2 = sum(width * height for _, width, height, _, _ in blocks) * 1e6
    bbox_mm2 = (xmax - xmin) * (ymax - ymin) * 1e6
    print(f"blocks: {len(blocks)}")
    print(f"bbox: {(xmax - xmin) * 1000:.3f} mm x {(ymax - ymin) * 1000:.3f} mm")
    print(f"covered area: {area_mm2:.3f} mm^2 / bbox {bbox_mm2:.3f} mm^2")
    print(f"wrote: {out}")


def path_label(path: Path) -> str:
    return path.stem.replace("_flp", "").replace("-", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a HotSpot floorplan.")
    parser.add_argument("flp", nargs="?", default=Path("GV100.flp"), type=Path)
    parser.add_argument("-o", "--out", default=Path("GV100_flp.png"), type=Path)
    parser.add_argument("--annotate", action="store_true", help="label each active block")
    args = parser.parse_args()

    blocks = read_flp(args.flp)
    plot_flp(blocks, args.out, args.annotate)


if __name__ == "__main__":
    main()
