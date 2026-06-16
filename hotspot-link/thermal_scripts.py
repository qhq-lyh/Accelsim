#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

base_dir = Path(__file__).resolve().parent

hotspot_bin = (
    base_dir.parent
    / "hotspot-link"
    / "hotspot"
)

config = base_dir / "GV100-hotspot.config"
ptrace = base_dir / "GV100_example_100.ptrace"
lcf = base_dir / "flp/gv100.lcf"
out_dir = base_dir / "output"
steady_file = out_dir / "temp.steady"
grid_steady_file = out_dir / "temp.grid.steady"
plot_script = base_dir / "plot_gv100_grid_heatmap.py"
heatmap_png = out_dir / "gv100_grid_heatmap_celsius.png"

out_dir.mkdir(parents=True, exist_ok=True)

cmd = [
    str(hotspot_bin),
    "-c", str(config.relative_to(base_dir)),
    "-p", str(ptrace.relative_to(base_dir)),
    "-grid_layer_file", str(lcf.relative_to(base_dir)),
    "-model_type", "grid",
    "-steady_file", str(steady_file.relative_to(base_dir)),
    "-grid_steady_file", str(grid_steady_file.relative_to(base_dir)),
]

print("Running HotSpot:")
print(" ".join(cmd))

subprocess.run(cmd, cwd=base_dir, check=True)

plot_cmd = [
    sys.executable,
    str(plot_script.relative_to(base_dir)),
    "--grid-steady", str(grid_steady_file.relative_to(base_dir)),
    "--out", str(heatmap_png.relative_to(base_dir)),
]

print("Plotting heatmap:")
print(" ".join(plot_cmd))

subprocess.run(plot_cmd, cwd=base_dir, check=True)

print(f"Done. Heatmap saved to: {heatmap_png}")