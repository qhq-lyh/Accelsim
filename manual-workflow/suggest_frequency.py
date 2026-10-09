#!/usr/bin/env python3
"""Suggest (not guarantee) a uniform-domain frequency from SM block temperatures."""
import argparse
import math
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--steady', type=Path, required=True)
p.add_argument('--grid', action='store_true', help='read active-silicon grid maximum instead of SM block center maximum; both input files must be grid files')
p.add_argument('--rows', type=int, default=64)
p.add_argument('--cols', type=int, default=64)
p.add_argument('--freq', type=float, required=True, help='current MHz')
p.add_argument('--threshold-c', type=float, required=True)
p.add_argument('--ambient-c', type=float, default=45.)
p.add_argument('--previous-steady', type=Path)
p.add_argument('--previous-freq', type=float)
p.add_argument('--min-mhz', type=float, default=300.)
p.add_argument('--max-mhz', type=float, default=4000.)
a = p.parse_args()
def peak(path):
    values = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if a.grid and len(parts) == 2:
            values.append(float(parts[1]) - 273.15)
            if len(values) == a.rows * a.cols:
                break
        elif len(parts) == 2 and parts[0].startswith('layer_0_SM_'):
            values.append(float(parts[1]) - 273.15)
    if not values:
        p.error(f'no active-layer SM temperatures in {path}')
    if a.grid and len(values) != a.rows * a.cols:
        p.error('grid has fewer cells than rows*cols')
    return max(values)
t = peak(a.steady)
if (a.previous_steady is None) != (a.previous_freq is None):
    p.error('provide both previous-steady and previous-freq')
if not (a.freq > 0 and a.threshold_c > a.ambient_c and 0 < a.min_mhz <= a.max_mhz <= 4294):
    p.error('invalid frequency bounds or threshold at/below ambient')
if a.previous_steady is not None:
    old_t = peak(a.previous_steady)
    if a.freq == a.previous_freq:
        p.error('two distinct frequencies required')
    slope = (t - old_t) / (a.freq - a.previous_freq)
    if slope <= 1e-9:
        p.error('temperature-frequency slope is not positive; use more measurements or bracket search')
    candidate = a.freq + (a.threshold_c - t) / slope
    method = 'two-point local secant estimate'
else:
    if t <= a.ambient_c:
        p.error('non-positive temperature rise')
    candidate = a.freq * (a.threshold_c - a.ambient_c) / (t - a.ambient_c)
    method = 'single-point proportional approximation; assumes all mapped power scales with frequency'
if not math.isfinite(candidate):
    p.error('non-finite candidate')
clamped = max(a.min_mhz, min(a.max_mhz, candidate))
print(f'Peak={t:.4f} C; target={a.threshold_c:.4f} C; delta={a.threshold_c-t:.4f} C; method={method}; raw={candidate:.3f} MHz; bounded={clamped:.3f} MHz. Rerun required.', file=sys.stderr)
print(f'{clamped:.3f}')
