#!/usr/bin/env python3
"""Invert steady G/B thermal model under per-block P=P0+k*(f/fref)^exponent."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]


def load_triplets(path):
    data = np.loadtxt(path, ndmin=2)
    # Exporter appends dimensions, not another nonzero entry.
    shape = tuple(data[-1, :2].astype(int))
    entries = data[:-1]
    return csc_matrix((entries[:, 2], (entries[:, 0].astype(int), entries[:, 1].astype(int))), shape=shape)


def load_model(directory):
    ptr = np.loadtxt(directory / 'Amatrixcolptr', dtype=int)
    n = len(ptr) - 1
    g = csc_matrix((np.loadtxt(directory / 'Amatrixnzval'),
                    np.loadtxt(directory / 'Amatrixrowind', dtype=int), ptr), shape=(n, n))
    b = load_triplets(directory / 'Bmatrix')
    l = load_triplets(directory / 'Lmatrix').T
    if b.shape[0] != n or l.shape[1] != n:
        raise ValueError('inconsistent G/B/L dimensions')
    return g, b, l


def average_power(path):
    with path.open() as file:
        names = file.readline().split()
        data = np.loadtxt(file, ndmin=2)
    if data.size == 0 or data.shape[1] != len(names):
        raise ValueError(f'{path}: invalid power table')
    if not np.all(np.isfinite(data)) or np.any(data < 0):
        raise ValueError(f'{path}: invalid power values')
    return names, data.mean(axis=0)


def threshold_frequency(base_c, rise_at_reference, target_c, reference_mhz, exponent):
    """All nodes must be <=target; the first binding node determines frequency."""
    if np.any(base_c > target_c):
        raise ValueError('fixed/background power alone exceeds threshold')
    active = rise_at_reference > 1e-12
    if not np.any(active):
        raise ValueError('temperature does not depend on frequency in this model')
    if np.any(rise_at_reference < -1e-8):
        raise ValueError('non-monotone thermal response')
    ratios = np.full(len(base_c), np.inf)
    ratios[active] = (target_c - base_c[active]) / rise_at_reference[active]
    index = int(np.argmin(ratios))
    return reference_mhz * float(ratios[index]) ** (1 / exponent), index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrices', type=Path, default=ROOT / 'hotspot-link/model_extract')
    parser.add_argument('--flp', type=Path, default=ROOT / 'hotspot-link/flp/gv100_new.flp')
    parser.add_argument('--sample', action='append', required=True, help='MHz=power.ptrace; repeat to fit multiple samples')
    parser.add_argument('--exponent', type=float, default=1., help='default 1: fixed-voltage, frozen-activity approximation of AccelWattch energy/time; other exponents are explicit user hypotheses')
    parser.add_argument('--fit-offset', action='store_true', help='fit nonnegative fixed/background power; requires >=2 distinct frequencies')
    parser.add_argument('--scale', choices=['all', 'sm'], default='all', help='which blocks depend on frequency; others remain at highest-frequency sample power')
    parser.add_argument('--threshold-c', type=float, required=True)
    parser.add_argument('--ambient-c', type=float, default=45.)
    parser.add_argument('--rows', type=int, default=64)
    parser.add_argument('--cols', type=int, default=64)
    parser.add_argument('--max-mhz', type=float, help='optional upper bound; report unreachable if threshold is beyond it')
    parser.add_argument('--reference-grid', type=Path, help='HotSpot grid result for the highest-frequency sample, for reconstruction verification')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.exponent <= 0 or args.threshold_c <= args.ambient_c:
        parser.error('positive exponent and threshold above ambient required')
    if args.rows <= 0 or args.cols <= 0 or (args.max_mhz is not None and args.max_mhz <= 0):
        parser.error('positive grid dimensions and frequency bounds required')
    samples = []
    for item in args.sample:
        frequency, path = item.split('=', 1)
        frequency = float(frequency)
        if not np.isfinite(frequency) or frequency <= 0:
            parser.error('sample frequency must be positive and finite')
        names, power = average_power(Path(path))
        samples.append((frequency, names, power))
    samples.sort(key=lambda sample: sample[0])
    names = samples[-1][1]
    if any(sample[1] != names for sample in samples):
        parser.error('sample power headers/order differ')
    flp_names = [line.split()[0] for line in args.flp.read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith('#')]
    if names != flp_names:
        parser.error('power order must match the active-layer floorplan order used to export B')
    g, b, l = load_model(args.matrices)
    if b.shape[1] != len(names) or args.rows * args.cols > g.shape[0]:
        parser.error('matrix dimensions do not match power/grid dimensions')
    if args.fit_offset and len(set(sample[0] for sample in samples)) < 2:
        parser.error('fit-offset requires >=2 distinct frequencies')
    reference = samples[-1][0]
    frequencies = np.array([sample[0] for sample in samples])
    observed = np.array([sample[2] for sample in samples])
    x = (frequencies / reference) ** args.exponent
    p0 = np.zeros(len(names))
    coefficient = np.zeros(len(names))
    selected = np.array([args.scale == 'all' or name.startswith('SM_') for name in names])
    for index in range(len(names)):
        if not selected[index]:
            p0[index] = observed[-1, index]
        elif args.fit_offset:
            p0[index], coefficient[index] = nnls(np.column_stack([np.ones(len(x)), x]), observed[:, index])[0]
        else:
            coefficient[index] = np.dot(x, observed[:, index]) / np.dot(x, x)
    fitted = p0[None, :] + x[:, None] * coefficient[None, :]
    lu = splu(g)
    thermal_fixed = lu.solve(b @ p0)
    thermal_scaled = lu.solve(b @ coefficient)
    count = args.rows * args.cols
    base = args.ambient_c + thermal_fixed[:count]
    candidate, binding = threshold_frequency(base, thermal_scaled[:count], args.threshold_c, reference, args.exponent)
    chosen = min(candidate, args.max_mhz) if args.max_mhz is not None else candidate
    factor = (chosen / reference) ** args.exponent
    predicted_power = p0 + coefficient * factor
    predicted_rise = thermal_fixed + thermal_scaled * factor
    reconstructed = args.ambient_c + lu.solve(b @ observed[-1])[:count]
    errors = []
    for frequency, obs, fit in zip(frequencies, observed, fitted):
        obs_temp = args.ambient_c + lu.solve(b @ obs)[:count]
        fit_temp = args.ambient_c + lu.solve(b @ fit)[:count]
        errors.append(dict(freq_mhz=float(frequency), observed_mapped_watts=float(obs.sum()),
                           fitted_mapped_watts=float(fit.sum()), observed_peak_c=float(obs_temp.max()),
                           fitted_peak_c=float(fit_temp.max()),
                           power_vector_relative_l2_error=float(np.linalg.norm(obs-fit) / max(np.linalg.norm(obs), 1e-12))))
    result = dict(model=f'P=P0+k*(f/{reference})^{args.exponent}',
                  power_model_basis=('AccelWattch energy/time motivates exponent 1 only with fixed energy coefficients and frozen per-cycle activity; supplied samples calibrate the approximation, not exported internal coefficients'
                                     if args.exponent == 1. else 'Explicit user power-law hypothesis; not the frequency law implemented by current AccelWattch'),
                  exponent=args.exponent, scale=args.scale, fit_offset=args.fit_offset,
                  G_shape=list(g.shape), B_shape=list(b.shape), L_shape=list(l.shape),
                  C_used=False, ambient_c=args.ambient_c, threshold_c=args.threshold_c,
                  reference_mhz=reference, reconstructed_reference_peak_c=float(reconstructed.max()),
                  raw_threshold_frequency_mhz=candidate, selected_frequency_mhz=chosen,
                  threshold_reachable_within_bound=args.max_mhz is None or candidate <= args.max_mhz,
                  predicted_selected_peak_c=float((args.ambient_c+predicted_rise[:count]).max()),
                  binding_grid_index=binding, binding_grid_row=binding // args.cols, binding_grid_col=binding % args.cols,
                  mapped_predicted_watts=float(predicted_power.sum()),
                  simulator_core_freq_uint32_compatible=0 < chosen * 1e6 <= np.iinfo(np.uint32).max,
                  calibration=errors,
                  warning='Exact inverse within the assumed power model only. Unmapped GPU power is excluded; rerun Accel-Sim to verify.')
    if args.reference_grid:
        hot = np.loadtxt(args.reference_grid, ndmin=2)[:count, 1] - 273.15
        if len(hot) != count:
            parser.error('reference grid too short')
        result['reference_hotspot_max_abs_error_c'] = float(np.max(np.abs(hot - reconstructed)))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.out_dir / 'frequency_mhz.txt').write_text(f'{chosen:.9f}\n')
    with (args.out_dir / 'power_coefficients.csv').open('w') as file:
        writer = csv.writer(file)
        writer.writerow(['block', 'fixed_watts', 'watts_at_reference_scaling', 'predicted_watts'])
        writer.writerows(zip(names, p0, coefficient, predicted_power))
    with (args.out_dir / 'predicted_power.ptrace').open('w') as file:
        file.write('\t'.join(names) + '\n')
        file.write('\t'.join(f'{value:.12g}' for value in predicted_power) + '\n')
    np.savetxt(args.out_dir / 'predicted_active_grid.csv',
               np.column_stack([np.arange(count), args.ambient_c+predicted_rise[:count]]),
               delimiter=',', header='grid_index,temperature_c', comments='')
    np.savetxt(args.out_dir / 'predicted_L_observations_c.csv',
               args.ambient_c + l @ predicted_rise, header='area_weighted_observation_c', comments='')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
