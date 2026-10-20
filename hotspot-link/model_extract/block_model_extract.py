#!/usr/bin/env python3
import argparse
import os

import numpy as np
from scipy.io import savemat
from scipy.linalg import solve
from scipy.sparse import csc_matrix, eye, spdiags
from scipy.sparse.linalg import spsolve


def load_trace(path):
    with open(path) as f:
        headers = f.readline().split()
        data = np.loadtxt(f)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return headers, data


def parse_config(path):
    params = {}
    if not path:
        return params
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line.startswith("-"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0][1:]
            try:
                params[key] = float(parts[1])
            except ValueError:
                params[key] = parts[1]
    return params


def export_block_mats(block_dir, output_dir, num_units):
    G_dense = np.loadtxt(os.path.join(block_dir, "Gmatrix_block"))
    C_diag = np.loadtxt(os.path.join(block_dir, "Cmatrix_block"))

    if G_dense.shape[0] != G_dense.shape[1]:
        raise ValueError(f"Gmatrix_block must be square, got {G_dense.shape}")
    if C_diag.shape[0] != G_dense.shape[0]:
        raise ValueError("Cmatrix_block length must match Gmatrix_block size")
    if num_units > G_dense.shape[0]:
        raise ValueError("trace/floorplan unit count is larger than block node count")

    num_nodes = G_dense.shape[0]
    G = csc_matrix(G_dense)
    C = spdiags(C_diag, 0, num_nodes, num_nodes, format="csc")

    # Block HotSpot stores floorplan-unit temperatures in the first n nodes.
    # This is the block equivalent of grid B: it injects trace power into those nodes.
    B = csc_matrix(
        (np.ones(num_units), (np.arange(num_units), np.arange(num_units))),
        shape=(num_nodes, num_units),
    )
    L = B.T.tocsc()

    os.makedirs(output_dir, exist_ok=True)
    savemat(os.path.join(output_dir, "G.mat"), {"G": G})
    savemat(os.path.join(output_dir, "C.mat"), {"C": C})
    savemat(os.path.join(output_dir, "B.mat"), {"B": B})
    savemat(os.path.join(output_dir, "L.mat"), {"L": L})

    G_inv_B = spsolve(G, B.toarray())
    A = B.T @ G_inv_B
    savemat(os.path.join(output_dir, "A.mat"), {"A": A})

    return G_dense, C_diag, B


def plot_verification(headers, reference_c, simulated_c, output_path, max_units=6):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    err = simulated_c - reference_c
    steps = np.arange(reference_c.shape[0])
    units = min(max_units, reference_c.shape[1])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for idx in range(units):
        axes[0].plot(steps, reference_c[:, idx], "--", linewidth=1.4, label=f"{headers[idx]} HotSpot")
        axes[0].plot(steps, simulated_c[:, idx], linewidth=1.2, label=f"{headers[idx]} block mat")
        axes[1].plot(steps, err[:, idx], linewidth=1.2, label=headers[idx])

    axes[0].set_ylabel("Temperature (C)")
    axes[0].set_title("HotSpot Block Trace vs Extracted Block Matrix Model")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2, fontsize="small")

    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Error (C)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=3, fontsize="small")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def verify(block_dir, temp_trace, dt, init_temp_k, plot_path=None, max_plot_units=6):
    headers, reference_c = load_trace(temp_trace)
    num_units = len(headers)
    G = np.loadtxt(os.path.join(block_dir, "Gmatrix_block"))
    C = np.loadtxt(os.path.join(block_dir, "Cmatrix_block"))
    P = np.loadtxt(os.path.join(block_dir, "Pmatrix_block"))

    lhs = G + np.diag(C / dt)
    rhs_cap = C / dt
    temp_k = np.full(G.shape[0], init_temp_k)
    simulated_c = np.empty((reference_c.shape[0], num_units))

    for step in range(reference_c.shape[0]):
        temp_k = solve(lhs, P + rhs_cap * temp_k, assume_a="gen")
        simulated_c[step] = temp_k[:num_units] - 273.15

    err = simulated_c - reference_c
    if plot_path:
        plot_verification(headers, reference_c, simulated_c, plot_path, max_plot_units)
    return {
        "steps": reference_c.shape[0],
        "units": num_units,
        "min_ref_c": float(np.min(reference_c)),
        "max_ref_c": float(np.max(reference_c)),
        "min_sim_c": float(np.min(simulated_c)),
        "max_sim_c": float(np.max(simulated_c)),
        "max_abs_err_c": float(np.max(np.abs(err))),
        "mean_abs_err_c": float(np.mean(np.abs(err))),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export HotSpot block G/C/B/L/A .mat files and optionally verify them."
    )
    parser.add_argument("--block_dir", default="..", help="directory containing Gmatrix_block/Cmatrix_block/Pmatrix_block")
    parser.add_argument("--output_dir", default="block_mat", help="directory for generated .mat files")
    parser.add_argument("--power_trace", help="power trace; only the header is used to get unit count")
    parser.add_argument("--temp_trace", help="HotSpot ttrace used for verification")
    parser.add_argument("--config", help="HotSpot config used to get init_temp if --init_temp is omitted")
    parser.add_argument("--dt", type=float, default=0.001, help="sampling interval in seconds")
    parser.add_argument("--init_temp", type=float, help="initial temperature in Kelvin")
    parser.add_argument("--plot", help="save a verification plot to this path")
    parser.add_argument("--max_plot_units", type=int, default=6, help="number of units to draw in the plot")
    args = parser.parse_args()

    if not args.power_trace and not args.temp_trace:
        raise SystemExit("need --power_trace or --temp_trace to determine the floorplan unit count")

    if args.power_trace:
        headers, _ = load_trace(args.power_trace)
    else:
        headers, _ = load_trace(args.temp_trace)

    config = parse_config(args.config)
    init_temp_k = args.init_temp if args.init_temp is not None else config.get("init_temp", 318.15)

    G, C, B = export_block_mats(args.block_dir, args.output_dir, len(headers))
    print(f"wrote {args.output_dir}/G.mat C.mat B.mat L.mat A.mat")
    print(f"G: {G.shape}, C: ({C.shape[0]}, {C.shape[0]}), B: {B.shape}, units: {len(headers)}")

    if args.temp_trace:
        stats = verify(args.block_dir, args.temp_trace, args.dt, init_temp_k, args.plot, args.max_plot_units)
        print("verification against HotSpot block ttrace:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        if args.plot:
            print(f"saved plot: {args.plot}")


if __name__ == "__main__":
    main()
