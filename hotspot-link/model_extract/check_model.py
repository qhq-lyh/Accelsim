import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.linalg import solve
import os
import argparse

def load_trace(filename):
    """Loads HotSpot trace files and returns unit names and data."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Trace file not found: {filename}")
    with open(filename, 'r') as f:  
        headers = f.readline().split()
        data = np.loadtxt(f)
    return headers, data

def verify_extracted_model(mat_dir='.', 
                           power_trace_path='../power_core.trace', 
                           temp_trace_path='../temperature_core.trace', 
                           sampling_intvl=0.001, 
                           ambient_temp=35):
    """
    Runs a transient simulation using extracted G, C, B matrices 
    and compares the result with the original HotSpot temperature trace.
    """
    print(f"--- Starting Model Verification ---")
    
    # 1. Load Extracted Matrices
    print("Loading matrices...")
    G = sio.loadmat(os.path.join(mat_dir, 'G.mat'))['G']
    # C is stored as diagonal elements in Cmatrix file, then converted to .mat
    C_diag = sio.loadmat(os.path.join(mat_dir, 'C.mat'))['C'].diagonal()
    B = sio.loadmat(os.path.join(mat_dir, 'B.mat'))['B']
    
    # 2. Load Reference Traces
    print("Loading reference traces...")
    _, p_data = load_trace(power_trace_path)
    t_headers, t_ref = load_trace(temp_trace_path)

    num_nodes = G.shape[0]
    num_steps = min(p_data.shape[0], t_ref.shape[0])
    
    # 3. Setup Solver (Backward Euler)
    # (C/dt + G) * T_new = (C/dt) * T_old + B * P_new
    C_inv_dt = np.diag(C_diag / sampling_intvl)
    LHS = C_inv_dt + G
    
    # Initialize state (starts at ambient, so rise is 0)
    T_state = np.zeros((num_nodes, 1))
    simulated_temp = np.zeros((num_steps, t_ref.shape[1]))

    # 4. Run Simulation
    print(f"Simulating {num_steps} steps...")
    for k in range(num_steps):
        P_k = p_data[k, :].reshape(-1, 1)
        
        # Solve system of equations
        rhs = (C_inv_dt @ T_state) + (B @ P_k)
        T_state = solve(LHS, rhs)
        
        # Map state rise to core units and add ambient
        # Temperature rise Tc = B^T * T_state
        Tc_k = (B.T @ T_state).flatten()
        simulated_temp[k, :] = Tc_k + ambient_temp

    # 5. Visualization
    plt.figure(figsize=(12, 6))
    # Plotting first 4 cores for clarity (you can adjust this)
    num_plots = min(4, t_ref.shape[1])
    for i in range(num_plots):
        plt.plot(t_ref[:num_steps, i], '--', alpha=0.7, label=f"{t_headers[i]} (HotSpot Ref)")
        plt.plot(simulated_temp[:, i], label=f"{t_headers[i]} (Simulated)")

    plt.title(f"Transient Temperature Comparison (Ambient={ambient_temp}, dt={sampling_intvl}s)")
    plt.xlabel("Time Step")
    plt.ylabel("Temperature (K or °C)")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def parse_hotspot_config(config_path):
    """Parses HotSpot .config file for key parameters."""
    params = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('-'):
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0][1:]
                        try:
                            params[key] = float(parts[1])
                        except ValueError:
                            params[key] = parts[1]
    return params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify extracted HotSpot model.')
    parser.add_argument('--config', type=str, help='Path to HotSpot config file')
    parser.add_argument('--mat_dir', type=str, default='.', help='Directory containing .mat files')
    parser.add_argument('--power_trace', type=str, default='power_core.trace', help='Path to power trace')
    parser.add_argument('--temp_trace', type=str, default='temperature_core.trace', help='Path to temperature trace')
    parser.add_argument('--sampling_intvl', type=float, help='Sampling interval (overrides config)')
    parser.add_argument('--ambient', type=float, help='Ambient temperature (overrides config)')
    parser.add_argument('--grid_layer_file', type=str, default='DDR_16core/cores.lcf', help='Grid layer file')
    args = parser.parse_args()

    config = parse_hotspot_config(args.config)
    s_intvl = args.sampling_intvl or config.get('sampling_intvl', 0.001)
    amb = args.ambient or config.get('ambient', 35.0)
    
    # HotSpot config is in Kelvin, traces are in Celsius. Convert if necessary.
    if amb > 200:
        amb -= 273.15

    verify_extracted_model(mat_dir=args.mat_dir, power_trace_path=args.power_trace, 
                           temp_trace_path=args.temp_trace, sampling_intvl=s_intvl, ambient_temp=amb)
