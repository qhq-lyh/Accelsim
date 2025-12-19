# 将该脚本置于输出文件夹中运行
import matplotlib.pyplot as plt
import re
import os
import glob
NUM_SM = 80
out_dir = "Core_Change_PNG"
os.makedirs(out_dir, exist_ok=True)
log_files = glob.glob("*.o[0-9][0-9]")

if len(log_files) == 0:
    raise RuntimeError("No .oXX log file found in current directory.")
elif len(log_files) > 1:
    raise RuntimeError(f"Multiple .oXX log files found: {log_files}")

log_file = log_files[0]
print(f"Using log file: {log_file}")
current_kernel_name = None
current_uid = None
cycles = []
active_sm = []
idle_sm = []

def flush_and_plot():
    """把当前 kernel 的数据画图并清空"""
    global cycles, active_sm, current_kernel_name, current_uid

    if not cycles:
        return

    if cycles[0] != 0:
        cycles.insert(0, 0)
        active_sm.insert(0, 0.0)

    idle_sm = [NUM_SM - x for x in active_sm]

    plt.figure(figsize=(6, 4))
    plt.step(cycles, idle_sm, where="post")
    plt.xlabel("Kernel cycle")
    plt.ylabel("Idle SM count")
    plt.title(f"{current_kernel_name} (uid={current_uid})")
    plt.grid(True)
    plt.tight_layout()

    fname = f"{current_kernel_name}_{current_uid}.png"
    fname = fname.replace("/", "_")
    plt.savefig(os.path.join(out_dir, fname), dpi=300)
    plt.close()

    print(f"Saved {fname}")

    cycles.clear()
    active_sm.clear()

with open(log_file, "r") as f:
    for line in f:
        if "launching kernel name:" in line:
            flush_and_plot()
            m = re.search(r"launching kernel name:\s+(\S+)\s+uid:\s+(\d+)", line)
            if not m:
                continue
            current_kernel_name = m.group(1)
            current_uid = int(m.group(2))
            print(f"Start kernel {current_kernel_name}, uid={current_uid}")
            cycles = [0]
            active_sm = [0.0]
        elif "Lyhong_print(has changed)" in line and current_kernel_name is not None:
            try:
                sm = float(line.split("sum_active_sm =")[1].split(",")[0].strip())
                cycle_part = line.split("cycle =")[1]
                cycle = int(cycle_part.split(",")[0].strip())
                cycles.append(cycle)
                active_sm.append(sm)
            except (IndexError, ValueError):
                pass

flush_and_plot()