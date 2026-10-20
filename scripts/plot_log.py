# 将该脚本置于输出文件夹中运行
import os
import re
import matplotlib.pyplot as plt

# ========= 配置 =========
log_file = "lyhong_power_report.log"
out_dir = "PNG"
SMOOTH_WINDOW = 1   # 小窗口平滑，尽量保持真实性
# ========================

power_keys = [
    "rt_power.readOp.dynamic", "Total Power",
    "gpu_IBP", "gpu_ICP", "gpu_DCP", "gpu_TCP", "gpu_CCP",
    "gpu_SHRDP", "gpu_RFP", "gpu_INTP", "gpu_FPUP", "gpu_DPUP",
    "gpu_INT_MUL24P", "gpu_INT_MUL32P", "gpu_INT_MULP",
    "gpu_INT_DIVP", "gpu_FP_MULP", "gpu_FP_DIVP", "gpu_FP_SQRTP",
    "gpu_FP_LGP", "gpu_FP_SINP", "gpu_FP_EXP",
    "gpu_DP_MULP", "gpu_DP_DIVP", "gpu_TENSORP", "gpu_TEXP",
    "gpu_SCHEDP", "gpu_L2CP", "gpu_MCP", "gpu_NOCP",
    "gpu_DRAMP", "gpu_PIPEP", "gpu_IDLE_COREP",
    "gpu_CONSTP", "gpu_STATICP",
]
gpu_star_keys = [
    "gpu_IBP", "gpu_ICP", "gpu_DCP", "gpu_TCP", "gpu_CCP",
    "gpu_SHRDP", "gpu_RFP", "gpu_INTP", "gpu_FPUP", "gpu_DPUP",
    "gpu_INT_MUL24P", "gpu_INT_MUL32P", "gpu_INT_MULP",
    "gpu_INT_DIVP", "gpu_FP_MULP", "gpu_FP_DIVP", "gpu_FP_SQRTP",
    "gpu_FP_LGP", "gpu_FP_SINP", "gpu_FP_EXP",
    "gpu_DP_MULP", "gpu_DP_DIVP", "gpu_TENSORP", "gpu_TEXP",
    "gpu_SCHEDP", "gpu_L2CP", "gpu_MCP", "gpu_NOCP",
    "gpu_DRAMP", "gpu_PIPEP",
]

power_data = {k: [] for k in power_keys}
idle_cores = []

def smooth(data, window):
    if window <= 1 or len(data) < window:
        return data
    out = []
    for i in range(len(data)):
        s = max(0, i - window + 1)
        out.append(sum(data[s:i+1]) / (i - s + 1))
    return out

# ---------- 解析 log ----------
with open(log_file, "r") as f:
    for line in f:
        line = line.strip()
        # num_idle_cores
        m = re.search(r"num_idle_cores:\s*([0-9.]+)", line)
        if m:
            idle_cores.append(float(m.group(1)))
        # rt_power.readOp.dynamic（单独处理，忽略尾注释）
        if line.startswith("rt_power.readOp.dynamic"):
            m = re.search(r":\s*([0-9.]+)", line)
            if m:
                power_data["rt_power.readOp.dynamic"].append(float(m.group(1)))
            continue
        # 其余功耗项
        for key in power_keys:
            if key == "rt_power.readOp.dynamic":
                continue
            if line.startswith(key):
                try:
                    value = float(line.split(":")[-1].strip())
                    power_data[key].append(value)
                except ValueError:
                    pass

# ---------- 创建输出目录 ----------
os.makedirs(out_dir, exist_ok=True)

# ---------- idle cores ----------
if idle_cores:
    data = smooth(idle_cores, SMOOTH_WINDOW)
    plt.figure()
    plt.plot(data)
    plt.xlabel("Sample index")
    plt.ylabel("Idle cores")
    plt.title("Idle cores over time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "idle_cores.png"), dpi=300)
    plt.close()
# ---------- idle_core vs static power ----------
if idle_cores and power_data["gpu_STATICP"]:
    n = min(len(idle_cores), len(power_data["gpu_STATICP"]))

    x = idle_cores[:n]
    y = power_data["gpu_STATICP"][:n]

    plt.figure()
    plt.scatter(x, y, s=6, alpha=0.6)
    plt.xlabel("Idle cores")
    plt.ylabel("gpu_STATICP")
    plt.title("Idle cores vs Static Power")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "idle_core_vs_staticp.png"), dpi=300)
    plt.close()
# ---------- gpu_*P sum ----------
gpu_star_sum = []

num_samples = max(len(v) for v in power_data.values() if v)

for i in range(num_samples):
    s = 0.0
    for k in gpu_star_keys:
        if i < len(power_data[k]):
            s += power_data[k][i]
    gpu_star_sum.append(s)
# ---------- gpu_*P sum vs idle cores ----------
if idle_cores and gpu_star_sum:
    n = min(len(idle_cores), len(gpu_star_sum))

    x = idle_cores[:n]
    y = gpu_star_sum[:n]

    plt.figure()
    plt.scatter(x, y, s=6, alpha=0.6)
    plt.xlabel("Idle cores")
    plt.ylabel("Sum of gpu_*P")
    plt.title("gpu_*P (sum) vs Idle cores")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gpu_starP_vs_idle_core.png"), dpi=300)
    plt.close()
# ---------- gpu_IDLE_COREP vs idle cores ----------
if idle_cores and power_data["gpu_IDLE_COREP"]:
    n = min(len(idle_cores), len(power_data["gpu_IDLE_COREP"]))

    x = idle_cores[:n]
    y = power_data["gpu_IDLE_COREP"][:n]

    plt.figure()
    plt.scatter(x, y, s=6, alpha=0.6)
    plt.xlabel("Idle cores")
    plt.ylabel("gpu_IDLE_COREP")
    plt.title("gpu_IDLE_COREP vs Idle cores")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gpu_IDLE_COREP_vs_idle_core.png"), dpi=300)
    plt.close()

# ---------- 功耗项 ----------
for key, values in power_data.items():
    if not values:
        continue

    data = smooth(values, SMOOTH_WINDOW)

    plt.figure()
    plt.plot(data)
    plt.xlabel("Sample index")
    plt.ylabel("Power")
    plt.title(key)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{key}.png"), dpi=300)
    plt.close()

print(f"All figures saved to ./{out_dir}/")
