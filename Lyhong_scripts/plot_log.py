# 将该脚本置于输出文件夹中运行
import os
import re
import matplotlib.pyplot as plt

# ========= 配置 =========
log_file = "lyhong_power_report.log"   # ← 改成你的 .log 文件名
out_dir = "PNG"
# ========================
power_keys = [
    "Dynamic Power Total",
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

power_data = {k: [] for k in power_keys}
idle_cores = []

# ---------- 解析 log ----------
with open(log_file, "r") as f:
    for line in f:
        line = line.strip()

        # num_idle_cores（允许浮点）
        if "num_idle_cores:" in line:
            tokens = line.replace(",", " ").split()
            for i, t in enumerate(tokens):
                if t == "num_idle_cores:":
                    try:
                        idle_cores.append(float(tokens[i + 1]))
                    except ValueError:
                        pass
                    break

        # 功耗字段（统一从冒号后取）
        for key in power_keys:
            if line.startswith(key):
                try:
                    value = float(line.split(":")[-1].strip())
                    power_data[key].append(value)
                except ValueError:
                    pass

# ---------- 创建输出目录 ----------
os.makedirs(out_dir, exist_ok=True)

# ---------- idle core 数量 ----------
if idle_cores:
    plt.figure()
    plt.plot(range(1, len(idle_cores) + 1), idle_cores)
    plt.xlabel("Sample index")
    plt.ylabel("Idle cores")
    plt.title("Idle cores over time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "idle_cores.png"), dpi=300)
    plt.close()

# ---------- 功耗项 ----------
for key, values in power_data.items():
    if not values:
        continue

    plt.figure()
    plt.plot(range(1, len(values) + 1), values)
    plt.xlabel("Sample index")
    plt.ylabel("Power")
    plt.title(key)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{key}.png"), dpi=300)
    plt.close()

print(f"All figures saved to ./{out_dir}/")