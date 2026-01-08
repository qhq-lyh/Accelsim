import re
import os
import matplotlib.pyplot as plt

log_file = "lyhong_power_report.log"  
out_dir = "PNG"
# 存储数据
tsc_list = []
static_p_list = []
static_p_scaled_list = []
per_active_core_list = []

current_per_active_core = None
current_tsc = None

with open(log_file, "r") as f:
    for line in f:
        # 提取 per_active_core
        if line.startswith("lyhong_print:"):
            m = re.search(r"per_active_core:\s*([0-9.eE+-]+)", line)
            if m:
                current_per_active_core = float(m.group(1))

        # 提取 TSC
        elif line.startswith("Total_sample_count"):
            m = re.search(r"TSC\):\s*(\d+)", line)
            if m:
                current_tsc = int(m.group(1))

        # 提取 static power
        elif line.startswith("gpu_STATICP"):
            m = re.search(r"gpu_STATICP:\s*([0-9.eE+-]+)", line)
            if m and current_per_active_core is not None and current_tsc is not None:
                static_p = float(m.group(1))
                scaled_p = static_p * current_per_active_core

                tsc_list.append(current_tsc)
                static_p_list.append(static_p)
                static_p_scaled_list.append(scaled_p)
                per_active_core_list.append(current_per_active_core)

                # 清空，防止错配
                current_per_active_core = None
                current_tsc = None

# ====== 画图 ======
plt.figure(figsize=(8, 4))

plt.plot(
    tsc_list,
    static_p_list,
    label="Static Power",
    linewidth=1.2
)

plt.plot(
    tsc_list,
    static_p_scaled_list,
    label="Static Power × per_active_core",
    linewidth=1.2,
    linestyle="--"
)

plt.xlabel("Sample Count (TSC)")
plt.ylabel("Power")
plt.title("Static Power vs Active-Core-Scaled Static Power")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(out_dir,"static_power_compare.png"), dpi=300)


