import os
import sys
import matplotlib.pyplot as plt

# ========= 配置 =========
sm_log_file = "lyhong_SM_power_report.log"
out_dir = "PNG/SM"
SMOOTH_WINDOW = 1
# ========================

if os.path.isdir(out_dir) and any(os.scandir(out_dir)):
    print(f"[Skip] {out_dir} already exists and is not empty. Skip plotting.")
    sys.exit(0)

os.makedirs(out_dir, exist_ok=True)


def smooth(data, window):
    if window <= 1 or len(data) < window:
        return data
    out = []
    for i in range(len(data)):
        s = max(0, i - window + 1)
        out.append(sum(data[s:i+1]) / (i - s + 1))
    return out

# ---------- 读取 SM 表格 ----------
with open(sm_log_file, "r") as f:
    lines = [l.strip() for l in f if l.strip()]

# header
headers = lines[0].split()
num_cols = len(headers)

# 初始化列数据
columns = {h: [] for h in headers}

# 数据行
for line in lines[1:]:
    values = line.split()
    if len(values) != num_cols:
        continue
    for h, v in zip(headers, values):
        columns[h].append(float(v))

# ---------- 按指标分目录绘图 ----------
# header 示例：SM_0_INT_MULP
for col_name, values in columns.items():
    if not values:
        continue

    # 拆解列名
    # SM_0_INT_MULP -> sm_id=0, key=INT_MULP
    parts = col_name.split("_", 2)
    if len(parts) != 3:
        continue

    sm_id = parts[1]
    key = parts[2]

    key_dir = os.path.join(out_dir, key)
    os.makedirs(key_dir, exist_ok=True)

    data = smooth(values, SMOOTH_WINDOW)

    plt.figure()
    plt.plot(data)
    plt.xlabel("Sample index")
    plt.ylabel("Power")
    plt.title(f"SM {sm_id} {key}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(key_dir, f"SM_{sm_id}_{key}.png"),
        dpi=300
    )
    plt.close()

print(f"SM plots saved to {out_dir}")
