# 将该脚本置于输出文件夹中运行
import matplotlib.pyplot as plt
import glob
log_files = glob.glob("*.o??")

if len(log_files) == 0:
    raise RuntimeError("No .oXX log file found in current directory.")
elif len(log_files) > 1:
    raise RuntimeError(f"Multiple .oXX log files found: {log_files}")


log_file = log_files[0]
values = []

with open(log_file, "r") as f:
    for line in f:
        if "Lyhong_print: Sum of lyhong_active_sm" in line:
            try:
                value = 80 - float(line.split("=")[-1].strip())
                values.append(value)
            except ValueError:
                pass

if not values:
    print("No lyhong_idle_sm data found.")
    exit(0)

plt.figure()
plt.plot(range(1, len(values) + 1), values)
plt.xlabel("Sample index")
plt.ylabel("Sum of lyhong_idle_sm")
plt.title("Sum of lyhong_idle_sm over time")
plt.grid(True)
plt.tight_layout()
plt.savefig("lyhong_idle_sm.png", dpi=300)
plt.close()

print("Figure saved as lyhong_idle_sm.png")
