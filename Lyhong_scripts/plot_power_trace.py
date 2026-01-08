import os
import glob
import gzip
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def find_and_decompress(prefix):
    """
    找到 prefix_*.log 或 prefix_*.log.gz
    如果是 gz，则解压
    返回 log 文件名
    """
    logs = glob.glob(f"{prefix}_*.log")
    if logs:
        return logs[0]

    gz_logs = glob.glob(f"{prefix}_*.log.gz")
    if not gz_logs:
        raise FileNotFoundError(f"No {prefix}_*.log(.gz) found")

    gz_file = gz_logs[0]
    log_file = gz_file[:-3]

    print(f"Decompressing {gz_file} -> {log_file}")
    with gzip.open(gz_file, "rb") as fin, open(log_file, "wb") as fout:
        shutil.copyfileobj(fin, fout)

    return log_file


def deep_scalar(x):
    """
    无限展开 tuple / list / ndarray
    """
    while isinstance(x, (tuple, list, np.ndarray)):
        if len(x) == 0:
            return np.nan
        x = x[0]
    return x


def process_and_plot(prefix, out_dir):
    """
    通用处理流程：读取 → 清洗 → 逐列画图
    """
    log_file = find_and_decompress(prefix)
    print(f"Using {log_file}")

    os.makedirs(out_dir, exist_ok=True)

    # 读取 CSV（python engine 必须）
    df = pd.read_csv(
        log_file,
        sep=",",
        engine="python"
    )

    # 删除因行尾逗号产生的空列
    df = df.dropna(axis=1, how="all")

    # 深度清洗
    df = df.applymap(deep_scalar)
    df = df.apply(pd.to_numeric, errors="coerce")

    x = np.arange(len(df))

    for col in df.columns:
        y = df[col].values.astype(float)

        if np.all(np.isnan(y)):
            continue

        plt.figure()
        plt.plot(x, y)
        plt.xlabel("Sample Index")
        plt.ylabel(col)
        plt.title(col)
        plt.grid(True)

        plt.savefig(
            os.path.join(out_dir, f"{col}.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    print(f"All figures saved to ./{out_dir}/")

if __name__ == "__main__":
    process_and_plot(
        prefix="gpgpusim_power_trace_report",
        out_dir="power_figure"
    )

    process_and_plot(
        prefix="gpgpusim_metric_trace_report",
        out_dir="metric_figure"
    )
