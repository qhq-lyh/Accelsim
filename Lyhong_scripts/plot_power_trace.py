import os
import glob
import gzip
import shutil
import pandas as pd
import matplotlib.pyplot as plt

def gunzip_if_needed(gz_pattern):
    """
    如果存在 .log.gz 且 .log 不存在，则解压
    返回所有 .log 文件路径
    """
    log_files = []

    for gz_file in glob.glob(gz_pattern):
        log_file = gz_file[:-3]
        if not os.path.exists(log_file):
            print(f"[INFO] Decompressing {gz_file}")
            with gzip.open(gz_file, 'rb') as f_in, open(log_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            print(f"[INFO] {log_file} already exists, skip decompression")
        log_files.append(log_file)

    return log_files


def ensure_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def fix_power_header_if_needed(log_file):
    """
    修复 power_trace 第一行 CONSTPSTATICP -> CONSTP,STATICP,
    若已修复则不处理
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()

    if not lines:
        return

    header = lines[0]

    if "CONSTPSTATICP" in header:
        print(f"[FIX] Fixing header in {log_file}")
        header = header.replace("CONSTPSTATICP", "CONSTP,STATICP,")
        lines[0] = header
        with open(log_file, 'w') as f:
            f.writelines(lines)
    else:
        print(f"[INFO] Power header already fixed: {log_file}")

def fix_metric_header_if_needed(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    if not lines:
        return

    header = lines[0]

    if "constant_power\n" in header:
        print(f"[FIX] Fixing header in {log_file}")
        header = header.replace("constant_power\n", "constant_power,\n")
        lines[0] = header
        with open(log_file, 'w') as f:
            f.writelines(lines)
    else:
        print(f"[INFO] metric header already fixed: {log_file}")\

def plot_csv_columns(log_file, out_dir, title_prefix):
    """
    每一列单独画一张折线图
    x轴：采样编号
    y轴：该列的值
    """
    ensure_dir(out_dir)

    # 末尾多一个逗号，用 python engine 更稳妥
    df = pd.read_csv(log_file, engine="python")

    x = range(len(df))

    for col in df.columns:
        plt.figure(figsize=(8, 4))
        plt.plot(x, df[col].values)
        plt.xlabel("Sample Index")
        plt.ylabel(col)
        plt.title(f"{title_prefix}: {col}")
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{col}.png")
        plt.savefig(out_path)
        plt.close()


def main():
    # ---------- Power trace ----------
    power_logs = gunzip_if_needed("gpgpusim_power_trace_report_*.log.gz")

    for log in power_logs:
        fix_power_header_if_needed(log)
        plot_csv_columns(
            log_file=log,
            out_dir="power_figure",
            title_prefix="Power Trace"
        )

    # ---------- Metric trace ----------
    metric_logs = gunzip_if_needed("gpgpusim_metric_trace_report_*.log.gz")

    for log in metric_logs:
        fix_metric_header_if_needed(log)
        plot_csv_columns(
            log_file=log,
            out_dir="metric_figure",
            title_prefix="Metric Trace"
        )


if __name__ == "__main__":
    main()
