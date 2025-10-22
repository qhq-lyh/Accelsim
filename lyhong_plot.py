import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

SEARCH_DIR = "80-1-1628MHz-result-2"
LOG_NAME = "lyhong_power_report.log"
PLOT_DIR = "plot_80-1-1628MHz-result-2"

SM_CORE_METRICS = {
    'RFP', 'INTP', 'FPUP', 'DPUP', 'INT_MUL24P', 'INT_MUL32P', 
    'INT_MULP', 'INT_DIVP', 'FP_MULP', 'FP_DIVP', 'FP_SQRTP',
    'FP_LGP', 'FP_SINP', 'FP_EXP', 'DP_MULP', 'DP_DIVP', 'TENSORP',
    'TEXP', 'SCHEDP'
}

os.makedirs(PLOT_DIR, exist_ok=True)

pattern = re.compile(
    r"gpu_(\w+)[,\s]*\s*[:]\s*([\d.eE+-]+)",  
    flags=re.UNICODE
)

# split benchmark name
def extract_folder_head(path): 
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))

def process_log_file(log_path): # every power plot
    benchmark_name = extract_folder_head(log_path)
    data = defaultdict(list)
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                key, value = match.groups()
                try:
                    data[key].append(float(value))
                except ValueError:
                    continue

    if not data:
        print(f"no data in :{log_path}")
        return

    plt.figure(figsize=(16, 10))
    for metric, values in data.items():
        plt.plot(values, label=metric)
    plt.title(f"{benchmark_name}_power_log")
    plt.xlabel("time_step")
    plt.ylabel("power")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.grid(True)
    plot_path = os.path.join(PLOT_DIR, f"{benchmark_name}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"generate png: {plot_path}")

def process_smcore_log(log_path):
    folder_head = extract_folder_head(log_path)
    data = defaultdict(list)
    sm_core_buffer = []
    with open(log_path, "r", encoding="utf-8") as f:
        current_sample = {
            'sm_total': 0.0,
            'sample_valid': False
        } 
        for line in f:
            line = line.strip()
            if line.startswith("Total_sample_count"):
                if current_sample['sample_valid']:
                    sm_core_buffer.append(current_sample['sm_total'])
                    current_sample = {'sm_total': 0.0, 'sample_valid': False}
                continue
            match = pattern.search(line)
            if match:
                metric, value = match.groups()
                try:
                    val = float(value)
                except ValueError:
                    continue
                data[metric].append(val)
                if metric in SM_CORE_METRICS:
                    current_sample['sm_total'] += val
                    current_sample['sample_valid'] = True
            if not line :
                if current_sample['sample_valid']:
                    sm_core_buffer.append(current_sample['sm_total'])
                    current_sample = {'sm_total': 0.0, 'sample_valid': False}
    if current_sample['sample_valid']:
        sm_core_buffer.append(current_sample['sm_total'])
    if sm_core_buffer:
        data['SM_CORE_TOTAL'] = sm_core_buffer
    filtered_data = {k: v for k, v in data.items() if k not in SM_CORE_METRICS}
    total_power_sum = [sum(values) for values in zip(*filtered_data.values())]
    plt.figure(figsize=(16, 10))
    colors = plt.cm.tab20.colors
    for idx, (metric, values) in enumerate(filtered_data.items()):
        if metric == 'SM_CORE_TOTAL':
            continue
        plt.plot(values, label=metric, 
                color=colors[idx % 20], alpha=0.8, linewidth=1.5)
    if 'SM_CORE_TOTAL' in filtered_data:
        plt.plot(filtered_data['SM_CORE_TOTAL'], 
                label='SM Core Total', 
                color='#FF4500', linewidth=2)
    plt.plot(total_power_sum, label='Total Power Sum', 
        color='#0000FF', linewidth=2)
    plt.title(f"{folder_head} power")
    plt.xlabel("timestep")
    plt.ylabel("power")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle=':', alpha=0.6)
    all_values = total_power_sum + [v for sublist in filtered_data.values() for v in sublist]
    ymin, ymax = min(all_values), max(all_values)
    plt.ylim(ymin * 0.95, 180) 
    plot_path = os.path.join(PLOT_DIR, f"{folder_head}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

def main():
    log_files = []
    for root,_,files in os.walk(SEARCH_DIR):
        if LOG_NAME in files:
            log_files.append(os.path.join(root, LOG_NAME))
    
    for log_file in log_files:
        # process_log_file(log_file)
        process_smcore_log(log_file)

if __name__ == "__main__":
    main()