#!/usr/bin/env python3
"""Controlled Core=ICNT=L2 frequency sweep; report sampled GPU and mapped power."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100'
TRACE = ROOT / 'hw_run/rodinia_2.0-ft/9.1/backprop-bfs-heartwall-rodinia-2.0-ft/backprop_bfs_heartwall/traces'
OLD = ROOT / 'sim_run_12.8-3.3G-compare/backprop-bfs-heartwall-rodinia-2.0-ft/backprop_bfs_heartwall/QV100-Lyhong_SIM'


def summarize(directory, frequency):
    log = (directory / 'sim.log').read_text(errors='ignore')
    if '*** exit detected ***' not in log:
        raise ValueError(f'{directory}: missing normal exit')
    def last(key):
        return float(re.findall(r'\b' + key + r'\s*=\s*([0-9.eE+-]+)', log)[-1])
    with (directory / 'lyhong_power_report.log').open() as file:
        names = file.readline().split()
        values = [[float(x) for x in line.split()] for line in file if line.strip()]
    if not values or any(len(row) != len(names) for row in values):
        raise ValueError('invalid mapped power samples')
    records, current = [], {}
    pattern = re.compile(r'^([A-Za-z0-9_]+):\s*([-+0-9.eE]+)\s*$')
    for line in (directory / 'lyhong_SM_power_report.log').read_text().splitlines():
        match = pattern.match(line)
        if not match:
            continue
        key, value = match.groups()
        current[key] = float(value)
        if key == 'Total_GPU_Power':
            records.append(current)
            current = {}
    if len(records) != len(values):
        raise ValueError(f'component samples {len(records)} != mapped samples {len(values)}')
    means = {key: sum(record[key] for record in records) / len(records) for key in records[0]}
    means['mapped_power_w'] = sum(map(sum, values)) / len(values)
    result = dict(frequency_mhz=frequency, samples=len(records), cycles=int(last('gpu_tot_sim_cycle')),
                  instructions=int(last('gpu_tot_sim_insn')), ipc=last('gpu_tot_ipc'),
                  simulated_time_us=last('gpu_tot_sim_cycle') / frequency,
                  unsampled_tail_cycles=int(last('gpu_tot_sim_cycle')) % 500, **means)
    result['max_total_component_sum_error_w'] = max(abs(r['Total_GPU_Power'] - r['Total_SM_Power'] - r['Total_Non_SM_Power']) for r in records)
    (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with (directory / 'block_average_power.csv').open('w') as file:
        writer = csv.writer(file)
        writer.writerow(['block','average_power_w'])
        writer.writerows((name, sum(row[i] for row in values) / len(values)) for i,name in enumerate(names))
    return result


def run(root, frequency):
    directory = root / f'{frequency:g}MHz'
    if (directory / 'summary.json').exists():
        return json.loads((directory / 'summary.json').read_text())
    directory.mkdir(exist_ok=True)
    for path in root.joinpath('inputs').iterdir():
        shutil.copy2(path, directory / path.name)
    config = (directory / 'gpgpusim.config').read_text()
    config, count = re.subn(r'(?m)^-gpgpu_clock_domains\s+[^\n]+$',
                            f'-gpgpu_clock_domains {frequency}:{frequency}:{frequency}:850.0', config)
    if count != 1:
        raise ValueError('expected one clock line')
    # Change only the first three clocks in the per-run copy. Shared base stays untouched.
    (directory / 'gpgpusim.config').write_text(config)
    command = [str(ROOT / 'gpu-simulator/bin/release/accel-sim.out'),
               '-config', './gpgpusim.config', '-config', './trace.config',
               '-config', './power.config', '-trace', str(TRACE / 'kernelslist.g')]
    print(f'Starting {frequency:g} MHz', flush=True)
    environment = os.environ.copy()
    libraries = sorted((ROOT / 'gpu-simulator/gpgpu-sim/lib').glob('*/cuda-*/release/libcudart.so'))
    if len(libraries) != 1:
        raise ValueError('source setup_environment first or resolve the simulator library explicitly')
    environment['LD_LIBRARY_PATH'] = str(libraries[0].parent) + ':/home/lyhong/cuda-12.8/lib64:' + environment.get('LD_LIBRARY_PATH', '')
    with (directory / 'sim.log').open('w') as file:
        subprocess.run(command, cwd=directory, stdout=file, stderr=subprocess.STDOUT, check=True, env=environment)
    result = summarize(directory, frequency)
    print(f'Finished {frequency:g} MHz: GPU={result["Total_GPU_Power"]:.6f} W, mapped={result["mapped_power_w"]:.6f} W, cycles={result["cycles"]}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frequencies', default='800,1132,1400,1698,2200,2800,3396,4000')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    frequencies = list(map(float, args.frequencies.split(',')))
    if any(not 0 < f <= 4294 for f in frequencies) or len(set(frequencies)) != len(frequencies) or args.jobs < 1:
        parser.error('distinct positive frequencies <=4294 MHz and positive jobs required')
    root = args.out_dir.resolve()
    if not root.exists():
        inputs = root / 'inputs'
        inputs.mkdir(parents=True)
        shutil.copy2(BASE / 'gpgpusim.config', inputs)
        shutil.copy2(ROOT / 'gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config', inputs)
        for pattern in ['*.xml', '*.icnt']:
            for path in BASE.glob(pattern):
                shutil.copy2(path, inputs)
        shutil.copy2(OLD / 'stream_sm_alloc.txt', inputs)
        (inputs / 'power.config').write_text('''-gpgpu_n_clusters 80
-gpgpu_n_cores_per_cluster 1
-gpgpu_concurrent_kernel_sm 1
-gpgpu_concurrent_stream_sm_config_file ./stream_sm_alloc.txt
-gpgpu_concurrent_stream_sm_reload_interval 100
-gpgpu_concurrent_stream_sm_debug 0
-power_simulation_enabled 1
-power_simulation_mode 0
-accelwattch_xml_file ./accelwattch_sass_sim.xml
-dvfs_enabled 0
-gpgpu_runtime_stat 500:0
-power_per_cycle_dump 0
''')
        manifest = dict(trace=str(TRACE), dram_mhz=850, sampling_cycles=500,
                        note='three clocks equal; all other run files fixed; historical physical SM mapping',
                        binary_sha256=hashlib.sha256((ROOT/'gpu-simulator/bin/release/accel-sim.out').read_bytes()).hexdigest(),
                        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()},
                        trace_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in TRACE.iterdir() if p.is_file()})
        (root / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda f: run(root, f), frequencies))
    results.sort(key=lambda row: row['frequency_mhz'])
    with (root / 'sweep.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)


if __name__ == '__main__':
    main()
