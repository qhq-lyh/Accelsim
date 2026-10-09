#!/usr/bin/env python3
"""Sweep single benchmarks and fit per-SM sampled peaks from lyhong_power_report.log."""
import argparse, concurrent.futures, csv, hashlib, json, os, re, shutil, subprocess
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def peaks(path):
    with path.open() as handle:
        names = handle.readline().split()
        data = np.loadtxt(handle, ndmin=2)
    columns = [names.index(f'SM_{i}') for i in range(80)]
    if data.shape[1] != len(names) or not np.isfinite(data).all():
        raise ValueError(f'Invalid power data: {path}')
    sm = data[:, columns]
    return sm.max(axis=0), sm.argmax(axis=0), len(sm)

def run(root, name, trace, frequency):
    directory = root/name/f'{frequency:g}MHz'
    if not (directory/'complete.json').exists():
        directory.mkdir(parents=True, exist_ok=True)
        for p in (root/'inputs').iterdir():
            shutil.copy2(p, directory/p.name)
        config = (directory/'gpgpusim.config').read_text()
        config, count = re.subn(r'(?m)^-gpgpu_clock_domains\s+[^\n]+$', f'-gpgpu_clock_domains {frequency}:{frequency}:{frequency}:850.0', config)
        if count != 1: raise ValueError('Expected one clock line')
        (directory/'gpgpusim.config').write_text(config)
        env = os.environ.copy()
        libs = sorted((ROOT/'gpu-simulator/gpgpu-sim/lib').glob('*/cuda-*/release/libcudart.so'))
        if len(libs) != 1: raise ValueError('Cannot identify simulator libcudart')
        env['LD_LIBRARY_PATH'] = str(libs[0].parent)+':/home/lyhong/cuda-12.8/lib64:'+env.get('LD_LIBRARY_PATH','')
        command = [str(ROOT/'gpu-simulator/bin/release/accel-sim.out'), '-config','./gpgpusim.config','-config','./trace.config','-config','./power.config','-trace',str(trace/'kernelslist.g')]
        print(f'Start {name} {frequency:g} MHz', flush=True)
        with (directory/'sim.log').open('w') as handle:
            subprocess.run(command,cwd=directory,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
        log = (directory/'sim.log').read_text(errors='replace')
        if '*** exit detected ***' not in log: raise ValueError(f'Abnormal exit: {directory}')
        cycles = int(re.findall(r'gpu_tot_sim_cycle\s*=\s*(\d+)',log)[-1])
        instructions = int(re.findall(r'gpu_tot_sim_insn\s*=\s*(\d+)',log)[-1])
        maximum, indices, samples = peaks(directory/'lyhong_power_report.log')
        (directory/'complete.json').write_text(json.dumps(dict(cycles=cycles,instructions=instructions,samples=samples,frequency_mhz=frequency,benchmark=name),indent=2)+'\n')
        print(f'Done {name} {frequency:g} MHz: {samples} windows, max SM {maximum.max():.3f} W',flush=True)
    metadata = json.loads((directory/'complete.json').read_text())
    maximum, indices, samples = peaks(directory/'lyhong_power_report.log')
    return [dict(benchmark=name,frequency_mhz=frequency,sm_id=i,peak_power_w=float(maximum[i]),peak_sample_index_0based=int(indices[i]),sample_count=samples,cycles=metadata['cycles'],instructions=metadata['instructions'],source=str(directory/'lyhong_power_report.log')) for i in range(80)]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--frequencies',default='800,1132,1400,1698,2200,2800,3396,4000')
    parser.add_argument('--benchmarks',default='all',help='all available single benchmarks or comma-separated names')
    parser.add_argument('--jobs',type=int,default=3)
    args=parser.parse_args()
    frequencies=sorted(map(float,args.frequencies.split(',')))
    if len(set(frequencies))!=len(frequencies) or any(not 0<f<=4294 for f in frequencies) or args.jobs<1: parser.error('Invalid frequency/jobs')
    traces={p.parents[1].name.removesuffix('-rodinia-2.0-ft'):p for p in sorted((ROOT/'hw_run/rodinia_2.0-ft/9.1').glob('*/*/traces')) if '-' not in p.parents[1].name.removesuffix('-rodinia-2.0-ft')}
    if args.benchmarks!='all': traces={name:traces[name] for name in args.benchmarks.split(',')}
    root=args.out_dir.resolve(); inputs=root/'inputs'
    if not inputs.exists():
        inputs.mkdir(parents=True)
        shutil.copy2(BASE/'gpgpusim.config',inputs)
        shutil.copy2(ROOT/'gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config',inputs)
        for pattern in ['*.xml','*.icnt']:
            for p in BASE.glob(pattern): shutil.copy2(p,inputs)
        (inputs/'power.config').write_text('''-gpgpu_n_clusters 80
-gpgpu_n_cores_per_cluster 1
-gpgpu_concurrent_kernel_sm 0
-gpgpu_concurrent_stream_sm_debug 0
-power_simulation_enabled 1
-power_simulation_mode 0
-accelwattch_xml_file ./accelwattch_sass_sim.xml
-dvfs_enabled 0
-gpgpu_runtime_stat 500:0
-power_per_cycle_dump 0
''')
        manifest=dict(sampling_cycles=500,dram_mhz=850,voltage='fixed; dvfs_enabled=0',sm_allocation='single benchmark eligible on all 80 SMs; no partition',binary_sha256=digest(ROOT/'gpu-simulator/bin/release/accel-sim.out'),input_sha256={p.name:digest(p) for p in inputs.iterdir()},traces={name:dict(path=str(trace),sha256={p.name:digest(p) for p in trace.iterdir() if p.is_file()}) for name,trace in traces.items()})
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    tasks=[(name,trace,f) for name,trace in traces.items() for f in frequencies]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results=list(pool.map(lambda task:run(root,*task),tasks))
    rows=[row for result in results for row in result]
    with (root/'sm_peak_samples.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'Wrote {len(rows)} per-SM peaks from {len(tasks)} runs',flush=True)

if __name__=='__main__': main()
