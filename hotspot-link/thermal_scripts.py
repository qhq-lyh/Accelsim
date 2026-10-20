#!/usr/bin/env python3
# 作用：串联算法1分配 SM、算法2确定物理位置、首次 Accel-Sim 性能/功耗仿真、
# 算法3频率反算，以及降频后的实际仿真与 HotSpot grid 稳态温度验证。
# 输入：--run-dir 提供配置和混合 trace 模板；--gcb-dir 提供对应 G/B 矩阵。
# --allocate-first 开启完整流程，--benchmarks 按 CUDA stream 1、2、3…排序；
# 不加该开关则接入 --run-dir 中已经完成的首次仿真及其活动量、功耗输出。
# 输出：--out-dir 下的 placement/、initial_sim/、各轮 verification_XX/sim/、
# thermal/temp.grid.steady（K）、grid_heatmap.png（°C）、summary.json、iterations.csv。
# 保持放置和 DRAM 频率不变，Core/ICNT/L2 同步降频；实际超温则更新活动继续降频。
# 用法：python3 hotspot-link/thermal_scripts.py --allocate-first \
#         --benchmarks backprop,bfs,heartwall --initial-frequency-mhz 3396 \
#         --run-dir <配置与混合trace模板目录> --out-dir <新的输出目录> --threshold-c 90
from __future__ import annotations
import argparse
import csv
import json
import lzma
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'hotspot-link'
sys.path.insert(0,str(ROOT/'scripts'))
from Alg3_compute_safe_frequency import (DEFAULT_GCB, DEFAULT_MODEL, GridModel,
    PowerWindows, config_values, dump_json, floorplan, sha256, solve_frequency, write_ptrace)


def run_hotspot(args, names, power, directory, plot=True):
    directory.mkdir(parents=True,exist_ok=True)
    # The model-extract build writes matrices relative to cwd; keep these local.
    (directory/'model_extract').mkdir(exist_ok=True)
    ptrace=directory/'worst_window.ptrace'
    write_ptrace(ptrace,names,power)
    command=[str(args.hotspot_bin),'-c',str(args.thermal_config),'-p',str(ptrace),
             '-grid_layer_file',str(args.lcf),'-model_type','grid','-detailed_3D','on',
             '-steady_file',str(directory/'temp.steady'),
             '-grid_steady_file',str(directory/'temp.grid.steady')]
    with (directory/'hotspot.log').open('w') as log:
        subprocess.run(command,cwd=directory,stdout=log,stderr=subprocess.STDOUT,check=True)
    grid=np.loadtxt(directory/'temp.grid.steady',ndmin=2)
    if grid.shape[1]!=2 or not np.isfinite(grid).all():
        raise ValueError('Invalid HotSpot grid result')
    values=grid[:,1]-273.15
    if plot:
        env=os.environ.copy();env['MPLCONFIGDIR']=str(directory/'matplotlib')
        subprocess.run([sys.executable,str(BASE/'plot_gv100_grid_heatmap.py'),
            '--grid-steady',str(directory/'temp.grid.steady'),'--flp',str(args.flp),
            '--rows',str(args.rows),'--cols',str(args.cols),'--extrema-region','all',
            '--out',str(directory/'grid_heatmap.png')],env=env,check=True,
            stdout=subprocess.DEVNULL)
    return values


def snapshot_thermal_inputs(args):
    dest=args.out_dir/'thermal_inputs';dest.mkdir(parents=True,exist_ok=True)
    config=config_values(args.thermal_config)
    if config.get('leakage_used','0')!='0' or config.get('package_model_used','0')!='0':
        raise ValueError('Linear GCB requires leakage_used=0 and package_model_used=0')
    args.rows=int(config.get('grid_rows',64));args.cols=int(config.get('grid_cols',64))
    args.ambient_c=float(config['ambient'])-273.15
    shutil.copyfile(args.thermal_config,dest/'hotspot.config')
    args.thermal_config=(dest/'hotspot.config').resolve()
    lines=[s.split('#',1)[0].strip() for s in args.lcf.read_text().splitlines()]
    lines=[s for s in lines if s]
    if len(lines)%7:raise ValueError('Expected seven records per LCF layer')
    powered=[]
    for i in range(0,len(lines),7):
        original=Path(lines[i+6])
        if not original.is_absolute():original=(args.lcf.parent/original).resolve()
        if lines[i+2].upper()=='Y':
            powered.append(i)
            original=args.flp
        target=dest/f'layer_{lines[i]}.flp'
        shutil.copyfile(original,target)
        lines[i+6]=str(target.resolve())
    if len(powered)!=1 or powered[0]!=0:
        raise ValueError('Expected exactly one powered silicon layer, layer 0')
    args.flp=Path(lines[6])
    (dest/'layers.lcf').write_text('\n'.join(lines)+'\n')
    args.lcf=(dest/'layers.lcf').resolve()
    dump_json(dest/'hashes.json',{p.name:sha256(p) for p in dest.iterdir() if p.is_file() and p.name!='hashes.json'})


def validate_gcb(args, thermal, names):
    checks=[]
    # Different spatial probes check mapping as well as total thermal resistance.
    for label,power in [('zero',np.zeros(len(names))),
                        ('sm_corners',np.array([10. if n in ('SM_0','SM_79') else 0. for n in names])),
                        ('shared',np.array([10. if not n.startswith('SM_') else 0. for n in names]))]:
        actual=run_hotspot(args,names,power,args.out_dir/'gcb_checks'/label,plot=False)
        predicted=args.ambient_c+thermal.H@power
        if len(actual)>len(predicted) or len(actual)<args.rows*args.cols:
            raise ValueError('GCB/HotSpot grid node count mismatch')
        error=float(abs(actual-predicted[:len(actual)]).max())
        checks.append({'probe':label,'max_abs_error_c':error,'hotspot_grid_nodes':len(actual)})
        if error>args.gcb_tolerance_c:
            dump_json(args.out_dir/'gcb_checks.json',checks)
            raise ValueError(f'GCB does not match thermal inputs: {label} error {error:.6g} C. Re-extract matching matrices.')
    dump_json(args.out_dir/'gcb_checks.json',checks)


def analyze_thermal(args, thermal, windows, directory):
    directory.mkdir(parents=True,exist_ok=True)
    write_ptrace(directory/'active_windows.ptrace',windows.names,windows.reference_power)
    index,node,predicted=windows.worst(thermal,windows.reference_mhz,args.ambient_c)
    actual=run_hotspot(args,windows.names,windows.reference_power[index],directory)
    expected=args.ambient_c+thermal.H@windows.reference_power[index]
    error=float(abs(actual-expected[:len(actual)]).max())
    if error>args.gcb_tolerance_c:
        raise ValueError(f'Actual worst-window GCB/HotSpot mismatch {error:.6g} C')
    offset=np.zeros(len(expected));offset[:len(actual)]=actual-expected[:len(actual)]
    max_all=max(float(actual.max()),float(expected[len(actual):].max()) if len(actual)<len(expected) else -1e100)
    result={'frequency_mhz':windows.reference_mhz,'worst_sample':int(windows.samples[index]),
            'predicted_max_all_nodes_c':predicted,'hotspot_max_grid_c':float(actual.max()),
            'max_including_package_c':max_all,'grid_nodes':len(actual),
            'max_gcb_hotspot_error_c':error,'simultaneous_worst_window_total_w':float(windows.reference_power[index].sum()),
            'power_audit':windows.audit}
    dump_json(directory/'thermal_summary.json',result)
    return result,offset


def prepare_simulator(args):
    directory=args.out_dir/'simulator';directory.mkdir(exist_ok=True)
    if not args.library.is_file() or not args.accel_sim.is_file():
        raise ValueError('Missing simulator/library; build the intended project first')
    for source,name in [(args.accel_sim,'accel-sim.out'),(args.library,'libcudart.so')]:
        shutil.copyfile(source,directory/name)
        if name=='accel-sim.out':(directory/name).chmod(0o755)
    env=os.environ.copy();env['LD_LIBRARY_PATH']=str(directory)
    env['ACCELWATTCH_SM_ACTIVITY']='1'
    env.pop('ACCELWATTCH_AGGREGATE_REFERENCE',None)
    check=subprocess.run(['ldd','-r',str(directory/'accel-sim.out')],env=env,
                         text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (directory/'ldd.txt').write_text(check.stdout)
    if check.returncode or re.search(r'undefined symbol:|not found',check.stdout):
        raise ValueError('Simulator and library do not match; see simulator/ldd.txt')
    dump_json(directory/'hashes.json',{n:sha256(directory/n) for n in ('accel-sim.out','libcudart.so')})
    return directory/'accel-sim.out',env


def run_accel(args, executable, env, frequency, directory):
    directory.mkdir(parents=True,exist_ok=False)
    for item in args.run_dir.iterdir():
        if item.is_file() and (item.suffix in ('.xml','.icnt') or item.name=='stream_sm_alloc.txt'):
            shutil.copyfile(item,directory/item.name)
    traces=(args.run_dir/'traces').resolve(strict=True)
    (directory/'traces').symlink_to(traces,target_is_directory=True)
    text=(args.run_dir/'gpgpusim.config').read_text()
    def replace_clock(match):
        clocks=match.group(1).split(':')
        return '-gpgpu_clock_domains '+':'.join([f'{frequency:.12g}']*3+[clocks[3]])
    text,count=re.subn(r'^\s*-gpgpu_clock_domains\s+([^\s#]+)',replace_clock,text,flags=re.M)
    if count!=1:raise ValueError('Expected exactly one active gpgpu_clock_domains setting')
    text=re.sub(r'(^\s*-gpgpu_concurrent_stream_sm_config_file\s+)\S+',
                r'\g<1>./stream_sm_alloc.txt',text,flags=re.M)
    (directory/'gpgpusim.config').write_text(text)
    command=[str(executable),'-config','./gpgpusim.config','-trace','./traces/kernelslist.g']
    print(f'Running fresh Accel-Sim at {frequency:g} MHz -> {directory}',flush=True)
    started=time.monotonic()
    with (directory/'simulation.o').open('w') as out,(directory/'simulation.e').open('w') as err:
        result=subprocess.run(command,cwd=directory,env=env,stdout=out,stderr=err)
    log=(directory/'simulation.o').read_text()
    if result.returncode or '*** exit detected ***' not in log:
        raise RuntimeError(f'Accel-Sim failed/incomplete ({result.returncode}); inspect {directory}/simulation.e')
    def last(pattern):
        values=re.findall(pattern,log)
        return float(values[-1]) if values else None
    cycles=last(r'gpu_tot_sim_cycle\s*=\s*(\d+)')
    instructions=last(r'gpu_tot_sim_insn\s*=\s*(\d+)')
    if cycles is None or instructions is None:raise ValueError('Missing completed performance statistics')
    perf={'frequency_mhz':frequency,'total_cycles':int(cycles),'total_instructions':int(instructions),
          'ipc':instructions/cycles,'simulated_core_time_s':cycles/(frequency*1e6),
          'host_elapsed_s':time.monotonic()-started,'exit_code':result.returncode}
    dump_json(directory/'performance.json',perf)
    return perf


def initial_performance(run_dir, frequency):
    logs=[p for p in run_dir.iterdir() if p.is_file() and
          (p.name=='simulation.o' or re.search(r'\.o\d+$',p.name))]
    logs.sort(key=lambda p:p.stat().st_mtime,reverse=True)
    if not logs:raise ValueError('First run needs a completed .o job log or simulation.o')
    path=logs[0];text=path.read_text()
    if '*** exit detected ***' not in text:
        raise ValueError(f'Latest first-run job did not complete: {path}')
    def last(pattern):
        values=re.findall(pattern,text)
        if not values:raise ValueError(f'Missing first-run performance field: {path}')
        return int(values[-1])
    cycles=last(r'gpu_tot_sim_cycle\s*=\s*(\d+)')
    instructions=last(r'gpu_tot_sim_insn\s*=\s*(\d+)')
    return {'frequency_mhz':frequency,'total_cycles':cycles,'total_instructions':instructions,
            'ipc':instructions/cycles,'simulated_core_time_s':cycles/(frequency*1e6),
            'source_log':str(path)}


def allocation_first_run(args, executable, env):
    """Run Alg1/Alg2 in isolated files, then actually run their placement."""
    benchmarks=[n.strip() for n in (args.benchmarks or '').split(',') if n.strip()]
    if len(benchmarks)<2 or len(set(benchmarks))!=len(benchmarks):
        raise ValueError('--allocate-first requires >=2 distinct --benchmarks in CUDA stream 1,2,... order')
    traces=(args.run_dir/'traces').resolve(strict=True)
    listed=[s.strip() for s in (traces/'kernelslist.g').read_text().splitlines()
            if s.strip().endswith(('.traceg','.traceg.xz'))]
    # The mixed-trace generator preserves benchmark names in kernel filenames.
    # Verify the user's ordering against actual CUDA stream IDs before placing.
    matched=set()
    for stream,benchmark in enumerate(benchmarks,1):
        files=[name for name in listed if Path(name).name.startswith(f'kernel-{benchmark}-')]
        if not files:raise ValueError(f'No named trace kernels for {benchmark}; use the mixed-trace generator first')
        for name in files:
            path=traces/name
            opener=lzma.open if path.suffix=='.xz' else open
            found=None
            with opener(path,'rt') as file:
                for line in file:
                    m=re.match(r'-cuda stream id\s*=\s*(\d+)',line)
                    if m:found=int(m.group(1));break
                    if line.startswith('#'):break
            if found!=stream:
                raise ValueError(f'Benchmark order/stream mismatch: {name} is stream {found}, expected {stream}')
            matched.add(name)
    if matched!=set(listed):raise ValueError('Trace contains kernels not included in --benchmarks')
    placement=args.out_dir/'placement';placement.mkdir()
    commands=[
        [sys.executable,str(ROOT/'scripts/Alg1_compute_sm_allocation.py'),
         '--ipc-csv',str(args.ipc_csv),'--config',str(args.run_dir/'gpgpusim.config'),
         '--benchmarks',','.join(benchmarks),'--alpha',str(args.allocation_alpha),
         '--out-csv',str(placement/'sm_allocation.csv')],
        [sys.executable,str(ROOT/'scripts/Alg2_compute_sm_placement.py'),
         '--allocation-csv',str(placement/'sm_allocation.csv'),'--flp',str(args.flp),
         '--out-dir',str(placement)],
    ]
    for index,command in enumerate(commands,1):
        with (placement/f'algorithm{index}.log').open('w') as log:
            plot_env=env.copy();plot_env['MPLCONFIGDIR']=str(placement/'matplotlib')
            subprocess.run(command,cwd=ROOT,env=plot_env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (placement/'sm_map.csv').open() as file:
        sm_map={row['benchmark']:[int(n) for n in row['sm_ids'].replace(',',' ').split()]
                for row in csv.DictReader(file)}
    ids=[sm for bench in benchmarks for sm in sm_map[bench]]
    expected=[int(b[0][3:]) for b in floorplan(args.flp) if b[0].startswith('SM_')]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):
        raise ValueError('Alg2 placement must cover all SMs exactly once')
    prepared=args.out_dir/'prepared_first_input';prepared.mkdir()
    for item in args.run_dir.iterdir():
        if item.is_file() and (item.suffix in ('.xml','.icnt') or item.name=='gpgpusim.config'):
            shutil.copyfile(item,prepared/item.name)
    (prepared/'traces').symlink_to(traces,target_is_directory=True)
    allocation=','.join(f'{i}:{"|".join(map(str,sm_map[name]))}' for i,name in enumerate(benchmarks,1))
    (prepared/'stream_sm_alloc.txt').write_text(allocation+'\n')
    config=(prepared/'gpgpusim.config').read_text()
    for key in ('gpgpu_concurrent_kernel_sm','gpgpu_concurrent_stream_sm_config_file',
                'gpgpu_concurrent_stream_sm_count','gpgpu_concurrent_stream_sm_reload_interval'):
        config=re.sub(r'^\s*-'+key+r'\s+[^\n]*','',config,flags=re.M)
    config+='\n-gpgpu_concurrent_kernel_sm 1\n-gpgpu_concurrent_stream_sm_config_file ./stream_sm_alloc.txt\n-gpgpu_concurrent_stream_sm_reload_interval 1\n'
    (prepared/'gpgpusim.config').write_text(config)
    args.run_dir=prepared
    frequency=args.initial_frequency_mhz
    if frequency is None:
        frequency=float(config_values(prepared/'gpgpusim.config')['gpgpu_clock_domains'].split(':')[0])
    if not np.isfinite(frequency) or frequency<=0:raise ValueError('Initial frequency must be positive and finite')
    perf=run_accel(args,executable,env,frequency,args.out_dir/'initial_sim')
    args.run_dir=args.out_dir/'initial_sim'
    dump_json(placement/'workflow.json',{'benchmarks_in_stream_order':benchmarks,
        'stream_sm_allocation':allocation,'initial_performance':perf,
        'ipc_lookup_sha256':sha256(args.ipc_csv)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True,help='Completed first run, or input template when --allocate-first is set')
    p.add_argument('--allocate-first',action='store_true',help='Run Alg1 -> Alg2 -> first simulation before Alg3')
    p.add_argument('--benchmarks',help='With --allocate-first: comma-separated benchmark names in CUDA stream 1,2,... order')
    p.add_argument('--ipc-csv',type=Path,default=ROOT/'IPC-sm/ipc_lookup.csv')
    p.add_argument('--allocation-alpha',type=float,default=.5)
    p.add_argument('--initial-frequency-mhz',type=float,help='With --allocate-first: initial Core/ICNT/L2 MHz, DRAM unchanged')
    p.add_argument('--out-dir',type=Path,required=True,help='New, empty output directory')
    p.add_argument('--gcb-dir',type=Path,default=DEFAULT_GCB)
    p.add_argument('--activity-model',type=Path,default=DEFAULT_MODEL)
    p.add_argument('--ptrace',type=Path,help='Optional first-run Ptrace, same complete windows as activity export')
    p.add_argument('--flp',type=Path,default=BASE/'flp/gv100_new.flp')
    p.add_argument('--lcf',type=Path,default=BASE/'flp/gv100.lcf')
    p.add_argument('--thermal-config',type=Path,default=BASE/'GV100-hotspot.config')
    p.add_argument('--hotspot-bin',type=Path,default=BASE/'hotspot')
    p.add_argument('--accel-sim',type=Path,default=ROOT/'gpu-simulator/bin/release/accel-sim.out')
    p.add_argument('--library',type=Path,default=ROOT/'gpu-simulator/gpgpu-sim/lib/gcc-13.3.0/cuda-12080/release/libcudart.so')
    p.add_argument('--threshold-c',type=float,default=90.)
    p.add_argument('--frequency-step-mhz',type=float,default=1.)
    p.add_argument('--minimum-frequency-mhz',type=float,default=100.)
    p.add_argument('--safety-margin-c',type=float,default=.1)
    p.add_argument('--gcb-tolerance-c',type=float,default=.03)
    p.add_argument('--max-verification-runs',type=int,default=5)
    args=p.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.expanduser().resolve())
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        p.error('--out-dir must be new or empty; preserve prior results')
    if args.max_verification_runs<1:p.error('--max-verification-runs must be positive')
    if not args.allocate_first and (args.benchmarks or args.initial_frequency_mhz is not None):
        p.error('--benchmarks/--initial-frequency-mhz require --allocate-first')
    if args.allocate_first and args.ptrace:p.error('--ptrace cannot override a newly generated first run')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    summary={'status':'running','threshold_c':args.threshold_c,'iterations':[],
             'scope':'Highest frequency under frozen activities; the final safe result is verified, not a proof of the global maximum over changing activities.'}
    try:
        snapshot_thermal_inputs(args)
        blocks=floorplan(args.flp)
        thermal=GridModel(args.gcb_dir,len(blocks))
        dump_json(args.out_dir/'gcb_model.json',thermal.metadata)
        validate_gcb(args,thermal,[b[0] for b in blocks])
        executable,env=prepare_simulator(args)
        if args.allocate_first:
            allocation_first_run(args,executable,env)
            summary['placement_file']=str(args.out_dir/'placement/sm_map.csv')
        summary['first_run_dir']=str(args.run_dir)
        windows=PowerWindows(args.run_dir,blocks,args.activity_model,args.ptrace)
        summary['initial_performance']=initial_performance(args.run_dir,windows.reference_mhz)
        dump_json(args.out_dir/'initial_performance.json',summary['initial_performance'])
        initial,offset=analyze_thermal(args,thermal,windows,args.out_dir/'initial_thermal')
        summary['initial']=initial
        upper=windows.reference_mhz
        summary['initial_frequency_mhz']=upper
        for iteration in range(1,args.max_verification_runs+1):
            plan=solve_frequency(thermal,windows,args.ambient_c,args.threshold_c,
                    args.frequency_step_mhz,args.minimum_frequency_mhz,upper,
                    args.safety_margin_c,offset)
            frequency=plan['frequency_mhz']
            base=args.out_dir/f'verification_{iteration:02d}'
            base.mkdir()
            dump_json(base/'frequency_plan.json',plan)
            index,_,_=windows.worst(thermal,frequency,args.ambient_c)
            write_ptrace(base/'predicted_worst.ptrace',windows.names,windows.power(frequency)[index])
            perf=run_accel(args,executable,env,frequency,base/'sim')
            actual_windows=PowerWindows(base/'sim',blocks,args.activity_model)
            actual,actual_offset=analyze_thermal(args,thermal,actual_windows,base/'thermal')
            success=max(actual['max_including_package_c'],actual['predicted_max_all_nodes_c'])<=args.threshold_c-args.safety_margin_c
            summary['iterations'].append({'iteration':iteration,'plan':plan,'performance':perf,
                                          'actual_thermal':actual,'verified_safe':success})
            dump_json(args.out_dir/'summary.json',summary)
            print(f'Verification {iteration}: {frequency:g} MHz, actual max {actual["max_including_package_c"]:.3f} C, safe={success}',flush=True)
            if success:
                summary.update(status='verified_safe',final_frequency_mhz=frequency,
                    final_grid_file=str(base/'thermal/temp.grid.steady'),
                    final_heatmap=str(base/'thermal/grid_heatmap.png'),
                    final_performance=str(base/'sim/performance.json'),
                    final_ptrace=str(base/'thermal/worst_window.ptrace'))
                break
            upper=frequency-args.frequency_step_mhz
            windows,offset=actual_windows,actual_offset
        else:
            raise RuntimeError('Verification budget exhausted; no verified-safe result')
    except Exception as exc:
        summary.update(status='failed',error=str(exc))
        dump_json(args.out_dir/'summary.json',summary)
        raise
    dump_json(args.out_dir/'summary.json',summary)
    with (args.out_dir/'iterations.csv').open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['iteration','frequency_mhz','cycles','instructions','ipc','core_time_s','max_grid_c','max_all_nodes_c','verified_safe'])
        perf=summary['initial_performance'];t=summary['initial']
        writer.writerow([0,perf['frequency_mhz'],perf['total_cycles'],perf['total_instructions'],perf['ipc'],perf['simulated_core_time_s'],t['hotspot_max_grid_c'],t['max_including_package_c'],t['max_including_package_c']<=args.threshold_c-args.safety_margin_c])
        for row in summary['iterations']:
            perf=row['performance'];t=row['actual_thermal']
            writer.writerow([row['iteration'],perf['frequency_mhz'],perf['total_cycles'],perf['total_instructions'],perf['ipc'],perf['simulated_core_time_s'],t['hotspot_max_grid_c'],t['max_including_package_c'],row['verified_safe']])
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
