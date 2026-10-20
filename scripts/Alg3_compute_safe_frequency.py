#!/usr/bin/env python3
# 作用：算法3，根据首次仿真的活动量、功耗和 grid 稳态温度，结合 GCB
# 矩阵反算所有热节点不超过阈值（默认 90°C）的最高候选频率，向下取整。
#
# 输入：
#   1. --run-dir：首次运行目录，包含 gpgpusim.config、lyhong_power_report.log、
#      power_components.csv、sm_activity.csv.gz（或解压后的 CSV）。
#   2. --grid-steady：最热同时窗口对应的 HotSpot grid 稳态温度文件，单位 K。
#   3. --gcb-dir：G/B 导出矩阵目录；--flp、--thermal-config：对应布局和热配置。
#   4. --activity-model：已标定的活动功耗系数；--threshold-c：温度阈值，单位 °C。
#
# 方法：删除全 SM 零活动窗口，保留每个窗口完整的同时功耗分布；补全全局
# 功耗并核对守恒。冻结参考活动量，建立 P_w(f)=k_w*f_GHz+b_w，利用
# T_w=Tamb+G^(-1)*B*P_w 对全部窗口/节点求频率上限。C 在稳态不参与求解。
# 默认按 1 MHz 向下取整，预留 0.1°C 裕量，不高于参考频率。
# 静态/常量功耗按 SM 面积分摊，MCP/NoC 映射到 L2，DRAM 分给两侧 HBM。
# 候选频率仍需重新仿真验证；不能由冻结活动模型证明真实全局最高频率。
#
# 输出：--out-dir/frequency.json（候选频率、限制节点及窗口等）、
#       predicted_worst.ptrace（候选频率下预测的最热同时窗口功耗）。
# 用法：python3 scripts/Alg3_compute_safe_frequency.py --run-dir <首次运行目录> \
#         --grid-steady <最热窗口.grid.steady> --out-dir <输出目录>
# 完整算法1/2/3、性能功耗及 grid 热仿真流程由 hotspot-link/thermal_scripts.py 调用。
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GCB = ROOT/'hotspot-link/model_extract'
DEFAULT_MODEL = ROOT/'scripts/models/qv100_activity_power.json'
from sm_activity_power import FEATURES, COUNTERS, activity_rates


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def config_values(path):
    result = {}
    for raw in Path(path).read_text().splitlines():
        fields = raw.split('#', 1)[0].split()
        if len(fields) >= 2 and fields[0].startswith('-'):
            result[fields[0].lstrip('-')] = fields[1]
    return result


def floorplan(path):
    blocks = []
    for raw in Path(path).read_text().splitlines():
        fields = raw.split('#', 1)[0].split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f'Expected name/width/height/x/y: {path}: {raw}')
        blocks.append((fields[0], *map(float, fields[1:])))
    if not blocks or len({b[0] for b in blocks}) != len(blocks):
        raise ValueError('Empty floorplan or duplicate names')
    return blocks


def write_ptrace(path, names, values):
    values = np.atleast_2d(values)
    if not np.isfinite(values).all() or (values < -1e-9).any():
        raise ValueError('Ptrace contains nonfinite/negative power')
    np.savetxt(path, np.maximum(values, 0), delimiter='\t', fmt='%.12f',
               header='\t'.join(names), comments='')


class GridModel:
    def __init__(self, directory, n_blocks):
        directory = Path(directory)
        raw = ['Amatrixcolptr', 'Amatrixrowind', 'Amatrixnzval', 'Bmatrix']
        if all((directory/name).is_file() for name in raw):
            ptr = np.loadtxt(directory/raw[0], dtype=np.int64)
            row = np.loadtxt(directory/raw[1], dtype=np.int64)
            val = np.loadtxt(directory/raw[2])
            n = len(ptr)-1
            if ptr[0] != 0 or ptr[-1] != len(val) or len(row) != len(val):
                raise ValueError('Invalid G CSC indices')
            self.G = csc_matrix((val, row, ptr), shape=(n,n))
            triplets = np.loadtxt(directory/'Bmatrix')
            shape = tuple(triplets[-1,:2].astype(int))
            self.B = csc_matrix((triplets[:-1,2], (triplets[:-1,0].astype(int),
                                  triplets[:-1,1].astype(int))), shape=shape)
            sources = raw
        else:
            self.G = csc_matrix(loadmat(directory/'G.mat')['G'])
            self.B = csc_matrix(loadmat(directory/'B.mat')['B'])
            sources = ['G.mat','B.mat']
        if self.G.shape[0] != self.G.shape[1] or self.B.shape != (self.G.shape[0], n_blocks):
            raise ValueError(f'G/B dimensions disagree with floorplan: {self.G.shape}, {self.B.shape}, {n_blocks}')
        self.H = splu(self.G).solve(self.B.toarray())
        if not np.isfinite(self.H).all() or self.H.min() < -1e-8:
            raise ValueError('Thermal response is not finite/nonnegative')
        self.H = np.maximum(self.H, 0)
        self.metadata = {'G_shape':list(self.G.shape), 'B_shape':list(self.B.shape),
                         'files':{name:sha256(directory/name) for name in sources},
                         'B_column_sum_max_error':float(abs(np.asarray(self.B.sum(axis=0))-1).max()),
                         'C_usage':'C*dT/dt is zero at steady state; C is intentionally unused'}


class PowerWindows:
    def __init__(self, run_dir, flp, model_path=DEFAULT_MODEL, ptrace=None):
        self.run_dir = Path(run_dir)
        self.names = [b[0] for b in flp]
        self.reference_mhz = float(config_values(self.run_dir/'gpgpusim.config')['gpgpu_clock_domains'].split(':')[0])
        clocks = list(map(float, config_values(self.run_dir/'gpgpusim.config')['gpgpu_clock_domains'].split(':')))
        if not np.allclose(clocks[:3], clocks[0]) or len(clocks) != 4:
            raise ValueError('Core/ICNT/L2 frequencies must be equal, with a fixed DRAM clock')
        self.dram_mhz = clocks[3]
        cfg = config_values(self.run_dir/'gpgpusim.config')
        if cfg.get('dvfs_enabled','0') != '0':
            raise ValueError('Fixed-voltage power model does not support enabled DVFS')
        ptrace = Path(ptrace) if ptrace else self.run_dir/'lyhong_power_report.log'
        with ptrace.open() as file:
            headers = file.readline().split()
            measured = np.loadtxt(file, ndmin=2)
        if set(headers) != set(self.names) or measured.shape[1] != len(headers):
            raise ValueError('Ptrace columns must exactly match powered floorplan block names')
        measured = measured[:,[headers.index(n) for n in self.names]]
        if not np.isfinite(measured).all() or (measured < -1e-9).any():
            raise ValueError('Ptrace power must be finite and nonnegative')
        activity_path = self.run_dir/'sm_activity.csv.gz'
        if not activity_path.exists():
            activity_path = self.run_dir/'sm_activity.csv'
        activity = pd.read_csv(activity_path) # Compressed input also checks gzip trailer.
        model = json.loads(Path(model_path).read_text())
        if model['features'] != FEATURES:
            raise ValueError('Activity model feature schema mismatch')
        sm_ids = sorted(int(n[3:]) for n in self.names if n.startswith('SM_'))
        if sm_ids != list(range(len(sm_ids))):
            raise ValueError('Expected contiguous physical SM IDs')
        n_sm = len(sm_ids)
        expected_rows = len(measured)*n_sm
        if len(activity) != expected_rows or activity.duplicated(['sample','sm_id']).any():
            raise ValueError('Activity/Ptrace window count mismatch or duplicate rows')
        activity = activity.sort_values(['sample','sm_id']).reset_index(drop=True)
        if not np.isfinite(activity.select_dtypes(include=[np.number]).to_numpy()).all():
            raise ValueError('Nonfinite activity export')
        if not np.array_equal(activity.sm_id.to_numpy(), np.tile(sm_ids, len(measured))):
            raise ValueError('Some windows do not include all SMs')
        if not np.allclose(activity.frequency_hz, self.reference_mhz*1e6):
            raise ValueError('Activity and config frequencies differ')
        coefficients = np.array(model['coefficients'])
        k_sm = (activity_rates(activity)@coefficients[:-1]).reshape(-1,n_sm)
        b_sm = (activity.idle_fraction.to_numpy()*coefficients[-1]).reshape(-1,n_sm)
        actual_sm = activity.sm_power_w.to_numpy().reshape(-1,n_sm)
        model_error = float(abs(k_sm*self.reference_mhz/1000+b_sm-actual_sm).max())
        if model_error > 1e-5:
            raise ValueError(f'Activity model does not match this simulator/config: max error {model_error:.6g} W; refit model first')
        # Keep exact reference power while retaining the fitted idle intercept.
        k_sm = (actual_sm-b_sm)/(self.reference_mhz/1000)
        if k_sm.min() < -1e-8:
            raise ValueError('Negative activity power slope')
        self.k = np.zeros_like(measured)
        self.b = np.zeros_like(measured)
        for sm in sm_ids:
            col = self.names.index(f'SM_{sm}')
            if np.max(abs(measured[:,col]-actual_sm[:,sm])) > 1e-8:
                raise ValueError('Activity SM targets disagree with Ptrace')
            self.k[:,col] = np.maximum(k_sm[:,sm],0)
            self.b[:,col] = b_sm[:,sm]
        components = pd.read_csv(self.run_dir/'power_components.csv').sort_values('sample')
        if not np.isfinite(components.to_numpy()).all():
            raise ValueError('Nonfinite global component export')
        if len(components) != len(measured):
            raise ValueError('Global component window count mismatch')
        samples = activity['sample'].to_numpy().reshape(-1,n_sm)[:,0]
        component_samples = components['sample'].to_numpy()
        # Existing global CSV uses the incremented counter; require one constant offset.
        if not np.all(component_samples-samples == component_samples[0]-samples[0]):
            raise ValueError('Global/local sample alignment is inconsistent')
        l2 = self.names.index('L2_cache')
        counters = ['counter_'+name for name in COUNTERS]+['counter_PIPE_A']
        active = (activity[counters].abs().sum(axis=1).to_numpy().reshape(-1,n_sm)>1e-12).any(axis=1)
        # MC has a routine-work term proportional to executionTime, hence a
        # constant W intercept after dividing by time (memoryctrl.cc).
        quiet=(~active)&(components[['L2CP','DRAMP','NOCP']].abs().sum(axis=1).to_numpy()<1e-12)
        mcp_fixed=float(components.loc[quiet,'MCP'].min()) if quiet.any() else 0.
        if mcp_fixed<0 or (components.MCP.to_numpy()<mcp_fixed-1e-9).any():
            raise ValueError('Invalid measured MCP background intercept')
        shared = components[['L2CP','MCP','NOCP']].sum(axis=1).to_numpy()
        self.k[:,l2] = (shared-mcp_fixed)/(self.reference_mhz/1000)
        self.b[:,l2] = mcp_fixed
        for name in ('HBM2_0','HBM2_1'):
            self.k[:,self.names.index(name)] = components.DRAMP.to_numpy()/2/(self.reference_mhz/1000)
        # Floorplan has no separate static/constant/NoC/controller blocks.
        # Explicit area policy preserves TOTAL rather than silently dropping heat.
        sm_areas = np.array([flp[self.names.index(f'SM_{sm}')][1]*flp[self.names.index(f'SM_{sm}')][2] for sm in sm_ids])
        sm_areas /= sm_areas.sum()
        for sm in sm_ids:
            self.b[:,self.names.index(f'SM_{sm}')] += (components.STATICP+components.CONSTP).to_numpy()*sm_areas[sm]
        full = self.k*self.reference_mhz/1000+self.b
        conservation_error = float(abs(full.sum(axis=1)-components.TOTAL.to_numpy()).max())
        if conservation_error > 1e-7:
            raise ValueError(f'Full thermal mapping fails power conservation: {conservation_error}')
        # Local event/duty activity; shared counters in this export are zero placeholders.
        if not active.any():
            raise ValueError('No non-idle sampling windows')
        self.samples = samples[active].astype(int)
        self.k = self.k[active]
        self.b = self.b[active]
        self.reference_power = full[active]
        self.audit = {'total_windows':len(measured),'retained_windows':int(active.sum()),
                      'discarded_wholly_idle_windows':int((~active).sum()),
                      'max_activity_reconstruction_error_w':model_error,
                      'max_full_power_conservation_error_w':conservation_error,
                      'component_sample_offset':int(component_samples[0]-samples[0]),
                      'mcp_fixed_background_w':mcp_fixed,
                      'mcp_background_measured_from_idle':bool(quiet.any()),
                      'mapping':{'L2_cache':'L2CP+MCP+NOCP','HBM2_0/HBM2_1':'DRAMP/2 each',
                                 'SM_i':'local measured power + area share of STATICP+CONSTP'},
                      'idle_policy':'Drop wholly idle windows; keep local idle/static heat in retained windows. Individual idle cycles inside a window cannot be removed from window-only data.',
                      'power_model':'Fixed measured accesses/core-cycle, duty and idle; shared event power scales with core frequency, static/constant fixed; DRAM clock unchanged',
                      'activity_model_sha256':sha256(model_path)}

    def power(self, mhz):
        return self.k*(mhz/1000)+self.b

    def worst(self, thermal, mhz, ambient_c):
        temp = ambient_c+thermal.H@self.power(mhz).T
        node,window = np.unravel_index(temp.argmax(), temp.shape)
        return int(window), int(node), float(temp[node,window])


def quantize_down(value, step):
    # No epsilon up-rounding: nextafter protects the strictly downward bound.
    value = min(value, math.floor(value/step)*step)
    return float(value)


def solve_frequency(thermal, windows, ambient_c, threshold_c=90., step_mhz=1.,
                    minimum_mhz=100., upper_mhz=None, margin_c=0.1, offset=None):
    upper = windows.reference_mhz if upper_mhz is None else min(upper_mhz,windows.reference_mhz)
    if step_mhz <= 0 or minimum_mhz <= 0 or upper < minimum_mhz or margin_c < 0:
        raise ValueError('Invalid frequency range, step, or margin')
    slope = thermal.H@windows.k.T/1000 # degrees C / MHz
    intercept = ambient_c+thermal.H@windows.b.T
    if offset is not None:
        intercept += np.asarray(offset)[:,None]
    limit = threshold_c-margin_c
    if (intercept+minimum_mhz*slope).max() > limit:
        raise ValueError('Even the minimum frequency exceeds the thermal limit; no safe solution in this range')
    positive = slope > 1e-14
    limits = np.full(slope.shape, np.inf)
    limits[positive] = (limit-intercept)[positive]/slope[positive]
    continuous = min(upper,float(limits.min()))
    candidate = quantize_down(continuous,step_mhz)
    if candidate < minimum_mhz:
        raise ValueError('No downward-quantized frequency in the requested range')
    temperature = intercept+candidate*slope
    node,window = np.unravel_index(temperature.argmax(), temperature.shape)
    next_f = candidate+step_mhz
    return {'continuous_limit_mhz':continuous,'frequency_mhz':candidate,
            'threshold_c':threshold_c,'safety_margin_c':margin_c,
            'predicted_max_all_nodes_c':float(temperature.max()),
            'limiting_node':int(node),'worst_sample':int(windows.samples[window]),
            'next_step_mhz':next_f,
            'next_step_allowed':bool(next_f<=upper and (intercept+next_f*slope).max()<=limit),
            'upper_bound_mhz':upper,'frequency_step_mhz':step_mhz,
            'meaning':'Highest quantized frequency under frozen reference activities, all retained windows/all thermal nodes. Requires a fresh simulation to validate.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--grid-steady',type=Path,required=True,help='HotSpot grid result for the simultaneous worst-window Ptrace')
    p.add_argument('--gcb-dir',type=Path,default=DEFAULT_GCB)
    p.add_argument('--flp',type=Path,default=ROOT/'hotspot-link/flp/gv100_new.flp')
    p.add_argument('--thermal-config',type=Path,default=ROOT/'hotspot-link/GV100-hotspot.config')
    p.add_argument('--activity-model',type=Path,default=DEFAULT_MODEL)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--threshold-c',type=float,default=90.)
    p.add_argument('--frequency-step-mhz',type=float,default=1.)
    p.add_argument('--minimum-frequency-mhz',type=float,default=100.)
    p.add_argument('--safety-margin-c',type=float,default=.1)
    args=p.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    blocks=floorplan(args.flp);thermal=GridModel(args.gcb_dir,len(blocks))
    windows=PowerWindows(args.run_dir,blocks,args.activity_model)
    ambient=float(config_values(args.thermal_config)['ambient'])-273.15
    w,_,_=windows.worst(thermal,windows.reference_mhz,ambient)
    grid=np.loadtxt(args.grid_steady,ndmin=2)[:,1]-273.15
    prediction=ambient+thermal.H@windows.reference_power[w]
    if len(grid)>len(prediction) or np.max(abs(grid-prediction[:len(grid)]))>.03:
        raise ValueError('Supplied grid output is not the matching simultaneous worst-window/GCB model')
    offset=np.zeros(len(prediction));offset[:len(grid)]=grid-prediction[:len(grid)]
    result=solve_frequency(thermal,windows,ambient,args.threshold_c,args.frequency_step_mhz,
                           args.minimum_frequency_mhz,margin_c=args.safety_margin_c,offset=offset)
    selected,_,_=windows.worst(thermal,result['frequency_mhz'],ambient)
    write_ptrace(args.out_dir/'predicted_worst.ptrace',windows.names,windows.power(result['frequency_mhz'])[selected])
    dump_json(args.out_dir/'frequency.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
