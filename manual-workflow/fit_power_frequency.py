#!/usr/bin/env python3
"""Fit controlled frequency sweep, holding 1400/2800 MHz out of training."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


MODELS = {
    'affine': (lambda x,a,b: a+b*x, 2),
    'quadratic': (lambda x,a,b,c: a+b*x+c*x*x, 3),
    'saturating': (lambda x,a,b,c: a+b*x/(1+c*x), 3),
    'offset_cubic': (lambda x,a,b: a+b*x**3, 2),
}


def fit(name, x, y):
    if name in ['affine','quadratic','offset_cubic']:
        cols = [np.ones(len(x)), x**3 if name=='offset_cubic' else x]
        if name == 'quadratic': cols.append(x*x)
        return np.linalg.lstsq(np.column_stack(cols),y,rcond=None)[0]
    coefficients,_=curve_fit(MODELS[name][0],x,y,p0=[y.min(),10.,0.1],bounds=([-np.inf,0.,0.],[np.inf,np.inf,10.]),maxfev=20000)
    return coefficients


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    paths=sorted(args.root.glob('*MHz/summary.json'),key=lambda p:float(p.parent.name[:-3]))
    records=[json.loads(p.read_text()) for p in paths]
    if len(records)<8: raise SystemExit('need all eight controlled points')
    x=np.array([r['frequency_mhz']/1000. for r in records])
    validation=np.isin(x,[1.4,2.8]); train=~validation
    output={'frequency_unit':'GHz','power_unit':'W','validation_frequencies_mhz':[1400,2800],
            'scope':'backprop+bfs+heartwall; fixed trace/physical SM map; Core=ICNT=L2=f, DRAM=850 MHz; sampled mean power, 500-core-cycle windows',
            'outputs':{}}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for axis,key in zip(axes,['Total_GPU_Power','mapped_power_w']):
        y=np.array([r[key] for r in records]); candidates=[]
        for name,(fn,count) in MODELS.items():
            coeff=fit(name,x[train],y[train]); pred=fn(x[validation],*coeff)
            cv=[]
            for index in np.flatnonzero(train):
                subset=train.copy();subset[index]=False
                c=fit(name,x[subset],y[subset]); cv.append(float(abs(fn(x[index],*c)-y[index])/y[index]))
            candidates.append(dict(name=name,parameter_count=count,training_coefficients=coeff.tolist(),
                                   validation_max_relative_error=float(np.max(np.abs(pred-y[validation])/y[validation])),
                                   validation_rmse_w=float(np.sqrt(np.mean((pred-y[validation])**2))),
                                   training_loo_max_relative_error=max(cv),validation_predictions_w=pred.tolist()))
        # Prefer simpler model only within 0.05 percentage points of the best held-out error.
        best=min(c['validation_max_relative_error'] for c in candidates)
        shortlist=[c for c in candidates if c['validation_max_relative_error']<=best+0.0005]
        chosen=min(shortlist,key=lambda c:(c['parameter_count'],c['validation_max_relative_error']))
        name=chosen['name'];coeff=fit(name,x,y);fn=MODELS[name][0]
        error=np.abs(fn(x,*coeff)-y)
        output['outputs'][key]=dict(selected_model=name,all_sample_coefficients=coeff.tolist(),
                                   all_sample_max_abs_error_w=float(error.max()),all_sample_max_relative_error=float(np.max(error/y)),
                                   candidates=candidates,observed_power_w=y.tolist())
        axis.scatter(x[train],y[train],label='Training points',color='#1f77b4')
        axis.scatter(x[validation],y[validation],label='Held-out points',color='#d62728',marker='s')
        grid=np.linspace(x.min(),x.max(),200)
        axis.plot(grid,fn(grid,*coeff),label=f'{name}: refit all points',color='#222222')
        axis.set(xlabel='Core = ICNT = L2 frequency (GHz)',ylabel='Sampled mean power (W)',title=key)
        axis.grid(alpha=.25);axis.legend(fontsize=8)
    output['frequencies_mhz']=[r['frequency_mhz'] for r in records]
    # Preserve spatial power information for subsequent G/B thermal calculations.
    block_names=None;block_values=[]
    for path in paths:
        with (path.parent/'block_average_power.csv').open() as file:
            rows=list(csv.DictReader(file))
        names=[row['block'] for row in rows]
        if block_names is not None and names!=block_names:raise ValueError('block order changed')
        block_names=names;block_values.append([float(row['average_power_w']) for row in rows])
    blocks=np.array(block_values);design=np.column_stack([np.ones(len(x)),x,x*x])
    block_coeff=np.linalg.lstsq(design,blocks,rcond=None)[0]
    with (args.root/'block_quadratic_models.csv').open('w') as file:
        writer=csv.writer(file);writer.writerow(['block','a_w','b_w_per_ghz','c_w_per_ghz2'])
        writer.writerows((name,*block_coeff[:,i]) for i,name in enumerate(block_names))
    train_coeff=np.linalg.lstsq(design[train],blocks[train],rcond=None)[0]
    pred=design[validation]@train_coeff
    output['spatial_quadratic_validation']={'power_vector_relative_l2_errors':(np.linalg.norm(pred-blocks[validation],axis=1)/np.linalg.norm(blocks[validation],axis=1)).tolist(),
                                           'min_predicted_block_power_in_range_w':float((np.column_stack([np.ones(101),np.linspace(x.min(),x.max(),101),np.linspace(x.min(),x.max(),101)**2])@block_coeff).min()),
                                           'note':'83 per-block quadratic fits; do not replace their spatial vector with GPU total'}
    (args.root/'fit.json').write_text(json.dumps(output,indent=2)+'\n')
    fig.tight_layout();fig.savefig(args.root/'power_frequency.png',dpi=180);plt.close(fig)
    lines=['# 实测频率—功耗关系', '',
           '实验仅修改每轮 gpgpusim.config 中 gpgpu_clock_domains 的前三项，三者保持相等；DRAM 固定 850 MHz。原基础配置未改动。', '',
           '固定 backprop+bfs+heartwall trace、80 个 SM、原有物理映射（43/9/28 个）、XML、电压设置和 500 核心周期功耗采样。', '',
           'P 是已采样窗口的算术平均 W；f 以下使用 GHz，即配置中的 MHz/1000。各窗口时间相同；末尾不足 500 周期未计入该平均。该模型不是所有任务/SM 分配通用公式。', '',
           '| f (MHz) | GPU total (W) | SM/L2/HBM mapped (W) | Cycles | Unsampled tail (cycles) |',
           '|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f'| {r["frequency_mhz"]:g} | {r["Total_GPU_Power"]:.6f} | {r["mapped_power_w"]:.6f} | {r["cycles"]} | {r["unsampled_tail_cycles"]} |')
    lines+=['','1400、2800 MHz 在初始模型拟合时排除，用于独立频率点验证。另对训练点进行留一验证。若简单模型的验证最大相对误差在最佳模型的 0.05 个百分点内，优先简单模型。选型后使用全部频率点重拟合最终系数。','']
    for key,r in output['outputs'].items():
        c=r['all_sample_coefficients'];name=r['selected_model']
        if name=='affine': formula=f'P = {c[0]:.9f} + ({c[1]:.9f}) f'
        elif name=='quadratic':formula=f'P = {c[0]:.9f} + ({c[1]:.9f}) f + ({c[2]:.9f}) f²'
        elif name=='saturating':formula=f'P = {c[0]:.9f} + ({c[1]:.9f}) f / (1 + ({c[2]:.9f}) f)'
        else:formula=f'P = {c[0]:.9f} + ({c[1]:.9f}) f³'
        lines += [f'## {key}', '', f'选中模型：**{name}**。', '', f'`{formula}`', '',
                  f'最终全部样本最大偏差 {r["all_sample_max_abs_error_w"]:.6f} W，最大相对偏差 {100*r["all_sample_max_relative_error"]:.4f}%。', '',
                  '| Candidate | Held-out max error (%) | Held-out RMSE (W) | Training LOO max error (%) |',
                  '|---|---:|---:|---:|']
        for candidate in r['candidates']:
            lines.append(f'| {candidate["name"]} | {100*candidate["validation_max_relative_error"]:.4f} | {candidate["validation_rmse_w"]:.6f} | {100*candidate["training_loo_max_relative_error"]:.4f} |')
        lines.append('')
    lines += ['## 解释与适用范围','',
              '源码仍是组件能量/采样时间；升频会改变每周期活动数、等待、活跃 SM 与静态功率选择，所以并不保证严格恒定 q、b。拟合模型对本实验 800–4000 MHz 范围内的输出作经验近似，不用于范围外保证温度或功耗。', '',
              '不能把仅 SM/L2/HBM 的 mapped 功率当成完整 GPU total；后者还包含 MCP、NoC、CONSTP、STATICP 等。', '',
              'CONSTP 为 XML 常量 32.3252 W；STATICP 的平均值随活动状态变化。IDLE_COREP 分项汇总日志存在遗漏累加而显示 0 的问题，但 Total_SM_Power 和逐 SM 功耗包含空闲项。本实验没有修改源代码，也没有用该错误分项拟合。', '',
              '矩阵温度反解需要逐块功耗，而不是只将 GPU total 标量放入 B。block_quadratic_models.csv 同时提供 83 个热映射块的二次系数（f 使用 GHz）；它们仍只适用于本实验范围，应检查逐块拟合误差并用候选频率复跑验证。', '',
              '复现：`python3 manual-workflow/explore_power_frequency.py --out-dir <新目录> --jobs 2`，然后 `python3 manual-workflow/fit_power_frequency.py --root <结果目录>`。保存的 inputs/ 与 manifest.json 给出了配置、trace、二进制校验信息。']
    (args.root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
