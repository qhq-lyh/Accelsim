#!/usr/bin/env python3
"""Fit each benchmark/SM peak separately; use held-out frequencies for validation."""
import argparse, csv, json, os, re
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/accel-hotsim-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from explore_sm_peak_frequency import peaks

MODELS={'affine':(0,1),'quadratic':(0,1,2),'cubic':(0,1,2,3),'offset_f_cubed':(0,3)}
def design(x,powers): return np.column_stack([x**p for p in powers])
def fit(x,y,powers): return np.linalg.lstsq(design(x,powers),y,rcond=None)[0]
def predict(x,coefficients,powers): return design(x,powers)@coefficients

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--mixed-dir',type=Path)
    args=parser.parse_args();root=args.data_dir.resolve()
    with (root/'sm_peak_samples.csv').open() as handle: rows=list(csv.DictReader(handle))
    if args.mixed_dir:
        manifest=json.loads((args.mixed_dir/'manifest.json').read_text())
        if (root/'manifest.json').exists():
            single_manifest=json.loads((root/'manifest.json').read_text())
            for name,value in single_manifest['input_sha256'].items():
                if name.endswith(('.xml','.icnt')) or name=='trace.config':
                    if manifest['input_sha256'].get(name)!=value: raise ValueError(f'Model/config mismatch: {name}')
            if manifest['binary_sha256']!=single_manifest['binary_sha256']: raise ValueError('Simulator binary mismatch')
        if manifest['sampling_cycles']!=500 or manifest['dram_mhz']!=850: raise ValueError('Mixed data sampling/clocks differ from this experiment')
        for directory in sorted(args.mixed_dir.glob('*MHz')):
            clocks=list(map(float,re.findall(r'(?m)^-gpgpu_clock_domains\s+([^\n]+)',(directory/'gpgpusim.config').read_text())[-1].strip().split(':')))
            if clocks[:3]!=[float(directory.name[:-3])]*3 or clocks[3]!=850: raise ValueError(f'Unexpected clock settings: {directory}')
            maximum,indices,count=peaks(directory/'lyhong_power_report.log')
            for i in range(80): rows.append(dict(benchmark='mixed_backprop_bfs_heartwall_43_9_28',frequency_mhz=float(directory.name[:-3]),sm_id=i,peak_power_w=float(maximum[i]),peak_sample_index_0based=int(indices[i]),sample_count=count,source=str((directory/'lyhong_power_report.log').resolve())))
    # The requested worst-case view: maximum across benchmark conditions,
    # after taking temporal maximum within each run. Preserve the winning source.
    envelope={}
    for row in rows:
        key=(float(row['frequency_mhz']),int(row['sm_id']))
        if key not in envelope or float(row['peak_power_w'])>float(envelope[key]['peak_power_w']): envelope[key]=row
    with (root/'sm_peak_envelope.csv').open('w') as handle:
        fields=['frequency_mhz','sm_id','peak_power_w','winning_benchmark','source']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for (frequency,sm),row in sorted(envelope.items()):
            writer.writerow(dict(frequency_mhz=frequency,sm_id=sm,peak_power_w=row['peak_power_w'],winning_benchmark=row['benchmark'],source=row.get('source','')))
            rows.append(dict(benchmark='envelope_all_sampled_conditions',frequency_mhz=frequency,sm_id=sm,peak_power_w=row['peak_power_w']))
    with (root/'sm_peak_all_conditions.csv').open('w') as handle:
        fields=['benchmark','frequency_mhz','sm_id','peak_power_w']
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    data={}
    for row in rows:
        data.setdefault((row['benchmark'],int(row['sm_id'])),[]).append((float(row['frequency_mhz'])/1000,float(row['peak_power_w'])))
    models=[]; candidates=[];out=root/'fits';out.mkdir(exist_ok=True)
    for (benchmark,sm),samples in sorted(data.items()):
        x,y=np.array(sorted(samples)).T
        held=np.isclose(x,1.4)|np.isclose(x,2.8);train=~held
        if len(x)<6 or train.sum()<4 or not held.any(): raise ValueError('Require >=6 points including 1400/2800 MHz holdouts')
        scores=[]
        model_specs={'constant':(0,)} if np.ptp(y)<=0.00100001 else MODELS
        for name,powers in model_specs.items():
            if train.sum() <= len(powers): continue
            tx,ty=x[train],y[train];errors=[]
            for j in range(len(tx)):
                keep=np.arange(len(tx))!=j
                coefficients=fit(tx[keep],ty[keep],powers)
                errors.append(float(predict(tx[j:j+1],coefficients,powers)[0]-ty[j]))
            coefficients=fit(tx,ty,powers)
            hold_errors=predict(x[held],coefficients,powers)-y[held]
            all_coefficients=fit(x,y,powers)
            dense=np.linspace(x.min(),x.max(),1000)
            valid=bool(np.min(predict(dense,all_coefficients,powers))>=0)
            record=dict(benchmark=benchmark,sm_id=sm,model=name,powers=list(powers),train_loocv_rmse_w=float(np.sqrt(np.mean(np.square(errors)))),heldout_max_abs_error_w=float(np.max(abs(hold_errors))),heldout_max_relative_error_pct=float(np.max(abs(hold_errors)/y[held])*100),valid_nonnegative=valid)
            candidates.append(record);scores.append((record,all_coefficients))
        valid=[item for item in scores if item[0]['valid_nonnegative']]
        best=min(item[0]['train_loocv_rmse_w'] for item in valid)
        # 1 mW log resolution; prefer simpler model within 2 mW CV RMSE.
        selected,coefficients=min((item for item in valid if item[0]['train_loocv_rmse_w']<=best+0.002),key=lambda item:(len(item[0]['powers']),item[0]['train_loocv_rmse_w']))
        residual=predict(x,coefficients,selected['powers'])-y
        models.append(dict(**selected,coefficients_w_per_ghz_power=coefficients.tolist(),frequency_min_ghz=float(x.min()),frequency_max_ghz=float(x.max()),all_points_max_abs_error_w=float(np.max(abs(residual))),peak_range_w=[float(y.min()),float(y.max())],samples=[dict(frequency_ghz=float(f),peak_power_w=float(p)) for f,p in zip(x,y)]))
    (out/'models.json').write_text(json.dumps(dict(frequency_unit='GHz',power_unit='W',equation='P_peak(f) = sum(coefficients[j] * f ** powers[j])',sampling_cycles=500,validation='1400 and 2800 MHz held out; model selected using LOOCV on remaining six points; final coefficients refitted to all eight',models=models),indent=2)+'\n')
    with (out/'models.csv').open('w') as handle:
        fields=['benchmark','sm_id','model','c0_w','c1_w_per_ghz','c2_w_per_ghz2','c3_w_per_ghz3','train_loocv_rmse_w','heldout_max_abs_error_w','heldout_max_relative_error_pct','all_points_max_abs_error_w']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for model in models:
            coefficients=dict(zip(model['powers'],model['coefficients_w_per_ghz_power']))
            writer.writerow({**{key:model[key] for key in fields if key in model},**{key:coefficients.get(i,0) for i,key in enumerate(fields[3:7])}})
    with (out/'candidate_validation.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(candidates[0]));writer.writeheader();writer.writerows(candidates)
    benchmarks=sorted({m['benchmark'] for m in models})
    colors=plt.cm.turbo(np.linspace(0,1,80))
    for benchmark in benchmarks:
        fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
        for m in models:
            if m['benchmark']!=benchmark:continue
            x=np.array([s['frequency_ghz'] for s in m['samples']]);y=np.array([s['peak_power_w'] for s in m['samples']]);dense=np.linspace(x.min(),x.max(),200)
            ax.plot(dense*1000,predict(dense,m['coefficients_w_per_ghz_power'],m['powers']),color=colors[m['sm_id']],alpha=.6,lw=.8)
            ax.scatter(x*1000,y,color=colors[m['sm_id']],s=8,alpha=.7)
        ax.set(xlabel='Core = ICNT = L2 frequency (MHz); DRAM = 850 MHz',ylabel='Per-SM peak sampled power (W)',title=benchmark+' (80 SMs; 500-cycle windows)');ax.grid(alpha=.2)
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,79),cmap='turbo'),ax=ax,label='SM ID')
        fig.savefig(out/(benchmark+'.png'),dpi=160);plt.close(fig)
    # Every panel is one physical SM; colors distinguish benchmark conditions.
    fig,axes=plt.subplots(10,8,figsize=(24,25),sharex=True,layout='constrained')
    palette=plt.cm.tab20(np.linspace(0,1,len(benchmarks)))
    for sm,ax in enumerate(axes.flat):
        for j,benchmark in enumerate(benchmarks):
            m=next(m for m in models if m['benchmark']==benchmark and m['sm_id']==sm)
            x=np.array([s['frequency_ghz'] for s in m['samples']]);y=np.array([s['peak_power_w'] for s in m['samples']]);dense=np.linspace(x.min(),x.max(),100)
            ax.plot(dense*1000,predict(dense,m['coefficients_w_per_ghz_power'],m['powers']),color='black' if benchmark.startswith('envelope_') else palette[j],lw=1.5 if benchmark.startswith('envelope_') else .8,label=benchmark)
            ax.scatter(x*1000,y,color=palette[j],s=3)
        ax.set_title(f'SM_{sm}',fontsize=9);ax.tick_params(labelsize=6);ax.grid(alpha=.15)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.045),ncol=4,fontsize=8)
    fig.supxlabel('Core = ICNT = L2 frequency (MHz)');fig.supylabel('Per-SM peak sampled power (W)')
    fig.savefig(out/'all_sm_benchmarks.png',dpi=140,bbox_inches='tight');plt.close(fig)
    counts={name:sum(m['model']==name for m in models) for name in ['constant',*MODELS]}
    report=['# 每个 SM 峰值功耗与频率拟合','', '功耗来自 `lyhong_power_report.log` 的 SM_0…SM_79 列：每列取所有采样窗口的最大值。SM 列按现有实现包含 SM 内各模块、流水线和空闲功耗，未分摊 GPU 全局静态功耗及其他非 SM 项。不是 GPU 总功耗，也不是瞬时峰值。窗口固定为 500 个核心周期，频率变化时对应的实际窗口时间变化。', '', 'Core、ICNT、L2 同频，DRAM 固定 850 MHz，关闭 DVFS、保持电压配置。单 benchmark 可使用全部 80 SM；混合 benchmark 是已有 43/9/28 SM 分配，另列为独立条件。', '', '频率点：800、1132、1400、1698、2200、2800、3396、4000 MHz。1400 和 2800 MHz 留出验证；其余六点通过留一交叉验证选模型，最终系数用全部八点重新拟合。CV RMSE 相差不超过 0.002 W 时选参数更少的模型。拟合只适用于该区间与该 trace/调度条件。', '', f'共 {len(benchmarks)-1} 个负载条件，外加跨负载峰值包络，共 {len(models)} 条 SM 曲线。模型数量：`{counts}`。', '', '误差中位数与95分位统计非恒定曲线，避免大量空闲 SM 淹没活跃 SM 的误差。最大误差统计全部 SM。', '', '|benchmark|非恒定曲线数|最大绝对误差 W|相对误差中位数 %|95分位 %|最大 %|', '|---|---:|---:|---:|---:|---:|']
    for benchmark in benchmarks:
        subset=[m for m in models if m['benchmark']==benchmark]
        varying=[m for m in subset if m['model']!='constant']
        relative=[m['heldout_max_relative_error_pct'] for m in varying] or [0.]
        report.append(f"|{benchmark}|{len(varying)}|{max(m['heldout_max_abs_error_w'] for m in subset):.6f}|{np.median(relative):.3f}|{np.percentile(relative,95):.3f}|{max(m['heldout_max_relative_error_pct'] for m in subset):.3f}|")
    report+=['', '跨负载包络中，各频率所有 SM 的最大观测峰值：', '', '|频率 MHz|最大 SM 峰值 W|', '|---:|---:|']
    for frequency in sorted({key[0] for key in envelope}):
        report.append(f"|{frequency:g}|{max(float(row['peak_power_w']) for (f,sm),row in envelope.items() if f==frequency):.3f}|")
    report+=['', '跨负载包络中验证误差最大的五个 SM：', '', '|SM|模型|最大留出相对误差 %|观测最高峰值 W|该峰值频率 MHz|该峰值来源|', '|---:|---|---:|---:|---:|---|']
    envelope_models=sorted((m for m in models if m['benchmark']=='envelope_all_sampled_conditions'),key=lambda m:m['heldout_max_relative_error_pct'],reverse=True)
    for m in envelope_models[:5]:
        peak=max(m['samples'],key=lambda s:s['peak_power_w']);f=round(peak['frequency_ghz']*1000,6)
        winner=envelope[(f,m['sm_id'])]['benchmark']
        report.append(f"|{m['sm_id']}|{m['model']}|{m['heldout_max_relative_error_pct']:.3f}|{peak['peak_power_w']:.3f}|{f:g}|{winner}|")
    report+=['','拟合曲线不保证始终高于观测峰值，也不保证对未采样频率构成上界。', '', '模型公式与逐 SM 系数见 models.json / models.csv，全部候选模型的验证结果见 candidate_validation.csv。频率 f 的单位为 GHz，P 的单位为 W。constant 表示本次观测无法辨认频率依赖，不意味着 SM 在其他负载下恒定功耗。', '', 'envelope_all_sampled_conditions 进一步在每个频率、每个 SM 上取所有负载条件的最大值，来源记录在 sm_peak_envelope.csv。它只覆盖已采样的 traces，不能保证覆盖其他输入规模、SM 分配或所有潜在负载。', '', '每个 SM 的峰值可能发生在不同时间，80 个峰值拼成的向量不是同时发生的功耗快照。用于稳态热分析时只能解释为给定采样条件下的逐块峰值包络场景；不能把它直接当作实测稳态功耗。功耗日志通常保留三位小数，尾部不足一个窗口的活动未必进入报告。']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(conditions=len(benchmarks),curves=len(models),models=counts,heldout_max_abs_error_w=max(m['heldout_max_abs_error_w'] for m in models)),indent=2))
if __name__=='__main__':main()
