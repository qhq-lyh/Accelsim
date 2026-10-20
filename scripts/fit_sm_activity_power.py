#!/usr/bin/env python3
"""Learn a small activity-to-SM-power model, with entire frequencies held out."""
import argparse, csv, json, os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from sm_activity_power import COUNTERS, DIRECT, GROUPS, FEATURES, activity_rates, features
HOLDOUT = {1400.,2800.}

def stats(actual,pred):
    error=np.abs(actual-pred)
    return dict(count=len(error),mae_w=float(error.mean()),rmse_w=float(np.sqrt(np.mean(error**2))),
                max_abs_error_w=float(error.max()),p95_abs_error_w=float(np.percentile(error,95)),
                p95_relative_error_pct=float(np.percentile(error/np.maximum(actual,1e-8)*100,95)),
                max_relative_error_pct=float(np.max(error/np.maximum(actual,1e-8)*100)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--mixed-dir',type=Path)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    runs=[(p.parent.parent.name,float(p.parent.name[:-3]),p) for p in sorted(args.data_dir.glob('*/*MHz/sm_activity.csv.gz')) if (p.parent/'complete.json').exists()]
    if args.mixed_dir:
        runs += [('mixed_backprop_bfs_heartwall_43_9_28',float(p.parent.name[:-3]),p) for p in sorted(args.mixed_dir.glob('*MHz/sm_activity.csv.gz')) if (p.parent/'summary.json').exists()]
    required=['sm_id','sample','frequency_hz','window_cycles','idle_fraction','sm_power_w','counter_PIPE_A']+['counter_'+c for c in COUNTERS]
    train_x=[];train_y=[];peaks={};means={};export_checks=[];signed_counters=[]
    for name,f,path in runs:
        frame=pd.read_csv(path,usecols=required)
        if not np.isfinite(frame.to_numpy()).all():raise ValueError(f'Nonfinite: {path}')
        # Native miss = accesses - hits, with different MSHR_HIT inclusion.
        # These two model inputs can legitimately be signed algebraic values;
        # clipping them would change the preserved upstream power formula.
        if frame[['counter_'+c for c in COUNTERS if c not in ('DC_RM','DC_WM')]].to_numpy().min() < -1e-8:
            raise ValueError(f'Negative local activity counter: {path}')
        for counter in ('DC_RM','DC_WM'):
            values=frame['counter_'+counter].to_numpy()
            if values.min() < -1e-8:
                signed_counters.append(dict(benchmark=name,frequency_mhz=f,counter=counter,negative_rows=int((values < -1e-8).sum()),minimum=float(values.min())))
        if frame.idle_fraction.min() < -1e-6 or frame.idle_fraction.max() > 1+1e-6:
            raise ValueError(f'Invalid idle fraction: {path}')
        if not np.allclose(frame.frequency_hz,f*1e6,rtol=1e-12):raise ValueError(f'Clock mismatch: {path}')
        if frame.duplicated(['sm_id','sample']).any():raise ValueError(f'Duplicate rows: {path}')
        x=features(frame);y=frame.sm_power_w.to_numpy()
        mapped=np.loadtxt(path.parent/'lyhong_power_report.log',skiprows=1,ndmin=2)
        if len(y)!=mapped.shape[0]*80 or not np.array_equal(frame.sm_id.to_numpy(),np.tile(np.arange(80),mapped.shape[0])):
            raise ValueError(f'Unexpected SM/window layout: {path}')
        difference=float(np.max(abs(y.reshape(-1,80)-mapped[:,:80])))
        if difference>1e-8:raise ValueError(f'Activity target differs from mapped SM power: {path}: {difference}')
        export_checks.append(dict(benchmark=name,frequency_mhz=f,windows=mapped.shape[0],max_abs_error_w=difference))
        # Training excludes mixed runs completely, as well as both held-out frequencies.
        if name!='mixed_backprop_bfs_heartwall_43_9_28' and f not in HOLDOUT:
            indices=np.linspace(0,len(frame)-1,min(len(frame),4000),dtype=int)
            train_x.append(x[indices]);train_y.append(y[indices])
        selected=frame.groupby('sm_id').sm_power_w.idxmax().to_numpy()
        peaks[(name,f)]=(frame.loc[selected,'sm_id'].to_numpy(),x[selected],y[selected])
        means[(name,f)]=(np.arange(80),x.reshape(-1,80,x.shape[1]).mean(axis=0),y.reshape(-1,80).mean(axis=0))
    if not train_x:raise ValueError('No training data')
    (args.out_dir/'export_consistency.json').write_text(json.dumps(export_checks,indent=2)+'\n')
    (args.out_dir/'signed_derived_counters.json').write_text(json.dumps(signed_counters,indent=2)+'\n')
    x=np.concatenate(train_x);y=np.concatenate(train_y)
    scale=np.sqrt(np.mean(x*x,axis=0))
    unobserved=[name for name,value in zip(FEATURES,scale) if value==0]
    scale[scale==0]=1
    normalized=x/scale
    # NNLS on a factor of the Gram matrix is equivalent to fitting all training rows.
    gram=normalized.T@normalized;rhs=normalized.T@y
    values,vectors=np.linalg.eigh(gram)
    keep=values>values.max()*1e-12
    a=np.sqrt(values[keep])[:,None]*vectors[:,keep].T
    b=(vectors[:,keep].T@rhs)/np.sqrt(values[keep])
    coeff=nnls(a,b,maxiter=10000)[0]/scale
    model=dict(equation='P_sm_W = f_GHz * (sum(beta_r * counter_r / S) + beta_PIPE_A * PIPE_A) + beta_idle * idle_fraction',
               features=FEATURES,feature_groups=GROUPS,coefficients=coeff.tolist(),frequency_unit='GHz',training_rows=len(y),
               unobserved_features=unobserved,
               observed_frequency_range_mhz=[min(f for _,f,_ in runs),max(f for _,f,_ in runs)],
               source_manifest=json.loads((args.data_dir/'manifest.json').read_text()),
               train_frequencies_mhz=sorted({f for name,f,_ in runs if name!='mixed_backprop_bfs_heartwall_43_9_28' and f not in HOLDOUT}),
               heldout_frequencies_mhz=sorted(HOLDOUT),mixed_excluded_from_training=True,
               warning='Empirical coefficients for this fixed-voltage model/config; correlated features prevent interpreting beta as unique physical energies.')
    (args.out_dir/'activity_model.json').write_text(json.dumps(model,indent=2)+'\n')
    with (args.out_dir/'activity_coefficients.csv').open('w') as file:
        writer=csv.writer(file);writer.writerow(['feature','coefficient','input_definition'])
        for name,value in zip(FEATURES,coeff):
            definition='idle_fraction' if name=='idle_fraction' else ('f_GHz * PIPE_A' if name=='PIPE_A' else ('f_GHz * sum('+','.join(GROUPS[name])+') / window_cycles' if name in GROUPS else 'f_GHz * counter / window_cycles'))
            writer.writerow([name,value,definition])
    metrics=[];peak_rows=[]
    for name,f,path in runs:
        frame=pd.read_csv(path,usecols=required)
        pred=features(frame)@coeff;actual=frame.sm_power_w.to_numpy()
        kind='unseen_mixed' if name.startswith('mixed_') else ('heldout_frequency' if f in HOLDOUT else 'training_frequency')
        metrics.append(dict(benchmark=name,frequency_mhz=f,split=kind,**stats(actual,pred)))
        active=(frame[['counter_'+c for c in COUNTERS]].to_numpy()!=0).any(axis=1)|(frame.counter_PIPE_A.to_numpy()!=0)
        metrics[-1]['active_count']=int(active.sum())
        metrics[-1]['active_mae_w']=float(np.mean(np.abs(actual[active]-pred[active]))) if active.any() else 0.
        metrics[-1]['active_p95_relative_error_pct']=float(np.percentile(np.abs(actual[active]-pred[active])/np.maximum(actual[active],1e-8)*100,95)) if active.any() else 0.
        frame['predicted']=pred
        predicted_peak=frame.groupby('sm_id').predicted.max()
        ids,px,py=peaks[(name,f)]
        for sm,xp,yp in zip(ids,px,py):
            baseline=peaks.get((name,1698.))
            frozen=float('nan')
            if baseline is not None:
                j=np.flatnonzero(baseline[0]==sm)[0]
                fx=baseline[1][j].copy();fx[:-1]*=f/1698.
                frozen=float(fx@coeff)
            peak_rows.append(dict(benchmark=name,frequency_mhz=f,sm_id=int(sm),actual_peak_w=float(yp),
                                  activity_reconstruction_peak_w=float(predicted_peak.loc[sm]),
                                  frozen_1698_activity_peak_w=frozen,split=kind))
    for filename,rows in [('window_errors.csv',metrics),('peak_predictions.csv',peak_rows)]:
        with (args.out_dir/filename).open('w') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    profiles={};mean_profiles={};mean_rows=[]
    for (name,f),(ids,mx,my) in means.items():
        for sm,xm,ym in zip(ids,mx,my):
            mean_profiles.setdefault((name,int(sm)),[]).append(dict(frequency_mhz=f,
                k_w_per_ghz=float(xm[:-1]@coeff[:-1]/(f/1000)),
                b_w=float(xm[-1]*coeff[-1]),actual_mean_w=float(ym)))
    for filename,held in [('mean_activity_models.json',set()),('mean_activity_train_models.json',HOLDOUT)]:
        records=[dict(benchmark=name,sm_id=sm,points=sorted((p for p in points if p['frequency_mhz'] not in held),key=lambda p:p['frequency_mhz'])) for (name,sm),points in sorted(mean_profiles.items())]
        (args.out_dir/filename).write_text(json.dumps(dict(statistic='mean',heldout_frequencies_mhz=sorted(held),equation='P_mean(f_GHz) = f_GHz * interp(k(f)) + interp(b(f))',models=records),indent=2)+'\n')
    for (name,sm),points in mean_profiles.items():
        reference=sorted((p for p in points if p['frequency_mhz'] not in HOLDOUT),key=lambda p:p['frequency_mhz'])
        knots=[p['frequency_mhz'] for p in reference]
        for point in points:
            f=point['frequency_mhz']
            prediction=f/1000*np.interp(f,knots,[p['k_w_per_ghz'] for p in reference])+np.interp(f,knots,[p['b_w'] for p in reference])
            mean_rows.append(dict(benchmark=name,frequency_mhz=f,sm_id=sm,actual_mean_w=point['actual_mean_w'],
                                  interpolated_6ref_mean_w=float(prediction),heldout=f in HOLDOUT,
                                  active=point['k_w_per_ghz']>1e-9))
    with (args.out_dir/'mean_predictions.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(mean_rows[0]));writer.writeheader();writer.writerows(mean_rows)
    for (name,f),(ids,px,py) in peaks.items():
        for sm,xp,yp in zip(ids,px,py):
            profiles.setdefault((name,int(sm)),[]).append(dict(frequency_mhz=f,
                k_w_per_ghz=float(xp[:-1]@coeff[:-1]/(f/1000)),
                b_w=float(xp[-1]*coeff[-1]),actual_peak_w=float(yp)))
    for filename,held in [('frequency_activity_models.json',set()),('frequency_activity_train_models.json',HOLDOUT)]:
        records=[dict(benchmark=name,sm_id=sm,points=sorted((p for p in points if p['frequency_mhz'] not in held),key=lambda p:p['frequency_mhz'])) for (name,sm),points in sorted(profiles.items())]
        (args.out_dir/filename).write_text(json.dumps(dict(statistic='peak',equation='P_peak(f_GHz) = f_GHz * interp(k(f)) + interp(b(f)); interpolation is linear between reference frequencies',
                 heldout_frequencies_mhz=sorted(held),models=records),indent=2)+'\n')
    for row in peak_rows:
        reference=sorted((p for p in profiles[(row['benchmark'],row['sm_id'])] if p['frequency_mhz'] not in HOLDOUT),key=lambda p:p['frequency_mhz'])
        f=row['frequency_mhz'];knots=[p['frequency_mhz'] for p in reference]
        row['interpolated_6ref_activity_peak_w']=float(f/1000*np.interp(f,knots,[p['k_w_per_ghz'] for p in reference])+np.interp(f,knots,[p['b_w'] for p in reference]))
    # Rewrite after adding the interpolated prediction, which uses no target-frequency counters.
    with (args.out_dir/'peak_predictions.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(peak_rows[0]));writer.writeheader();writer.writerows(peak_rows)
    aggregate={}
    for kind in ['heldout_frequency','unseen_mixed']:
        subset=[r for r in peak_rows if r['split']==kind]
        if subset:
            actual=np.array([r['actual_peak_w'] for r in subset])
            aggregate[kind+'_activity_peak']=stats(actual,np.array([r['activity_reconstruction_peak_w'] for r in subset]))
            aggregate[kind+'_frozen_activity_peak']=stats(actual,np.array([r['frozen_1698_activity_peak_w'] for r in subset]))
            validation=[r for r in subset if r['frequency_mhz'] in HOLDOUT]
            if validation:
                aggregate[kind+'_interpolated_activity_peak']=stats(np.array([r['actual_peak_w'] for r in validation]),np.array([r['interpolated_6ref_activity_peak_w'] for r in validation]))
            # Avoid reporting a deceptively small relative error from idle-only SMs.
            active=[r for r in subset if r['actual_peak_w']>coeff[-1]+.001]
            if active:
                target=np.array([r['actual_peak_w'] for r in active])
                aggregate[kind+'_active_activity_peak']=stats(target,np.array([r['activity_reconstruction_peak_w'] for r in active]))
                aggregate[kind+'_active_frozen_activity_peak']=stats(target,np.array([r['frozen_1698_activity_peak_w'] for r in active]))
                validation=[r for r in active if r['frequency_mhz'] in HOLDOUT]
                if validation:
                    aggregate[kind+'_active_interpolated_activity_peak']=stats(np.array([r['actual_peak_w'] for r in validation]),np.array([r['interpolated_6ref_activity_peak_w'] for r in validation]))
    (args.out_dir/'validation.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    mean_validation={}
    for name in sorted({r['benchmark'] for r in mean_rows}):
        selected=[r for r in mean_rows if r['benchmark']==name and r['heldout']]
        if selected:
            mean_validation[name]=stats(np.array([r['actual_mean_w'] for r in selected]),np.array([r['interpolated_6ref_mean_w'] for r in selected]))
            active=[r for r in selected if r['active']]
            mean_validation[name]['active_count']=len(active)
            if active:
                error=stats(np.array([r['actual_mean_w'] for r in active]),np.array([r['interpolated_6ref_mean_w'] for r in active]))
                mean_validation[name]['active_p95_relative_error_pct']=error['p95_relative_error_pct']
                mean_validation[name]['active_max_relative_error_pct']=error['max_relative_error_pct']
    (args.out_dir/'mean_validation.json').write_text(json.dumps(mean_validation,indent=2)+'\n')
    report=['# 活动模型数值验证','',
            f'数据：{len(runs)} 组运行，{sum(r["windows"] for r in export_checks)} 个完整窗口，每个窗口 80 个 SM。',
            f'训练：{len(y)} 行，{len(FEATURES)} 个特征；1400/2800 MHz 和全部混合任务不参与活动权重训练。',
            f'全部逐 SM 窗口的最大重建误差：{max(r["max_abs_error_w"] for r in metrics):.6g} W。',
            f'诊断目标与 mapped SM 功耗列的最大差异：{max(r["max_abs_error_w"] for r in export_checks):.6g} W。','',
            '下表均为 1400/2800 MHz 留出点。峰值与平均值使用不同目标，不可互换。活动重建使用目标频率的计数；另外两种峰值预测与平均预测均不使用目标频率计数。','',
            '|benchmark|固定 1698 活动的峰值最大误差 %|六参考活动插值的峰值最大误差 %|平均功耗活动 SM 的95分位误差 %|平均功耗最大误差 %|',
            '|---|---:|---:|---:|---:|']
    for name in sorted(mean_validation):
        selected=[r for r in peak_rows if r['benchmark']==name and r['frequency_mhz'] in HOLDOUT]
        actual=np.array([r['actual_peak_w'] for r in selected])
        frozen=100*np.max(abs(np.array([r['frozen_1698_activity_peak_w'] for r in selected])-actual)/actual)
        interpolated=100*np.max(abs(np.array([r['interpolated_6ref_activity_peak_w'] for r in selected])-actual)/actual)
        mean=mean_validation[name]
        report.append(f'|{name}|{frozen:.4f}|{interpolated:.4f}|{mean.get("active_p95_relative_error_pct",0):.4f}|{mean["max_relative_error_pct"]:.4f}|')
    report+=['','同一 SM16、2800 MHz 的平均功耗留出预测示例：','','|benchmark|实际 W|预测 W|误差 %|','|---|---:|---:|---:|']
    for row in mean_rows:
        if row['sm_id']==16 and row['frequency_mhz']==2800 and row['benchmark'] in ['backprop','bfs','heartwall','mixed_backprop_bfs_heartwall_43_9_28']:
            actual,predicted=row['actual_mean_w'],row['interpolated_6ref_mean_w']
            report.append(f'|{row["benchmark"]}|{actual:.9f}|{predicted:.9f}|{100*abs(actual-predicted)/actual:.4f}|')
    report+=['','有效活动权重（省略零系数；完整精度见 activity_model.json）：','','|feature|coefficient|','|---|---:|']
    report += [f'|{name}|{value:.12g}|' for name,value in zip(FEATURES,coeff) if value>1e-12]
    report+=['','这些是当前计数口径的有效权重，不代表唯一的物理电容。计数已含 XML 执行单元标定，不能再次重复乘标定系数。',
             '平均功耗 profile 只适用于当前 trace、SM 分配、固定电压和 800–4000 MHz；反算候选频率应重新仿真验证。']
    (args.out_dir/'VALIDATION.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(aggregate,indent=2),flush=True)

if __name__=='__main__':main()
