#!/usr/bin/env python3
"""Predict or invert a per-SM peak or mean power profile inside its measured frequency range."""
import argparse,json
from pathlib import Path
import numpy as np

def evaluate(points,frequency_mhz):
    knots=np.array([p['frequency_mhz'] for p in points])
    if not knots[0]<=frequency_mhz<=knots[-1]:raise ValueError('Frequency outside measured profile range')
    k=float(np.interp(frequency_mhz,knots,[p['k_w_per_ghz'] for p in points]))
    b=float(np.interp(frequency_mhz,knots,[p['b_w'] for p in points]))
    return frequency_mhz/1000*k+b

def invert(points,target):
    roots=[];flat=[]
    for left,right in zip(points,points[1:]):
        lo,hi=left['frequency_mhz']/1000,right['frequency_mhz']/1000
        dk=(right['k_w_per_ghz']-left['k_w_per_ghz'])/(hi-lo)
        db=(right['b_w']-left['b_w'])/(hi-lo)
        polynomial=np.array([dk,left['k_w_per_ghz']-dk*lo+db,left['b_w']-db*lo-target])
        if np.max(abs(polynomial))<1e-10:
            flat.append([lo*1000,hi*1000]);continue
        polynomial=np.trim_zeros(polynomial,'f')
        for root in np.roots(polynomial):
            if abs(root.imag)<1e-9 and lo-1e-9<=root.real<=hi+1e-9:
                roots.append(float(np.clip(root.real,lo,hi)*1000))
    return sorted(set(round(r,9) for r in roots)),flat

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--benchmark',required=True)
    parser.add_argument('--sm',type=int,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--frequency-mhz',type=float)
    mode.add_argument('--target-power-w',type=float)
    args=parser.parse_args()
    profile=json.loads(args.profile.read_text());models=profile['models']
    matches=[m for m in models if m['benchmark']==args.benchmark and m['sm_id']==args.sm]
    if len(matches)!=1:parser.error('No unique matching benchmark/SM')
    points=matches[0]['points']
    result=dict(benchmark=args.benchmark,sm_id=args.sm,frequency_range_mhz=[points[0]['frequency_mhz'],points[-1]['frequency_mhz']],
                power_statistic=profile.get('statistic','peak'),
                warning='Applies to this trace, SM allocation, 500-cycle sampling and fixed voltage. Intermediate frequencies require simulation verification; per-SM peaks are not simultaneous.')
    if args.frequency_mhz is not None:
        try:power=evaluate(points,args.frequency_mhz)
        except ValueError as error:parser.error(str(error))
        result.update(frequency_mhz=args.frequency_mhz,predicted_power_w=power)
    else:
        roots,flat=invert(points,args.target_power_w)
        result.update(target_power_w=args.target_power_w,frequency_candidates_mhz=roots,flat_solution_intervals_mhz=flat)
        if not roots and not flat:result['status']='No solution within measured frequency range'
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
