#!/usr/bin/env python3
"""Predict SM power or invert a power target using one measured activity window."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sm_activity_power import FEATURES,activity_rates

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--activity',type=Path,required=True)
    parser.add_argument('--sm',type=int,required=True)
    parser.add_argument('--sample',type=int,help='Use this sample; default is the measured peak of this SM')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--frequency-mhz',type=float)
    mode.add_argument('--target-power-w',type=float)
    args=parser.parse_args()
    model=json.loads(args.model.read_text());assert model['features']==FEATURES
    coeff=np.array(model['coefficients'])
    selected=[]
    for chunk in pd.read_csv(args.activity,chunksize=10000):
        mask=chunk.sm_id==args.sm
        if args.sample is not None:mask &= chunk['sample']==args.sample
        if mask.any():selected.append(chunk.loc[mask])
    if not selected:parser.error('No matching SM/sample')
    frame=pd.concat(selected,ignore_index=True)
    row=frame.loc[frame.sm_power_w.idxmax()]
    rates=activity_rates(pd.DataFrame([row]))[0]
    k=float(rates@coeff[:-1]);b=float(row.idle_fraction*coeff[-1])
    if args.frequency_mhz is not None:
        if args.frequency_mhz<=0:parser.error('Frequency must be positive')
        frequency=args.frequency_mhz;power=b+k*frequency/1000
    else:
        if k<=0:parser.error('Activity has zero frequency slope; target cannot be uniquely inverted')
        frequency=1000*(args.target_power_w-b)/k;power=args.target_power_w
        if frequency<=0:parser.error('Target is below the fixed activity intercept')
    print(json.dumps(dict(sm_id=args.sm,sample=int(row['sample']),reference_frequency_mhz=row.frequency_hz/1e6,
                          k_w_per_ghz=k,b_w=b,frequency_mhz=frequency,predicted_power_w=power,
                          inside_observed_frequency_range=model['observed_frequency_range_mhz'][0]<=frequency<=model['observed_frequency_range_mhz'][1],
                          measured_reference_power_w=float(row.sm_power_w),
                          warning='Assumes the reference activity rates and idle fraction remain fixed. This is a conditional prediction, not a guarantee at a different frequency.'),indent=2))

if __name__=='__main__':main()
