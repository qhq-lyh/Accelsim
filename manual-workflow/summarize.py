#!/usr/bin/env python3
"""Summarize final simulator metrics and the thermal-mapped average power."""
import csv
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
freq = float(sys.argv[2])
text = (root / 'sim.log').read_text(errors='ignore')
def last(key):
    values = re.findall(r'\b' + key + r'\s*=\s*([0-9.eE+-]+)', text)
    if not values:
        raise SystemExit(f'missing simulator metric: {key}')
    return float(values[-1])
with (root / 'power.ptrace').open() as file:
    names = file.readline().split()
    sums, count = [0.] * len(names), 0
    for line in file:
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != len(names):
            raise SystemExit('malformed power row')
        sums = [x + y for x, y in zip(sums, values)]
        count += 1
if not count:
    raise SystemExit('power trace has no samples')
with (root / 'mapped_average_power.csv').open('w') as file:
    writer = csv.writer(file)
    writer.writerow(['block', 'average_watts'])
    writer.writerows((name, value / count) for name, value in zip(names, sums))
result = dict(freq_mhz=freq, total_cycles=last('gpu_tot_sim_cycle'),
              total_instructions=last('gpu_tot_sim_insn'), ipc=last('gpu_tot_ipc'),
              thermal_mapped_average_watts=sum(sums) / count, power_samples=count)
result['simulated_time_us'] = result['total_cycles'] / freq
(root / 'performance.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
