#!/usr/bin/env python3
"""Merge existing plain .traceg inputs into distinct streams, round robin."""
import argparse
import re
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', action='append', required=True, help='component=trace_directory; order defines stream IDs 1,2,...')
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
if len(a.input) < 2:
    p.error('at least two --input entries required')
inputs = []
for value in a.input:
    name, directory = value.split('=', 1)
    if not re.fullmatch(r'[A-Za-z0-9_]+', name):
        p.error('component names must contain only letters, digits and underscores')
    root = Path(directory).resolve()
    lines = (root / 'kernelslist.g').read_text().splitlines()
    if any(line.strip().startswith('kernel') and not line.strip().endswith('.traceg') for line in lines):
        p.error('only plain .traceg files supported')
    for line in lines:
        if line.strip().endswith('.traceg') and not (root / line.strip()).is_file():
            p.error(f'missing trace: {root / line.strip()}')
    inputs.append((name, root, lines))
if len({item[0] for item in inputs}) != len(inputs):
    p.error('component names must be unique')
if a.out.exists():
    p.error('output already exists; choose a fresh directory')
a.out.mkdir(parents=True)
all_chunks, tails = [], []
for stream, (name, root, lines) in enumerate(inputs, 1):
    chunks, pending = [], []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.endswith('.traceg'):
            dst = f'kernel-{name}-{Path(line).name.removeprefix("kernel-")}'
            text = (root / line).read_text()
            text, count = re.subn(r'^-cuda stream id = .*$', f'-cuda stream id = {stream}', text, flags=re.M)
            if count != 1:
                raise SystemExit(f'{line}: expected one CUDA stream header, got {count}')
            (a.out / dst).write_text(text)
            chunks.append(pending + [dst])
            pending = []
        else:
            pending.append(line)
    all_chunks.append(chunks)
    tails.extend(pending)
    print(f'stream {stream}: {name}, {len(chunks)} kernel launches')
combined = []
for index in range(max(map(len, all_chunks))):
    for chunks in all_chunks:
        if index < len(chunks):
            combined.extend(chunks[index])
(a.out / 'kernelslist.g').write_text('\n'.join(combined + tails) + '\n')
print(a.out.resolve() / 'kernelslist.g')
