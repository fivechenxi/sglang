"""Read-only preflight for pinned recovery patches; reject drift and mixed patches."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root, role):
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for path, expected in manifest['sources'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Bundle drift: ' + path)
    groups = defaultdict(set)
    for row in manifest['patches']:
        if row['role'] != role:
            continue
        actual = digest(root / row['path'])
        if actual == row['after_sha256']:
            groups[row['file']].add('after')
        elif actual == row['before_sha256']:
            groups[row['file']].add('before')
        else:
            raise ValueError('Unreviewed runtime source: ' + row['path'])
    pending = []
    for name, states in groups.items():
        if len(states) != 1:
            raise ValueError('Partially applied multi-file patch: ' + name)
        if states == {'before'}:
            pending.append(str(ROOT / name))
    if pending:
        subprocess.run(['git', '-C', str(root), 'apply', '--check', *pending], check=True)
    print('Preflight passed; no files changed. Pending:', pending)
    return pending


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--role', choices=['D', 'Router'], required=True)
    args = p.parse_args()
    check(args.source_root.resolve(), args.role)
