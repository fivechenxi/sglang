"""Read-only, fail-closed patch preflight for an isolated runtime source tree."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root, role):
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for path, expected in manifest['sources'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Bundle source drift: ' + path)
    pending = []
    for row in manifest['patches']:
        if row['role'] != role:
            continue
        actual = digest(root/row['path'])
        if actual == row['after_sha256']:
            print('Already measured version:', row['path'])
        elif actual == row['before_sha256']:
            pending.append(str(ROOT/row['file']))
        else:
            raise ValueError('Unreviewed source version: ' + row['path'])
    if pending:
        subprocess.run(['git', '-C', str(root), 'apply', '--check', *pending], check=True)
    print('Preflight passed; no source was modified.')
    return pending


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--role', choices=['P', 'D'], required=True)
    args = parser.parse_args()
    check(args.source_root.resolve(), args.role)
