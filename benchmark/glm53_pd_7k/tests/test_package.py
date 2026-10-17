import hashlib
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import check_sources

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT/'manifest.json').read_text())


class PackageTests(unittest.TestCase):
    def test_exact_runtime_and_kernel_source_bytes(self):
        for name, expected in MANIFEST['sources'].items():
            with self.subTest(path=name):
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), expected)

    def test_patch_paths_are_confined(self):
        for row in MANIFEST['patches']:
            self.assertFalse(Path(row['path']).is_absolute())
            self.assertNotIn('..', Path(row['path']).parts)
            delta = (ROOT/row['file']).read_text()
            self.assertEqual(re.findall(r'^--- a/(.+)$', delta, re.M), [row['path']])
            self.assertEqual(re.findall(r'^\+\+\+ b/(.+)$', delta, re.M), [row['path']])

    def test_source_drift_stops_before_git(self):
        with patch.object(check_sources, 'digest', return_value='unreviewed'), \
             patch.object(check_sources.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                check_sources.check(Path('/unused'), 'P')
            run.assert_not_called()

    def test_aggregate_accounting(self):
        result = json.loads((ROOT/'results/qualification.json').read_text())
        self.assertEqual(result['gpu_denominator'], 40)
        self.assertEqual(result['completed_requests'], 4144)
        self.assertEqual(result['errors'], 0)
        for row in result['runs']:
            window = row['window_latency']
            self.assertAlmostEqual(window['total_tps_per_gpu'], window['completed']*82520/420/40)
            self.assertAlmostEqual(window['fresh_input_tps'], window['completed']*8192/420)
            self.assertAlmostEqual(window['output_tps'], window['completed']*600/420)
            self.assertEqual(len(row['prefix_groups']), 4)
            self.assertLessEqual(row['conservative_total_tps_per_gpu'], window['total_tps_per_gpu'])

    def test_router_guards_exist_before_header_dispatch(self):
        delta = (ROOT/'patches/router-stream-load.patch').read_text()
        added = '\n'.join(s[1:] for s in delta.splitlines() if s.startswith('+') and not s.startswith('+++'))
        self.assertIn('let _prefill_guard = WorkerLoadGuard::new(prefill.clone(), headers)', added)
        self.assertIn('let _decode_guard = WorkerLoadGuard::new(decode.clone(), headers)', added)
        self.assertNotIn('!context.is_stream', added)


if __name__ == '__main__':
    unittest.main()
