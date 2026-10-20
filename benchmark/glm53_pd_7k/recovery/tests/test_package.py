import hashlib
import importlib.util
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('recovery_preflight', ROOT / 'check_sources.py')
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)
MANIFEST = json.loads((ROOT / 'manifest.json').read_text())


class RecoveryPackageTests(unittest.TestCase):
    def test_bundle_hashes(self):
        self.assertGreater(len(MANIFEST['sources']), 6)
        for path, digest in MANIFEST['sources'].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest, path)

    def test_patch_paths_match_manifest(self):
        for name in dict.fromkeys(r['file'] for r in MANIFEST['patches']):
            paths = [r['path'] for r in MANIFEST['patches'] if r['file'] == name]
            text = (ROOT / name).read_text()
            self.assertEqual(re.findall(r'^--- a/(.+)$', text, re.M), paths)
            self.assertEqual(re.findall(r'^\+\+\+ b/(.+)$', text, re.M), paths)
            for path in paths:
                self.assertFalse(Path(path).is_absolute())
                self.assertNotIn('..', Path(path).parts)

    def fake_digest(self, state, mixed=False):
        lookup = {str(ROOT / p): v for p, v in MANIFEST['sources'].items()}
        lookup.update({str(Path('/runtime') / r['path']): r[state + '_sha256']
                       for r in MANIFEST['patches']})
        if mixed:
            row = MANIFEST['patches'][0]
            lookup[str(Path('/runtime') / row['path'])] = row['after_sha256']
        return lambda p: lookup[str(p)]

    def test_preflight_deduplicates_multifile_patch(self):
        with patch.object(preflight, 'digest', side_effect=self.fake_digest('before')), \
             patch.object(preflight.subprocess, 'run') as run:
            pending = preflight.check(Path('/runtime'), 'D')
        self.assertEqual(len(pending), 3)
        self.assertEqual(len(set(pending)), 3)
        run.assert_called_once()
        self.assertIn('--check', run.call_args.args[0])

    def test_mixed_multifile_patch_rejected(self):
        with patch.object(preflight, 'digest', side_effect=self.fake_digest('before', mixed=True)), \
             patch.object(preflight.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'Partially applied'):
                preflight.check(Path('/runtime'), 'D')
            run.assert_not_called()

    def test_already_applied_does_not_apply_again(self):
        with patch.object(preflight, 'digest', side_effect=self.fake_digest('after')), \
             patch.object(preflight.subprocess, 'run') as run:
            self.assertEqual(preflight.check(Path('/runtime'), 'D'), [])
            run.assert_not_called()

    def test_unknown_source_rejected_before_git(self):
        with patch.object(preflight, 'digest', return_value='unknown'), \
             patch.object(preflight.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                preflight.check(Path('/runtime'), 'Router')
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
