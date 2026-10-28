"""Isolated event fixtures. Never calls SSH, Codex, or an experiment worker."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest


watch = runpy.run_path(str(Path(__file__).with_name('wake_codex_on_completion.py')))


class DiskEventChecks(unittest.TestCase):
    def run_probe(self, available=9, threshold=8, terminal=False, df_error=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ['launch.json', 'followup-launch.json']:
                (root / filename).write_text(json.dumps({'pid': os.getpid()}))
            if terminal:
                (root / 'followup-completed.json').write_text(json.dumps({'finished_at': 'fixture', 'results': {'search': {}}}))
            calls = []

            def fake_df(argv, **kwargs):
                calls.append(argv)
                if df_error:
                    raise subprocess.CalledProcessError(1, argv)
                return subprocess.CompletedProcess(argv, 0, f'1B-blocks Used Avail\n150 140 {available}\n')

            output = []
            from unittest.mock import patch
            # probe_script executes on Linux in production. Do not send its
            # POSIX signal-0 liveness probe to a Windows fixture process.
            with patch('subprocess.run', fake_df), patch('os.kill', return_value=None):
                exec(watch['probe_script'](str(root), '/data', threshold), {'print': output.append})
            return json.loads(output[0]), calls

    def test_default_does_not_query_disk(self):
        value, calls = self.run_probe(threshold=None)
        self.assertFalse(value['terminal'])
        self.assertEqual(calls, [])
        self.assertNotIn('disk', value)

    def test_df_low_space_is_attention_not_completion(self):
        value, calls = self.run_probe(available=7)
        self.assertFalse(value['terminal'])
        self.assertTrue(value['requires_attention'])
        self.assertEqual(value['reason'], 'low_disk_space')
        self.assertEqual(value['experiment_reason'], 'waiting')
        self.assertEqual(value['disk']['available_bytes'], 7)
        self.assertEqual(calls, [['df', '-B1', '--output=size,used,avail', '/data']])

    def test_sufficient_space_keeps_waiting(self):
        value, _ = self.run_probe(available=8)
        self.assertEqual(value['reason'], 'waiting')
        self.assertNotIn('requires_attention', value)

    def test_terminal_has_precedence_over_warning(self):
        value, _ = self.run_probe(available=0, terminal=True)
        self.assertTrue(value['terminal'])
        self.assertEqual(value['reason'], 'batch_finished')
        self.assertNotIn('requires_attention', value)

    def test_df_error_does_not_hide_completion(self):
        value, _ = self.run_probe(terminal=True, df_error=True)
        self.assertTrue(value['terminal'])
        self.assertEqual(value['reason'], 'batch_finished')
        self.assertIn('disk_error', value)


if __name__ == '__main__':
    unittest.main()
