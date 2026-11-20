"""Routing and preservation checks; no training or graph generation."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runtime


class RuntimeTests(unittest.TestCase):
    def test_scope_restores_original_modules(self):
        old = runtime.runner.model_factory, runtime.postprocess.validate, runtime.simple_runner.OUT
        with runtime.installed():
            self.assertEqual(runtime.runner.OUT, runtime.FAST_OUT)
            self.assertNotEqual(runtime.runner.model_factory, old[0])
            # Graph provenance continues to use the existing artifacts.
            import graphs
            self.assertEqual(graphs.OUT, runtime.ORIGINAL_OUT)
        self.assertEqual((runtime.runner.model_factory, runtime.postprocess.validate, runtime.simple_runner.OUT), old)

    def test_dry_run_never_trains_generates_or_checks_environment(self):
        with patch.object(runtime.runner, 'train_one', side_effect=AssertionError('training called')), \
             patch.object(runtime.simple_runner, 'generate', side_effect=AssertionError('generation called')), \
             patch.object(runtime.common, 'environment', side_effect=AssertionError('environment called')):
            for family in ['complement', 'local', 'random']:
                self.assertEqual(runtime.run(family, ['--dry-run']), 0)

    def test_original_completed_run_is_reused_without_training(self):
        with tempfile.TemporaryDirectory() as folder:
            old, new = Path(folder)/'old', Path(folder)/'new'
            row = dict(run_id='a', family='complement', model='gcn', seed=42)
            p = old/'runs/complement/a/status.json'
            p.parent.mkdir(parents=True)
            config = p.with_name('config.json')
            config.write_text(json.dumps(dict(compatibility={})))
            p.write_text(json.dumps(dict(status='complete', config=dict(path=str(config)))))

            def verify(rows, write=False):
                self.assertEqual(runtime.postprocess.OUT, old)
                self.assertIs(runtime.postprocess.code_hashes, runtime.common.code_hashes)
                return [dict(row)], dict(runs=[dict(run_id='a', family='complement', status='accepted')])

            with patch.object(runtime, 'ORIGINAL_OUT', old), patch.object(runtime, 'FAST_OUT', new), \
                 patch.object(runtime.postprocess, 'validate', side_effect=verify), \
                 patch.object(runtime.runner, 'train_one', side_effect=AssertionError('must skip training')):
                before = p.read_bytes()
                with runtime.installed():
                    self.assertEqual(runtime.simple_runner.train_one(row, {}, True), 'skipped')
                self.assertEqual(p.read_bytes(), before)
                self.assertFalse(new.exists())

    def test_failed_original_restarts_only_in_separate_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            old, new = Path(folder)/'old', Path(folder)/'new'
            row = dict(run_id='b', family='complement')
            p = old/'runs/complement/b/status.json'
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps(dict(status='failed')))
            def train(*args, **kwargs):
                self.assertEqual(runtime.runner.OUT, new)
                self.assertTrue(kwargs['retry_failed'])
                return 'complete'
            with patch.object(runtime, 'ORIGINAL_OUT', old), patch.object(runtime, 'FAST_OUT', new), \
                 patch.object(runtime.runner, 'train_one', side_effect=train):
                before = p.read_bytes()
                with runtime.installed():
                    self.assertEqual(runtime.simple_runner.train_one(row, {}, True), 'complete')
                self.assertEqual(p.read_bytes(), before)

    def test_duplicate_complete_versions_are_not_silently_selected(self):
        with tempfile.TemporaryDirectory() as folder:
            old, new = Path(folder)/'old', Path(folder)/'new'
            for base in [old, new]:
                p = base/'runs/complement/a/status.json'
                p.parent.mkdir(parents=True)
                p.write_text(json.dumps(dict(status='complete')))
            with patch.object(runtime, 'ORIGINAL_OUT', old), patch.object(runtime, 'FAST_OUT', new):
                with self.assertRaisesRegex(ValueError, 'Two completed versions'):
                    runtime.source_for(dict(run_id='a', family='complement'))

    def test_fast_completed_run_requires_fast_hashes_and_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            old, new = Path(folder)/'old', Path(folder)/'new'
            row = dict(run_id='c', family='complement', model='gat', seed=42)
            p = new/'runs/complement/c/status.json'
            p.parent.mkdir(parents=True)
            evidence = p.with_name('evidence.json')
            evidence.write_text(json.dumps(dict(passed=True, backend_sha256=runtime.common.sha(Path(runtime.__file__).with_name('sparse_backend.py')))))
            config = p.with_name('config.json')
            config.write_text(json.dumps(dict(compatibility=dict(execution_backend=runtime.BACKEND,
                                               sparse_verification=[runtime.common.artifact(evidence)]))))
            p.write_text(json.dumps(dict(status='complete', config=dict(path=str(config)))))
            def verify(rows, write=False):
                self.assertEqual(runtime.postprocess.OUT, new)
                self.assertIs(runtime.postprocess.code_hashes, runtime.fast_hashes)
                return [dict(row)], dict(runs=[dict(run_id='c', family='complement', status='accepted')])
            with patch.object(runtime, 'ORIGINAL_OUT', old), patch.object(runtime, 'FAST_OUT', new), \
                 patch.object(runtime.postprocess, 'validate', side_effect=verify):
                with runtime.installed():
                    accepted, report = runtime.simple_runner.validate([row], write=False)
                    self.assertEqual(accepted[0]['execution_backend'], runtime.BACKEND)
                    evidence.write_text('{}')
                    accepted, report = runtime.simple_runner.validate([row], write=False)
                    self.assertEqual(accepted, [])
                    self.assertEqual(report['runs'][0]['status'], 'invalid')


if __name__ == '__main__':
    unittest.main()
