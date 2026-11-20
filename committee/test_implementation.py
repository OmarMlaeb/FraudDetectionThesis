"""Synthetic tests and forward passes only. Never calls the real training loop."""
import contextlib
import io
import tempfile
import unittest
from unittest.mock import patch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from common import *
from plan import matrix, settings, families, graph_spec, training_config
from topology import canonical, symmetric, degree, complement, rewire, overlap, stats
from postprocess import signed_rank, holm
from runner import model_factory, check_completed


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.e = np.array([(i, i+1) for i in range(19)]+[(0, 19)], dtype=np.int64)
        self.e = self.e[np.lexsort((self.e[:, 1], self.e[:, 0]))]

    def test_complement_replaces_not_adds(self):
        a = complement(self.e, 40, 9876, 20)
        b = complement(self.e, 40, 9876, 20)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(a), 400)
        self.assertEqual(overlap(self.e, a, 40)['shared_edges'], 0)
        np.testing.assert_array_equal(canonical(symmetric(a), 40), a)
        self.assertEqual(stats(a, 40)['average_undirected_degree'], 20)

    def test_complement_impossible_fails(self):
        with self.assertRaises(ValueError):
            complement(self.e, 20, 0, 20)

    def test_rewire_exact_degrees_and_isolates(self):
        a, report = rewire(self.e, 24, 123, 10, 100)
        b, _ = rewire(self.e, 24, 123, 10, 100)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(degree(self.e, 24), degree(a, 24))
        self.assertEqual(report['successful_swaps'], 200)
        self.assertLess(report['jaccard'], 1)
        np.testing.assert_array_equal(canonical(symmetric(a), 24), a)

    def test_time_step_constraints(self):
        e = np.concatenate([self.e, self.e+20])
        strata = np.repeat([1, 2], 20)
        a, report = rewire(e, 40, 66, 5, 100, strata=strata)
        self.assertTrue(np.all(strata[a[:, 0]] == strata[a[:, 1]]))
        np.testing.assert_array_equal(degree(e, 40), degree(a, 40))
        self.assertEqual(len(report['strata']), 2)

    def test_window_constraint(self):
        times = np.arange(20)*1000
        a, _ = rewire(self.e, 20, 3, 5, 100, times=times, max_delta=19000)
        self.assertTrue(np.all(np.abs(times[a[:, 0]]-times[a[:, 1]]) <= 19000))
        with self.assertRaises(ValueError):
            rewire(self.e, 20, 3, times=times, max_delta=100)

    def test_clique_cannot_rewire_is_reported(self):
        clique = np.array([(i, j) for i in range(5) for j in range(i+1, 5)])
        a, report = rewire(clique, 6, 5, 10, 20)
        np.testing.assert_array_equal(a, clique)
        self.assertEqual(report['successful_swaps'], 0)
        self.assertEqual(report['attempts'], 200)
        self.assertTrue(report['limited_randomization'])

    def test_empty_graph(self):
        a, report = rewire(np.empty((0, 2), dtype=np.int64), 4, 3)
        self.assertEqual(len(a), 0)
        self.assertTrue(report['limited_randomization'])
        self.assertEqual(stats(a, 4)['components'], 4)

    def test_reject_directed_duplicate_loop(self):
        for bad in [np.array([[0], [1]]), np.array([[0], [0]]), np.array([[0, 1, 0, 1], [1, 0, 1, 0]])]:
            with self.assertRaises(ValueError):
                canonical(bad, 4)


class ModelAndPlanTests(unittest.TestCase):
    def test_matrix_and_complete_families(self):
        rows = matrix()
        self.assertEqual(len(rows), 300)
        self.assertEqual(len({r['run_id'] for r in rows}), 300)
        self.assertEqual({f: sum(r['family'] == f for r in rows) for f in ['complement', 'local', 'random']}, dict(complement=60, local=60, random=180))
        self.assertEqual(len({r['graph_id'] for r in rows}), 9)
        self.assertEqual(len(families()), 30)
        self.assertEqual(sum(len(f['comparisons']) for f in families()), 180)
        self.assertEqual(len({r['reference_run_id'] for r in rows}), 210)
        self.assertEqual(len(matrix(optional=True)), 360)

    def test_saved_reference_integrity_and_schema_mapping(self):
        verify_lock_files(['analysis/supervisor_2026-10-13/run_manifest.json'])
        mapping = lock()['feature_mapping']
        self.assertEqual(mapping['local_model_raw_indices'], list(range(2, 95)))
        self.assertEqual(mapping['excluded_aggregate_raw_indices'], list(range(95, 167)))
        self.assertEqual(len(mapping['local_loader_columns']), 93)
        self.assertEqual(len(mapping['excluded_loader_columns']), 72)
        self.assertEqual(training_config('transformer')['selection_metric'], 'pr_auc')
        self.assertEqual(training_config('transformer')['gradient_clip'], 1.0)

    def test_all_model_forward_shapes(self):
        torch.manual_seed(42)
        edge = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        with torch.no_grad():
            for dataset, dimension, models in [('elliptic', 93, MODELS), ('elliptic', 165, MODELS), ('ieee', 358, GNN)]:
                for name in models:
                    model = model_factory(name, dataset, dimension).eval()
                    x = torch.randn(4, 10, dimension) if name in ['lstm', 'transformer'] else torch.randn(4, dimension)
                    output = model(x, edge) if name in GNN else model(x)
                    self.assertEqual(tuple(output.shape), (4,), (dataset, name, dimension))
                    self.assertTrue(bool(torch.isfinite(output).all()))

    def test_sequence_all_targets_padding_unknown_mask(self):
        x = np.arange(14*93, dtype=np.float32).reshape(14, 93)
        y = np.arange(14) % 2
        mask = np.array([False, True, True, False]+[True]*10)
        ds = frozen('elliptic_sequence').EllipticSequenceDataset(x, y, mask, 10)
        selected = x[mask]
        self.assertEqual(len(ds), int(mask.sum()))
        for i in range(len(ds)):
            window, target = ds[i]
            expected = selected[max(0, i-9):i+1]
            expected = np.concatenate([np.repeat(expected[:1], 10-len(expected), axis=0), expected])
            np.testing.assert_array_equal(window.numpy(), expected)
            self.assertEqual(target.item(), y[mask][i])

    def test_cli_dry_parse_and_optional_isolation(self):
        from cli import parser
        p = parser().parse_args(['list', '--family', 'local', '--model', 'lstm', '--seed', '51'])
        self.assertEqual(p.action, 'list')
        self.assertFalse(p.retry_failed)


class StatisticsAndStorageTests(unittest.TestCase):
    def test_prediction_validation_accepts_matching_and_rejects_wrong_ids(self):
        import postprocess
        from runner import metric_values
        row = matrix()[0]
        y = np.array([0, 1, 0, 1], dtype=np.float32)
        probs = np.array([.1, .8, .3, .9], dtype=np.float32)
        threshold, vf1 = frozen('metrics').find_best_threshold(y, probs)
        ref = dict(run_id=row['reference_run_id'], prediction_file='reference.npz')
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            attempt = base/'runs'/row['family']/row['run_id']/'attempt_001'
            attempt.mkdir(parents=True)
            ck = attempt/'synthetic.pt'; ck.write_bytes(b'fixture only')
            np.savez_compressed(base/'reference.npz', val_ids=np.arange(4), val_y=y, test_ids=np.arange(4)+10, test_y=y)
            prediction_path = attempt/'predictions.npz'
            def save_predictions(wrong=False):
                np.savez_compressed(prediction_path, val_ids=np.arange(4), val_y=y, val_probs=probs,
                    test_ids=np.arange(4)+(11 if wrong else 10), test_y=y, test_probs=probs, threshold=threshold)
            save_predictions()
            m = {k: row[k] for k in ['run_id', 'seed', 'family', 'dataset', 'construction', 'model', 'graph_seed', 'feature_set', 'reference_run_id']}
            m.update(metric_values(y, probs, threshold))
            m.update(threshold=threshold, validation_f1_at_threshold=vf1, best_epoch=1, best_validation_score=m['pr_auc'],
                     checkpoint_sha256=sha(ck), input_dimension=358, validation_count=4, test_count=4, effective_graph_sha256='fixture')
            atomic_json(attempt/'metrics.json', m)
            graph = dict(config={}, topology={}, source_graph=artifact(ck), effective_graph_sha256='fixture')
            comp = dict(row=row, training=training_config(row['model']), code_hashes=code_hashes(), packages=lock()['packages'],
                        topology={}, graph_config={}, effective_graph_sha256='fixture', reference=ref)
            atomic_json(attempt/'config.json', {'compatibility': comp})
            def save_status():
                status = dict(status='complete', compatibility_sha256=digest(comp), config=artifact(attempt/'config.json'),
                              artifacts=dict(metrics=artifact(attempt/'metrics.json'), predictions=artifact(prediction_path), checkpoint=artifact(ck)))
                atomic_json(attempt.parent/'status.json', status)
            save_status()
            with patch.object(postprocess, 'OUT', base), patch.object(postprocess, 'AUDIT', base), \
                 patch.object(postprocess, 'read_graph_manifest', return_value=graph), patch.object(postprocess, 'reference', return_value=ref):
                accepted, report = postprocess.validate([row], write=False)
                self.assertTrue(report['complete'], report)
                self.assertEqual(len(accepted), 1)
                save_predictions(wrong=True); save_status()
                accepted, report = postprocess.validate([row], write=False)
                self.assertFalse(report['complete'])
                self.assertIn('IDs differ', report['runs'][0]['error'])

    def test_post_analysis_complete_families_on_synthetic_metrics(self):
        import postprocess
        rows = matrix()
        synthetic = []
        for row in rows:
            r = {k: row[k] for k in ['run_id', 'family', 'dataset', 'construction', 'model', 'seed', 'graph_seed', 'feature_set', 'reference_run_id']}
            r.update({k: .5+row['seed']/10000 for k in METRICS})
            r.update(dict(true_negative=100, false_positive=5, false_negative=5, true_positive=20, threshold=.5))
            synthetic.append(r)
        with tempfile.TemporaryDirectory() as d, patch.object(postprocess, 'OUT', Path(d)), \
             patch.object(postprocess, 'validate', return_value=(synthetic, {'complete': True})), \
             patch.object(postprocess, 'reference', side_effect=lambda row: dict({m: '0.4' for m in METRICS}, run_id=row['reference_run_id'])), \
             contextlib.redirect_stdout(io.StringIO()):
            postprocess.analyze(settings())
            tests = read(Path(d)/'analysis/wilcoxon_holm.json')
            self.assertEqual(len(tests), 180)
            self.assertTrue(all(t['n_pairs'] == 10 for t in tests))
            self.assertTrue(all(t['p_raw'] == 2/1024 for t in tests))
            self.assertEqual(len(read(Path(d)/'analysis/random_per_seed.json')), 180)
            self.assertEqual(len(read(Path(d)/'analysis/local_summary.json')), 6)

    def test_post_analysis_missing_has_null_p_and_keeps_all_hypotheses(self):
        import postprocess
        with tempfile.TemporaryDirectory() as d, patch.object(postprocess, 'OUT', Path(d)), \
             patch.object(postprocess, 'validate', return_value=([], {'complete': False})), contextlib.redirect_stdout(io.StringIO()):
            postprocess.analyze(settings())
            tests = read(Path(d)/'analysis/wilcoxon_holm.json')
            self.assertEqual(len(tests), 180)
            self.assertTrue(all(t['p_raw'] is None and t['p_holm'] is None for t in tests))
            self.assertTrue(all(r['n_seeds'] == 0 and not r['complete'] for r in read(Path(d)/'analysis/complement_summary.json')))

    def test_exact_sign_rank_no_ties_matches_scipy(self):
        from scipy.stats import wilcoxon
        d = np.array([-1, 2, 3, -4, 5, 6, 7, 8, -9, 10.])
        self.assertEqual(signed_rank(d)['p_raw'], wilcoxon(d, method='exact').pvalue)

    def test_ties_zero_and_holm_full_family(self):
        self.assertEqual(signed_rank([0]*10)['p_raw'], 1)
        self.assertEqual(signed_rank([1]*10)['p_raw'], 2/1024)
        self.assertEqual(signed_rank([0, 1, 1])['p_raw'], .5)
        np.testing.assert_allclose(holm([.01, .03, .04]), [.03, .06, .06])
        adjusted = holm([.01, None, .04])
        self.assertEqual(adjusted, [.03, None, .08])

    def test_atomic_json_preserves_old_on_serialization_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'result.json'
            atomic_json(path, {'a': 1})
            with self.assertRaises(ValueError):
                atomic_json(path, {'a': float('nan')})
            self.assertEqual(read(path), {'a': 1})

    def test_resume_failure_and_no_checkpoint_overwrite_with_mocked_loop(self):
        import runner
        row = matrix()[0]
        fake_graph = dict(config={}, topology={}, source_graph={}, effective_graph_sha256='synthetic')
        fake_env = dict(code_hashes={}, packages={}, python='test', gpu='test', cuda='test', cudnn='test')
        with tempfile.TemporaryDirectory() as d, patch.object(runner, 'OUT', Path(d)), \
             patch.object(runner, 'read_graph_manifest', return_value=fake_graph), \
             patch.object(runner, 'environment', return_value=fake_env), \
             patch.object(runner, 'reference', return_value={}), \
             patch('torch.cuda.is_available', return_value=True), contextlib.redirect_stdout(io.StringIO()):
            with patch.object(runner, 'train_loop', side_effect=RuntimeError('SYNTHETIC failure; no training executed')):
                with self.assertRaises(RuntimeError):
                    runner.train_one(row, settings())
            directory = Path(d)/'runs'/row['family']/row['run_id']
            self.assertEqual(read(directory/'status.json')['status'], 'failed')
            with self.assertRaises(ValueError):
                runner.train_one(row, settings())
            def fake_loop(row, graph, attempt):
                ck = attempt/'synthetic_checkpoint.pt'
                ck.write_bytes(b'synthetic fixture, NOT weights')
                (attempt/'predictions.npz').write_bytes(b'synthetic fixture')
                (attempt/'training.jsonl').write_text('synthetic fixture')
                return {'fixture': True}, ck
            with patch.object(runner, 'train_loop', side_effect=fake_loop) as mocked:
                self.assertEqual(runner.train_one(row, settings(), retry_failed=True), 'complete')
                self.assertEqual(runner.train_one(row, settings(), retry_failed=True), 'skipped')
                self.assertEqual(mocked.call_count, 1)
            self.assertEqual(len(list(directory.glob('attempt_*'))), 2)
            status = read(directory/'status.json')
            check_completed(status)
            with self.assertRaises(ValueError):
                check_completed(status, {'different': 'configuration'})
            Path(status['artifacts']['checkpoint']['path']).write_bytes(b'corruption')
            with self.assertRaises(ValueError):
                check_completed(status)


if __name__ == '__main__':
    unittest.main(verbosity=2)
