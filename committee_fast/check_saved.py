"""Compare sparse inference with completed complement predictions; never train."""
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'committee'), str(Path(__file__).resolve().parent)]
import numpy as np
import torch
from common import OUT, atomic_json, read, sha, frozen
from graphs import load_inputs, read_graph_manifest
from plan import matrix, settings
from postprocess import validate
from runner import model_factory, metric_values
from sparse_backend import accelerate, BACKEND


def check():
    rows = [r for r in matrix(settings()) if r['family'] == 'complement'
            and r['dataset'] == 'ieee' and r['model'] == 'gcn']
    accepted, validation = validate(rows, write=False)
    ids = {r['run_id'] for r in accepted}
    rows = [r for r in rows if r['run_id'] in ids]
    if not rows:
        raise RuntimeError('No verified completed checkpoints to compare.')
    graph = read_graph_manifest(rows[0]['graph_spec'])
    g, mapping = load_inputs(rows[0]['graph_spec'], graph)
    g = g.to('cuda')
    checks = []
    for index, row in enumerate(rows):
        status = read(OUT / 'runs/complement' / row['run_id'] / 'status.json')
        metrics = read(status['artifacts']['metrics']['path'])
        model = model_factory('gcn', 'ieee', g.x.shape[1]).cuda().eval()
        model.load_state_dict(torch.load(status['artifacts']['checkpoint']['path'], weights_only=True))
        original_seconds = None
        if index == 0:
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                baseline = torch.sigmoid(model(g.x, g.edge_index)).cpu().numpy()
            torch.cuda.synchronize()
            original_seconds = time.perf_counter() - start
        model = accelerate(model, 'gcn')
        times = []
        for repeat in range(2):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.no_grad():
                probs = torch.sigmoid(model(g.x, g.edge_index)).cpu().numpy()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        record = dict(seed=row['seed'], run_id=row['run_id'], checkpoint=status['artifacts']['checkpoint'],
                      original_forward_seconds=original_seconds,
                      sparse_first_forward_seconds=times[0], sparse_cached_forward_seconds=times[1],
                      sparse_peak_bytes=torch.cuda.max_memory_allocated(), splits={})
        with np.load(status['artifacts']['predictions']['path']) as saved:
            for split in ['val', 'test']:
                mask = getattr(g, split+'_mask').cpu().numpy()
                current, previous = probs[mask], saved[split+'_probs']
                assert np.array_equal(mapping['node_ids'][mask], saved[split+'_ids'])
                np.testing.assert_allclose(current, previous, rtol=2e-4, atol=2e-6)
                old = metric_values(saved[split+'_y'], previous, metrics['threshold'])
                new = metric_values(saved[split+'_y'], current, metrics['threshold'])
                changes = int(np.count_nonzero((current >= metrics['threshold']) != (previous >= metrics['threshold'])))
                record['splits'][split] = dict(max_probability_difference=float(np.max(np.abs(current-previous))),
                                               threshold_classification_changes=changes,
                                               metric_differences={k:new[k]-old[k] for k in old})
                assert changes == 0
            threshold, vf1 = frozen('metrics').find_best_threshold(saved['val_y'], probs[g.val_mask.cpu().numpy()], strategy='f1')
            assert threshold == metrics['threshold']
            record['validation_selected_threshold_unchanged'] = True
        if index == 0:
            record['fresh_original_max_probability_difference'] = float(np.max(np.abs(baseline-probs)))
        checks.append(record)
        print(json.dumps(record), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    report = dict(backend=BACKEND, backend_sha256=sha(Path(__file__).with_name('sparse_backend.py')),
                  passed=True, optimizer_steps=0, completed_checkpoints=len(checks),
                  graph_sha256=graph['effective_graph_sha256'], checks=checks,
                  limitation='Inference equivalence does not prove bitwise-identical future training trajectories.')
    atomic_json(ROOT / 'results/committee_runtime_checks/saved_checkpoint_equivalence.json', report)


if __name__ == '__main__':
    check()
