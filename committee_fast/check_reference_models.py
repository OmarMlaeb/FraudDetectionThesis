"""Check sparse inference against six verified August checkpoints; no training."""
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'committee'), str(Path(__file__).resolve().parent)]
import numpy as np
import torch
from common import AUDIT, atomic_json, read, sha, reference, frozen
from graphs import load_reference
from plan import matrix, settings
from runner import model_factory, metric_values
from sparse_backend import accelerate, BACKEND


def check():
    rows = [r for r in matrix(settings()) if r['family'] == 'complement' and r['seed'] == 42]
    checks = []
    for row in rows:
        ref = reference(row)
        g, mapping, _ = load_reference(row['graph_spec'])
        checkpoint = ROOT / ref['checkpoint_path']
        assert sha(checkpoint) == ref['checkpoint_sha256']
        g = g.to('cuda')
        model = model_factory(row['model'], row['dataset'], g.x.shape[1]).cuda().eval()
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        model = accelerate(model, row['model'])
        with torch.no_grad():
            probs = torch.sigmoid(model(g.x, g.edge_index)).cpu().numpy()
        record = dict(dataset=row['dataset'], model=row['model'], reference_run_id=ref['run_id'],
                      checkpoint_sha256=sha(checkpoint), graph_sha256=ref['feature_graph_sha256'], splits={})
        with np.load(AUDIT / ref['prediction_file']) as saved:
            for split in ['val', 'test']:
                mask = getattr(g, split+'_mask').cpu().numpy()
                current, previous = probs[mask], saved[split+'_probs']
                assert np.array_equal(mapping['node_ids'][mask], saved[split+'_ids'])
                np.testing.assert_allclose(current, previous, rtol=2e-4, atol=2e-6)
                threshold = float(ref['threshold'])
                changes = int(np.count_nonzero((current >= threshold) != (previous >= threshold)))
                old, new = [metric_values(saved[split+'_y'], p, threshold) for p in [previous, current]]
                record['splits'][split] = dict(max_probability_difference=float(np.max(np.abs(current-previous))),
                                               threshold_classification_changes=changes,
                                               metric_differences={k:new[k]-old[k] for k in old})
                assert changes == 0
            threshold, _ = frozen('metrics').find_best_threshold(saved['val_y'], probs[g.val_mask.cpu().numpy()], strategy='f1')
            assert threshold == float(ref['threshold'])
            record['validation_selected_threshold_unchanged'] = True
        checks.append(record)
        print(json.dumps(record), flush=True)
        del model, g
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(ROOT / 'results/committee_runtime_checks/reference_checkpoint_equivalence.json',
                dict(backend=BACKEND, backend_sha256=sha(Path(__file__).with_name('sparse_backend.py')),
                     passed=True, optimizer_steps=0, checks=checks))


if __name__ == '__main__':
    check()
