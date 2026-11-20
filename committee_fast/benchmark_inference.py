"""Time full-graph inference only; no optimizer, training or graph generation."""
import gc
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'committee'), str(Path(__file__).resolve().parent)]
import torch
from common import atomic_json, reference, sha
from graphs import read_graph_manifest, load_inputs
from plan import matrix, settings
from runner import model_factory
from sparse_backend import accelerate, BACKEND


def main():
    rows = [r for r in matrix(settings()) if r['family'] == 'complement'
            and r['dataset'] == 'ieee' and r['seed'] == 42]
    graph = read_graph_manifest(rows[0]['graph_spec'])
    g, _ = load_inputs(rows[0]['graph_spec'], graph)
    g = g.cuda()
    records = []
    for row in rows:
        ref = reference(row)
        checkpoint = ROOT / ref['checkpoint_path']
        assert sha(checkpoint) == ref['checkpoint_sha256']
        for backend in (['original_edge_index', BACKEND] if row['model'] == 'gcn' else [BACKEND]):
            model = model_factory(row['model'], row['dataset'], g.x.shape[1]).cuda().eval()
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            if backend == BACKEND:
                model = accelerate(model, row['model'])
            times = []
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for repeat in range(4):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    output = model(g.x, g.edge_index)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)
                    assert torch.isfinite(output).all()
                    del output
            record = dict(model=row['model'], backend=backend, setup_forward_seconds=times[0],
                          warm_forward_seconds=times[1:], warm_median_seconds=statistics.median(times[1:]),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated())
            records.append(record)
            print(json.dumps(record), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    atomic_json(ROOT / 'results/committee_runtime_checks/inference_benchmark.json',
                dict(optimizer_steps=0, training=False, dataset='ieee', graph='sampled_complement',
                     graph_sha256=graph['effective_graph_sha256'], nodes=g.num_nodes,
                     edges=g.edge_index.shape[1], backend_sha256=sha(Path(__file__).with_name('sparse_backend.py')),
                     gpu=torch.cuda.get_device_name(), records=records,
                     limitation='Fixed pretrained weights; inference only. Does not estimate end-to-end training time.'))


if __name__ == '__main__':
    main()
