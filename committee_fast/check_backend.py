"""Synthetic forward/backward checks only: no optimizer and no training runs."""
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'committee'), str(Path(__file__).resolve().parent)]
import torch
from common import atomic_json, sha
from runner import model_factory
from sparse_backend import accelerate, BACKEND, EdgeLayout, AttentionSum


def check():
    # Finite-difference check of the custom GAT derivatives on a directed graph.
    small_edges = torch.tensor([[0, 1, 3, 2, 1], [1, 2, 1, 0, 1]])
    layout = EdgeLayout(small_edges, 5)
    gx = torch.randn(5, 2, 3, dtype=torch.double, requires_grad=True)
    ga = torch.randn(5, 2, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, a: AttentionSum.apply(x, a, layout), (gx, ga))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(811)
    n = 3000
    edges = torch.randint(n - 1, (2, 30000), device=device)
    edges = edges[:, edges[0] != edges[1]]
    edges = torch.cat([edges, edges.flip(0)], dim=1)
    edges = torch.unique(edges, dim=1)  # Symmetric; last node is isolated.
    output = []
    for dataset in ['ieee', 'elliptic']:
        for name in ['gcn', 'sage', 'gat']:
            torch.manual_seed(812)
            dimension = 358 if dataset == 'ieee' else 93
            original = model_factory(name, dataset, dimension).to(device)
            fast = accelerate(copy.deepcopy(original), name)
            assert list(original.state_dict()) == list(fast.state_dict())
            x = torch.randn(n, dimension, device=device)
            for training in [False, True]:
                outputs, gradients, timings, peaks = [], [], [], []
                for model in [original, fast]:
                    model.train(training)
                    model.zero_grad(set_to_none=True)
                    inp = x.detach().clone().requires_grad_(True)
                    # The same RNG state preserves the node dropout masks.
                    torch.manual_seed(913)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    pred = model(inp, edges)
                    pred.square().mean().backward()
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    timings.append(time.perf_counter() - start)
                    peaks.append(torch.cuda.max_memory_allocated() if device == 'cuda' else None)
                    outputs.append(pred.detach())
                    gradients.append([inp.grad] + [p.grad for p in model.parameters()])
                torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-4, atol=2e-6)
                for before, after in zip(*gradients):
                    torch.testing.assert_close(before, after, rtol=3e-3, atol=2e-6)
                result = dict(dataset=dataset, model=name, training_mode=training,
                              max_output_difference=float((outputs[0]-outputs[1]).abs().max()),
                              max_gradient_difference=max(float((a-b).abs().max()) for a,b in zip(*gradients)),
                              original_seconds=timings[0], sparse_seconds=timings[1],
                              original_peak_bytes=peaks[0], sparse_peak_bytes=peaks[1])
                output.append(result)
                print(json.dumps(result), flush=True)
    report = dict(backend=BACKEND, backend_sha256=sha(Path(__file__).with_name('sparse_backend.py')),
                  device=device, passed=True, optimizer_steps=0, synthetic_nodes=n,
                  caution='Synthetic timings include cold start and adjacency setup; not full-run speed estimates.',
                  attention_finite_difference_gradcheck=True, checks=output)
    atomic_json(ROOT / 'results/committee_runtime_checks/synthetic_equivalence.json', report)
    return report


if __name__ == '__main__':
    check()
