# Faster execution with the same experiment settings

From the repository terminal, run:

```powershell
python run_sampled_complement_fast.py
```

All ten seeds, models, full graphs, feature columns, losses, optimizer settings,
100-epoch GNN limit, patience, validation selection and thresholds are unchanged.
The completed original results are reused after their normal provenance checks.
An interrupted run restarts from its seed because the old checkpoints contain
model weights but not the optimizer/RNG state required for an exact continuation.

New checkpoints and combined old-format CSVs are in
`results/committee_fast_experiments/`. Original result files are untouched.
The companion execution-provenance CSV identifies each run's implementation;
combined summaries are explicitly a mixture of verified execution backends when
completed original runs are reused. This is not a claim of identical training
trajectories. Small rounding differences can affect validation ranking, checkpoint
selection and subsequent training. New sparse runs are a later code change, not
original August implementation outputs.

## What was changed

GCN normalizes its fixed graph once using the original PyG normalization and
self-loop rules, then uses sparse CSR multiplication. GraphSAGE uses sparse mean
aggregation. GAT retains the original feature transform, attention calculation,
edge ordering, softmax, attention dropout, self-loops and node dropout. Only its
weighted neighbor sum is replaced by sparse multiplication. Its backward pass
uses the same analytic derivatives with bounded edge chunks to avoid materializing
an enormous edge-by-feature tensor. No neighborhood sampling or mixed precision
is introduced. MLP, LSTM and Transformer use their original computation.

This addresses the large temporary tensors created by gather/scatter aggregation
on the complement's 11,810,800 directed IEEE-CIS edges. The memory issue is described
in the [PyG sparse aggregation documentation](https://pytorch-geometric.readthedocs.io/en/latest/advanced/sparse_tensor.html).

The frozen `committee/*.py` files are unchanged. `runtime.py` installs scoped
adapters in the fast process; it calls the original training loop. Original results
are checked against their original code hashes; sparse results require those
hashes plus the fast adapter hashes and immutable verification evidence. The
original graph manifests/hashes remain authoritative. The original workspace lock
prevents simultaneous use of the old and fast launchers.

## Verification and its limits

Reports are in `results/committee_runtime_checks/`:

- `synthetic_equivalence.json`: forward outputs, input/parameter gradients, and
  identical-RNG dropout checks for all three GNNs and both dataset implementations;
  includes isolated nodes and finite-difference checks of GAT's custom derivatives.
- `saved_checkpoint_equivalence.json`: all six completed complement GCN checkpoints
  against their saved validation/test predictions, IDs, metrics and thresholds.
- `reference_checkpoint_equivalence.json`: GCN, GraphSAGE and GAT seed 42 checkpoints
  on the verified IEEE-CIS main and native Elliptic graphs.

These checks use **no optimizer steps**. Inference checks establish agreement at
fixed saved weights, not identical future optimization trajectories. Probability
agreement uses `rtol=2e-4, atol=2e-6`, and the checks additionally require zero
threshold classification changes and identical validation-selected thresholds.
Small AP differences from float ties are recorded, not hidden.

The old 100-epoch IEEE-CIS complement GCN runs usually took about 44 minutes.
On this RTX 3080 Ti, `inference_benchmark.json` records these full-complement
forward passes (median of three warm calls, after a separate setup call):

| Implementation | Seconds per forward | Peak allocated GPU memory |
|---|---:|---:|
| Original GCN | 10.977 | 13.88 GiB |
| Sparse GCN | 0.0266 | 2.57 GiB |
| Sparse GraphSAGE | 0.615 | 2.83 GiB |
| Sparse GAT | 0.0361 | 2.54 GiB |

The original GCN allocation exceeded the GPU's 12 GiB physical capacity. Avoiding
those large temporary tensors removes that source of memory pressure. These
measurements are **not training-time predictions**: backward costs, metrics and
checkpoint writes are excluded. Actual epoch times should be assessed when you
launch the remaining runs. Full training was not run by Codex.

To repeat checks without training:

```powershell
python committee_fast/check_backend.py
python committee_fast/check_saved.py
python committee_fast/check_reference_models.py
python committee_fast/benchmark_inference.py
python -m unittest discover -s committee_fast -p 'test_*.py' -v
```

Use the same `_fast.py` commands with `--dry-run`, `--graphs-only` or
`--summarize-only` as needed. Keep the original and new folders together when
sharing results: the new CSVs reference the existing graphs and completed runs.
