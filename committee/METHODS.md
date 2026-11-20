# Committee additional experiments: detailed methods

For the simple three-command workflow, see [README.md](README.md). The commands below describe the optional advanced interface. The simple scripts prepare inputs, train, and save CSVs automatically.

Implementation date: 13 October 2026. **Training and full-dataset graph generation have NOT been run by Codex.** Only inspection of existing artifacts, synthetic tests, model forward passes and dry planning are authorized during implementation. The user executes the generation and training commands below.

All new experiment outputs go to `results/committee_additional_experiments/`. Original source, datasets, checkpoints, caches, thesis and `analysis/supervisor_2026-10-13/` are read-only inputs. No existing training entry point is invoked. The new runner imports frozen, side-effect-free architecture, sequence and metric modules copied from the August source snapshot.

## Experiment matrix

| Family | Dataset / reference construction | Models | Features | Graph seed | Training seeds | Runs |
|---|---|---|---|---|---|---:|
| Complement | IEEE-CIS main | GCN, GraphSAGE, GAT | All 358 | 20261013 | 42–51 | 30 |
| Complement | Elliptic native | GCN, GraphSAGE, GAT | All 165 | 20261013 | 42–51 | 30 |
| Local only | Elliptic native | MLP, LSTM, Transformer, GCN, GraphSAGE, GAT | Local 93 | None: native graph reused | 42–51 | 60 |
| Degree preserving | IEEE-CIS main, card, address, email_domain, identifier_time_window-24h | GCN, GraphSAGE, GAT | All 358 | 20261013, separately applied to each reference | 42–51 | 150 |
| Degree preserving | Elliptic native | GCN, GraphSAGE, GAT | All 165 | 20261013 | 42–51 | 30 |
| **Required total** | | | | | | **300** |

There are nine prepared input artifacts: two complements, six rewired graphs, and one native-topology/local-feature specification. All 30 GNN runs for a particular generated topology share exactly the same topology file. Training seed and graph seed are separate fields. The ten training seeds measure training variability conditional on one graph realization, not uncertainty across random graphs. Optional matched full-feature Elliptic reruns add 60 runs and are excluded from the 300-run default plan.

## Exact PowerShell commands

Run from a PowerShell terminal. Use this actual working interpreter: the old `venv/Scripts/python.exe` points to a missing Python installation. The portable Python 3.10.0 below loads the existing venv packages. No installation or environment upgrade is required.

```powershell
Set-Location -LiteralPath 'C:\Users\user\Desktop\FraudDetectionThesis'
$py = 'C:\Users\user\Desktop\FraudDetectionThesis\analysis\supervisor_2026-10-13\runtime\python310\python.exe'
$env:CUDA_VISIBLE_DEVICES = '0'
& $py committee/cli.py preflight
& $py committee/test_implementation.py
& $py committee/cli.py list
```

`preflight` requires Python 3.10.0, torch 2.5.1+cu121, torch_geometric 2.7.0, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3 and scikit-learn 1.7.2. GPU information is recorded. Training requires CUDA and never silently falls back to CPU. `list` loads no graphs and trains nothing. CLI model spelling is `sage` for GraphSAGE. All commands default to graph seed 20261013, 10 successful swaps per original edge, and a maximum of 100 attempts per original edge.

Generate/prepare inputs explicitly. These commands can be run separately and are safe to repeat if existing manifests and hashes match:

```powershell
& $py committee/cli.py generate --family complement
if ($LASTEXITCODE -ne 0) { throw 'Complement preparation failed; inspect the failure report.' }
& $py committee/cli.py generate --family local
if ($LASTEXITCODE -ne 0) { throw 'Local-feature preparation failed; inspect the failure report.' }
& $py committee/cli.py generate --family random
if ($LASTEXITCODE -ne 0) { throw 'Random graph preparation failed; inspect the failure report.' }
```

Or prepare all nine artifacts with `& $py committee/cli.py generate --family all`. Inspect `graphs/*/manifest.json` and `graphs/*/attempt_*/statistics.json`, particularly any limited-randomization flags, before interpreting training results. Generation does not train. Training does not generate missing graphs.

One genuine training run for the initial user check (not executed during implementation):

```powershell
& $py committee/cli.py train --family local --dataset elliptic --model mlp --seed 42
```

Each entire experiment family:

```powershell
& $py committee/cli.py train --family complement --retry-failed --continue-on-error
& $py committee/cli.py train --family local --retry-failed --continue-on-error
& $py committee/cli.py train --family random --retry-failed --continue-on-error
```

Sequential execution of all 300 runs on the selected GPU, without background processes or concurrent models:

```powershell
& $py committee/cli.py train --family all --retry-failed --continue-on-error
```

Resume after interruption/failure by repeating the same command. Each completed run is skipped only after configuration, code, graph, checkpoint, prediction, metric and log hash checks. An incomplete run restarts from its original seed in a new attempt directory; this is **run-level resumption**, not optimizer-state continuation. Previously selected checkpoints and failed attempts are retained. OOM or other failures do not change any dimensions, features, edges or batches. The process returns a nonzero exit code if any run fails; see `last_runner_failures.json` and each attempt's failure log for an exact rerun command.

Example targeted retry:

```powershell
& $py committee/cli.py train --family random --dataset ieee --construction identifier_time_window-24h --model gat --seed 51 --retry-failed
```

An exclusive `.active.lock` prevents simultaneous mutation commands. A normally interrupted run releases it. If the Python process was killed, inspect the recorded PID and ensure it has ended before explicitly removing the stale lock:

```powershell
Get-Content -LiteralPath 'results/committee_additional_experiments/.active.lock'
& $py committee/cli.py unlock --confirm-no-running-process
```

After training, run validation, per-seed/mean/sample-SD summaries, paired comparisons and Wilcoxon/Holm calculations:

```powershell
& $py committee/cli.py validate
& $py committee/cli.py analyze
& $py committee/cli.py package
```

`validate` and `analyze` return exit code 2 for incomplete/invalid sets but write explicit reports. `analyze` always uses the full preregistered matrix; it cannot opportunistically shrink the test families. `package` creates a new timestamped review ZIP and `.zip.sha256`. It never trains or generates graphs. The current implementation review ZIP contains verification evidence and missing-result templates, not new trained-model results.

If deliberately using a different graph seed or budget, specify the same flags for `list`, `generate`, `train`, `validate` and `analyze`, for example `--graph-seed 20261014 --swaps-per-edge 10 --attempts-per-edge 100`. This creates different artifact/run identifiers. It is an additional graph realization, not a replacement for the declared primary plan. Do not select graph seeds or budgets using model outcomes. Analysis tables are current-plan views and are rewritten atomically; immutable run attempts remain preserved. Package one plan before switching these views to another plan.

Optional matched full-feature Elliptic reference runs, only if the user wants freshly trained controls in the current environment:

```powershell
& $py committee/cli.py list --family matched_full
& $py committee/cli.py generate --family matched_full
& $py committee/cli.py train --family matched_full --retry-failed --continue-on-error
& $py committee/cli.py validate --family matched_full
```

These use the same new loop with all 165 features and native edge order. They are excluded from default analysis, whose reference is the verified August results. Do not silently replace those original references with optional reruns. There is no identified configuration mismatch requiring these reruns; historical environment/RNG details cannot be reconstructed beyond the saved provenance and checkpoint reproduction evidence.

## Reference and configuration provenance

`reference_lock.json` pins 210 distinct original runs: all 150 IEEE construction-tagged GNN runs and all 60 Elliptic models/seeds. Complements reuse the corresponding main/native references. Each new run points to one accepted supervisor reproduction check and its original full-precision result row and prediction file. This includes Elliptic GAT seed 51 recovered from Git `2be96ee`; `recovered_result_provenance.json` is included in review packages. Tables 4.9/4.10's older batches are not used.

The IEEE main reference is `results/ieee_cis_graph_trainfit_main_max1000.pt`, with **406,769** unique undirected edges. It is the cache that reproduced the chosen August checkpoints. The earlier 406,767 reconstruction is not substituted or rebuilt; its discrepancy is described in the preserved supervisor `GRAPH_PROVENANCE.md`. The five IEEE reference edge counts are 406,769 / 304,889 / 7,424 / 16,657 / 345,272. Elliptic has 234,355 unique undirected edges. Saved graphs all use symmetric edge indices, have no stored self-loops/duplicates, and are treated as simple undirected graphs for these controls. Layers retain their frozen self-loop/root handling internally.

The loader reuses the pinned caches' features, labels and masks, whose preprocessing and ID alignment were independently verified by the supervisor analysis. It does not refit preprocessing or rebuild a reference graph. IEEE inputs are the 358 train-fit features, with TransactionID excluded and TransactionDT retained. Elliptic has 165 inputs, with both txId and time_step excluded. Unknown Elliptic labels remain present as graph nodes with placeholder zero labels but are excluded from every supervised loss/evaluation mask. Splits remain chronological: IEEE 472,432 / 59,054 / 59,054; Elliptic known-label populations 29,894 / 7,829 / 8,841. Full-graph transductive inference/training retains every reference node, including unknown Elliptic nodes.

GNNs retain 100 epochs, patience 15; the other models retain 50 epochs, patience 10. Adam uses lr 0.001 and weight decay 1e-5. The loss is BCEWithLogitsLoss with training negatives/positives as positive weight. MLP batches are 1,024; sequence batches are 512; shuffle is false. GNNs are full-batch. Architecture defaults come directly from the frozen source: GCN/GraphSAGE hidden 128, GAT 64 with two first-layer heads; MLP 256/128/64; LSTM hidden 128 and one layer; Transformer width 128, four heads, two layers, feedforward 256; dropout 0.3. Transformer gradient clipping is 1.0. No mixed precision, sampling, model-size or batching optimization has been introduced.

Every requested model selects its best checkpoint by strictly improving validation average precision, with patience measured on that score. Elliptic Transformer uses this same AP criterion. The August **IEEE Transformer** has a different capped-F1 rule but is outside these 300 experiments; its rule is not incorrectly applied to Elliptic. The decision threshold is selected on the best checkpoint's validation predictions from `numpy.arange(.01, 1, .01)`, maximizing (F1, recall, precision), retaining the first/lower threshold on a complete tie. Test predictions are obtained only after model and threshold selection. PR-AUC means `average_precision_score`, not trapezoidal PR integration. Undefined precision/recall/F1 follow the original whole-test `zero_division=0` behavior; these fixed populations contain both classes.

Elliptic sequence inputs preserve the corrected August behavior: sorted (time_step, txId), filter to labeled examples separately within each split, window length 10 ending at target i, left-pad with the first available split row, predict y[i], retain every target. Sequences may cross time steps within a split, never split boundaries; they are global chronological sequences, not wallet histories. Unknown-label rows are removed before windowing. The local ablation changes only selected columns and necessary input-layer dimensions. Seeds remain paired, although changing input dimension also changes the shape and random draws of input-layer initialization.

## Elliptic column mapping

The raw feature CSV has **167 columns**. Raw positions below are zero-based:

| Raw CSV position | Loader name | Treatment |
|---|---|---|
| 0 | txId | ID only; never a model feature |
| 1 | time_step | Ordering/split/rewiring metadata only |
| 2–94 | feature_0 … feature_92 | Retain: 93 local numeric features |
| 95–166 | feature_93 … feature_164 | Remove: 72 supplied neighbor-aggregate features |

The publisher describes 166 features, with time included among the first 94 local features and the last 72 representing neighbor aggregates. Combining that documentation with the verified raw schema and loader establishes the 93/72 numeric split. [Elliptic's dataset description, Claudio Bellei, 6 August 2019](https://medium.com/elliptic/the-elliptic-data-set-opening-up-machine-learning-on-the-blockchain-e0a343d99a14). Every individual included/excluded raw index and loader name is recorded in `reference_lock.json`. Removing the supplied neighbor columns does **not** establish that remaining local features contain no relational information; GNNs also still use native topology.

## Graph definitions and interpretation

**Complement:** replace the reference edge set with sampled nonedges, with no original non-self edge retained. Uniform unordered pair proposals are rejected for loops, original edges or previously sampled edges. NumPy's seeded generator is recorded by package version and graph seed. This is the same replacement definition as the original implementation; the new seed/RNG realization is not claimed to reproduce the June complement cache. Degree-5 and degree-20 old caches are never reused. The new target is exactly `n*20/2`: IEEE 5,905,400 unique edges and Elliptic 2,037,690, stored with both orientations. Average undirected degree is exactly 20; individual degrees are random. Density becomes `20/(n-1)` and changes substantially. Sampling is over all nodes, including cross-time pairs on Elliptic, preserving the existing global complement definition. Thus that comparison changes temporal connectivity as well as topology/density; the time-constrained random control addresses a different question.

**Degree preserving:** start from each pinned edge set and use simple-undirected double-edge swaps. Four distinct endpoints are required; either cross-pairing is proposed at random. Reject existing/duplicate edges, loops, unchanged sets and violations of the declared time constraint. Every node's non-self degree, node ID, isolated status and total edge count is checked exactly after generation. Edges are stored symmetrically. No label, feature value or model score guides proposals. Only the explicitly permitted temporal metadata constrains eligibility.

The preselected budget is 10 successful swaps per original edge, capped at 100 attempts per edge. For Elliptic the budget is applied separately within each time step, with deterministic step order and one graph-seeded RNG. Reference inspection found zero cross-step edges and 49 components. Rewiring preserves time-step separation, but may split components within a step. IEEE identifier_time_window-24h rejects new edges with absolute TransactionDT difference above 86,400 seconds; original inspection confirmed that bound. IEEE main/card/address/email_domain have no fixed time-gap restriction and are rewired across their active nodes. No construction artificially preserves connected components. Exact degrees automatically retain all originally ineligible/isolated nodes as isolated.

Shared card/address/email/device identities and original chain adjacency are deliberately **not** preserved: they are the topology being controlled, and using them to constrain all proposed edges would reintroduce feature-based guidance. Node eligibility and the explicit temporal limits above are preserved. In particular, a randomized identifier-window graph retains the 24-hour bound, not the shared-identifier semantics. This scope is essential when interpreting the control.

Statistics record actual swaps, attempts, acceptance, elapsed time, edge count/density, per-node degree fingerprints, isolated-node equality, components, retained edges and Jaccard similarity. A descriptive limited-randomization flag is set if any stratum misses its swap budget or at least 90% of original edges remain; this rule is fixed before outcomes. Constrained or dense graphs can exhaust the attempt budget. That limitation is retained in the manifest; the graph is neither silently replaced nor repeatedly regenerated to seek better results. Reaching a swap target does not establish uniform sampling or adequate mixing. Zero overlap is not required.

Prepared inputs store topology separately as NPZ to avoid copying the large feature matrix nine times. The source graph hash, resulting topology file hash, exact effective feature hash, node IDs, labels and split hashes define an `effective_graph_sha256`. This composition is the graph actually used by the runner; it is not a hash of an invented replacement `.pt` file. Local/full native preparations preserve the reference edge column order exactly.

## Statistical families fixed before training

`comparison_families.json` contains **180 planned hypotheses in 30 families**. Six metrics are tested separately: average precision, F1, recall, precision, ROC-AUC and FP count. Dataset and metric are never pooled.

| Family, for each metric | Comparisons in family |
|---|---:|
| IEEE complement versus own main reference, all three GNNs | 3 |
| Elliptic complement versus native reference, all three GNNs | 3 |
| Elliptic local versus full features, all six models | 6 |
| IEEE randomized versus own original construction, five constructions × three GNNs pooled | 15 |
| Elliptic randomized versus native reference, all three GNNs | 3 |

Paired difference is **new minus original**, in original metric units; positive is better except for FP. Pair by the same seed 42–51 using full-precision saved original metrics, including the recovered result. Report all paired values and their median, number of pairs, raw p and Holm-adjusted p. The two-sided signed-rank test drops exact-zero differences, uses average ranks for tied absolute differences and enumerates all signs of the remaining ranks (at most 2^10). P is twice the lower-tail probability at min(W+, W−), capped at one; all-zero differences give p=1. No rounding precedes ranking. Holm correction uses the entire planned family, alpha=.05. Missing comparisons use p=1 only inside adjustment and retain null reported p-values; no ten-seed inference is reported for an incomplete pair set. Summaries report mean, sample SD (`ddof=1`), available/expected counts and missing seeds. Discrete confusion counts are also summarized, even though only FP is in the six inferential metric families.

## Artifacts, validation and resumption

- `committee/`: implementation, frozen safe source modules, immutable reference/configuration lock, statistical definitions, this README and synthetic tests.
- `implementation_verification/`: actual saved-graph inspection, dry experiment matrix, environment, test output and preservation checks. No trained results are stored here.
- `graphs/<dataset__construction__features__variant__gseed__config-hash>/`: generation config, attempts, topology, statistics and completion manifest. Graph generation seed has no relationship to training seeds.
- `runs/<family>/<graph-id__model__trainseed>/attempt_NNN/`: exact config written before the loop, source commit/dirty state/code hashes/package versions, validation/test ID-label-score NPZ, metrics, selected-epoch checkpoints, per-epoch JSONL logs, completion/failure metadata.
- `runs/.../status.json`: atomic pointer to the latest attempt; old attempts are retained. Failed runs include a precise command to restart.
- `analysis/`: separate per-seed and summary CSV/JSON files for complement/local/random, complete family definitions, paired differences, Wilcoxon/Holm and validation reports.
- `committee_review_<UTC timestamp>.zip`: scripts, configuration, graph statistics, manifests, full-precision results, summaries, tests, logs, new predictions, the completed supervisor evidence/predictions, and the earlier `audit/` files referenced by that evidence. Large graph arrays, source datasets and checkpoints are excluded; `review_manifest.json` retains paths and SHA-256 values. Runtime binaries/venv are excluded; their environment/provenance is retained.

Validation rechecks all completed artifact hashes, exact run/config/reference identities, original versus new validation/test IDs and labels, finite probability ranges, validation-threshold selection, test metrics, confusion totals and selected-epoch bounds. It does not assume a filename proves provenance. Missing original artifacts or a hash mismatch stop dependent work with an explicit error. Atomic metadata writes and exclusive attempt creation prevent accidental overwrite. All improvements within an attempt get distinct epoch checkpoint names. Results are collected from unique run manifests, not appended repeatedly to a shared CSV.

The synthetic verification includes deterministic complements, exact degree preservation with isolates, temporal constraints, impossible swaps, loop/duplicate/direction rejection, sequence padding/target alignment, all required model forward shapes, exact signed-rank/Holm checks, and mocked failure/retry/hash-corruption handling. It does not perform optimizer steps. Successful lightweight tests establish implementation invariants, not successful completion, convergence or memory feasibility of any full training run.
