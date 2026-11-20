# Thesis experiment provenance audit

**PDF follow-up:** The subsequently supplied thesis has now been inspected. See [the extracted final tables](thesis/FINAL_TABLES.md) and [their verified experiment sources](thesis/TABLE_PROVENANCE.md). The manuscript-availability limitation below describes the initial repository-only audit.

Audited local files and Git HEAD `ecfc284` on 2026-10-11. No thesis manuscript or table-to-run manifest is present, so this identifies the saved experiment batches, not which batch was ultimately printed in the submitted thesis. Here, **original ten-seed batch** means the June 28 experiments preserved in `2f66446`; the still-earlier May/June archive is listed separately.

The ten explicitly recorded seeds are **42, 43, 44, 45, 46, 47, 48, 49, 50, 51**. They are model-training seeds, not ten different train/test splits. `notes2.txt` lines 47 and 49 record the explicit ten-seed commands. The default all-model runners use only 42, 43, 44 unless overridden.

## Saved results and generations

| Batch | Source | Coverage | Code/history reference |
|---|---|---|---|
| Early implementation archive | `results/model_results_old.csv`, lines 2–103 | 102 rows dated May 24–June 6; no seed column, not an identified ten-seed experiment | Archived by `2f66446`; later pre-June-28 source hardcodes seed 42, but these rows do not identify their seeds |
| Original June repeated experiments | `results/model_results.csv`, lines 2–127, **excluding lines 14–19** | 120 rows: 6 models × 2 datasets × 10 seeds | `2f66446`, June 28 |
| June sampled-complement experiments | Same CSV, lines 14–19 | 6 rows, seed 42 only | `2f66446`; separate stress-test graph variants |
| August baseline reruns | Same CSV, lines 128–246 | 119 current rows: IEEE-CIS 60, Elliptic 59 | Added in `2be96ee`, August 30 |
| August graph-construction experiments | `results/ieee_cis_graph_model_results.csv`, lines 2–151 | 150 rows: GCN/GraphSAGE/GAT × 5 constructions × 10 seeds | `5333c55`, August 30 |

The missing August row is **Elliptic-GAT, seed 51**, timestamp `2026-08-30T13:24:40`. It was present at line 247 in `2be96ee` and deleted by `5333c55`. Its threshold is **0.92**, validation F1 **0.5374800637958532**, test PR-AUC **0.28934854624693335**, test F1 **0.35786802030456855**, and confusion matrix `[[8194,239],[267,141]]`. A copy is in [recovered_elliptic_gat_seed51.json](recovered_elliptic_gat_seed51.json). The source CSV was not changed.

The source commit references identify the snapshots containing these batches and methods. Runs do not record an executed commit hash, so they are not cryptographic proof of the exact code used.

## Graphs and checkpoint names

All paths below are relative to `results/`. `s` ranges from 42 to 51, and `m` is `gcn`, `sage`, or `gat`. These graph caches are shared across training seeds.

| Batch | Graph file | Checkpoint pattern |
|---|---|---|
| June IEEE-CIS GNNs | `ieee_cis_graph.pt` | `best_{m}_seed{s}_model.pt` (name later reused) |
| August IEEE-CIS initial GNN reruns | `ieee_cis_graph_trainfit.pt` | `best_{m}_seed{s}_model.pt` |
| June/August Elliptic GNNs | `elliptic_graph.pt` | `best_elliptic_{m}_seed{s}_model.pt` (name reused) |
| August IEEE-CIS main | `ieee_cis_graph_trainfit_main_max1000.pt` | `best_{m}_main_seed{s}_model.pt` |
| August IEEE-CIS card | `ieee_cis_graph_trainfit_card_max1000.pt` | `best_{m}_card_seed{s}_model.pt` |
| August IEEE-CIS address | `ieee_cis_graph_trainfit_address_max1000.pt` | `best_{m}_address_seed{s}_model.pt` |
| August IEEE-CIS email | `ieee_cis_graph_trainfit_email_domain_max1000.pt` | `best_{m}_email_domain_seed{s}_model.pt` |
| August IEEE-CIS identifier/window | `ieee_cis_graph_trainfit_identifier_time_window_24h.pt` | `best_{m}_identifier_time_window_24h_seed{s}_model.pt` |
| IEEE-CIS MLP/LSTM/Transformer | Not applicable | `best_{mlp,lstm,transformer}_seed{s}_model.pt` |
| Elliptic MLP/LSTM/Transformer | Not applicable | `best_elliptic_{mlp,lstm,transformer}_seed{s}_model.pt` |

All **270 seed-specific checkpoint files currently have August 30 modification timestamps** (120 baseline + 150 construction checkpoints). There are another 18 unseeded checkpoints dated June 28. The 288 inspected checkpoints contain parameter state dictionaries only, with no saved seed, threshold, validation score, epoch, graph hash, test IDs, or predictions. Naming and modification times support the August association, but exact metric reproduction was not run. Do not treat the current baseline checkpoint filenames as independently preserved June weights. The unseeded June files are candidates for older/default-seed runs, not an archive of ten seeds.

Graphs and checkpoints are ignored by Git (`*.pt`); Git cannot restore overwritten June weights. [artifact_inventory.json](artifact_inventory.json) records all current checkpoint SHA-256 hashes, sizes, timestamps and tensor shapes, plus the graph attributes, dimensions and mask fingerprints.

The following graph dimensions were read from the actual `.pt` archives. Edge columns store both directions; the last column is half that count, matching the graph summaries in the notes.

| Graph file | Nodes | Features | Directed edge columns | Undirected edges |
|---|---:|---:|---:|---:|
| `elliptic_graph.pt` | 203,769 | 165 | 468,710 | 234,355 |
| `elliptic_graph_sampled_complement_deg20_seed42.pt` | 203,769 | 165 | 4,075,380 | 2,037,690 |
| `ieee_cis_graph.pt` | 590,540 | 358 | 813,538 | 406,769 |
| `ieee_cis_graph_sampled_complement_deg20_seed42.pt` | 590,540 | 358 | 11,810,800 | 5,905,400 |
| `ieee_cis_graph_sampled_complement_deg5_seed42.pt` | 590,540 | 358 | 2,952,700 | 1,476,350 |
| `ieee_cis_graph_trainfit.pt` | 590,540 | 358 | 813,538 | 406,769 |
| `ieee_cis_graph_trainfit_address_max1000.pt` | 590,540 | 358 | 14,848 | 7,424 |
| `ieee_cis_graph_trainfit_card_max1000.pt` | 590,540 | 358 | 609,778 | 304,889 |
| `ieee_cis_graph_trainfit_email_domain_max1000.pt` | 590,540 | 358 | 33,314 | 16,657 |
| `ieee_cis_graph_trainfit_identifier_time_window_24h.pt` | 590,540 | 358 | 690,544 | 345,272 |
| `ieee_cis_graph_trainfit_main_max1000.pt` | 590,540 | 358 | 813,538 | 406,769 |

The four construction edge-list CSVs are `ieee_cis_graph_edges_card_max1000.csv`, `...address_max1000.csv`, `...email_domain_max1000.csv`, and `...identifier_time_window_24h.csv`. Their columns are **node indices** (`source_node,target_node`), not transaction IDs. The README-mentioned `ieee_cis_graph_edges_main_max1000.csv` is **absent**, although the main `.pt` exists.

Complement graphs are a separate June experiment. Elliptic has `elliptic_graph_sampled_complement_deg20_seed42.pt`; IEEE-CIS has both `ieee_cis_graph_sampled_complement_deg20_seed42.pt` and `...deg5_seed42.pt`. The `seed42` in a graph filename is the **graph-sampling seed**. The result rows do not record the complement degree, and complement checkpoint names do not distinguish degrees. Exact IEEE-CIS degree-to-result matching is therefore unresolved. Existing complement checkpoints are the unseeded `best_{m}_sampled_complement_model.pt` and `best_elliptic_{m}_sampled_complement_model.pt`; the seeded names that the June commit would generate are absent. This is another reason not to infer complete run provenance from that commit alone.

## Test transaction IDs

No historical standalone test-ID files were saved. The audit exports below are **new reconstructions** using the current raw data and historical split rules. The graph caches contain masks, not original IDs.

| Dataset and generation | Evaluated test population | Reconstructed list |
|---|---|---|
| IEEE-CIS original MLP/GNNs | 59,054 IDs, **3518486–3577539 inclusive**; 2,213 fraud | [IDs](ieee_original_full_test_ids.json) |
| IEEE-CIS original LSTM/Transformer | 59,045 IDs, **3518495–3577539 inclusive**; 2,213 fraud | [IDs](ieee_original_sequence_test_ids.json) |
| IEEE-CIS August all models/constructions | 59,054 IDs, **3518486–3577539 inclusive**; 2,213 fraud | [IDs](ieee_august_all_test_ids.json) |
| Elliptic original MLP/GNNs and all August models | 8,841 labeled transactions at time steps **42–49**; 408 illicit | [IDs](elliptic_full_test_ids.json) |
| Elliptic original LSTM/Transformer | 8,832 transactions from that ordered set after dropping its first 9; 408 illicit | [IDs](elliptic_original_sequence_test_ids.json) |

IEEE-CIS uses the last 10% of `train_transaction.csv`, not Kaggle's unlabeled `test_transaction.csv`. Saved masks select zero-based nodes 531486–590539. The current raw transaction order exactly equals sorting by `(TransactionDT, TransactionID)`, so August's explicit sorting does not change these IDs on the present dataset. Every IEEE-CIS cache has the same test-mask fingerprint.

Elliptic nodes are ordered by `(time_step, txId)`. Its reconstructed test indices exactly match the test-mask SHA-256 fingerprint in both saved graphs. The first nine IDs omitted by the original sequence models are `268186, 295623, 971497, 992596, 1153751, 1154526, 1720404, 1744093, 1876609`; original sequence evaluation starts with `1876617`. These IDs are not a contiguous numeric interval.

Every saved result row, including the early archive and recovered row, has a confusion-matrix total matching its reconstructed test-set count. This verifies population counts, not historical per-transaction probabilities.

## Prediction scores and thresholds

**Per-transaction prediction probabilities and validation probability vectors are absent.** Training computes sigmoid probabilities in memory, then saves only aggregate recall, precision, F1, PR-AUC, ROC-AUC, confusion-matrix counts, selected threshold, threshold strategy, validation F1 and seed. Some console logs contain probability summaries, not per-ID scores. Checkpoints alone do not store those scores. Producing new scores would require compatible model code, graph/features and preprocessing, followed by inference; those would be reconstructed predictions, not recovered original predictions. June baseline weights are not independently preserved for all ten seeds.

All June ten-seed thresholds and August GNN thresholds use validation F1 selection on a 0.01–0.99 grid, with recall then precision as tie-breakers. Checkpoints generally use validation PR-AUC early stopping. The **August IEEE-CIS Transformer differs**: it selects maximum validation F1 subject to a 20% validation alert-rate cap, adds probability-quantile threshold candidates, and early-stops on capped validation F1. Its saved strategy is `f1_max_alert_rate_0.20`. The CSV's `validation_f1_at_threshold` is not the best validation PR-AUC or a prediction score.

Exact saved numeric strings, result source lines, graph/checkpoint candidates, checkpoint hashes, and test-ID list paths for every row are in [run_manifest.json](run_manifest.json). The tables below display the selected thresholds in seed order; `0.83` and `0.8300000000000001` are shown identically here but preserved verbatim in the manifest.

### Original June thresholds

| Model | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Elliptic-MLP | 0.48 | 0.91 | 0.5 | 0.49 | 0.48 | 0.51 | 0.91 | 0.91 | 0.51 | 0.49 |
| Elliptic-LSTM | 0.98 | 0.94 | 0.85 | 0.83 | 0.92 | 0.94 | 0.98 | 0.91 | 0.96 | 0.99 |
| Elliptic-Transformer | 0.89 | 0.77 | 0.91 | 0.97 | 0.94 | 0.94 | 0.87 | 0.94 | 0.98 | 0.94 |
| Elliptic-GCN | 0.5 | 0.5 | 0.5 | 0.51 | 0.52 | 0.52 | 0.5 | 0.54 | 0.5 | 0.54 |
| Elliptic-GraphSAGE | 0.8 | 0.83 | 0.85 | 0.84 | 0.76 | 0.82 | 0.82 | 0.81 | 0.85 | 0.52 |
| Elliptic-GAT | 0.91 | 0.88 | 0.93 | 0.93 | 0.9 | 0.89 | 0.87 | 0.9 | 0.88 | 0.92 |
| MLP | 0.87 | 0.87 | 0.81 | 0.86 | 0.88 | 0.85 | 0.8 | 0.88 | 0.83 | 0.85 |
| LSTM | 0.89 | 0.87 | 0.94 | 0.91 | 0.88 | 0.88 | 0.89 | 0.86 | 0.94 | 0.9 |
| Transformer | 0.46 | 0.96 | 0.91 | 0.01 | 0.01 | 0.94 | 0.94 | 0.01 | 0.01 | 0.01 |
| GCN | 0.78 | 0.98 | 0.72 | 0.99 | 0.58 | 0.72 | 0.65 | 0.58 | 0.61 | 0.61 |
| GraphSAGE | 0.62 | 0.8 | 0.57 | 0.76 | 0.74 | 0.88 | 0.74 | 0.71 | 0.62 | 0.97 |
| GAT | 0.95 | 0.82 | 0.99 | 0.65 | 0.75 | 0.9 | 0.99 | 0.99 | 0.74 | 0.82 |

### August baseline thresholds

| Model | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | 0.93 | 0.88 | 0.92 | 0.85 | 0.9 | 0.89 | 0.87 | 0.87 | 0.89 | 0.92 |
| LSTM | 0.92 | 0.88 | 0.88 | 0.9 | 0.9 | 0.91 | 0.9 | 0.89 | 0.89 | 0.91 |
| Transformer | 0.926534 | 0.921243 | 0.934546 | 0.93 | 0.87 | 0.92 | 0.93048 | 0.9 | 0.95095 | 0.91 |
| GCN | 0.83 | 0.87 | 0.87 | 0.88 | 0.74 | 0.84 | 0.85 | 0.86 | 0.88 | 0.72 |
| GraphSAGE | 0.87 | 0.9 | 0.88 | 0.9 | 0.9 | 0.85 | 0.89 | 0.86 | 0.87 | 0.9 |
| GAT | 0.87 | 0.83 | 0.88 | 0.85 | 0.88 | 0.84 | 0.84 | 0.84 | 0.83 | 0.8 |
| Elliptic-MLP | 0.48 | 0.91 | 0.5 | 0.49 | 0.48 | 0.51 | 0.91 | 0.91 | 0.51 | 0.49 |
| Elliptic-LSTM | 0.95 | 0.96 | 0.89 | 0.98 | 0.97 | 0.9 | 0.98 | 0.94 | 0.94 | 0.97 |
| Elliptic-Transformer | 0.96 | 0.91 | 0.88 | 0.94 | 0.74 | 0.8 | 0.77 | 0.73 | 0.8 | 0.93 |
| Elliptic-GCN | 0.5 | 0.5 | 0.5 | 0.51 | 0.52 | 0.52 | 0.5 | 0.54 | 0.5 | 0.54 |
| Elliptic-GraphSAGE | 0.8 | 0.83 | 0.85 | 0.84 | 0.76 | 0.82 | 0.82 | 0.81 | 0.85 | 0.52 |
| Elliptic-GAT | 0.91 | 0.88 | 0.93 | 0.93 | 0.9 | 0.89 | 0.87 | 0.9 | 0.88 | 0.92* |

### August graph-construction thresholds

| Model | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GCN-main | 0.83 | 0.87 | 0.87 | 0.88 | 0.74 | 0.83 | 0.85 | 0.86 | 0.87 | 0.72 |
| GraphSAGE-main | 0.87 | 0.9 | 0.87 | 0.9 | 0.87 | 0.85 | 0.88 | 0.87 | 0.87 | 0.89 |
| GAT-main | 0.86 | 0.83 | 0.88 | 0.85 | 0.88 | 0.84 | 0.84 | 0.84 | 0.84 | 0.8 |
| GCN-card | 0.88 | 0.85 | 0.89 | 0.68 | 0.79 | 0.88 | 0.88 | 0.88 | 0.87 | 0.78 |
| GraphSAGE-card | 0.86 | 0.88 | 0.88 | 0.89 | 0.88 | 0.87 | 0.89 | 0.89 | 0.87 | 0.89 |
| GAT-card | 0.88 | 0.75 | 0.88 | 0.81 | 0.89 | 0.89 | 0.8 | 0.7 | 0.85 | 0.8 |
| GCN-address | 0.86 | 0.83 | 0.85 | 0.7 | 0.68 | 0.88 | 0.87 | 0.84 | 0.87 | 0.83 |
| GraphSAGE-address | 0.87 | 0.87 | 0.88 | 0.86 | 0.91 | 0.86 | 0.85 | 0.87 | 0.88 | 0.93 |
| GAT-address | 0.84 | 0.8 | 0.75 | 0.76 | 0.78 | 0.85 | 0.9 | 0.78 | 0.8 | 0.81 |
| GCN-email_domain | 0.86 | 0.84 | 0.84 | 0.7 | 0.68 | 0.85 | 0.87 | 0.87 | 0.87 | 0.71 |
| GraphSAGE-email_domain | 0.87 | 0.87 | 0.87 | 0.86 | 0.91 | 0.9 | 0.9 | 0.92 | 0.87 | 0.88 |
| GAT-email_domain | 0.83 | 0.81 | 0.76 | 0.74 | 0.76 | 0.84 | 0.89 | 0.67 | 0.8 | 0.77 |
| GCN-identifier_time_window-24h | 0.92 | 0.83 | 0.88 | 0.89 | 0.71 | 0.88 | 0.88 | 0.87 | 0.92 | 0.78 |
| GraphSAGE-identifier_time_window-24h | 0.91 | 0.92 | 0.88 | 0.93 | 0.92 | 0.88 | 0.88 | 0.89 | 0.89 | 0.91 |
| GAT-identifier_time_window-24h | 0.83 | 0.8 | 0.81 | 0.83 | 0.81 | 0.79 | 0.77 | 0.84 | 0.78 | 0.81 |

`*` August Elliptic-GAT seed 51 was recovered from Git, not from the current CSV.

June complement seed-42 thresholds: IEEE-CIS GCN **0.53**, GAT **0.17**, GraphSAGE **0.60**; Elliptic GCN **0.77**, GAT **0.60**, GraphSAGE **0.57**. Early-archive rows have a mixture of populated and blank threshold fields; blanks are not validation-selected thresholds.

## Which derived outputs belong to which implementation?

| Output | Provenance and interpretation |
|---|---|
| `model_results_old.csv`, `model_rankings_old.csv`, `model_rankings_by_f1_old.csv`, `model_rankings_by_recall_old.csv`, `wilcoxon_model_comparisons_old.csv` | Early May/June implementation; archived June 28. The result archive lacks explicit run seeds. |
| `results.txt`, June portion of `model_results.csv`, `new_results.txt`, `notes2.txt` before its later appended construction notes | Original implementation console/results artifacts. `results.txt` was last changed June 6; `new_results.txt` is a partial June console transcript, not a complete ten-seed record. `notes2.txt` mixes June and August information. |
| `wilcoxon_model_comparisons.csv` | June analysis, last changed in `2f66446`; not recomputed for August. For example, its IEEE-CIS MLP PR-AUC mean is 0.42845456701080487 (June), versus 0.4248846279291471 (August). |
| `model_rankings_v2.csv`, current `model_rankings_by_f1.csv`, current `model_rankings_by_recall.csv` | August baseline rankings plus retained June complement entries. These rank the latest single result per model, not ten-seed means. They still contain the subsequently deleted August Elliptic-GAT seed-51 result. |
| `ieee_cis_graph_model_results.csv`, four construction edge CSVs, construction-specific graphs/checkpoints | Later August construction experiments in `5333c55`. |
| `ieee_cis_graph_combined_rankings.csv`, `ieee_cis_graph_vs_baseline_comparisons.csv` | Later August comparisons. Combine latest-per-seed August baselines with construction results, replacing unsuffixed GNN baselines with `-main` when present. June complement entries remain as single-run rows. Combined rankings use per-seed means/std, unlike the baseline rankings. |

Re-running the current general Wilcoxon script against the mixed baseline CSV would select the latest row per model and seed. Because August Elliptic-GAT seed 51 is missing, that procedure would silently fall back to its June seed-51 row while using August for the other seeds. Filter by generation and resolve the missing row before interpreting a rerun as an August-only comparison.

The key method changes in `2be96ee` are IEEE-CIS preprocessing fitted on training data only (earlier code fitted missingness filtering, encodings, medians and scaling on the full dataset), explicit temporal sorting, padded sequences retaining the first nine examples, the new `ieee_cis_graph_trainfit.pt` cache, and the Transformer threshold/early-stopping changes above. Elliptic sequence handling also changed; its graph and non-sequence data pipeline remained shared. Some rerun metrics may be identical without belonging to the same run.

`5333c55` adds the five IEEE-CIS construction choices, separate graph-result CSV, and construction-specific checkpoint/cache names. The current default construction is **card**. Running today's default commands does not recreate the original main-graph experiment.

The label `graph_variant="original"` only means the graph is not a sampled complement. It does **not** mean that its preprocessing, construction code, checkpoint or metrics belong to the original June implementation.

## Verification and remaining limits

Inspected all 299 `.pt` archives (11 graphs, 288 checkpoint dictionaries) without executing pickle globals, and all 497 currently saved result rows across the three result CSVs. Also recovered one historical deleted row. Checked all result confusion-matrix totals against reconstructed test populations. Derived audit files are separate; original code/results/data/checkpoints were not modified, and no training or inference was performed.

The repository virtual environment points to an absent Python 3.10 installation. Audit extraction used the bundled Python runtime and inert pickle metadata readers; no dependency installation was needed. Tensor contents were not run through models. File timestamps and source naming are supporting evidence, not immutable run records. There is no saved preprocessing object, environment lock, executed Git SHA, graph hash per result, or historical dataset hash to prove full run reproduction. The manuscript is needed to establish which of these identifiable result batches its tables actually used.
