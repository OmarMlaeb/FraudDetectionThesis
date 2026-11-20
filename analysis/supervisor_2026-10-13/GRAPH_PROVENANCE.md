# Main-graph provenance discrepancy

The selected August `-main` GNN experiments are reproduced using the existing `results/ieee_cis_graph_trainfit_main_max1000.pt`, with **406,769 unique undirected non-self edges**. All 30 main GNN checkpoints pass the whole-test/validation reproduction gate on this cache. The older `ieee_cis_graph.pt` and `ieee_cis_graph_trainfit.pt` have exactly the same transaction-ID edge set (their features/serialization differ). None was rebuilt or replaced for inference.

## Reproducing the reported 406,767

The inner graph-construction operation sorts each identifier group only by TransactionDT, with default quicksort. Equal timestamps have no explicit secondary key. Changing NumPy’s CPU sorting dispatch changes the order of ties and therefore which consecutive transactions are joined. The initial outer sort by TransactionID does not prevent an unstable inner sort from reordering those ties.

Controlled diagnostic results:

| Diagnostic | NumPy / pandas | Inner sort | Edges | Fraud–fraud | Fraud–legitimate | Legitimate–legitimate |
|---|---|---|---:|---:|---:|---:|
| Saved experiment cache | existing artifact | existing | 406,769 | 8,476 | 18,647 | 379,646 |
| Experiment packages, normal CPU dispatch | 2.2.6 / 2.3.3 | quicksort | 406,769 | 8,476 | 18,647 | 379,646 |
| Same packages, AVX2/AVX512F disabled | 2.2.6 / 2.3.3 | quicksort | 406,767 | 8,475 | 18,644 | 379,648 |
| Experiment packages, stable sort | 2.2.6 / 2.3.3 | mergesort | 406,764 | 8,475 | 18,644 | 379,645 |
| Bundled audit packages | 2.3.5 / 3.0.1 | quicksort | 406,764 | 8,475 | 18,644 | 379,645 |

Setting `NPY_DISABLE_CPU_FEATURES=AVX2,AVX512F` before starting the portable Python process reproduces **406,767 edges and exactly the thesis Table 4.4 class counts 8,475 / 18,644 / 379,648**. This identifies a concrete computational mechanism for the earlier reconstruction’s discrepancy. It does not establish which CPU dispatch was used historically: the original 406,767-edge file was not preserved. The reconstructed version is explicitly diagnostic, not a recovered training artifact.

## Transaction-ID edge comparison

The normal-dispatch reconstruction matches the saved cache edge-for-edge. Compared with that cache, the scalar-sort 406,767-edge reconstruction removes **28** transaction-ID edges and adds **26**, for **54** differing edges and a net count difference of two. It is not simply deletion of two edges. `graph_diagnostics/transaction_id_edge_differences.csv` lists all differences, endpoint IDs, timestamps and labels. Stable-sort alternatives remove 24 and add 19 edges.

The saved and scalar-reconstructed versions have the same set of nodes with a non-self neighbor, so connected/isolated membership is unchanged. Their neighborhoods differ, which can affect GNN predictions; this is why inference retains the saved experiment graph. No inference was performed on a diagnostic replacement.

## Hashes

| Version | Canonical transaction-edge SHA-256 | Artifact SHA-256 |
|---|---|---|
| `results/ieee_cis_graph_trainfit_main_max1000.pt` | `113cea140702632b0e8a00f16fcd13993e9f8b6deb40f99adeb0080752df6293` | `6118895d98d28d37695abe0f0be7813029bd198bedef9f3b3e1a165ea8b3987b` |
| `diagnostic_experiment_scalar_sort_quicksort.npz` | `6d45824bfdba0b2454f965440afceb74de3287e3d4774586a4a78ec0495f8708` | `14b137f86785701bec651476e2ee4e170804e40dd1e67fb0c90a7de2810fe62e` |

Canonical edge hashes encode sorted, unique, undirected TransactionID pairs as little-endian int64 pairs. Full artifact versions, hashes and comparisons are in `graph_diagnostics/graph_versions.json` and `edge_set_comparisons.json`; all required experiment graphs/data/checkpoints have paths and SHA-256 hashes in `artifact_hashes_before.json`. The hardcoded 406,767 expectation in the later connectivity script is not a manifest of the August inference graph. No existing source or graph was changed to force either count.
