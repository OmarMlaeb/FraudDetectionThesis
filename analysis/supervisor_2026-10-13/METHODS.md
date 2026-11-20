# Analysis specification (fixed before inference and interpretation)

Analysis date: 2026-10-13. Original experimental files will not be modified. No model training, graph replacement or test-threshold optimization is permitted.

## Experiment selection

Seeds 42 through 51, paired by explicit seed. IEEE-CIS: August MLP, LSTM and Transformer, plus the August construction-tagged main/card/address/email_domain/identifier_time_window-24h GNN runs. Elliptic: August six-model runs, including the GAT seed-51 row recovered from Git commit 2be96ee (removed in 5333c55). Exactly 240 selected run rows are expected. Source CSV numeric strings are preserved; analysis uses their full double precision, never rounded thesis cells.

## Inference acceptance gate

Use existing checkpoint weights and saved graph caches. Reconstruct preprocessing separately, without saving over experiment artifacts, and compare the resulting features/labels/masks with the caches. Use evaluation mode, no gradients, the original architectures, and the original validation-selected threshold from each saved row. Record checkpoint/graph/data/source hashes and test IDs.

A checkpoint is accepted for subgroup analysis only if its test population/labels match, all four whole-test confusion counts match exactly, classification metrics differ by no more than 1e-10, average precision and ROC-AUC differ by no more than 2e-5, validation F1 at the saved threshold differs by no more than 1e-10, and the reselected validation threshold differs by no more than 1e-6. The small ranking-metric tolerance permits floating-point tie-order differences; it is not a tolerance on classifications. Report all measured differences. Recompute thresholds only on validation to check provenance; always apply the originally saved threshold to test groups. A failed gate is an explicit mismatch, not permission to substitute another checkpoint or batch.

LSTM/Transformer inference is an additional reproduction check. Their original versus corrected sequence behavior must be established from the selected August source and populations, not from Table 4.10. Missing or mismatching checkpoints do not prevent separate source-verified statistical analysis of the saved run metrics, which must be labeled separately from an inference-verified analysis.

## Connected versus isolated

Degree is computed from the full graph used in inference, including train, validation, test and (for Elliptic) unknown-label nodes. A test node is isolated iff it has no non-self neighbor. Ignore self-loops for grouping even if a model adds them internally. Use identical test IDs for GCN, GraphSAGE, GAT and MLP within each graph construction. Compute MLP predictions once per seed/dataset and reuse them across IEEE-CIS constructions. Ten-seed summaries use arithmetic means and sample SD (ddof=1), and display available/expected seed counts.

PR-AUC retains the project's definition: sklearn.metrics.average_precision_score (average precision), not trapezoidal integration. Compute F1, recall, precision and false positives at the saved whole-run threshold. Preserve confusion counts. Empty groups have no rates/metrics; groups with no positive labels have undefined recall and average precision. Undefined precision (no predicted positives) and F1 (no actual or predicted positives) are flagged; do not silently turn missing values into zeros. For whole-test reproduction use the original zero_division=0 convention. Subgroup confusion matrices must partition the whole-test confusion matrix exactly. Fraud rate = positive labels / subgroup size.

## Complete Wilcoxon/Holm families

Metrics: average precision (`pr_auc`), F1, recall, precision, ROC-AUC, and false positives (counts). Higher is better for the first five; lower is better for false positives. Every metric is a separate family; datasets are never pooled.

1. **Core model family**, separately for IEEE-CIS and Elliptic: all 15 unordered pairwise comparisons among MLP, LSTM, Transformer, GCN, GraphSAGE and GAT. IEEE-CIS uses the construction-tagged `-main` GNNs. Six metrics produce 180 hypotheses across the two datasets.
2. **IEEE-CIS construction family**, separately for each metric: all ten unordered construction pairs within each of GCN, GraphSAGE and GAT, pooled into one 30-hypothesis family per metric. Constructions: main, card, address, email_domain, identifier_time_window-24h. Six metrics produce 180 hypotheses.
3. **IEEE-CIS alternative-GNN versus MLP family**, separately for each metric: all 12 alternative construction/architecture combinations (four non-main constructions times three GNNs) versus MLP. Six metrics produce 72 hypotheses. Main-GNN versus MLP comparisons are already in the core family.

Total: 432 planned comparisons in 24 families. Apply Holm step-down adjustment to each complete family, alpha=0.05. Missing comparisons remain explicit and retain the planned family size (conservative p=1 placeholder for adjustment only); never shrink a family opportunistically. Report no ten-seed conclusion for incomplete pairs.

Paired difference is A minus B in original metric units; report median paired difference and also means and mean difference for interpretation. Pair by seed 42–51 only. Two-sided Wilcoxon signed-rank test, zero_method='wilcox' (drop exact-zero differences). Use exhaustive sign permutation of observed average ranks of absolute nonzero differences, including ties (at most 2^10 assignments). Two-sided p = min(1, 2 * lower-tail probability at min(W+, W-)); all-zero differences give p=1 and statistic=0. Record number of total pairs, zeros, nonzero pairs and tied absolute differences. No rounding of differences before ranking. Test complete families, not just comparisons printed in the thesis.

Produce a source-verified saved-results analysis immediately and a separate inference-verified analysis after checkpoint checks; the latter flags excluded runs rather than silently replacing them. Sequence reproduction failures are treated the same way as any other failed gate.

## Graph discrepancy

Hash and compare saved graph versions first, using transaction-ID edge pairs. A diagnostic reconstruction of the connectivity script is allowed only in memory/separate analysis outputs and must be labeled as such. Never use a reconstruction as a substitute inference graph. Investigate ordering of transactions sharing identical TransactionDT values and runtime versions; record observed evidence rather than assuming a two-edge discrepancy is harmless.
