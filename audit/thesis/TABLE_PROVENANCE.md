# Which saved experiments appear in the final thesis?

Source: `C:/Users/user/Desktop/AAI_Thesis.pdf`, 104 PDF pages. All 13 Chapter 4 results tables have been extracted into [FINAL_TABLES.md](FINAL_TABLES.md) and [final_tables.json](final_tables.json), with both printed and PDF page numbers. The table images were visually inspected. The literature-review comparison in Table 2.1 is outside this results extraction.

**The thesis combines several experiment generations. Its ten-run aggregate performance tables use the August experiments, while the Elliptic final-run table uses older results.** There is no single batch or seed that describes all its final-run tables.

All numeric matches below are at the thesis's displayed precision, using every metric in the relevant row. Matching a rounded value supports numerical provenance; it does not prove which file an author copied. When multiple saved rows match, that ambiguity is retained in [table_matches.json](table_matches.json), which includes source lines, seeds, timestamps, selected thresholds, and graph/checkpoint/ID-list paths.

| Thesis table | Printed page (PDF page) | Identified source |
|---|---|---|
| 4.1 Graph statistics | 53 (68) | Original/native and June sampled-complement graph descriptions. IEEE-CIS original edge count differs from the saved caches; see below. |
| 4.2 Alternative graph properties | 54 (69) | August card, address, email-domain and 24-hour identifier constructions. Nodes/edges match the saved graph artifacts; displayed summaries correspond to `notes2.txt`. |
| 4.3 Isolated transactions | 55 (70) | All five isolation counts and percentages reproduce from the August construction graph caches. |
| 4.4 Class-specific connectivity | 55 (70) | All four alternative-construction rows reproduce from the August graph caches. The main row has different edge-class counts, while its two node percentages match. |
| 4.5 GNN performance by construction | 57 (72) | All 15 rows match the 150 August rows in `results/ieee_cis_graph_model_results.csv`, seeds 42–51, mean and sample SD. |
| 4.6 False positives by construction | 58 (73) | All 15 rows match the same 150 August construction results, mean and sample SD. |
| 4.7 IEEE-CIS ten-run means | 60 (75) | August MLP/LSTM/Transformer from `model_results.csv`, plus August `GraphSAGE-main`, `GCN-main`, `GAT-main` from `ieee_cis_graph_model_results.csv`. All 30 metric values match. |
| 4.8 Elliptic ten-run means | 62 (77) | All 30 metric values match August seeds 42–51, including the deleted GAT seed-51 row recovered from `2be96ee`. MLP and GCN also match June at displayed precision, so those individual rows are not uniquely attributable. |
| 4.9 IEEE-CIS final run | 64 (79) | August baseline seed 51 in `model_results.csv`. GraphSAGE and GAT are the earlier unsuffixed August GNN runs, not the later `-main` runs used in Table 4.7. GCN also matches `GCN-main` seed 51 at displayed precision. |
| 4.10 Elliptic final run | 65 (80) | Five rows match June seed 42, with additional archival matches. GAT matches only the archived May 30 row described below. This table does not represent the final August seed-51 batch. |
| 4.11 Sampled-complement ablation | 67 (82) | All six rows match the June seed-42 complement results in `model_results.csv`, lines 14–19. The PDF claims graph degree 20, but the result rows do not independently establish the IEEE-CIS graph degree. |
| 4.12 IEEE-CIS Wilcoxon findings | 70 (85) | Every displayed p-value, higher-mean model and significance label reproduces from August paired seeds. Both the unsuffixed and `-main` August GNN alternatives give the same printed p-values for these comparisons, so this table cannot distinguish those GNN batches. It does not match the saved June Wilcoxon table as a whole. |
| 4.13 Elliptic Wilcoxon findings | 71 (86) | Every displayed p-value, higher-mean model and significance label reproduces from August paired seeds, including recovered GAT seed 51. The Transformer/LSTM PR-AUC p-value distinguishes this from the June analysis. |

## Exact final-run rows and validation-selected thresholds

Table 4.9 maps to these August 30 seed-51 rows in `results/model_results.csv`:

| Printed model | CSV line | Selected threshold | Checkpoint in `results/` |
|---|---:|---:|---|
| LSTM | 183 | 0.91 | `best_lstm_seed51_model.pt` |
| Transformer | 184 | 0.91 | `best_transformer_seed51_model.pt` |
| MLP | 182 | 0.92 | `best_mlp_seed51_model.pt` |
| GraphSAGE | 186 | 0.90 | `best_sage_seed51_model.pt` |
| GAT | 187 | 0.80 | `best_gat_seed51_model.pt` |
| GCN | 185 | 0.72 | `best_gcn_seed51_model.pt` |

The GNN graph for this unsuffixed August batch is `results/ieee_cis_graph_trainfit.pt`. GCN also matches `results/ieee_cis_graph_model_results.csv` line 11, threshold 0.72, with `best_gcn_main_seed51_model.pt` and `ieee_cis_graph_trainfit_main_max1000.pt`. All Table 4.9 rows evaluate 59,054 IEEE-CIS IDs, 3518486–3577539 inclusive. Transformer uses the later capped-validation-F1 strategy; the other listed models use validation F1 selection.

Table 4.10 maps as follows. Source lines are physical file lines, including the header.

| Printed model | Seed recorded in matching June batch | Matching June `model_results.csv` line | Threshold | Other provenance detail |
|---|---:|---:|---:|---|
| Transformer | 42 | 4 | 0.89 | Also matches older archived rows; 8,832 test transactions. |
| LSTM | 42 | 3 | 0.98 | Also matches older archived rows; 8,832 test transactions. |
| MLP | 42 | 2 | 0.48 | Also matches August seed 42 and older archived rows; 8,841 test transactions. |
| GraphSAGE | 42 | 6 | 0.80 | Also matches older archived rows; 8,841 test transactions. |
| GAT | **Not recorded** | **No June-batch match** | 0.91 | Matches `results/model_results_old.csv` line **85**, timestamp **2026-05-30T12:31:59**. |
| GCN | 42 | 5 | 0.50 | Also matches August seed 42 and older archived rows; 8,841 test transactions. |

The archived GAT row has test confusion matrix `[[8100,333],[245,163]]` and PR-AUC `0.3251` when rounded as in the thesis. Its archival CSV has no seed field. The May 30 source snapshot hardcodes seed 42, which is consistent with that run, but the row does not itself establish its seed. It should not be relabeled as a verified June or August seed-42 row.

For the original Elliptic sequences use [the 8,832-ID list](../elliptic_original_sequence_test_ids.json). Other Elliptic models use [the 8,841-ID list](../elliptic_full_test_ids.json). The shared native GNN graph is `results/elliptic_graph.pt`. The currently saved seed-specific checkpoints are August files, so their names do not preserve independently verified weights for Table 4.10's older sequence runs or the May GAT run. No per-transaction prediction scores were found in the earlier audit.

## Differences and limits revealed by the PDF

1. **The aggregate and final-run tables are not drawn consistently from the same batch.** IEEE-CIS Table 4.7 uses construction-tagged `-main` GNNs; Table 4.9 uses the earlier August GraphSAGE/GAT reruns. Elliptic Table 4.8 matches August means, but Table 4.10 contains older results with original sequence warm-up exclusions.

2. **The removed Elliptic-GAT result was used in the aggregate.** Table 4.8's GAT row reproduces from ten August runs only when the recovered seed-51 result is included. Its threshold is 0.92; its test PR-AUC is 0.28934854624693335. The recovery remains separate from the original CSV at [recovered_elliptic_gat_seed51.json](../recovered_elliptic_gat_seed51.json).

3. **The saved general Wilcoxon CSV is older than the thesis's statistical tables.** Recomputed exact two-sided signed-rank tests reproduce all 21 printed comparisons from August data. Each comparison has ten nonzero differences with no tied absolute ranks, so all 1,024 possible rank-sign assignments were enumerated. For example, IEEE-CIS MLP/LSTM PR-AUC is 0.2324 in the thesis/August calculation, versus 0.0020 for June. Elliptic Transformer/LSTM PR-AUC is 0.0273 in the thesis/August calculation, versus 0.0020 for June. These matches do not turn the saved `wilcoxon_model_comparisons.csv` into an August output.

4. **The printed main-graph edge counts do not exactly match the saved graph.** Table 4.1 prints **406,767** IEEE-CIS main edges. The saved main cache has **406,769** undirected edges and no self-loop columns. Table 4.4 prints main edge-class counts **8,475 / 18,644 / 379,648**, summing to 406,767. The saved `ieee_cis_graph_trainfit_main_max1000.pt` yields **8,476 / 18,647 / 379,646**, summing to 406,769. This is an unresolved numerical difference, not a transcription correction. All four alternative-construction class-count rows match exactly. See [graph_statistics_check.json](graph_statistics_check.json).

5. **Graph degree remains a provenance limitation for the IEEE-CIS complement runs.** The PDF consistently describes degree 20, and Table 4.11 identifies the same six June result rows. Both degree-5 and degree-20 IEEE-CIS caches exist, while the result and checkpoint names do not record degree. The text establishes the thesis's stated experiment, but cannot prove which cache generated each saved row.

No source PDF, experimental code, original result CSV, checkpoint, or dataset was changed. The extraction preserves the printed values, and this companion report records differences separately. Full row-level links to the earlier artifact audit are in [table_matches.json](table_matches.json).
