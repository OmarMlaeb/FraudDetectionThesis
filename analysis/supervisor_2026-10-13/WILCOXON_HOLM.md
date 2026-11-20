# Wilcoxon signed-rank tests with Holm correction

All 240 selected runs passed checkpoint reproduction. Tests use full-precision saved per-seed metrics from those accepted runs. The source-verified and inference-verified analyses now agree for every comparison. Exact sign permutations use average absolute ranks, drop exact zeros and retain all ten paired observations in the reported pair count. See METHODS.md for definitions fixed before results were interpreted.

## Complete family coverage

| Dataset | Family | Hypotheses per metric | Metrics | Total | Significant after Holm |
|---|---|---:|---:|---:|---:|
| IEEE-CIS | core | 15 | 6 | 90 | 43 |
| Elliptic | core | 15 | 6 | 90 | 56 |
| IEEE-CIS | construction | 30 | 6 | 180 | 0 |
| IEEE-CIS | alternative_gnn_vs_mlp | 12 | 6 | 72 | 50 |

The metrics are average precision, F1, recall, precision, ROC-AUC and false positives. There are **432 tests in 24 families**, each with ten paired seeds. Holm is applied within each dataset/family/metric, never across a selected subset of printed findings. Construction comparisons pool all 30 within-architecture graph pairs; the alternative-GNN/MLP family includes all 12 non-main graph/architecture variants.

## Selected changes to interpretation

| Dataset/family | A minus B | Metric | Median paired difference | Raw p | Holm p | Significant after Holm |
|---|---|---|---:|---:|---:|---|
| IEEE-CIS__core__pr_auc | MLP minus GraphSAGE-main | AP | 0.09014334 | 0.00585938 | 0.03515625 | Yes |
| IEEE-CIS__construction__pr_auc | GraphSAGE-main minus GraphSAGE-email_domain | AP | -0.03233220 | 0.00195312 | 0.05859375 | No |
| IEEE-CIS__alternative_gnn_vs_mlp__pr_auc | GraphSAGE-email_domain minus MLP | AP | -0.04179113 | 0.06445312 | 0.09765625 | No |
| Elliptic__core__pr_auc | LSTM minus Transformer | AP | -0.02012550 | 0.02734375 | 0.08203125 | No |

None of the construction comparisons survives the prespecified 30-hypothesis-per-metric family. With ten nonzero paired differences, the smallest possible two-sided exact signed-rank p-value is 2/1024 = 0.001953125; the first Holm step for 30 tests is at least 0.05859375. This is a limitation of the sample size and chosen family, not evidence of equivalence or absence of a construction effect. The family was not narrowed after observing this result.

Elliptic Transformer versus LSTM PR-AUC has raw p=0.02734375 but Holm p=0.08203125 in the complete core AP family. The median difference shown above is LSTM minus Transformer, so its negative sign favors Transformer descriptively. A non-significant adjusted comparison does not establish equal performance.

Every comparison, statistic, raw/adjusted p-value, total/nonzero/zero pair count, tie flag and median paired difference is in `wilcoxon_holm_inference_verified.csv`. `paired_differences_inference_verified.json` exposes the ten full-precision differences for every comparison. False-positive differences are counts; negative A-minus-B favors A for that metric.
