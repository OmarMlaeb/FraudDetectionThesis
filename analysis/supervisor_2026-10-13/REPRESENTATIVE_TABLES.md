# Proposed replacement Tables 4.9 and 4.10

These tables use **seed 51 from the same accepted August experiment sets as Tables 4.7 and 4.8**. IEEE-CIS GNNs are construction-tagged main runs. All twelve checkpoints passed reproduction. Original thesis/result files were not modified. Metrics below are formatted to four decimals; the companion CSV retains original full precision.

## Proposed Table 4.9: IEEE-CIS, seed 51

| Model | Recall | Precision | F1 | AP | ROC-AUC | TN | FP | FN | TP | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 0.3972 | 0.5072 | 0.4455 | 0.4287 | 0.8562 | 55,987 | 854 | 1,334 | 879 | 0.91 |
| Transformer | 0.4180 | 0.4695 | 0.4423 | 0.3975 | 0.8389 | 55,796 | 1,045 | 1,288 | 925 | 0.91 |
| MLP | 0.3439 | 0.4961 | 0.4062 | 0.3298 | 0.8616 | 56,068 | 773 | 1,452 | 761 | 0.92 |
| GraphSAGE-main | 0.3913 | 0.3775 | 0.3843 | 0.3446 | 0.8549 | 55,413 | 1,428 | 1,347 | 866 | 0.89 |
| GAT-main | 0.4189 | 0.1921 | 0.2634 | 0.1374 | 0.8177 | 52,943 | 3,898 | 1,286 | 927 | 0.8 |
| GCN-main | 0.3678 | 0.1729 | 0.2353 | 0.1232 | 0.8004 | 52,948 | 3,893 | 1,399 | 814 | 0.72 |
## Proposed Table 4.10: Elliptic, seed 51

| Model | Recall | Precision | F1 | AP | ROC-AUC | TN | FP | FN | TP | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transformer | 0.3725 | 0.8994 | 0.5269 | 0.4887 | 0.8010 | 8,416 | 17 | 256 | 152 | 0.93 |
| LSTM | 0.4142 | 0.7191 | 0.5257 | 0.5060 | 0.8434 | 8,367 | 66 | 239 | 169 | 0.97 |
| MLP | 0.4167 | 0.3491 | 0.3799 | 0.4242 | 0.7626 | 8,116 | 317 | 238 | 170 | 0.49 |
| GAT | 0.3456 | 0.3711 | 0.3579 | 0.2893 | 0.8166 | 8,194 | 239 | 267 | 141 | 0.92 |
| GraphSAGE | 0.3382 | 0.2942 | 0.3147 | 0.3485 | 0.7735 | 8,102 | 331 | 270 | 138 | 0.52 |
| GCN | 0.5760 | 0.1391 | 0.2240 | 0.1751 | 0.7841 | 6,978 | 1,455 | 173 | 235 | 0.54 |

Elliptic-GAT seed 51 comes from `git:2be96ee:results/model_results.csv`, line 247, timestamp 2026-08-30T13:24:40, deleted by 5333c55. It is explicitly recovered, not substituted. All Elliptic models here evaluate 8,841 targets; all IEEE-CIS models evaluate 59,054. The older Table 4.10 warm-up exclusion does not apply.
