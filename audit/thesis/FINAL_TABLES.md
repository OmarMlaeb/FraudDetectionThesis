# Final results tables extracted from AAI_Thesis.pdf

All 13 empirical-results tables in Chapter 4, transcribed from the supplied final PDF and visually checked against rendered table images. Table 2.1 is a literature comparison and is outside this results extraction. Values and row order are preserved as printed, including discrepancies. Only spacing, exponent notation and merged dataset labels are normalized. Source: `C:/Users/user/Desktop/AAI_Thesis.pdf`.

PDF page numbers count from the first file page. Printed thesis page numbers are also provided. [Experiment-source matching and discrepancies](TABLE_PROVENANCE.md) is separate from this faithful extraction.

## Table 4.1: Graph Statistics for Original and Sampled Complement Graphs

Printed page **53**; PDF page **68**.

| Dataset | Graph Variant | Nodes | Edges | Average Degree | Connected Components | Density | Features |
|---|---|---|---|---|---|---|---|
| IEEE-CIS | Original | 590,540 | 406,767 | 1.3776 | 255,381 | 0.00000233 | 358 |
| IEEE-CIS | Sampled Complement | 590,540 | 5,905,400 | 20.0000 | 1 | 0.00003387 | 358 |
| Elliptic | Original | 203,769 | 234,355 | 2.3002 | 49 | 0.00001129 | 165 |
| Elliptic | Sampled Complement | 203,769 | 2,037,690 | 20.0000 | 1 | 0.00009815 | 165 |

## Table 4.2: Structural Properties of Alternative IEEE-CIS Graph Constructions

Printed page **54**; PDF page **69**.

| Construction | Nodes | Edges | Average Degree | Components | Density |
|---|---|---|---|---|---|
| Card | 590,540 | 304,889 | 1.0326 | 285,651 | 1.75 × 10^-6 |
| Address | 590,540 | 7,424 | 0.0251 | 583,116 | 4.26 × 10^-8 |
| Email domain | 590,540 | 16,657 | 0.0564 | 574,435 | 9.55 × 10^-8 |
| Identifier, 24 hours | 590,540 | 345,272 | 1.1693 | 553,796 | 1.98 × 10^-6 |

## Table 4.3: Isolated Transactions in IEEE-CIS Graph Constructions

Printed page **55**; PDF page **70**.

| Construction | Isolated Nodes | Percentage (%) |
|---|---|---|
| Main | 251,474 | 42.5837 |
| Card | 274,960 | 46.5608 |
| Address | 582,899 | 98.7061 |
| Email domain | 574,412 | 97.2689 |
| Identifier, 24 hours | 549,109 | 92.9842 |

**Context / audit note:** Isolated means no non-self neighbors. The denominator is all 590,540 transactions.

## Table 4.4: Class-Specific Connectivity of Alternative IEEE-CIS Graphs

Printed page **55**; PDF page **70**.

| Construction | Fraud–Fraud | Fraud–Legitimate | Legitimate–Legitimate | Fraud Nodes with Any Neighbor (%) | Fraud Nodes with a Fraud Neighbor (%) |
|---|---|---|---|---|---|
| Main | 8,475 | 18,644 | 379,648 | 65.67 | 50.41 |
| Card | 5,124 | 11,623 | 288,142 | 54.16 | 35.46 |
| Address | 260 | 467 | 6,697 | 2.45 | 1.76 |
| Email domain | 275 | 539 | 15,843 | 2.54 | 1.82 |
| Identifier, 24 hours | 16,584 | 28,657 | 300,031 | 24.20 | 22.30 |

**Context / audit note:** Unique undirected edges, excluding self-loops. Both percentages use all fraudulent transactions as denominator. The thesis describes this as retrospective label-based structural analysis, not label-based graph construction or optimization.

## Table 4.5: IEEE-CIS GNN Performance Across Graph Constructions: Mean ± Standard Deviation Across 10 Runs

Printed page **57**; PDF page **72**.

| Model | Construction | PR-AUC | F1-score | Recall |
|---|---|---|---|---|
| GraphSAGE | Main | 0.3432±0.0534 | 0.3788±0.0190 | 0.3972±0.0172 |
| GraphSAGE | Card | 0.3174±0.0547 | 0.3706±0.0229 | 0.3951±0.0102 |
| GraphSAGE | Address | 0.3665±0.0572 | 0.3917±0.0146 | 0.4175±0.0371 |
| GraphSAGE | Email domain | 0.3912±0.0142 | 0.4025±0.0100 | 0.4099±0.0325 |
| GraphSAGE | Identifier, 24 h | 0.3284±0.0491 | 0.3900±0.0125 | 0.3815±0.0254 |
| GCN | Main | 0.2939±0.0899 | 0.3470±0.0569 | 0.3645±0.0290 |
| GCN | Card | 0.2746±0.1083 | 0.3360±0.0718 | 0.3641±0.0243 |
| GCN | Address | 0.2777±0.1041 | 0.3328±0.0564 | 0.4033±0.0394 |
| GCN | Email domain | 0.2624±0.1042 | 0.3243±0.0579 | 0.3997±0.0386 |
| GCN | Identifier, 24 h | 0.3256±0.1061 | 0.3573±0.0634 | 0.3884±0.0418 |
| GAT | Main | 0.2480±0.1002 | 0.2964±0.0640 | 0.3306±0.0757 |
| GAT | Card | 0.2176±0.0993 | 0.2745±0.0688 | 0.3096±0.0984 |
| GAT | Address | 0.2032±0.0837 | 0.2727±0.0514 | 0.3046±0.0635 |
| GAT | Email domain | 0.1711±0.0552 | 0.2578±0.0401 | 0.3251±0.0857 |
| GAT | Identifier, 24 h | 0.2342±0.0926 | 0.2942±0.0635 | 0.3296±0.0841 |

**Context / audit note:** The preceding thesis text specifies ten runs using seeds 42–51. The saved results reproduce these standard deviations with ddof=1 (sample standard deviation).

## Table 4.6: IEEE-CIS Test-Set False Positives Across Graph Constructions: Mean ± Sample Standard Deviation Across 10 Runs

Printed page **58**; PDF page **73**.

| Model | Construction | False Positives |
|---|---|---|
| GraphSAGE | Main | 1,561.4±308.8 |
| GraphSAGE | Card | 1,649.9±344.4 |
| GraphSAGE | Address | 1,586.3±364.9 |
| GraphSAGE | Email domain | 1,391.4±325.6 |
| GraphSAGE | Identifier, 24 h | 1,273.7±236.9 |
| GCN | Main | 1,771.1±1,017.8 |
| GCN | Card | 2,049.0±1,448.9 |
| GCN | Address | 2,459.0±1,374.4 |
| GCN | Email domain | 2,603.1±1,542.1 |
| GCN | Identifier, 24 h | 1,950.7±1,370.9 |
| GAT | Main | 2,048.9±866.7 |
| GAT | Card | 2,156.7±1,360.2 |
| GAT | Address | 2,042.2±403.2 |
| GAT | Email domain | 2,617.5±978.4 |
| GAT | Identifier, 24 h | 2,006.1±737.0 |

**Context / audit note:** Ten runs. False positives are measured at each run’s validation-F1-selected threshold; standard deviations are sample standard deviations.

## Table 4.7: Mean Performance Across 10 Runs on IEEE-CIS

Printed page **60**; PDF page **75**.

| Model | Recall | Precision | F1-score | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| MLP | 0.3709 | 0.5482 | 0.4407 | 0.4249 | 0.8586 |
| LSTM | 0.4006 | 0.4682 | 0.4310 | 0.4151 | 0.8558 |
| Transformer | 0.4076 | 0.4911 | 0.4433 | 0.4104 | 0.8485 |
| GraphSAGE | 0.3972 | 0.3648 | 0.3788 | 0.3432 | 0.8585 |
| GCN | 0.3645 | 0.3455 | 0.3470 | 0.2939 | 0.8431 |
| GAT | 0.3306 | 0.2798 | 0.2964 | 0.2480 | 0.8187 |

**Context / audit note:** Ten-run means. Printed GNN names omit the -main suffix used by the matching saved results.

## Table 4.8: Mean Performance Across 10 Runs on Elliptic

Printed page **62**; PDF page **77**.

| Model | Recall | Precision | F1-score | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Transformer | 0.4304 | 0.8149 | 0.5620 | 0.5125 | 0.7994 |
| LSTM | 0.4152 | 0.6647 | 0.5102 | 0.4880 | 0.8452 |
| MLP | 0.4556 | 0.3430 | 0.3827 | 0.3901 | 0.7783 |
| GraphSAGE | 0.4184 | 0.4112 | 0.4115 | 0.4089 | 0.7965 |
| GAT | 0.3875 | 0.3696 | 0.3709 | 0.3421 | 0.8185 |
| GCN | 0.4625 | 0.1746 | 0.2472 | 0.2233 | 0.7793 |

**Context / audit note:** Ten-run means. Reproducing the August GAT mean requires the seed-51 row recovered from Git.

## Table 4.9: Final-Run Results and Confusion Matrices on IEEE-CIS

Printed page **64**; PDF page **79**.

| Model | Recall | Precision | F1-score | PR-AUC | ROC-AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|
| LSTM | 0.3972 | 0.5072 | 0.4455 | 0.4287 | 0.8562 | 55,987 | 854 | 1,334 | 879 |
| Transformer | 0.4180 | 0.4695 | 0.4423 | 0.3975 | 0.8389 | 55,796 | 1,045 | 1,288 | 925 |
| MLP | 0.3439 | 0.4961 | 0.4062 | 0.3298 | 0.8616 | 56,068 | 773 | 1,452 | 761 |
| GraphSAGE | 0.3755 | 0.3976 | 0.3862 | 0.3451 | 0.8551 | 55,582 | 1,259 | 1,382 | 831 |
| GAT | 0.4248 | 0.1947 | 0.2670 | 0.1391 | 0.8182 | 52,953 | 3,888 | 1,273 | 940 |
| GCN | 0.3678 | 0.1729 | 0.2353 | 0.1232 | 0.8004 | 52,948 | 3,893 | 1,399 | 814 |

**Context / audit note:** The thesis calls these final-run results and ranks them by F1-score. Matching current baseline rows have seed 51.

## Table 4.10: Final-Run Results and Confusion Matrices on Elliptic

Printed page **65**; PDF page **80**.

| Model | Recall | Precision | F1-score | PR-AUC | ROC-AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|
| Transformer | 0.4363 | 0.7946 | 0.5633 | 0.5490 | 0.8539 | 8,378 | 46 | 230 | 178 |
| LSTM | 0.4216 | 0.6880 | 0.5228 | 0.5197 | 0.8641 | 8,346 | 78 | 236 | 172 |
| MLP | 0.4510 | 0.3333 | 0.3833 | 0.4069 | 0.7679 | 8,065 | 368 | 224 | 184 |
| GraphSAGE | 0.4363 | 0.4331 | 0.4347 | 0.4177 | 0.7883 | 8,200 | 233 | 230 | 178 |
| GAT | 0.3995 | 0.3286 | 0.3606 | 0.3251 | 0.8181 | 8,100 | 333 | 245 | 163 |
| GCN | 0.4657 | 0.1698 | 0.2489 | 0.2939 | 0.7725 | 7,504 | 929 | 218 | 190 |

**Context / audit note:** The PDF footnote states LSTM and Transformer use 8,832 test transactions because nine sequence warm-up targets are excluded. Other models use 8,841 transactions.

## Table 4.11: Sampled Complement Graph Ablation Results

Printed page **67**; PDF page **82**.

| Dataset | Model Variant | Recall | Precision | F1-score | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| IEEE-CIS | GraphSAGE-sampled complement | 0.3159 | 0.2212 | 0.2602 | 0.1424 | 0.7999 |
| IEEE-CIS | GCN-sampled complement | 0.3692 | 0.1338 | 0.1964 | 0.1027 | 0.7097 |
| IEEE-CIS | GAT-sampled complement | 0.9986 | 0.0375 | 0.0722 | 0.0407 | 0.5055 |
| Elliptic | GraphSAGE-sampled complement | 0.3578 | 0.3842 | 0.3706 | 0.2483 | 0.7581 |
| Elliptic | GCN-sampled complement | 0.3382 | 0.0864 | 0.1376 | 0.0827 | 0.6743 |
| Elliptic | GAT-sampled complement | 0.4877 | 0.0852 | 0.1451 | 0.0828 | 0.6628 |

**Context / audit note:** One sampled-complement ablation row per dataset/model. The surrounding text claims average graph degree 20. The saved result rows do not record degree, so this claim does not resolve the IEEE-CIS degree-5 versus degree-20 artifact ambiguity.

## Table 4.12: Main Wilcoxon Findings on IEEE-CIS

Printed page **70**; PDF page **85**.

| Comparison | Metric | Higher Mean Model | p-value | Significant at α=0.05 |
|---|---|---|---|---|
| MLP vs LSTM | PR-AUC | MLP | 0.2324 | No |
| MLP vs LSTM | F1-score | MLP | 0.3750 | No |
| MLP vs GCN | PR-AUC | MLP | 0.0020 | Yes |
| MLP vs GraphSAGE | PR-AUC | MLP | 0.0059 | Yes |
| MLP vs GAT | PR-AUC | MLP | 0.0020 | Yes |
| MLP vs Transformer | PR-AUC | MLP | 0.2324 | No |
| LSTM vs GCN | PR-AUC | LSTM | 0.0020 | Yes |
| LSTM vs GraphSAGE | PR-AUC | LSTM | 0.0020 | Yes |
| LSTM vs GAT | PR-AUC | LSTM | 0.0020 | Yes |
| LSTM vs Transformer | PR-AUC | LSTM | 0.5566 | No |

**Context / audit note:** Significance threshold α = 0.05. These are selected pairwise findings, not a complete matrix of comparisons.

## Table 4.13: Main Wilcoxon Findings on Elliptic

Printed page **71**; PDF page **86**.

| Comparison | Metric | Higher Mean Model | p-value | Significant at α=0.05 |
|---|---|---|---|---|
| Transformer vs LSTM | PR-AUC | Transformer | 0.0273 | Yes |
| Transformer vs LSTM | F1-score | Transformer | 0.0020 | Yes |
| Transformer vs MLP | PR-AUC | Transformer | 0.0020 | Yes |
| Transformer vs GraphSAGE | PR-AUC | Transformer | 0.0020 | Yes |
| Transformer vs GAT | PR-AUC | Transformer | 0.0020 | Yes |
| Transformer vs GCN | PR-AUC | Transformer | 0.0020 | Yes |
| LSTM vs GraphSAGE | PR-AUC | LSTM | 0.0020 | Yes |
| LSTM vs GAT | PR-AUC | LSTM | 0.0020 | Yes |
| LSTM vs GCN | PR-AUC | LSTM | 0.0020 | Yes |
| GraphSAGE vs GCN | PR-AUC | GraphSAGE | 0.0020 | Yes |
| GraphSAGE vs GAT | PR-AUC | GraphSAGE | 0.0059 | Yes |

**Context / audit note:** Significance threshold α = 0.05. These are selected pairwise findings, not a complete matrix of comparisons.

