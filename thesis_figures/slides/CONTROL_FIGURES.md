# Additional thesis figures

Run `python make_thesis_control_figures.py` from the project root.

This script checks all ten seeds, verifies prediction hashes and paired test IDs, recomputes subgroup AP and control AP/F1, and checks the saved summaries. No training is performed. PDFs are vector figures; PNGs are 300 dpi. Numeric data are in the accompanying CSVs. `control_figures_provenance.json` records input hashes and graph settings. `control_figures.tex` contains insertion-ready LaTeX and captions.

## main/06_connected_vs_isolated

IEEE-CIS connected/isolated AP on identical test IDs across models within each construction. Ten-seed means ± sample SD; group sizes and fraud rates are shown. Degree excludes self-loops. Address and email connected groups contain only 37 and 65 fraud cases; PR-AUC is not comparable across groups because prevalence differs. MLP predictions are reused across constructions.

## card/06_connected_vs_isolated

IEEE-CIS connected/isolated AP on identical test IDs across models within each construction. Ten-seed means ± sample SD; group sizes and fraud rates are shown. Degree excludes self-loops. Address and email connected groups contain only 37 and 65 fraud cases; PR-AUC is not comparable across groups because prevalence differs. MLP predictions are reused across constructions.

## address/06_connected_vs_isolated

IEEE-CIS connected/isolated AP on identical test IDs across models within each construction. Ten-seed means ± sample SD; group sizes and fraud rates are shown. Degree excludes self-loops. Address and email connected groups contain only 37 and 65 fraud cases; PR-AUC is not comparable across groups because prevalence differs. MLP predictions are reused across constructions.

## email_domain/06_connected_vs_isolated

IEEE-CIS connected/isolated AP on identical test IDs across models within each construction. Ten-seed means ± sample SD; group sizes and fraud rates are shown. Degree excludes self-loops. Address and email connected groups contain only 37 and 65 fraud cases; PR-AUC is not comparable across groups because prevalence differs. MLP predictions are reused across constructions.

## identifier_time_window-24h/06_connected_vs_isolated

IEEE-CIS connected/isolated AP on identical test IDs across models within each construction. Ten-seed means ± sample SD; group sizes and fraud rates are shown. Degree excludes self-loops. Address and email connected groups contain only 37 and 65 fraud cases; PR-AUC is not comparable across groups because prevalence differs. MLP predictions are reused across constructions.

## local/07_elliptic_feature_topology_controls

Elliptic controls: full (165) versus local-only (93) features for all six models; original versus degree-preserving rewired topology for the three GNNs. Circles show AP, open squares F1. Mean ± sample SD over seeds 42–51. Rewired edges remain within time steps; full feature values are retained. One graph seed, 20261013, is reused, so error bars measure training-seed variation, not graph-realization variation. Original historical runs and the validated sparse GNN control backend can differ in floating-point reduction order.

## random/07_elliptic_feature_topology_controls

Elliptic controls: full (165) versus local-only (93) features for all six models; original versus degree-preserving rewired topology for the three GNNs. Circles show AP, open squares F1. Mean ± sample SD over seeds 42–51. Rewired edges remain within time steps; full feature values are retained. One graph seed, 20261013, is reused, so error bars measure training-seed variation, not graph-realization variation. Original historical runs and the validated sparse GNN control backend can differ in floating-point reduction order.
