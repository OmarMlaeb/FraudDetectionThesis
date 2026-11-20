# Supervisor analyses, 2026-10-13

Both priority analyses are complete. **240/240 August checkpoints passed** the predefined whole-test and validation reproduction gate, including all 200 GNN/MLP runs needed for subgroups and all 40 sequence runs. No retraining or original-result/graph/checkpoint replacement occurred.

- [Connected-versus-isolated report](SUBGROUP_ANALYSIS.md): 12 groups, 480 per-seed metric rows, 48 mean/sample-SD summaries, and 240 exact confusion-count reconciliations. MLP uses identical group IDs and the same saved predictions across constructions.
- [Wilcoxon/Holm report](WILCOXON_HOLM.md): all 432 comparisons across 24 prespecified dataset/metric/family combinations, ten paired seeds each, raw/adjusted p-values and median paired differences.
- [Graph discrepancy](GRAPH_PROVENANCE.md): runtime-dependent tie ordering reproduces the 406,767 report; inference uses the verified 406,769 saved graph. Transaction-ID edge differences and hashes are supplied.
- [Representative tables](REPRESENTATIVE_TABLES.md): proposed Tables 4.9/4.10 use seed 51 from these same verified experiment sets, including explicitly recovered Elliptic-GAT.
- [Sequence inputs](SEQUENCE_INPUTS.md): selected August sequences retain all targets with split-local left-padding. The older final-run table is not used to infer their behavior.
- [Predeclared methods/families](METHODS.md), plus `comparison_families.json` with every planned comparison.

## Essential files

| Output | Content |
|---|---|
| `selected_per_seed_results.csv`, `selected_runs.json` | 240 selected rows with original full-precision strings, seed, source file/line and historical source snapshot. |
| `run_manifest.json` | Selected rows joined to verified checkpoints, graph hashes, test-ID mappings and predictions. |
| `recovered_result_provenance.json` | Exact Elliptic-GAT seed-51 row, Git commit/source line, historical CSV hash and deletion diff. |
| `artifact_hashes_before.json` | Paths and SHA-256 hashes for required data, graph caches, checkpoints and code. |
| `additional_referenced_artifact_hashes.json` | Hashes/paths of other saved graph/checkpoint/result files referenced by the archived audit. |
| `original_artifact_preservation.json` | All 304 original input artifacts rehashed after analysis; no changes. |
| `feature_checks_*.json`, `reconstructed_ieee_preprocessor.json` | Exact feature/label/split reconstruction checks and explicitly reconstructed train-fitted preprocessing parameters. |
| `checkpoint_reproduction_checks.csv`, `checks/*.json` | Whole-test metric deltas, exact confusion checks, validation F1, reselected threshold and acceptance per run. |
| `predictions/*.npz` | New inference probabilities with validation/test transaction IDs and labels, saved threshold; these are not claimed to be originally saved predictions. |
| `node_mappings/*.npz`, `subgroup_ids/*.npz` | Full node/ID/label/split mappings and exact subgroup IDs. |
| `subgroup_counts.csv`, `subgroup_per_seed.csv`, `subgroup_summary.csv` | Counts/fraud rates, all individual-seed metrics, mean and sample SD, defined-value counts. JSON equivalents preserve nulls. |
| `subgroup_confusion_reconciliation.csv` | Every subgroup pair sums exactly to its saved whole-test confusion matrix. |
| `wilcoxon_holm_inference_verified.csv` | Final complete statistical analysis, including all comparison families. |
| `wilcoxon_holm_source_verified.csv` | Analysis prepared before inference; identical p-values/effects after all runs passed. |
| `paired_differences_*.json` | Every full-precision seed pair and A-minus-B difference. |
| `representative_seed51.csv` | Full-precision proposed representative rows, source references and verification status. |
| `graph_diagnostics/*` | Explicit diagnostic reconstructions, transaction-ID edge differences, counts and fingerprints. |
| `source_snapshots/*` | Relevant August source files from Git. |

## Unresolved issues and interpretation

No missing selected run or failed checkpoint blocks either priority analysis. Elliptic's isolated labeled test group is empty, so its performance metrics are undefined. The original 406,767-edge reconstruction file was not saved; the controlled reconstruction reproduces its reported total and class counts but cannot prove historical edge identity. All selected experiments instead reproduce on their saved caches. Ranking scores have small GPU floating-point/tie differences: maximum absolute AP difference 1.77607274794e-05, ROC-AUC 1.58996246391e-07; every confusion matrix matches exactly. See `blockers_and_limitations.json`.

The complete 30-comparison construction families yield no Holm-significant results at alpha .05 with ten seeds. Do not describe this as evidence of equivalence. The full report explains the discrete p-value limit and reports median effects.

## Reproduction

Run from the original repository root with the packages in `environment.json`: Python 3.10.0, torch 2.5.1+cu121, PyG 2.7.0, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2 and SciPy 1.15.3. Analysis used an NVIDIA RTX 3080 Ti. A portable official Python runtime was downloaded under this analysis folder because the venv's original executable was missing; it reuses the existing venv packages. The downloaded runtime and packages are excluded from the ZIP. Record of its URL and hash is in `environment.json`.

Use `python` below for that compatible interpreter. Scripts write only inside the separate analysis folder. Existing per-run check files are resumable; use a fresh copy of the analysis scripts and required inputs to rerun inference from scratch. Do not rerun blindly over a changed experiment set: verify the saved input hashes first.

```powershell
python analysis/supervisor_2026-10-13/prepare.py
python analysis/supervisor_2026-10-13/wilcoxon_holm.py --mode source_verified
python analysis/supervisor_2026-10-13/verify_features.py IEEE-CIS
python analysis/supervisor_2026-10-13/verify_features.py Elliptic
python analysis/supervisor_2026-10-13/infer.py IEEE-CIS
python analysis/supervisor_2026-10-13/infer.py Elliptic
python analysis/supervisor_2026-10-13/subgroups.py
python analysis/supervisor_2026-10-13/infer.py IEEE-CIS --kind sequence
python analysis/supervisor_2026-10-13/infer.py Elliptic --kind sequence
python analysis/supervisor_2026-10-13/verify_sequences.py
python analysis/supervisor_2026-10-13/wilcoxon_holm.py --mode inference_verified
python analysis/supervisor_2026-10-13/diagnose_graph.py experiment_packages
$env:NPY_DISABLE_CPU_FEATURES='AVX2,AVX512F'
python analysis/supervisor_2026-10-13/diagnose_graph.py experiment_scalar_sort
Remove-Item Env:NPY_DISABLE_CPU_FEATURES
```

The additional `audit_runtime` diagnostic uses bundled Python 3.12, NumPy 2.3.5 and pandas 3.0.1. Then run `compare_graphs.py`, `finalize.py` and `package.py` from this folder with the experiment-compatible interpreter. The scripts require the original raw datasets/checkpoints at their recorded paths; preparation also requires the original Git history for recovery. They never call model training routines.

## Package contents

The ZIP preserves the workspace-relative layout and includes **the entire earlier `audit/` tree**, not only `audit/thesis/`, this complete analysis folder (except portable runtime and Python caches), source code, and the small result CSVs. Large source datasets, full tensor graph caches and checkpoints are excluded and represented by their paths/hashes. Newly computed inference probabilities and diagnostic edge arrays are included. The original PDF is represented by its source path/hash and the prior audit's extracted text/tables/rendered previews.

The earlier audit is a historical snapshot. Since that audit, commits b464cb9/55a8d11 updated connectivity analysis and the general Wilcoxon CSV. The current manifest and reproduction outputs supersede the earlier audit's statements that inference was unavailable or that the then-current Wilcoxon file was necessarily still June-only. The older audit is retained as provenance, not silently rewritten.
