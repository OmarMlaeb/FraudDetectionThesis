# Run the three experiments

Open PowerShell in `C:\Users\user\Desktop\FraudDetectionThesis`.

First, run this once in your terminal so `python` uses the working environment:

```powershell
Set-Alias python "$PWD\analysis\supervisor_2026-10-13\runtime\python310\python.exe"
```

Then use one command for each task:

```powershell
python run_sampled_complement_fast.py
python run_elliptic_local_features_fast.py
python run_random_graph_fast.py
```

**Use these fast commands for the remaining runs.** They use sparse graph operations for GCN, GraphSAGE and GAT while keeping all ten seeds, degree 20, epochs, patience, model sizes and other experiment settings. Your six completed complement results are checked and reused. Interrupted seed 48 starts again; its previous attempt is preserved.

Run them one at a time. Each script prepares its graphs, checks them, trains all its models across **seeds 42-51**, and saves results automatically. Repeat the same command to resume; completed runs are checked and skipped. The original commands without `_fast` remain available and unchanged.

| Command | What it runs | Runs |
|---|---|---:|
| `python run_sampled_complement_fast.py` | IEEE-CIS main and Elliptic; GCN, GraphSAGE, GAT; sampled nonedges replacing original edges, average undirected degree **20** | 60 |
| `python run_elliptic_local_features_fast.py` | Elliptic; all six models; **93 local features**, excluding the 72 neighbor-aggregate features | 60 |
| `python run_random_graph_fast.py` | All five IEEE-CIS constructions and Elliptic; GCN, GraphSAGE, GAT; **exactly the original degree of every node** | 180 |

## Your CSV files

Find the combined CSVs from the fast commands in **`results/committee_fast_experiments/`**:

- `sampled_complement_results.csv`
- `elliptic_local_features_results.csv`
- `random_graph_results.csv`

These use **exactly the same columns and column order as the old `results/model_results.csv`**: timestamp, model, recall, precision, F1, PR-AUC, ROC-AUC, TN/FP/FN/TP, threshold, threshold strategy, validation F1, and seed. A row is saved after each completed and checked run. Resuming does not duplicate rows. Dataset, construction and experiment are identified in the model name.

Each script also saves:

- `*_graph_statistics.csv`: original/new edge counts, average degrees, **density**, isolated nodes, components, overlap, degree-preservation checks and rewiring information.
- `*_summary.csv`: mean and sample SD across the ten seeds, with missing seeds clearly listed.
- `*_comparisons.csv`: paired comparisons against the verified original results, including median differences and Wilcoxon/Holm p-values.
- `*_failures.csv`: failed runs, if any. Detailed checkpoints, predictions and logs are retained in the same experiment folder.
- `*_execution_provenance.csv`: identifies reused original runs and new sparse runs, with artifact paths and hashes. Original files remain in `results/committee_additional_experiments/`.

The `*` is the same name as the results file, e.g. `sampled_complement_graph_statistics.csv`.

## Optional commands

Check the plan without doing any training or graph generation:

```powershell
python run_sampled_complement_fast.py --dry-run
```

Generate the graphs and see their density/statistics **before** training:

```powershell
python run_sampled_complement_fast.py --graphs-only
```

Refresh the CSV summaries from saved runs, without training:

```powershell
python run_sampled_complement_fast.py --summarize-only
```

These options work with all three scripts. You do not need them for the normal workflow.

The graph seed is fixed at **20261013**, independently of training seeds. Each generated graph is reused across all ten training seeds. Complement graphs change density; randomized graphs preserve every node's degree. Randomized Elliptic edges stay within time steps, and the IEEE 24-hour construction retains its time limit. Original results and the completed supervisor analyses are preserved. Full configuration and statistical details remain in [METHODS.md](METHODS.md).

Sparse operations passed synthetic output/gradient checks and saved-checkpoint inference checks. Reduction order introduces small floating-point differences; future training is not guaranteed bitwise identical. See [the speed-check details](../committee_fast/README.md). No full training-time estimate is claimed from inference timings.

**Training remains yours to launch. Codex ran verification and inference only, with no optimizer updates or full-dataset graph generation.**
