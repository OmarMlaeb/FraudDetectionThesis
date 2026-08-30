# Financial Fraud Detection Using GNNs and Deep Learning

This project is part of a thesis titled:

**Detecting Financial Fraud Using Graph Neural Networks in Comparison with Advanced Deep Learning Models**

The goal of this project is to compare traditional deep learning models such as MLP, LSTM, and Transformer with Graph Neural Network models such as GCN, GraphSAGE, and GAT for financial fraud detection.

The experiments use benchmark fraud detection datasets:

- IEEE-CIS Fraud Detection Dataset
- Elliptic Bitcoin Dataset

The models are evaluated using metrics suitable for imbalanced fraud detection:

- Recall
- Precision
- F1-score
- PR-AUC
- ROC-AUC

Graph datasets are summarized using:

- Number of nodes
- Number of edges
- Average degree
- Connected components
- Graph density

Training uses PR-AUC based early stopping:

- MLP, LSTM, Transformer: 50 maximum epochs, patience 10
- GCN, GraphSAGE, GAT: 100 maximum epochs, patience 15

## Project Structure

```text
fraud-detection-thesis/
|-- data/
|   |-- ieee-cis/
|   `-- elliptic_bitcoin_dataset/
|-- src/
|   |-- common/
|   |   `-- metrics.py
|   |-- ieee_cis/
|   |   |-- preprocessing.py
|   |   |-- models.py
|   |   |-- gnn_models.py
|   |   |-- graph_data.py
|   |   `-- train_*.py
|   |-- elliptic/
|   |   |-- data.py
|   |   |-- models.py
|   |   |-- gnn_models.py
|   |   |-- sequence.py
|   |   `-- train_*.py
|   |-- run_ieee_all.py
|   `-- run_elliptic_all.py
|-- results/
|-- requirements.txt
`-- README.md
```

## Setup
1. Create a virtual environment
On Windows:
```powershell
python -m venv venv
```

2. Activate the virtual environment
On Windows:
```powershell
venv\Scripts\activate
```

3. Upgrade pip:
```powershell
pip install --upgrade pip
```

4. Install Requiremnts:
```powershell
pip install -r requirements.txt
```

5. Then install PyTorch separately with:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

6. To test if gpu is recognized:
```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Run Experiments

Run every IEEE-CIS model:

```powershell
python src/run_ieee_all.py
```

Run every Elliptic Bitcoin model:

```powershell
python src/run_elliptic_all.py
```

By default, these commands run each model with seeds `42`, `43`, and `44`.
To choose different seeds for repeated experiments, pass them explicitly:

```powershell
python src/run_ieee_all.py --seeds 45 46 47
python src/run_elliptic_all.py --seeds 45 46 47
```

Build graph data and print graph statistics:

```powershell
$env:PYTHONPATH="src"; python -m ieee_cis.build_graph --graph-construction card --rebuild-graph
$env:PYTHONPATH="src"; python -m elliptic.build_graph --rebuild-graph
```

IEEE-CIS supports the original main graph plus four meaningful graph constructions.
Build each one alone:

```powershell
$env:PYTHONPATH="src"; python -m ieee_cis.build_graph --graph-construction main --rebuild-graph
$env:PYTHONPATH="src"; python -m ieee_cis.build_graph --graph-construction card --rebuild-graph
$env:PYTHONPATH="src"; python -m ieee_cis.build_graph --graph-construction address --rebuild-graph
$env:PYTHONPATH="src"; python -m ieee_cis.build_graph --graph-construction email_domain --rebuild-graph
$env:PYTHONPATH="src"; python -m ieee_cis.build_graph --graph-construction identifier_time_window --identifier-window-hours 24 --rebuild-graph
```

Each IEEE-CIS graph build saves a separate edge-list CSV in `results/`, for
example:

- `results/ieee_cis_graph_edges_main_max1000.csv`
- `results/ieee_cis_graph_edges_card_max1000.csv`
- `results/ieee_cis_graph_edges_address_max1000.csv`
- `results/ieee_cis_graph_edges_email_domain_max1000.csv`
- `results/ieee_cis_graph_edges_identifier_time_window_24h.csv`

Train a GNN on one IEEE-CIS construction at a time:

```powershell
$env:PYTHONPATH="src"; python -m ieee_cis.train_gnn --model gcn --graph-construction main --rebuild-graph
$env:PYTHONPATH="src"; python -m ieee_cis.train_gnn --model gcn --graph-construction card --rebuild-graph
$env:PYTHONPATH="src"; python -m ieee_cis.train_gnn --model gcn --graph-construction address --rebuild-graph
$env:PYTHONPATH="src"; python -m ieee_cis.train_gnn --model gcn --graph-construction email_domain --rebuild-graph
$env:PYTHONPATH="src"; python -m ieee_cis.train_gnn --model gcn --graph-construction identifier_time_window --identifier-window-hours 24 --rebuild-graph
```

IEEE-CIS graph-model metrics are saved separately to:

- `results/ieee_cis_graph_model_results.csv`

To choose another results file, pass `--output-path`:

```powershell
$env:PYTHONPATH="src"; python -m ieee_cis.train_gnn --model sage --graph-construction card --output-path results/ieee_cis_card_results.csv
```

Run graph models only for seeds `42` through `51` across the corrected main
graph and all four IEEE-CIS graph constructions:

```powershell
python src/run_ieee_graph_constructions.py --rebuild-graph
```

After the graph runs finish, rank IEEE-CIS models and compare the constructed
graphs against the corrected main graph and non-graph baselines:

```powershell
python src/compare_ieee_graph_constructions.py
```

This writes:

- `results/ieee_cis_graph_combined_rankings.csv`
- `results/ieee_cis_graph_vs_baseline_comparisons.csv`

For a shorter identifier graph, use a one-hour window:

```powershell
$env:PYTHONPATH="src"; python -m ieee_cis.train_gnn --model gcn --graph-construction identifier_time_window --identifier-window-hours 1 --rebuild-graph
```

Random sampled-complement graphs are not used as the main IEEE-CIS graph
construction because they do not represent meaningful transaction relationships.
They remain available only as a stress-test graph variant:

```powershell
$env:PYTHONPATH="src"; python -m elliptic.build_graph --rebuild-graph --graph-variant sampled_complement --complement-average-degree 20
$env:PYTHONPATH="src"; python -m elliptic.train_gnn --model gat --graph-variant sampled_complement --rebuild-graph
```

For individual GNN runs, use `--seed` to produce paired repeated runs:

```powershell
$env:PYTHONPATH="src"; python -m ieee_cis.train_gnn --model gcn --seed 43
$env:PYTHONPATH="src"; python -m elliptic.train_gnn --model sage --seed 43
```

Graph models compare validation thresholds before final testing. By default they
select the threshold that maximizes validation F1:

```powershell
$env:PYTHONPATH="src"; python -m ieee_cis.train_gnn --model gcn --threshold-strategy f1
$env:PYTHONPATH="src"; python -m elliptic.train_gnn --model gat --threshold-strategy f1
```

To prefer a threshold that balances recall and precision, use:

```powershell
$env:PYTHONPATH="src"; python -m ieee_cis.train_gnn --model sage --threshold-strategy balanced
$env:PYTHONPATH="src"; python -m elliptic.train_gnn --model sage --threshold-strategy balanced
```

All final metrics are appended to `results/model_results.csv`.

Rank the latest result for each model on each dataset:

```powershell
python src/rank_models.py
```

Rankings are printed in the terminal and saved to:

- `results/model_rankings.csv` for PR-AUC ranking
- `results/model_rankings_by_f1.csv` for F1 ranking
- `results/model_rankings_by_recall.csv` for recall ranking

Run Wilcoxon signed-rank tests across repeated model runs:

```powershell
python src/run_wilcoxon_tests.py
```

The test pairs models by run order within each dataset and metric, then saves
pairwise significance results to:

- `results/wilcoxon_model_comparisons.csv`
