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

Build graph data and print graph statistics:

```powershell
$env:PYTHONPATH="src"; python -m ieee_cis.build_graph --rebuild-graph
$env:PYTHONPATH="src"; python -m elliptic.build_graph --rebuild-graph
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
