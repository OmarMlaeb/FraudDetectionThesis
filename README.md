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

## Run Experiments

Run every IEEE-CIS model:

```powershell
python src/run_ieee_all.py
```

Run every Elliptic Bitcoin model:

```powershell
python src/run_elliptic_all.py
```

All final metrics are appended to `results/model_results.csv`.
