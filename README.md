# Financial Fraud Detection Using GNNs and Deep Learning

This project is part of a thesis titled:

**Detecting Financial Fraud Using Graph Neural Networks in Comparison with Advanced Deep Learning Models**

The goal of this project is to compare traditional deep learning models such as MLP, LSTM, and Transformer with Graph Neural Network models such as GCN, GraphSAGE, and GAT for financial fraud detection.

The experiments use benchmark fraud detection datasets such as:

- IEEE-CIS Fraud Detection Dataset
- Elliptic Bitcoin Dataset

The models are evaluated using metrics suitable for imbalanced fraud detection:

- Recall
- Precision
- F1-score
- PR-AUC
- ROC-AUC

---

## Project Structure

```text
fraud-detection-thesis/
│
├── data/
│   └── ieee-cis/
│       ├── train_transaction.csv
│       └── train_identity.csv
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── metrics.py
│   └── train_mlp.py
│
├── results/
│
├── requirements.txt
└── README.md