import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from common.metrics import evaluate_binary_classification, save_results_to_csv
from .data import load_or_build_elliptic_graph
from .gnn_models import create_gnn_model


EPOCHS = 20
LEARNING_RATE = 0.001


def evaluate(model, data, mask):
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.sigmoid(logits[mask]).cpu().numpy()
        targets = data.y[mask].cpu().numpy()

    return evaluate_binary_classification(targets, probs)


def train(model_name, epochs=EPOCHS, rebuild_graph=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Training Elliptic GNN model: {model_name.upper()}")

    data = load_or_build_elliptic_graph(rebuild=rebuild_graph)
    data = data.to(device)

    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.edge_index.size(1)}")
    print(f"Features: {data.x.size(1)}")
    print(f"Train nodes: {int(data.train_mask.sum())}")
    print(f"Validation nodes: {int(data.val_mask.sum())}")
    print(f"Test nodes: {int(data.test_mask.sum())}")

    model = create_gnn_model(model_name, data.x.size(1)).to(device)

    train_targets = data.y[data.train_mask]
    positive_count = train_targets.sum()
    negative_count = len(train_targets) - positive_count
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_val_pr_auc = 0
    best_model_path = f"results/best_elliptic_{model_name}_model.pt"
    Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        val_results = evaluate(model, data, data.val_mask)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {loss.item():.4f} "
            f"Val Recall: {val_results['Recall']:.4f} "
            f"Val F1: {val_results['F1-score']:.4f} "
            f"Val PR-AUC: {val_results['PR-AUC']:.4f} "
            f"Val ROC-AUC: {val_results['ROC-AUC']:.4f}"
        )

        if val_results["PR-AUC"] > best_val_pr_auc:
            best_val_pr_auc = val_results["PR-AUC"]
            torch.save(model.state_dict(), best_model_path)

    print("\nLoading best model for testing...")

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    test_results = evaluate(model, data, data.test_mask)

    model_label = {
        "gcn": "Elliptic-GCN",
        "sage": "Elliptic-GraphSAGE",
        "gat": "Elliptic-GAT",
    }[model_name]

    print(f"\nFinal Test Results - {model_label}")
    print("--------------------------------")
    print(f"Recall: {test_results['Recall']:.4f}")
    print(f"Precision: {test_results['Precision']:.4f}")
    print(f"F1-score: {test_results['F1-score']:.4f}")
    print(f"PR-AUC: {test_results['PR-AUC']:.4f}")
    print(f"ROC-AUC: {test_results['ROC-AUC']:.4f}")
    print("Confusion Matrix:")
    print(test_results["Confusion Matrix"])

    save_results_to_csv(model_label, test_results)
    print("\nSaved test results to results/model_results.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Elliptic Bitcoin GNN baselines.")
    parser.add_argument("--model", choices=["gcn", "sage", "gat"], required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--rebuild-graph", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_name=args.model,
        epochs=args.epochs,
        rebuild_graph=args.rebuild_graph,
    )
