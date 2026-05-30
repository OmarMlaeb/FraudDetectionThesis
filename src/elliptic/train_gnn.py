import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common.early_stopping import EarlyStopping
from common.metrics import (
    evaluate_binary_classification,
    find_best_threshold,
    print_threshold_comparison,
    save_results_to_csv,
)
from .data import load_or_build_elliptic_graph
from .gnn_models import create_gnn_model


EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LEARNING_RATE = 0.001


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict(model, data, mask):
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.sigmoid(logits[mask]).cpu().numpy()
        targets = data.y[mask].cpu().numpy()

    return targets, probs


def evaluate(model, data, mask, threshold=0.5):
    targets, probs = predict(model, data, mask)
    return evaluate_binary_classification(targets, probs, threshold=threshold)


def train(model_name, epochs=EPOCHS, rebuild_graph=False, threshold_strategy="f1"):
    set_seed(42)

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

    best_model_path = f"results/best_elliptic_{model_name}_model.pt"
    Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

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

        if early_stopping.step(val_results["PR-AUC"], model, best_model_path, epoch + 1):
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best Val PR-AUC: {early_stopping.best_score:.4f} "
                f"at epoch {early_stopping.best_epoch}."
            )
            break

    print("\nLoading best model for testing...")

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

    val_targets, val_probs = predict(model, data, data.val_mask)
    best_threshold, best_val_f1 = find_best_threshold(
        val_targets,
        val_probs,
        strategy=threshold_strategy,
    )

    print_threshold_comparison(
        val_targets,
        val_probs,
        selected_threshold=best_threshold,
        strategy=threshold_strategy,
    )

    print(f"\nSelected threshold strategy: {threshold_strategy}")
    print(f"Selected threshold: {best_threshold:.2f}")
    print(f"Best validation F1: {best_val_f1:.4f}")

    test_results = evaluate(model, data, data.test_mask, threshold=best_threshold)

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

    save_results_to_csv(
        model_label,
        test_results,
        threshold=best_threshold,
        threshold_strategy=threshold_strategy,
        validation_f1=best_val_f1,
    )
    print("\nSaved test results to results/model_results.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Elliptic Bitcoin GNN baselines.")
    parser.add_argument("--model", choices=["gcn", "sage", "gat"], required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--rebuild-graph", action="store_true")
    parser.add_argument(
        "--threshold-strategy",
        choices=["f1", "balanced"],
        default="f1",
        help="Select validation threshold by max F1 or by balanced recall/precision.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_name=args.model,
        epochs=args.epochs,
        rebuild_graph=args.rebuild_graph,
        threshold_strategy=args.threshold_strategy,
    )
