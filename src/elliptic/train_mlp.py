import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from torch.utils.data import DataLoader, TensorDataset

from common.metrics import evaluate_binary_classification, find_best_threshold, save_results_to_csv
from .data import load_elliptic_tabular_data
from .models import MLPFraudDetector


BATCH_SIZE = 1024
EPOCHS = 20
LEARNING_RATE = 0.001


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    data = load_elliptic_tabular_data()
    X = data["X"]
    y = data["y"]
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    X_train_tensor = torch.tensor(X[train_mask])
    y_train_tensor = torch.tensor(y[train_mask])
    X_val_tensor = torch.tensor(X[val_mask]).to(device)
    X_test_tensor = torch.tensor(X[test_mask]).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MLPFraudDetector(X.shape[1]).to(device)

    positive_count = y[train_mask].sum()
    negative_count = len(y[train_mask]) - positive_count
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_val_pr_auc = 0
    best_model_path = "results/best_elliptic_mlp_model.pt"
    Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_val_tensor)).cpu().numpy()

        val_results = evaluate_binary_classification(y[val_mask], val_probs)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {total_loss:.4f} "
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
    model.eval()

    with torch.no_grad():
        val_probs = torch.sigmoid(model(X_val_tensor)).cpu().numpy()
        test_probs = torch.sigmoid(model(X_test_tensor)).cpu().numpy()

    best_threshold, best_val_f1 = find_best_threshold(y[val_mask], val_probs)

    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best validation F1: {best_val_f1:.4f}")

    test_results = evaluate_binary_classification(
        y[test_mask],
        test_probs,
        threshold=best_threshold,
    )

    print("\nFinal Test Results - Elliptic MLP")
    print("---------------------------------")
    print(f"Recall: {test_results['Recall']:.4f}")
    print(f"Precision: {test_results['Precision']:.4f}")
    print(f"F1-score: {test_results['F1-score']:.4f}")
    print(f"PR-AUC: {test_results['PR-AUC']:.4f}")
    print(f"ROC-AUC: {test_results['ROC-AUC']:.4f}")
    print("Confusion Matrix:")
    print(test_results["Confusion Matrix"])

    save_results_to_csv("Elliptic-MLP", test_results)
    print("\nSaved test results to results/model_results.csv")


if __name__ == "__main__":
    train()
