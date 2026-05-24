import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from torch.utils.data import TensorDataset, DataLoader

from common.metrics import evaluate_binary_classification, save_results_to_csv
from .models import MLPFraudDetector
from .preprocessing import load_ieee_cis, preprocess_ieee_cis, temporal_train_val_test_split


TRANSACTION_PATH = "data/ieee-cis/train_transaction.csv"
IDENTITY_PATH = "data/ieee-cis/train_identity.csv"

BATCH_SIZE = 1024
EPOCHS = 20
LEARNING_RATE = 0.001


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    df = load_ieee_cis(TRANSACTION_PATH, IDENTITY_PATH)
    X, y = preprocess_ieee_cis(df)

    X_train, y_train, X_val, y_val, X_test, y_test = temporal_train_val_test_split(X, y)

    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)

    X_val_tensor = torch.tensor(X_val).to(device)
    y_val_tensor = torch.tensor(y_val).to(device)

    X_test_tensor = torch.tensor(X_test).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train.shape[1]
    model = MLPFraudDetector(input_dim).to(device)

    positive_count = y_train.sum()
    negative_count = len(y_train) - positive_count
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_val_pr_auc = 0
    best_model_path = "results/best_mlp_model.pt"
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
            val_logits = model(X_val_tensor)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        val_results = evaluate_binary_classification(y_val, val_probs)

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
        test_logits = model(X_test_tensor)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()

    test_results = evaluate_binary_classification(y_test, test_probs)

    print("\nFinal Test Results - MLP")
    print("-------------------------")
    print(f"Recall: {test_results['Recall']:.4f}")
    print(f"Precision: {test_results['Precision']:.4f}")
    print(f"F1-score: {test_results['F1-score']:.4f}")
    print(f"PR-AUC: {test_results['PR-AUC']:.4f}")
    print(f"ROC-AUC: {test_results['ROC-AUC']:.4f}")
    print("Confusion Matrix:")
    print(test_results["Confusion Matrix"])

    save_results_to_csv("MLP", test_results)
    print("\nSaved test results to results/model_results.csv")


if __name__ == "__main__":
    train()
