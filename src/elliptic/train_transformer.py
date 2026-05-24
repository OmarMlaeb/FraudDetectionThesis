import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from torch.utils.data import DataLoader

from common.metrics import evaluate_binary_classification, save_results_to_csv
from .data import load_elliptic_tabular_data
from .models import TransformerFraudDetector
from .sequence import EllipticSequenceDataset, predict_sequence_model


BATCH_SIZE = 512
EPOCHS = 20
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 10


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    data = load_elliptic_tabular_data()
    X = data["X"]
    y = data["y"]

    train_dataset = EllipticSequenceDataset(X, y, data["train_mask"], SEQUENCE_LENGTH)
    val_dataset = EllipticSequenceDataset(X, y, data["val_mask"], SEQUENCE_LENGTH)
    test_dataset = EllipticSequenceDataset(X, y, data["test_mask"], SEQUENCE_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TransformerFraudDetector(X.shape[1], SEQUENCE_LENGTH).to(device)

    train_targets = train_dataset.y[SEQUENCE_LENGTH - 1:].numpy()
    positive_count = train_targets.sum()
    negative_count = len(train_targets) - positive_count
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_val_pr_auc = 0
    best_model_path = "results/best_elliptic_transformer_model.pt"
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

        y_val_seq, val_probs = predict_sequence_model(model, val_loader, device)
        val_results = evaluate_binary_classification(y_val_seq, val_probs)

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
    y_test_seq, test_probs = predict_sequence_model(model, test_loader, device)
    test_results = evaluate_binary_classification(y_test_seq, test_probs)

    print("\nFinal Test Results - Elliptic Transformer")
    print("-----------------------------------------")
    print(f"Recall: {test_results['Recall']:.4f}")
    print(f"Precision: {test_results['Precision']:.4f}")
    print(f"F1-score: {test_results['F1-score']:.4f}")
    print(f"PR-AUC: {test_results['PR-AUC']:.4f}")
    print(f"ROC-AUC: {test_results['ROC-AUC']:.4f}")
    print("Confusion Matrix:")
    print(test_results["Confusion Matrix"])

    save_results_to_csv("Elliptic-Transformer", test_results)
    print("\nSaved test results to results/model_results.csv")


if __name__ == "__main__":
    train()
