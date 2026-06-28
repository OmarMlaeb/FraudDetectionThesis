import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from torch.utils.data import Dataset, DataLoader

from common.early_stopping import EarlyStopping
from common.metrics import evaluate_binary_classification, find_best_threshold, save_results_to_csv
from .models import TransformerFraudDetector
from .preprocessing import load_ieee_cis, preprocess_ieee_cis, temporal_train_val_test_split


TRANSACTION_PATH = "data/ieee-cis/train_transaction.csv"
IDENTITY_PATH = "data/ieee-cis/train_identity.csv"

BATCH_SIZE = 512
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 10
GRADIENT_CLIP_NORM = 1.0


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FraudSequenceDataset(Dataset):
    def __init__(self, X, y, sequence_length):
        if len(X) < sequence_length:
            raise ValueError("Dataset is smaller than the requested sequence length.")

        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length + 1

    def __getitem__(self, index):
        end = index + self.sequence_length
        return self.X[index:end], self.y[end - 1]


def predict(model, data_loader, device):
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            probs = torch.sigmoid(logits).cpu()

            all_probs.append(probs)
            all_targets.append(batch_y)

    return torch.cat(all_targets).numpy(), torch.cat(all_probs).numpy()


def train(seed=42):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Seed: {seed}")

    df = load_ieee_cis(TRANSACTION_PATH, IDENTITY_PATH)
    X, y = preprocess_ieee_cis(df)

    X_train, y_train, X_val, y_val, X_test, y_test = temporal_train_val_test_split(X, y)

    train_dataset = FraudSequenceDataset(X_train, y_train, SEQUENCE_LENGTH)
    val_dataset = FraudSequenceDataset(X_val, y_val, SEQUENCE_LENGTH)
    test_dataset = FraudSequenceDataset(X_test, y_test, SEQUENCE_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train.shape[1]
    model = TransformerFraudDetector(input_dim, SEQUENCE_LENGTH).to(device)

    sequence_targets = y_train[SEQUENCE_LENGTH - 1:]
    positive_count = sequence_targets.sum()
    negative_count = len(sequence_targets) - positive_count
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_model_path = f"results/best_transformer_seed{seed}_model.pt"
    Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            if not torch.isfinite(loss):
                print("Stopping early because the training loss became non-finite.")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()

            total_loss += loss.item()
        else:
            y_val_seq, val_probs = predict(model, val_loader, device)
            if not np.isfinite(val_probs).all():
                print("Stopping early because validation probabilities became non-finite.")
                break

            val_results = evaluate_binary_classification(y_val_seq, val_probs)

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Loss: {total_loss:.4f} "
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
            continue

        break

    print("\nLoading best model for testing...")

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

    y_val_seq, val_probs = predict(model, val_loader, device)
    best_threshold, best_val_f1 = find_best_threshold(y_val_seq, val_probs)

    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best validation F1: {best_val_f1:.4f}")

    y_test_seq, test_probs = predict(model, test_loader, device)

    print("Min probability:", test_probs.min())
    print("Max probability:", test_probs.max())
    print("Mean probability:", test_probs.mean())
    print("Fraud predictions:", (test_probs >= 0.5).sum())

    test_results = evaluate_binary_classification(
        y_test_seq,
        test_probs,
        threshold=best_threshold,
    )

    print("\nFinal Test Results - Transformer")
    print("--------------------------------")
    print(f"Recall: {test_results['Recall']:.4f}")
    print(f"Precision: {test_results['Precision']:.4f}")
    print(f"F1-score: {test_results['F1-score']:.4f}")
    print(f"PR-AUC: {test_results['PR-AUC']:.4f}")
    print(f"ROC-AUC: {test_results['ROC-AUC']:.4f}")
    print("Confusion Matrix:")
    print(test_results["Confusion Matrix"])

    save_results_to_csv(
        "Transformer",
        test_results,
        threshold=best_threshold,
        threshold_strategy="f1",
        validation_f1=best_val_f1,
        seed=seed,
    )
    print("\nSaved test results to results/model_results.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Train IEEE-CIS Transformer baseline.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(seed=args.seed)
