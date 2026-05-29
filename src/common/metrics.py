import numpy as np
import csv
from datetime import datetime
from pathlib import Path

from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


def evaluate_binary_classification(y_true, y_prob, threshold=0.5):
    if not np.isfinite(y_prob).all():
        raise ValueError("Predicted probabilities contain NaN or infinite values.")

    y_pred = (y_prob >= threshold).astype(int)

    results = {
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "Confusion Matrix": confusion_matrix(y_true, y_pred)
    }

    return results


def find_best_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_f1 = 0.0

    thresholds = np.arange(0.01, 0.99, 0.01)

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def save_results_to_csv(model_name, results, output_path="results/model_results.csv"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tn, fp, fn, tp = results["Confusion Matrix"].ravel()
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "recall": results["Recall"],
        "precision": results["Precision"],
        "f1_score": results["F1-score"],
        "pr_auc": results["PR-AUC"],
        "roc_auc": results["ROC-AUC"],
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    write_header = not output_path.exists()
    with output_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
