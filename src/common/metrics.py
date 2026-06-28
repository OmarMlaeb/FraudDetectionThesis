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
        "Confusion Matrix": confusion_matrix(y_true, y_pred, labels=[0, 1])
    }

    return results


def get_threshold_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "Threshold": float(threshold),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


def compare_thresholds(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    return [get_threshold_metrics(y_true, y_prob, threshold) for threshold in thresholds]


def _threshold_sort_key(row, strategy):
    if strategy == "f1":
        return row["F1-score"], row["Recall"], row["Precision"]
    if strategy == "balanced":
        return min(row["Recall"], row["Precision"]), row["F1-score"]
    raise ValueError("threshold_strategy must be either 'f1' or 'balanced'.")


def find_best_threshold(y_true, y_prob, strategy="f1", thresholds=None):
    best_threshold = 0.5
    best_score = None
    best_f1 = 0.0

    threshold_results = compare_thresholds(y_true, y_prob, thresholds=thresholds)

    for row in threshold_results:
        score = _threshold_sort_key(row, strategy)

        if best_score is None or score > best_score:
            best_score = score
            best_f1 = row["F1-score"]
            best_threshold = row["Threshold"]

    return best_threshold, best_f1


def print_threshold_comparison(y_true, y_prob, selected_threshold, strategy="f1"):
    fixed_thresholds = np.arange(0.10, 1.00, 0.10)
    thresholds = sorted(set(np.round(np.append(fixed_thresholds, selected_threshold), 2)))
    threshold_results = compare_thresholds(y_true, y_prob, thresholds=thresholds)
    top_results = sorted(
        compare_thresholds(y_true, y_prob),
        key=lambda row: _threshold_sort_key(row, strategy),
        reverse=True,
    )[:5]

    print("\nValidation threshold comparison")
    print("Threshold | Recall | Precision | F1-score | TP | FP | FN | TN")
    print("----------|--------|-----------|----------|----|----|----|----")

    for row in threshold_results:
        marker = "*" if round(row["Threshold"], 2) == round(selected_threshold, 2) else " "
        print(
            f"{marker}{row['Threshold']:.2f}      "
            f"| {row['Recall']:.4f} "
            f"| {row['Precision']:.4f} "
            f"| {row['F1-score']:.4f} "
            f"| {row['TP']} | {row['FP']} | {row['FN']} | {row['TN']}"
        )

    print(f"\nTop validation thresholds by {strategy}:")
    for row in top_results:
        print(
            f"threshold={row['Threshold']:.2f}, "
            f"recall={row['Recall']:.4f}, "
            f"precision={row['Precision']:.4f}, "
            f"f1={row['F1-score']:.4f}"
        )


def save_results_to_csv(
    model_name,
    results,
    output_path="results/model_results.csv",
    threshold=None,
    threshold_strategy=None,
    validation_f1=None,
    seed=None,
):
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
        "threshold": threshold,
        "threshold_strategy": threshold_strategy,
        "validation_f1_at_threshold": validation_f1,
        "seed": seed,
    }

    if output_path.exists():
        with output_path.open(newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            existing_rows = list(reader)
            fieldnames = list(reader.fieldnames or [])

        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)

        if set(fieldnames) != set(reader.fieldnames or []):
            with output_path.open("w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)
    else:
        fieldnames = list(row.keys())

    with output_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if output_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)
