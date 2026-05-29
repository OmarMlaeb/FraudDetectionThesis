import csv
from datetime import datetime
from pathlib import Path


DEFAULT_RESULTS_PATH = "results/model_results.csv"
DEFAULT_RANKINGS_PATH = "results/model_rankings.csv"
DEFAULT_RECALL_RANKINGS_PATH = "results/model_rankings_by_recall.csv"
RANKING_METRICS = ("pr_auc", "f1_score", "roc_auc", "recall", "precision")
NUMERIC_FIELDS = (
    "recall",
    "precision",
    "f1_score",
    "pr_auc",
    "roc_auc",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
)


def get_dataset_and_model(model_name):
    if model_name.startswith("Elliptic-"):
        return "Elliptic", model_name.replace("Elliptic-", "", 1)

    return "IEEE-CIS", model_name


def _parse_timestamp(value):
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.min


def _read_result_rows(results_path):
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"No results file found at {results_path}")

    with results_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    normalized_rows = []
    for row in rows:
        dataset, model = get_dataset_and_model(row["model"])
        row = row.copy()
        row["dataset"] = dataset
        row["model"] = model
        row["_timestamp"] = _parse_timestamp(row["timestamp"])

        for field in NUMERIC_FIELDS:
            row[field] = float(row[field])

        normalized_rows.append(row)

    return normalized_rows


def _latest_row_per_model(rows):
    latest_rows = {}
    for row in rows:
        key = (row["dataset"], row["model"])
        if key not in latest_rows or row["_timestamp"] > latest_rows[key]["_timestamp"]:
            latest_rows[key] = row

    return list(latest_rows.values())


def rank_model_results(
    results_path=DEFAULT_RESULTS_PATH,
    output_path=DEFAULT_RANKINGS_PATH,
    primary_metric="pr_auc",
    datasets=None,
):
    if primary_metric not in RANKING_METRICS:
        raise ValueError(f"primary_metric must be one of: {', '.join(RANKING_METRICS)}")

    rows = _latest_row_per_model(_read_result_rows(results_path))

    if datasets is not None:
        datasets = set(datasets)
        rows = [row for row in rows if row["dataset"] in datasets]

    rows.sort(
        key=lambda row: (
            row["dataset"],
            -row[primary_metric],
            -row["f1_score"],
            -row["roc_auc"],
            -row["recall"],
            -row["precision"],
        )
    )

    ranked_rows = []
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for rank, row in enumerate(dataset_rows, start=1):
            ranked_rows.append(
                {
                    "dataset": row["dataset"],
                    "rank": rank,
                    "model": row["model"],
                    "ranking_metric": primary_metric,
                    "timestamp": row["timestamp"],
                    "recall": row["recall"],
                    "precision": row["precision"],
                    "f1_score": row["f1_score"],
                    "pr_auc": row["pr_auc"],
                    "roc_auc": row["roc_auc"],
                    "true_negative": int(row["true_negative"]),
                    "false_positive": int(row["false_positive"]),
                    "false_negative": int(row["false_negative"]),
                    "true_positive": int(row["true_positive"]),
                }
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as csv_file:
        fieldnames = (
            "dataset",
            "rank",
            "model",
            "ranking_metric",
            "timestamp",
            *NUMERIC_FIELDS,
        )
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked_rows)

    return ranked_rows


def print_model_rankings(
    results_path=DEFAULT_RESULTS_PATH,
    output_path=DEFAULT_RANKINGS_PATH,
    primary_metric="pr_auc",
    datasets=None,
):
    ranked_rows = rank_model_results(
        results_path=results_path,
        output_path=output_path,
        primary_metric=primary_metric,
        datasets=datasets,
    )

    if not ranked_rows:
        print("No model results found to rank.")
        return

    print(f"\nModel rankings by {primary_metric.upper()}")
    current_dataset = None
    for row in ranked_rows:
        if row["dataset"] != current_dataset:
            current_dataset = row["dataset"]
            print(f"\n{current_dataset}")
            print("Rank  Model        PR-AUC   F1      ROC-AUC  Recall  Precision")

        print(
            f"{row['rank']:<5} "
            f"{row['model']:<11} "
            f"{row['pr_auc']:.4f}  "
            f"{row['f1_score']:.4f}  "
            f"{row['roc_auc']:.4f}   "
            f"{row['recall']:.4f}  "
            f"{row['precision']:.4f}"
        )

    print(f"\nSaved rankings to {output_path}")


def print_pr_auc_and_recall_rankings(results_path=DEFAULT_RESULTS_PATH, datasets=None):
    print_model_rankings(
        results_path=results_path,
        output_path=DEFAULT_RANKINGS_PATH,
        primary_metric="pr_auc",
        datasets=datasets,
    )
    print_model_rankings(
        results_path=results_path,
        output_path=DEFAULT_RECALL_RANKINGS_PATH,
        primary_metric="recall",
        datasets=datasets,
    )
