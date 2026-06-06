import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .ranking import get_dataset_and_model


DEFAULT_RESULTS_PATH = "results/model_results.csv"
DEFAULT_OUTPUT_PATH = "results/wilcoxon_model_comparisons.csv"
DEFAULT_METRICS = ("pr_auc", "f1_score", "roc_auc", "recall", "precision")
MIN_PAIRED_RUNS = 3


def _load_results(results_path):
    results = pd.read_csv(results_path)
    results = results.dropna(subset=["timestamp", "model"]).copy()

    parsed = results["model"].apply(get_dataset_and_model)
    results["dataset"] = parsed.apply(lambda item: item[0])
    results["model_name"] = parsed.apply(lambda item: item[1])
    results["timestamp"] = pd.to_datetime(results["timestamp"], errors="coerce")
    results = results.dropna(subset=["timestamp"])

    for metric in DEFAULT_METRICS:
        results[metric] = pd.to_numeric(results[metric], errors="coerce")

    results = results.dropna(subset=list(DEFAULT_METRICS))
    results = results.sort_values(["dataset", "model_name", "timestamp"])
    results["run_index"] = results.groupby(["dataset", "model_name"]).cumcount()

    return results


def _paired_metric_values(results, dataset, model_a, model_b, metric):
    model_a_rows = results[
        (results["dataset"] == dataset) & (results["model_name"] == model_a)
    ][["run_index", metric]]
    model_b_rows = results[
        (results["dataset"] == dataset) & (results["model_name"] == model_b)
    ][["run_index", metric]]

    paired = model_a_rows.merge(
        model_b_rows,
        on="run_index",
        suffixes=("_a", "_b"),
    )

    return paired[f"{metric}_a"].to_numpy(), paired[f"{metric}_b"].to_numpy()


def run_wilcoxon_tests(
    results_path=DEFAULT_RESULTS_PATH,
    output_path=DEFAULT_OUTPUT_PATH,
    metrics=DEFAULT_METRICS,
    alpha=0.05,
):
    results_path = Path(results_path)
    output_path = Path(output_path)

    results = _load_results(results_path)
    rows = []

    for dataset in sorted(results["dataset"].unique()):
        dataset_results = results[results["dataset"] == dataset]
        models = sorted(dataset_results["model_name"].unique())

        for model_a, model_b in itertools.combinations(models, 2):
            for metric in metrics:
                values_a, values_b = _paired_metric_values(
                    results,
                    dataset,
                    model_a,
                    model_b,
                    metric,
                )
                paired_runs = len(values_a)

                row = {
                    "dataset": dataset,
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": metric,
                    "paired_runs": paired_runs,
                    "mean_model_a": np.nan,
                    "mean_model_b": np.nan,
                    "mean_difference_a_minus_b": np.nan,
                    "wilcoxon_statistic": np.nan,
                    "p_value": np.nan,
                    "significant_at_alpha": False,
                    "better_mean_model": "",
                    "note": "",
                }

                if paired_runs < MIN_PAIRED_RUNS:
                    row["note"] = f"Need at least {MIN_PAIRED_RUNS} paired runs."
                    rows.append(row)
                    continue

                differences = values_a - values_b
                row["mean_model_a"] = float(np.mean(values_a))
                row["mean_model_b"] = float(np.mean(values_b))
                row["mean_difference_a_minus_b"] = float(np.mean(differences))
                row["better_mean_model"] = model_a if row["mean_difference_a_minus_b"] > 0 else model_b

                if np.allclose(differences, 0):
                    row["p_value"] = 1.0
                    row["wilcoxon_statistic"] = 0.0
                    row["note"] = "All paired differences are zero."
                    rows.append(row)
                    continue

                test_result = wilcoxon(
                    values_a,
                    values_b,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                row["wilcoxon_statistic"] = float(test_result.statistic)
                row["p_value"] = float(test_result.pvalue)
                row["significant_at_alpha"] = bool(test_result.pvalue < alpha)
                rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False)

    print(f"\nSaved Wilcoxon signed-rank comparisons to {output_path}")
    print(f"Alpha: {alpha}")

    significant = result_df[result_df["significant_at_alpha"]]
    if significant.empty:
        print("No statistically significant pairwise differences found at this alpha.")
    else:
        print("\nSignificant Wilcoxon pairwise differences")
        print("Dataset   Metric     Model A       Model B       Better Mean   p-value")
        print("--------  ---------  ------------  ------------  ------------  -------")
        for _, row in significant.iterrows():
            print(
                f"{row['dataset']:<8}  "
                f"{row['metric']:<9}  "
                f"{row['model_a']:<12}  "
                f"{row['model_b']:<12}  "
                f"{row['better_mean_model']:<12}  "
                f"{row['p_value']:.4g}"
            )

    return result_df


if __name__ == "__main__":
    run_wilcoxon_tests()
