import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common.ranking import get_dataset_and_model


DEFAULT_BASELINE_RESULTS_PATH = "results/model_results.csv"
DEFAULT_GRAPH_RESULTS_PATH = "results/ieee_cis_graph_model_results.csv"
DEFAULT_RANKING_OUTPUT_PATH = "results/ieee_cis_graph_combined_rankings.csv"
DEFAULT_COMPARISON_OUTPUT_PATH = "results/ieee_cis_graph_vs_baseline_comparisons.csv"
DEFAULT_METRICS = ("pr_auc", "f1_score", "roc_auc", "recall", "precision")
DEFAULT_SEEDS = list(range(42, 52))
NON_GRAPH_MODELS = ("MLP", "LSTM", "Transformer")
OLD_MAIN_GRAPH_MODELS = ("GCN", "GraphSAGE", "GAT")
MAIN_GRAPH_MODELS = ("GCN-main", "GraphSAGE-main", "GAT-main")
CONSTRUCTION_TAGS = (
    "-card",
    "-address",
    "-email_domain",
    "-identifier_time_window",
)
MIN_PAIRED_RUNS = 3


def _read_results(path):
    path = Path(path)
    if not path.exists():
        print(f"Skipping missing results file: {path}")
        return pd.DataFrame()

    results = pd.read_csv(path)
    if results.empty:
        return results

    results = results.dropna(subset=["timestamp", "model"]).copy()
    parsed = results["model"].apply(get_dataset_and_model)
    results["dataset"] = parsed.apply(lambda item: item[0])
    results["model_name"] = parsed.apply(lambda item: item[1])
    results["timestamp"] = pd.to_datetime(results["timestamp"], errors="coerce")
    results = results.dropna(subset=["timestamp"])

    for metric in DEFAULT_METRICS:
        results[metric] = pd.to_numeric(results[metric], errors="coerce")

    if "seed" in results.columns:
        results["seed"] = pd.to_numeric(results["seed"], errors="coerce")
    else:
        results["seed"] = np.nan

    return results.dropna(subset=[*DEFAULT_METRICS, "seed"])


def load_ieee_cis_results(results_paths, seeds):
    frames = [_read_results(path) for path in results_paths]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise FileNotFoundError("No result rows found in the requested CSV files.")

    results = pd.concat(frames, ignore_index=True)
    results = results[results["dataset"] == "IEEE-CIS"].copy()
    results = results[results["seed"].isin(seeds)]
    results = results.sort_values(["model_name", "seed", "timestamp"])
    results = results.drop_duplicates(
        subset=["dataset", "model_name", "seed"],
        keep="last",
    )

    if results["model_name"].isin(MAIN_GRAPH_MODELS).any():
        results = results[~results["model_name"].isin(OLD_MAIN_GRAPH_MODELS)]

    return results


def is_constructed_graph_model(model_name):
    return any(tag in model_name for tag in CONSTRUCTION_TAGS)


def rank_by_seed_mean(results, output_path, primary_metric="pr_auc"):
    grouped = results.groupby("model_name")
    summary = grouped.agg(
        runs=("seed", "count"),
        seed_min=("seed", "min"),
        seed_max=("seed", "max"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        f1_score_mean=("f1_score", "mean"),
        f1_score_std=("f1_score", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
    ).reset_index()

    summary = summary.sort_values(
        [f"{primary_metric}_mean", "f1_score_mean", "roc_auc_mean"],
        ascending=False,
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    summary.insert(1, "dataset", "IEEE-CIS")
    summary.insert(3, "ranking_metric", f"{primary_metric}_mean")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print(f"\nSaved IEEE-CIS combined ranking to {output_path}")
    print("Rank  Model                         Runs  PR-AUC mean  F1 mean  ROC-AUC mean")
    print("----  ----------------------------  ----  -----------  -------  ------------")
    for _, row in summary.iterrows():
        print(
            f"{int(row['rank']):<4}  "
            f"{row['model_name']:<28}  "
            f"{int(row['runs']):<4}  "
            f"{row['pr_auc_mean']:.4f}       "
            f"{row['f1_score_mean']:.4f}   "
            f"{row['roc_auc_mean']:.4f}"
        )

    return summary


def compare_constructed_graphs_to_baselines(results, output_path, metrics, alpha):
    available_models = set(results["model_name"])
    constructed_models = sorted(
        model
        for model in results["model_name"].unique()
        if is_constructed_graph_model(model)
    )
    baseline_models = [model for model in NON_GRAPH_MODELS if model in available_models]

    corrected_main_models = [
        model for model in MAIN_GRAPH_MODELS if model in available_models
    ]
    if corrected_main_models:
        baseline_models.extend(corrected_main_models)
    else:
        baseline_models.extend(
            model for model in OLD_MAIN_GRAPH_MODELS if model in available_models
        )

    rows = []
    for constructed_model in constructed_models:
        constructed_rows = results[results["model_name"] == constructed_model]

        for baseline_model in baseline_models:
            baseline_rows = results[results["model_name"] == baseline_model]
            paired = constructed_rows.merge(
                baseline_rows,
                on="seed",
                suffixes=("_constructed", "_baseline"),
            )

            for metric in metrics:
                row = {
                    "dataset": "IEEE-CIS",
                    "constructed_graph_model": constructed_model,
                    "comparison_model": baseline_model,
                    "metric": metric,
                    "paired_runs": len(paired),
                    "mean_constructed_graph": np.nan,
                    "mean_comparison_model": np.nan,
                    "mean_difference_constructed_minus_comparison": np.nan,
                    "wilcoxon_statistic": np.nan,
                    "p_value": np.nan,
                    "significant_at_alpha": False,
                    "better_mean_model": "",
                    "note": "",
                }

                if len(paired) < MIN_PAIRED_RUNS:
                    row["note"] = f"Need at least {MIN_PAIRED_RUNS} paired runs."
                    rows.append(row)
                    continue

                constructed_values = paired[f"{metric}_constructed"].to_numpy()
                baseline_values = paired[f"{metric}_baseline"].to_numpy()
                differences = constructed_values - baseline_values

                row["mean_constructed_graph"] = float(np.mean(constructed_values))
                row["mean_comparison_model"] = float(np.mean(baseline_values))
                row["mean_difference_constructed_minus_comparison"] = float(
                    np.mean(differences)
                )
                row["better_mean_model"] = (
                    constructed_model
                    if row["mean_difference_constructed_minus_comparison"] > 0
                    else baseline_model
                )

                if np.allclose(differences, 0):
                    row["wilcoxon_statistic"] = 0.0
                    row["p_value"] = 1.0
                    row["note"] = "All paired differences are zero."
                    rows.append(row)
                    continue

                test_result = wilcoxon(
                    constructed_values,
                    baseline_values,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                row["wilcoxon_statistic"] = float(test_result.statistic)
                row["p_value"] = float(test_result.pvalue)
                row["significant_at_alpha"] = bool(test_result.pvalue < alpha)
                rows.append(row)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparisons = pd.DataFrame(rows)
    comparisons.to_csv(output_path, index=False)

    print(f"\nSaved IEEE-CIS graph-vs-baseline comparisons to {output_path}")
    print(f"Alpha: {alpha}")
    return comparisons


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rank IEEE-CIS constructed graph models and compare them against "
            "main graph and non-graph model results."
        )
    )
    parser.add_argument(
        "--baseline-results-path",
        default=DEFAULT_BASELINE_RESULTS_PATH,
        help="CSV containing non-graph and main graph IEEE-CIS results.",
    )
    parser.add_argument(
        "--graph-results-path",
        default=DEFAULT_GRAPH_RESULTS_PATH,
        help="CSV containing constructed IEEE-CIS graph results.",
    )
    parser.add_argument(
        "--ranking-output-path",
        default=DEFAULT_RANKING_OUTPUT_PATH,
    )
    parser.add_argument(
        "--comparison-output-path",
        default=DEFAULT_COMPARISON_OUTPUT_PATH,
    )
    parser.add_argument(
        "--primary-metric",
        choices=DEFAULT_METRICS,
        default="pr_auc",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=DEFAULT_METRICS,
        default=list(DEFAULT_METRICS),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Seeds to include. Defaults to 42 through 51.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = load_ieee_cis_results(
        [args.baseline_results_path, args.graph_results_path],
        seeds=args.seeds,
    )
    rank_by_seed_mean(
        results,
        output_path=args.ranking_output_path,
        primary_metric=args.primary_metric,
    )
    compare_constructed_graphs_to_baselines(
        results,
        output_path=args.comparison_output_path,
        metrics=args.metrics,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
