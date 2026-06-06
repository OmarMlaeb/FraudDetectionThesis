import argparse

from common.wilcoxon import DEFAULT_METRICS, run_wilcoxon_tests


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Wilcoxon signed-rank tests between model results."
    )
    parser.add_argument("--results-path", default="results/model_results.csv")
    parser.add_argument("--output-path", default="results/wilcoxon_model_comparisons.csv")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        choices=list(DEFAULT_METRICS),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_wilcoxon_tests(
        results_path=args.results_path,
        output_path=args.output_path,
        metrics=args.metrics,
        alpha=args.alpha,
    )
