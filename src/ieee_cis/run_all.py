import argparse

from common.ranking import print_pr_auc_and_recall_rankings

from .train_gnn import train as train_gnn
from .train_lstm import train as train_lstm
from .train_mlp import train as train_mlp
from .train_transformer import train as train_transformer


DEFAULT_SEEDS = [42, 43, 44]


def run_all(seeds=None):
    seeds = seeds or DEFAULT_SEEDS

    for seed in seeds:
        print(f"\nRunning IEEE-CIS experiments with seed {seed}")
        print("=" * 45)
        train_mlp(seed=seed)
        train_lstm(seed=seed)
        train_transformer(seed=seed)
        train_gnn(model_name="gcn", seed=seed)
        train_gnn(model_name="sage", seed=seed)
        train_gnn(model_name="gat", seed=seed)

    print_pr_auc_and_recall_rankings()


def parse_args():
    parser = argparse.ArgumentParser(description="Run all IEEE-CIS experiments.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Random seeds to run. Use several seeds for paired Wilcoxon tests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_all(seeds=args.seeds)


if __name__ == "__main__":
    main()
