import argparse

from common.graph_stats import print_graph_statistics
from common.graph_variants import GRAPH_VARIANTS

from .graph_data import load_or_build_ieee_cis_graph
from .preprocessing import load_ieee_cis


TRANSACTION_PATH = "data/ieee-cis/train_transaction.csv"
IDENTITY_PATH = "data/ieee-cis/train_identity.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Build the IEEE-CIS transaction graph.")
    parser.add_argument("--rebuild-graph", action="store_true")
    parser.add_argument("--max-group-size", type=int, default=1000)
    parser.add_argument("--graph-variant", choices=GRAPH_VARIANTS, default="original")
    parser.add_argument("--complement-average-degree", type=int, default=20)
    parser.add_argument("--complement-seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = load_ieee_cis(TRANSACTION_PATH, IDENTITY_PATH)
    data = load_or_build_ieee_cis_graph(
        df,
        rebuild=args.rebuild_graph,
        max_group_size=args.max_group_size,
        graph_variant=args.graph_variant,
        complement_average_degree=args.complement_average_degree,
        complement_seed=args.complement_seed,
    )

    print(f"IEEE-CIS graph ready ({args.graph_variant})")
    print_graph_statistics(data)
    print(f"Features: {data.x.size(1)}")
    if args.graph_variant == "original":
        print("Saved to results/ieee_cis_graph.pt")
    else:
        print(
            "Saved to "
            f"results/ieee_cis_graph_{args.graph_variant}"
            f"_deg{args.complement_average_degree}_seed{args.complement_seed}.pt"
        )
