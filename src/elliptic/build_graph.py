import argparse

from common.graph_stats import print_graph_statistics
from common.graph_variants import GRAPH_VARIANTS

from .data import load_or_build_elliptic_graph


def parse_args():
    parser = argparse.ArgumentParser(description="Build the Elliptic Bitcoin transaction graph.")
    parser.add_argument("--rebuild-graph", action="store_true")
    parser.add_argument("--graph-variant", choices=GRAPH_VARIANTS, default="original")
    parser.add_argument("--complement-average-degree", type=int, default=20)
    parser.add_argument("--complement-seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = load_or_build_elliptic_graph(
        rebuild=args.rebuild_graph,
        graph_variant=args.graph_variant,
        complement_average_degree=args.complement_average_degree,
        complement_seed=args.complement_seed,
    )

    print(f"Elliptic graph ready ({args.graph_variant})")
    print_graph_statistics(data)
    print(f"Features: {data.x.size(1)}")
    print(f"Train nodes: {int(data.train_mask.sum())}")
    print(f"Validation nodes: {int(data.val_mask.sum())}")
    print(f"Test nodes: {int(data.test_mask.sum())}")
    if args.graph_variant == "original":
        print("Saved to results/elliptic_graph.pt")
    else:
        print(
            "Saved to "
            f"results/elliptic_graph_{args.graph_variant}"
            f"_deg{args.complement_average_degree}_seed{args.complement_seed}.pt"
        )
