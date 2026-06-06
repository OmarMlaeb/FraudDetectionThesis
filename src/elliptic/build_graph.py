import argparse

from common.graph_stats import print_graph_statistics

from .data import load_or_build_elliptic_graph


def parse_args():
    parser = argparse.ArgumentParser(description="Build the Elliptic Bitcoin transaction graph.")
    parser.add_argument("--rebuild-graph", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = load_or_build_elliptic_graph(rebuild=args.rebuild_graph)

    print("Elliptic graph ready")
    print_graph_statistics(data)
    print(f"Features: {data.x.size(1)}")
    print(f"Train nodes: {int(data.train_mask.sum())}")
    print(f"Validation nodes: {int(data.val_mask.sum())}")
    print(f"Test nodes: {int(data.test_mask.sum())}")
    print("Saved to results/elliptic_graph.pt")
