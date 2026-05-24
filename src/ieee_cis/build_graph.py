import argparse

from .graph_data import load_or_build_ieee_cis_graph
from .preprocessing import load_ieee_cis


TRANSACTION_PATH = "data/ieee-cis/train_transaction.csv"
IDENTITY_PATH = "data/ieee-cis/train_identity.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Build the IEEE-CIS transaction graph.")
    parser.add_argument("--rebuild-graph", action="store_true")
    parser.add_argument("--max-group-size", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = load_ieee_cis(TRANSACTION_PATH, IDENTITY_PATH)
    data = load_or_build_ieee_cis_graph(
        df,
        rebuild=args.rebuild_graph,
        max_group_size=args.max_group_size,
    )

    print("IEEE-CIS graph ready")
    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.edge_index.size(1)}")
    print(f"Features: {data.x.size(1)}")
    print("Saved to results/ieee_cis_graph.pt")
