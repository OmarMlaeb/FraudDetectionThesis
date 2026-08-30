import argparse

from common.graph_stats import print_graph_statistics
from common.graph_variants import GRAPH_VARIANTS

from .graph_data import (
    DEFAULT_IDENTIFIER_WINDOW_HOURS,
    GRAPH_CONSTRUCTIONS,
    get_default_ieee_cis_graph_edges_csv_path,
    get_default_ieee_cis_graph_cache_path,
    load_or_build_ieee_cis_graph,
    save_edge_index_to_csv,
)
from .preprocessing import load_ieee_cis


TRANSACTION_PATH = "data/ieee-cis/train_transaction.csv"
IDENTITY_PATH = "data/ieee-cis/train_identity.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Build the IEEE-CIS transaction graph.")
    parser.add_argument("--rebuild-graph", action="store_true")
    parser.add_argument("--max-group-size", type=int, default=1000)
    parser.add_argument(
        "--graph-construction",
        choices=GRAPH_CONSTRUCTIONS,
        default="card",
        help="Meaningful IEEE-CIS graph construction to build.",
    )
    parser.add_argument(
        "--identifier-window-hours",
        type=int,
        default=DEFAULT_IDENTIFIER_WINDOW_HOURS,
        help="Time window for identifier_time_window graph edges.",
    )
    parser.add_argument("--graph-variant", choices=GRAPH_VARIANTS, default="original")
    parser.add_argument("--complement-average-degree", type=int, default=20)
    parser.add_argument("--complement-seed", type=int, default=42)
    parser.add_argument(
        "--edge-csv-path",
        default=None,
        help="Optional path for the exported edge-list CSV. Defaults to a construction-specific file in results/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = load_ieee_cis(TRANSACTION_PATH, IDENTITY_PATH)
    data = load_or_build_ieee_cis_graph(
        df,
        rebuild=args.rebuild_graph,
        max_group_size=args.max_group_size,
        graph_construction=args.graph_construction,
        identifier_window_hours=args.identifier_window_hours,
        graph_variant=args.graph_variant,
        complement_average_degree=args.complement_average_degree,
        complement_seed=args.complement_seed,
    )

    print(f"IEEE-CIS graph ready ({args.graph_construction}, {args.graph_variant})")
    print_graph_statistics(data)
    print(f"Features: {data.x.size(1)}")
    print(f"Saved graph construction: {data.graph_construction}")
    if data.graph_construction == "identifier_time_window":
        print(f"Identifier window hours: {data.identifier_window_hours}")
    cache_path = get_default_ieee_cis_graph_cache_path(
        graph_construction=args.graph_construction,
        max_group_size=args.max_group_size,
        identifier_window_hours=args.identifier_window_hours,
        graph_variant=args.graph_variant,
        complement_average_degree=args.complement_average_degree,
        complement_seed=args.complement_seed,
    )
    edge_csv_path = args.edge_csv_path or get_default_ieee_cis_graph_edges_csv_path(
        graph_construction=args.graph_construction,
        max_group_size=args.max_group_size,
        identifier_window_hours=args.identifier_window_hours,
        graph_variant=args.graph_variant,
        complement_average_degree=args.complement_average_degree,
        complement_seed=args.complement_seed,
    )
    save_edge_index_to_csv(data.edge_index, edge_csv_path)

    print(f"Saved to {cache_path}")
    print(f"Saved edge CSV to {edge_csv_path}")
