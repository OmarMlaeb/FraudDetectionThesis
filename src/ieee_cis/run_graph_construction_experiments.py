import argparse

from .graph_data import (
    DEFAULT_IDENTIFIER_WINDOW_HOURS,
    GRAPH_CONSTRUCTIONS,
)
from .train_gnn import DEFAULT_GRAPH_RESULTS_PATH, train as train_gnn


DEFAULT_SEEDS = list(range(42, 52))
DEFAULT_MODELS = ("gcn", "sage", "gat")


def run_graph_construction_experiments(
    seeds=None,
    models=None,
    graph_constructions=None,
    identifier_window_hours=DEFAULT_IDENTIFIER_WINDOW_HOURS,
    max_group_size=1000,
    output_path=DEFAULT_GRAPH_RESULTS_PATH,
    rebuild_graph=False,
):
    seeds = seeds or DEFAULT_SEEDS
    models = models or DEFAULT_MODELS
    graph_constructions = graph_constructions or GRAPH_CONSTRUCTIONS

    for graph_construction in graph_constructions:
        for model_name in models:
            for seed_index, seed in enumerate(seeds):
                should_rebuild = rebuild_graph and model_name == models[0] and seed_index == 0
                print(
                    f"\nRunning IEEE-CIS {model_name.upper()} "
                    f"on {graph_construction} graph with seed {seed}"
                )
                print("=" * 70)

                train_gnn(
                    model_name=model_name,
                    rebuild_graph=should_rebuild,
                    max_group_size=max_group_size,
                    graph_construction=graph_construction,
                    identifier_window_hours=identifier_window_hours,
                    output_path=output_path,
                    seed=seed,
                )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run IEEE-CIS graph models across graph constructions and seeds."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Random seeds to run. Defaults to 42 through 51.",
    )
    parser.add_argument(
        "--models",
        choices=DEFAULT_MODELS,
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="GNN models to run.",
    )
    parser.add_argument(
        "--graph-constructions",
        choices=GRAPH_CONSTRUCTIONS,
        nargs="+",
        default=list(GRAPH_CONSTRUCTIONS),
        help="IEEE-CIS graph constructions to run.",
    )
    parser.add_argument(
        "--identifier-window-hours",
        type=int,
        default=DEFAULT_IDENTIFIER_WINDOW_HOURS,
        help="Time window for identifier_time_window graph edges.",
    )
    parser.add_argument("--max-group-size", type=int, default=1000)
    parser.add_argument(
        "--output-path",
        default=DEFAULT_GRAPH_RESULTS_PATH,
        help="CSV path for IEEE-CIS graph-model results.",
    )
    parser.add_argument(
        "--rebuild-graph",
        action="store_true",
        help="Rebuild each graph construction once before its first model run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_graph_construction_experiments(
        seeds=args.seeds,
        models=args.models,
        graph_constructions=args.graph_constructions,
        identifier_window_hours=args.identifier_window_hours,
        max_group_size=args.max_group_size,
        output_path=args.output_path,
        rebuild_graph=args.rebuild_graph,
    )


if __name__ == "__main__":
    main()
