import random

import torch


GRAPH_VARIANTS = ("original", "sampled_complement")


def _undirected_edge_set(edge_index):
    edges = edge_index.detach().cpu()
    edge_set = set()

    for source, target in edges.t().tolist():
        if source == target:
            continue
        edge_set.add((min(source, target), max(source, target)))

    return edge_set


def build_sampled_complement_edge_index(
    edge_index,
    num_nodes,
    average_degree=20,
    seed=42,
):
    if average_degree <= 0:
        raise ValueError("average_degree must be greater than zero.")

    original_edges = _undirected_edge_set(edge_index)
    max_possible_edges = num_nodes * (num_nodes - 1) // 2
    available_non_edges = max_possible_edges - len(original_edges)

    if available_non_edges <= 0:
        raise ValueError("The graph is already complete, so no complement edges are available.")

    target_edges = min(int(num_nodes * average_degree / 2), available_non_edges)
    rng = random.Random(seed)
    complement_edges = set()
    max_attempts = max(target_edges * 20, 1000)
    attempts = 0

    while len(complement_edges) < target_edges and attempts < max_attempts:
        source = rng.randrange(num_nodes)
        target = rng.randrange(num_nodes)
        attempts += 1

        if source == target:
            continue

        edge = (min(source, target), max(source, target))
        if edge in original_edges or edge in complement_edges:
            continue

        complement_edges.add(edge)

    if len(complement_edges) < target_edges:
        raise RuntimeError(
            "Could not sample enough complement edges. "
            "Try a lower complement average degree."
        )

    directed_edges = []
    for source, target in complement_edges:
        directed_edges.append((source, target))
        directed_edges.append((target, source))

    return torch.tensor(directed_edges, dtype=torch.long).t().contiguous()


def apply_graph_variant(
    edge_index,
    num_nodes,
    graph_variant="original",
    complement_average_degree=20,
    complement_seed=42,
):
    if graph_variant == "original":
        return edge_index

    if graph_variant == "sampled_complement":
        return build_sampled_complement_edge_index(
            edge_index,
            num_nodes=num_nodes,
            average_degree=complement_average_degree,
            seed=complement_seed,
        )

    raise ValueError(f"graph_variant must be one of: {', '.join(GRAPH_VARIANTS)}")
