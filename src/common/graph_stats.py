import torch


def _unique_undirected_edges(edge_index):
    if edge_index.numel() == 0:
        return edge_index

    edges = edge_index.detach().cpu()
    source = torch.minimum(edges[0], edges[1])
    target = torch.maximum(edges[0], edges[1])
    undirected_edges = torch.stack([source, target], dim=0)
    undirected_edges = undirected_edges[:, source != target]

    if undirected_edges.numel() == 0:
        return undirected_edges

    return torch.unique(undirected_edges.t(), dim=0).t().contiguous()


def compute_graph_statistics(data):
    num_nodes = int(data.num_nodes)
    undirected_edges = _unique_undirected_edges(data.edge_index)
    num_edges = int(undirected_edges.size(1)) if undirected_edges.numel() else 0

    average_degree = (2 * num_edges / num_nodes) if num_nodes else 0.0
    max_possible_edges = num_nodes * (num_nodes - 1) / 2
    graph_density = (num_edges / max_possible_edges) if max_possible_edges else 0.0

    adjacency = [[] for _ in range(num_nodes)]
    if num_edges:
        for source, target in undirected_edges.t().tolist():
            adjacency[source].append(target)
            adjacency[target].append(source)

    visited = [False] * num_nodes
    connected_components = 0

    for node in range(num_nodes):
        if visited[node]:
            continue

        connected_components += 1
        stack = [node]
        visited[node] = True

        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

    return {
        "number_of_nodes": num_nodes,
        "number_of_edges": num_edges,
        "average_degree": average_degree,
        "connected_components": connected_components,
        "graph_density": graph_density,
    }


def print_graph_statistics(data):
    stats = compute_graph_statistics(data)
    print(f"Number of nodes: {stats['number_of_nodes']}")
    print(f"Number of edges: {stats['number_of_edges']}")
    print(f"Average degree: {stats['average_degree']:.4f}")
    print(f"Connected components: {stats['connected_components']}")
    print(f"Graph density: {stats['graph_density']:.8f}")
