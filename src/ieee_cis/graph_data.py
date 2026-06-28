from pathlib import Path
import torch

from common.graph_variants import apply_graph_variant

from .preprocessing import preprocess_ieee_cis

try:
    from torch_geometric.data import Data
except ImportError as exc:
    raise ImportError(
        "PyTorch Geometric is required for the GNN baselines. "
        "Install it with: pip install torch-geometric"
    ) from exc


GRAPH_EDGE_COLUMNS = [
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "id_30",
    "id_31",
    "id_33",
]


def build_temporal_masks(num_nodes, train_ratio=0.80, val_ratio=0.10):
    train_end = int(num_nodes * train_ratio)
    val_end = int(num_nodes * (train_ratio + val_ratio))

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True

    return train_mask, val_mask, test_mask


def build_edges_from_shared_values(df, edge_columns=None, max_group_size=1000):
    edge_columns = edge_columns or GRAPH_EDGE_COLUMNS
    edge_pairs = set()

    sort_column = "TransactionDT" if "TransactionDT" in df.columns else None

    for column in edge_columns:
        if column not in df.columns:
            continue

        values = df[column]
        valid_values = values.notna()
        if not valid_values.any():
            continue

        for _, group in df.loc[valid_values].groupby(column, sort=False):
            if len(group) < 2 or len(group) > max_group_size:
                continue

            if sort_column:
                node_ids = group.sort_values(sort_column).index.to_numpy()
            else:
                node_ids = group.index.to_numpy()

            source_nodes = node_ids[:-1]
            target_nodes = node_ids[1:]

            for src, dst in zip(source_nodes, target_nodes):
                src = int(src)
                dst = int(dst)
                edge_pairs.add((src, dst))
                edge_pairs.add((dst, src))

    if not edge_pairs:
        raise ValueError("No graph edges were created. Check graph edge columns and missing values.")

    edge_index = torch.tensor(list(edge_pairs), dtype=torch.long).t().contiguous()
    return edge_index


def build_ieee_cis_graph(
    df,
    max_group_size=1000,
    graph_variant="original",
    complement_average_degree=20,
    complement_seed=42,
):
    df = df.reset_index(drop=True)

    X, y = preprocess_ieee_cis(df)
    edge_index = build_edges_from_shared_values(df, max_group_size=max_group_size)
    edge_index = apply_graph_variant(
        edge_index,
        num_nodes=len(df),
        graph_variant=graph_variant,
        complement_average_degree=complement_average_degree,
        complement_seed=complement_seed,
    )
    train_mask, val_mask, test_mask = build_temporal_masks(len(df))

    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    data.graph_variant = graph_variant

    return data


def load_or_build_ieee_cis_graph(
    df,
    cache_path="results/ieee_cis_graph.pt",
    max_group_size=1000,
    rebuild=False,
    graph_variant="original",
    complement_average_degree=20,
    complement_seed=42,
):
    cache_path = Path(cache_path)
    if graph_variant != "original" and cache_path == Path("results/ieee_cis_graph.pt"):
        cache_path = Path(
            f"results/ieee_cis_graph_{graph_variant}"
            f"_deg{complement_average_degree}_seed{complement_seed}.pt"
        )

    if cache_path.exists() and not rebuild:
        return torch.load(cache_path, weights_only=False)

    data = build_ieee_cis_graph(
        df,
        max_group_size=max_group_size,
        graph_variant=graph_variant,
        complement_average_degree=complement_average_degree,
        complement_seed=complement_seed,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)

    return data
