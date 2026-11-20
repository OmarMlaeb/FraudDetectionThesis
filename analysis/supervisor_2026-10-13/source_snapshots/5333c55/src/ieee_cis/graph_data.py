from pathlib import Path
import csv

import torch

from common.graph_variants import apply_graph_variant

from .preprocessing import preprocess_ieee_cis_with_train_fit, sort_ieee_cis_temporally

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
CARD_COLUMNS = ["card1", "card2", "card3", "card4", "card5", "card6"]
ADDRESS_COLUMNS = ["addr1", "addr2"]
EMAIL_DOMAIN_COLUMNS = ["P_emaildomain", "R_emaildomain"]
IDENTIFIER_COLUMNS = ["DeviceInfo", "id_30", "id_31", "id_33"]
GRAPH_CONSTRUCTIONS = (
    "main",
    "card",
    "address",
    "email_domain",
    "identifier_time_window",
)
DEFAULT_GRAPH_CONSTRUCTION = "card"
DEFAULT_IDENTIFIER_WINDOW_HOURS = 24
DEFAULT_GRAPH_CACHE_PATH = Path("results/ieee_cis_graph_trainfit.pt")
MISSING_VALUE = "__MISSING__"


def _get_graph_file_suffix(
    graph_construction=DEFAULT_GRAPH_CONSTRUCTION,
    max_group_size=1000,
    identifier_window_hours=DEFAULT_IDENTIFIER_WINDOW_HOURS,
    graph_variant="original",
    complement_average_degree=20,
    complement_seed=42,
):
    cache_suffix = graph_construction
    if graph_construction == "identifier_time_window":
        cache_suffix = f"{cache_suffix}_{identifier_window_hours}h"
    else:
        cache_suffix = f"{cache_suffix}_max{max_group_size}"

    if graph_variant != "original":
        cache_suffix = (
            f"{cache_suffix}_{graph_variant}"
            f"_deg{complement_average_degree}_seed{complement_seed}"
        )

    return cache_suffix


def get_default_ieee_cis_graph_cache_path(
    graph_construction=DEFAULT_GRAPH_CONSTRUCTION,
    max_group_size=1000,
    identifier_window_hours=DEFAULT_IDENTIFIER_WINDOW_HOURS,
    graph_variant="original",
    complement_average_degree=20,
    complement_seed=42,
):
    cache_suffix = _get_graph_file_suffix(
        graph_construction=graph_construction,
        max_group_size=max_group_size,
        identifier_window_hours=identifier_window_hours,
        graph_variant=graph_variant,
        complement_average_degree=complement_average_degree,
        complement_seed=complement_seed,
    )
    return Path(f"results/ieee_cis_graph_trainfit_{cache_suffix}.pt")


def get_default_ieee_cis_graph_edges_csv_path(
    graph_construction=DEFAULT_GRAPH_CONSTRUCTION,
    max_group_size=1000,
    identifier_window_hours=DEFAULT_IDENTIFIER_WINDOW_HOURS,
    graph_variant="original",
    complement_average_degree=20,
    complement_seed=42,
):
    file_suffix = _get_graph_file_suffix(
        graph_construction=graph_construction,
        max_group_size=max_group_size,
        identifier_window_hours=identifier_window_hours,
        graph_variant=graph_variant,
        complement_average_degree=complement_average_degree,
        complement_seed=complement_seed,
    )
    return Path(f"results/ieee_cis_graph_edges_{file_suffix}.csv")


def save_edge_index_to_csv(edge_index, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    edge_index = edge_index.detach().cpu()
    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["source_node", "target_node"])
        writer.writerows(edge_index.t().tolist())


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


def _add_temporal_chain_edges(edge_pairs, df, node_ids):
    if "TransactionDT" in df.columns:
        node_ids = df.loc[node_ids].sort_values("TransactionDT").index.to_numpy()

    source_nodes = node_ids[:-1]
    target_nodes = node_ids[1:]

    for src, dst in zip(source_nodes, target_nodes):
        src = int(src)
        dst = int(dst)
        edge_pairs.add((src, dst))
        edge_pairs.add((dst, src))


def build_edges_from_signature(df, columns, max_group_size=1000):
    columns = [column for column in columns if column in df.columns]
    if not columns:
        raise ValueError("None of the requested graph columns exist in the dataframe.")

    key_frame = df[columns].copy()
    valid_rows = key_frame.notna().any(axis=1)
    key_frame = key_frame[valid_rows].fillna(MISSING_VALUE).astype(str)
    edge_pairs = set()

    for _, group in key_frame.groupby(columns, sort=False):
        if len(group) < 2 or len(group) > max_group_size:
            continue

        _add_temporal_chain_edges(edge_pairs, df, group.index.to_numpy())

    if not edge_pairs:
        raise ValueError("No graph edges were created. Check graph columns and max group size.")

    return torch.tensor(list(edge_pairs), dtype=torch.long).t().contiguous()


def build_edges_from_shared_values_with_time_window(
    df,
    columns,
    window_hours=DEFAULT_IDENTIFIER_WINDOW_HOURS,
    max_group_size=1000,
):
    columns = [column for column in columns if column in df.columns]
    if not columns:
        raise ValueError("None of the requested graph columns exist in the dataframe.")
    if "TransactionDT" not in df.columns:
        raise ValueError("identifier_time_window graph requires TransactionDT.")
    if window_hours <= 0:
        raise ValueError("window_hours must be greater than zero.")

    max_seconds = window_hours * 60 * 60
    edge_pairs = set()

    for column in columns:
        values = df[column]
        valid_values = values.notna()
        if not valid_values.any():
            continue

        for _, group in df.loc[valid_values].groupby(column, sort=False):
            if len(group) < 2 or len(group) > max_group_size:
                continue

            group = group.sort_values("TransactionDT")
            node_ids = group.index.to_numpy()
            times = group["TransactionDT"].to_numpy()

            for target_index in range(1, len(node_ids)):
                source_index = target_index - 1
                while (
                    source_index >= 0
                    and times[target_index] - times[source_index] <= max_seconds
                ):
                    src = int(node_ids[source_index])
                    dst = int(node_ids[target_index])
                    edge_pairs.add((src, dst))
                    edge_pairs.add((dst, src))
                    source_index -= 1

    if not edge_pairs:
        raise ValueError(
            "No graph edges were created. Check identifier columns and time window."
        )

    return torch.tensor(list(edge_pairs), dtype=torch.long).t().contiguous()


def build_ieee_cis_edge_index(
    df,
    graph_construction=DEFAULT_GRAPH_CONSTRUCTION,
    max_group_size=1000,
    identifier_window_hours=DEFAULT_IDENTIFIER_WINDOW_HOURS,
):
    if graph_construction == "main":
        return build_edges_from_shared_values(
            df,
            GRAPH_EDGE_COLUMNS,
            max_group_size=max_group_size,
        )

    if graph_construction == "card":
        return build_edges_from_signature(
            df,
            CARD_COLUMNS,
            max_group_size=max_group_size,
        )

    if graph_construction == "address":
        return build_edges_from_signature(
            df,
            ADDRESS_COLUMNS,
            max_group_size=max_group_size,
        )

    if graph_construction == "email_domain":
        return build_edges_from_shared_values(
            df,
            EMAIL_DOMAIN_COLUMNS,
            max_group_size=max_group_size,
        )

    if graph_construction == "identifier_time_window":
        return build_edges_from_shared_values_with_time_window(
            df,
            IDENTIFIER_COLUMNS,
            window_hours=identifier_window_hours,
            max_group_size=max_group_size,
        )

    raise ValueError(
        f"graph_construction must be one of: {', '.join(GRAPH_CONSTRUCTIONS)}"
    )


def build_ieee_cis_graph(
    df,
    max_group_size=1000,
    graph_construction=DEFAULT_GRAPH_CONSTRUCTION,
    identifier_window_hours=DEFAULT_IDENTIFIER_WINDOW_HOURS,
    graph_variant="original",
    complement_average_degree=20,
    complement_seed=42,
):
    df = sort_ieee_cis_temporally(df).reset_index(drop=True)

    X, y = preprocess_ieee_cis_with_train_fit(df)
    edge_index = build_ieee_cis_edge_index(
        df,
        graph_construction=graph_construction,
        max_group_size=max_group_size,
        identifier_window_hours=identifier_window_hours,
    )
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
    data.graph_construction = graph_construction
    data.identifier_window_hours = identifier_window_hours
    data.graph_variant = graph_variant

    return data


def load_or_build_ieee_cis_graph(
    df,
    cache_path=DEFAULT_GRAPH_CACHE_PATH,
    max_group_size=1000,
    rebuild=False,
    graph_construction=DEFAULT_GRAPH_CONSTRUCTION,
    identifier_window_hours=DEFAULT_IDENTIFIER_WINDOW_HOURS,
    graph_variant="original",
    complement_average_degree=20,
    complement_seed=42,
):
    cache_path = Path(cache_path)
    if cache_path == DEFAULT_GRAPH_CACHE_PATH:
        cache_path = get_default_ieee_cis_graph_cache_path(
            graph_construction=graph_construction,
            max_group_size=max_group_size,
            identifier_window_hours=identifier_window_hours,
            graph_variant=graph_variant,
            complement_average_degree=complement_average_degree,
            complement_seed=complement_seed,
        )

    if cache_path.exists() and not rebuild:
        return torch.load(cache_path, weights_only=False)

    data = build_ieee_cis_graph(
        df,
        max_group_size=max_group_size,
        graph_construction=graph_construction,
        identifier_window_hours=identifier_window_hours,
        graph_variant=graph_variant,
        complement_average_degree=complement_average_degree,
        complement_seed=complement_seed,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)

    return data
