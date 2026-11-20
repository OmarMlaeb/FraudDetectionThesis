from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.graph_variants import apply_graph_variant


FEATURES_PATH = "data/elliptic_bitcoin_dataset/elliptic_txs_features.csv"
CLASSES_PATH = "data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv"
EDGELIST_PATH = "data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"


def load_elliptic_node_data(features_path=FEATURES_PATH, classes_path=CLASSES_PATH):
    first_row = pd.read_csv(features_path, header=None, nrows=1)
    feature_count = first_row.shape[1] - 2
    columns = ["txId", "time_step"] + [f"feature_{i}" for i in range(feature_count)]

    features_df = pd.read_csv(features_path, header=None, names=columns)
    classes_df = pd.read_csv(classes_path)

    df = features_df.merge(classes_df, on="txId", how="left")
    df["label"] = df["class"].astype(str).map({"1": 1.0, "2": 0.0})
    df = df.sort_values(["time_step", "txId"]).reset_index(drop=True)

    X = df[[f"feature_{i}" for i in range(feature_count)]].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].values.astype(np.float32)
    time_steps = df["time_step"].values.astype(np.int64)

    return df, X, y, time_steps


def build_chronological_masks(y, time_steps, train_ratio=0.70, val_ratio=0.15):
    known_mask_np = ~np.isnan(y)
    known_time_steps = np.sort(np.unique(time_steps[known_mask_np]))

    train_end = int(len(known_time_steps) * train_ratio)
    val_end = int(len(known_time_steps) * (train_ratio + val_ratio))

    train_steps = set(known_time_steps[:train_end])
    val_steps = set(known_time_steps[train_end:val_end])
    test_steps = set(known_time_steps[val_end:])

    train_mask_np = known_mask_np & np.isin(time_steps, list(train_steps))
    val_mask_np = known_mask_np & np.isin(time_steps, list(val_steps))
    test_mask_np = known_mask_np & np.isin(time_steps, list(test_steps))

    return (
        torch.tensor(train_mask_np, dtype=torch.bool),
        torch.tensor(val_mask_np, dtype=torch.bool),
        torch.tensor(test_mask_np, dtype=torch.bool),
    )


def load_elliptic_tabular_data():
    df, X, y, time_steps = load_elliptic_node_data()
    train_mask, val_mask, test_mask = build_chronological_masks(y, time_steps)

    return {
        "df": df,
        "X": X,
        "y": np.nan_to_num(y, nan=0.0).astype(np.float32),
        "time_steps": time_steps,
        "train_mask": train_mask.numpy(),
        "val_mask": val_mask.numpy(),
        "test_mask": test_mask.numpy(),
    }


def build_elliptic_edge_index(df, edgelist_path=EDGELIST_PATH):
    edges_df = pd.read_csv(edgelist_path)
    node_index = {tx_id: index for index, tx_id in enumerate(df["txId"].values)}

    source_nodes = edges_df["txId1"].map(node_index)
    target_nodes = edges_df["txId2"].map(node_index)
    valid_edges = source_nodes.notna() & target_nodes.notna()

    source_nodes = source_nodes[valid_edges].astype(int).to_numpy()
    target_nodes = target_nodes[valid_edges].astype(int).to_numpy()

    forward_edges = np.stack([source_nodes, target_nodes])
    reverse_edges = np.stack([target_nodes, source_nodes])
    edge_index = np.concatenate([forward_edges, reverse_edges], axis=1)

    return torch.tensor(edge_index, dtype=torch.long)


def build_elliptic_graph(
    graph_variant="original",
    complement_average_degree=20,
    complement_seed=42,
):
    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError(
            "PyTorch Geometric is required for the Elliptic GNN baselines. "
            "Install it with: pip install torch-geometric"
        ) from exc

    tabular_data = load_elliptic_tabular_data()
    df = tabular_data["df"]

    edge_index = build_elliptic_edge_index(df)
    edge_index = apply_graph_variant(
        edge_index,
        num_nodes=len(df),
        graph_variant=graph_variant,
        complement_average_degree=complement_average_degree,
        complement_seed=complement_seed,
    )

    data = Data(
        x=torch.tensor(tabular_data["X"], dtype=torch.float32),
        y=torch.tensor(tabular_data["y"], dtype=torch.float32),
        edge_index=edge_index,
        train_mask=torch.tensor(tabular_data["train_mask"], dtype=torch.bool),
        val_mask=torch.tensor(tabular_data["val_mask"], dtype=torch.bool),
        test_mask=torch.tensor(tabular_data["test_mask"], dtype=torch.bool),
    )
    data.graph_variant = graph_variant

    return data


def load_or_build_elliptic_graph(
    cache_path="results/elliptic_graph.pt",
    rebuild=False,
    graph_variant="original",
    complement_average_degree=20,
    complement_seed=42,
):
    cache_path = Path(cache_path)
    if graph_variant != "original" and cache_path == Path("results/elliptic_graph.pt"):
        cache_path = Path(
            f"results/elliptic_graph_{graph_variant}"
            f"_deg{complement_average_degree}_seed{complement_seed}.pt"
        )

    if cache_path.exists() and not rebuild:
        return torch.load(cache_path, weights_only=False)

    data = build_elliptic_graph(
        graph_variant=graph_variant,
        complement_average_degree=complement_average_degree,
        complement_seed=complement_seed,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)

    return data
