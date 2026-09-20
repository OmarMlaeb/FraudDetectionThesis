"""Retrospective label-connectivity analysis for IEEE-CIS graphs.

True labels are used only after graph construction to describe connectivity.
They are never used to construct an edge.
"""

from pathlib import Path
import argparse
import csv

import numpy as np
import pandas as pd


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


EXPECTED_EDGE_COUNTS = {
    "Main": 406_769,
    "Card": 304_889,
    "Address": 7_424,
    "Email domain": 16_657,
    "Identifier, 24 hours": 345_272,
}


def load_aligned_transactions(transaction_path, identity_path):
    transaction_columns = pd.read_csv(transaction_path, nrows=0).columns
    identity_columns = pd.read_csv(identity_path, nrows=0).columns
    transaction_usecols = [
        column
        for column in ["TransactionID", "TransactionDT", "isFraud", *GRAPH_EDGE_COLUMNS]
        if column in transaction_columns
    ]
    identity_usecols = [
        column
        for column in ["TransactionID", *GRAPH_EDGE_COLUMNS]
        if column in identity_columns and column != "TransactionID"
    ]

    transactions = pd.read_csv(transaction_path, usecols=transaction_usecols)
    identities = pd.read_csv(
        identity_path,
        usecols=["TransactionID", *identity_usecols],
    )
    return (
        transactions.merge(identities, on="TransactionID", how="left")
        .sort_values(["TransactionDT", "TransactionID"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_main_undirected_edges(df, max_group_size=1000):
    edges = set()
    for column in GRAPH_EDGE_COLUMNS:
        if column not in df.columns:
            continue
        valid = df[column].notna()
        for _, group in df.loc[valid].groupby(column, sort=False):
            if not 2 <= len(group) <= max_group_size:
                continue
            nodes = group.sort_values("TransactionDT").index.to_numpy()
            for source, target in zip(nodes[:-1], nodes[1:]):
                source, target = int(source), int(target)
                if source != target:
                    edges.add((min(source, target), max(source, target)))
    return edges


def load_undirected_edges(path):
    edges = set()
    with Path(path).open(newline="") as edge_file:
        for row in csv.DictReader(edge_file):
            source = int(row["source_node"])
            target = int(row["target_node"])
            if source != target:
                edges.add((min(source, target), max(source, target)))
    return edges


def connectivity_row(name, edges, labels):
    fraud = labels == 1
    fraud_with_neighbor = np.zeros(len(labels), dtype=bool)
    fraud_with_fraud_neighbor = np.zeros(len(labels), dtype=bool)
    fraud_fraud = fraud_legitimate = legitimate_legitimate = 0

    for source, target in edges:
        source_fraud = bool(fraud[source])
        target_fraud = bool(fraud[target])
        if source_fraud and target_fraud:
            fraud_fraud += 1
            fraud_with_fraud_neighbor[source] = True
            fraud_with_fraud_neighbor[target] = True
        elif source_fraud or target_fraud:
            fraud_legitimate += 1
        else:
            legitimate_legitimate += 1
        if source_fraud:
            fraud_with_neighbor[source] = True
        if target_fraud:
            fraud_with_neighbor[target] = True

    total_fraud = int(fraud.sum())
    edge_total = fraud_fraud + fraud_legitimate + legitimate_legitimate
    expected = EXPECTED_EDGE_COUNTS[name]
    if edge_total != expected:
        raise ValueError(f"{name}: calculated {edge_total:,} edges; expected {expected:,}")

    return {
        "construction": name,
        "fraud_fraud_edges": fraud_fraud,
        "fraud_legitimate_edges": fraud_legitimate,
        "legitimate_legitimate_edges": legitimate_legitimate,
        "fraud_nodes_with_any_neighbor_pct": 100 * fraud_with_neighbor.sum() / total_fraud,
        "fraud_nodes_with_fraud_neighbor_pct": (
            100 * fraud_with_fraud_neighbor.sum() / total_fraud
        ),
        "edge_count_check": edge_total,
        "total_fraud_nodes": total_fraud,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction-path", default="data/ieee-cis/train_transaction.csv")
    parser.add_argument("--identity-path", default="data/ieee-cis/train_identity.csv")
    parser.add_argument(
        "--output-path",
        default="results/ieee_cis_graph_connectivity.csv",
    )
    args = parser.parse_args()

    df = load_aligned_transactions(args.transaction_path, args.identity_path)
    labels = df["isFraud"].to_numpy()
    edge_sources = {
        "Main": build_main_undirected_edges(df),
        "Card": load_undirected_edges("results/ieee_cis_graph_edges_card_max1000.csv"),
        "Address": load_undirected_edges("results/ieee_cis_graph_edges_address_max1000.csv"),
        "Email domain": load_undirected_edges(
            "results/ieee_cis_graph_edges_email_domain_max1000.csv"
        ),
        "Identifier, 24 hours": load_undirected_edges(
            "results/ieee_cis_graph_edges_identifier_time_window_24h.csv"
        ),
    }
    results = pd.DataFrame(
        connectivity_row(name, edges, labels) for name, edges in edge_sources.items()
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, float_format="%.6f")
    print(results.to_string(index=False))
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
