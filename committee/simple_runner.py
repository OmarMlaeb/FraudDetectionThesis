"""One script per experiment; the detailed provenance checks stay behind the scenes."""
import argparse
import contextlib
import csv
import io

from common import OUT, LABELS, atomic_bytes, read, environment, require
from plan import matrix, settings
from cli import WorkspaceLock
from graphs import generate, read_graph_manifest
from runner import train_one
from postprocess import validate, analyze


EXPERIMENTS = {
    "complement": {
        "title": "Sampled complement: IEEE-CIS and Elliptic, GCN/GraphSAGE/GAT",
        "name": "sampled_complement",
        "script": "run_sampled_complement.py",
    },
    "local": {
        "title": "Elliptic local features: MLP/LSTM/Transformer/GCN/GraphSAGE/GAT",
        "name": "elliptic_local_features",
        "script": "run_elliptic_local_features.py",
    },
    "random": {
        "title": "Degree-preserving random graphs: five IEEE-CIS graphs and Elliptic",
        "name": "random_graph",
        "script": "run_random_graph.py",
    },
}

# Same columns, in the same order, as results/model_results.csv.
RESULT_COLUMNS = [
    "timestamp", "model", "recall", "precision", "f1_score", "pr_auc", "roc_auc",
    "true_negative", "false_positive", "false_negative", "true_positive",
    "threshold", "threshold_strategy", "validation_f1_at_threshold", "seed",
]
GRAPH_COLUMNS = [
    "dataset", "construction", "graph_variant", "graph_seed", "feature_set", "input_features",
    "nodes", "original_edges", "new_edges", "stored_directed_edges",
    "original_average_degree", "new_average_degree", "original_density", "new_density", "density_change",
    "original_isolated_nodes", "new_isolated_nodes", "original_components", "new_components",
    "exact_node_degrees_preserved", "isolated_node_ids_preserved", "symmetric", "self_loops", "duplicate_edges",
    "shared_original_edges", "jaccard", "time_constraint", "requested_swaps", "successful_swaps", "attempts",
    "acceptance_rate", "limited_randomization", "runtime_seconds", "source_graph_sha256",
    "topology_sha256", "effective_graph_sha256",
]


def save_csv(path, rows, columns):
    """Rewrite atomically from unique completed runs, so resuming never duplicates rows."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, buffer.getvalue().encode("utf-8"))


def model_label(row):
    label = LABELS[row["model"]]
    if row["dataset"] == "elliptic":
        label = "Elliptic-" + label
    else:
        label += "-" + row["construction"]
    suffix = {"complement": "sampled_complement", "local": "local_features", "random": "random_graph"}[row["family"]]
    return label + "-" + suffix


def export_results(family, rows, accepted):
    output = []
    for row in rows:
        if row["run_id"] not in accepted:
            continue
        metric = accepted[row["run_id"]]
        status = read(OUT / "runs" / family / row["run_id"] / "status.json")
        config = read(status["config"]["path"])
        record = {key: metric[key] for key in RESULT_COLUMNS if key not in ["timestamp", "model"]}
        record.update(timestamp=config["created_utc"], model=model_label(row))
        output.append(record)
    path = OUT / (EXPERIMENTS[family]["name"] + "_results.csv")
    save_csv(path, output, RESULT_COLUMNS)
    return path


def graph_csv_row(row, manifest):
    report = manifest["statistics"]
    original, new = report["original"], report["resulting"]
    swaps = report["rewiring"]
    return dict(
        dataset="IEEE-CIS" if row["dataset"] == "ieee" else "Elliptic",
        construction=row["construction"], graph_variant=row["graph_spec"]["graph_variant"],
        graph_seed=row["graph_seed"], feature_set=row["feature_set"],
        input_features=93 if row["feature_set"] == "local93" else 358 if row["dataset"] == "ieee" else 165,
        nodes=new["nodes"], original_edges=original["unique_undirected_edges"], new_edges=new["unique_undirected_edges"],
        stored_directed_edges=new["stored_directed_columns"],
        original_average_degree=original["average_undirected_degree"], new_average_degree=new["average_undirected_degree"],
        original_density=original["density"], new_density=new["density"], density_change=report["density_change"],
        original_isolated_nodes=original["isolated_nodes"], new_isolated_nodes=new["isolated_nodes"],
        original_components=original["components"], new_components=new["components"],
        exact_node_degrees_preserved=report["exact_per_node_degree_equality"],
        isolated_node_ids_preserved=report["isolated_node_ids_equal"],
        # generate() rejects loops/duplicates/asymmetry before completing the manifest.
        symmetric=True, self_loops=0, duplicate_edges=0,
        shared_original_edges=report["overlap"]["shared_edges"], jaccard=report["overlap"]["jaccard"],
        time_constraint=report["constraints"], requested_swaps=swaps.get("requested_swaps"),
        successful_swaps=swaps.get("successful_swaps"), attempts=swaps.get("attempts"),
        acceptance_rate=swaps.get("acceptance_rate"), limited_randomization=swaps.get("limited_randomization"),
        runtime_seconds=report["runtime_seconds"], source_graph_sha256=report["source_graph_sha256"],
        topology_sha256=manifest["topology"]["sha256"], effective_graph_sha256=manifest["effective_graph_sha256"],
    )


def finish_reports(family, rows, experiment_settings):
    """Keep the complete statistical families, but show the user their selected experiment."""
    with (OUT / (EXPERIMENTS[family]["name"] + "_analysis.log")).open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log):
            analyze(experiment_settings)
    records = read(OUT / "analysis" / (family + "_per_seed.json"))
    accepted = {row["run_id"]: row for row in records if row["status"] == "accepted"}
    result_path = export_results(family, rows, accepted)
    name = EXPERIMENTS[family]["name"]
    atomic_bytes(OUT / (name + "_summary.csv"), (OUT / "analysis" / (family + "_summary.csv")).read_bytes())
    tests = [row for row in read(OUT / "analysis/wilcoxon_holm.json") if row["experiment_family"] == family]
    save_csv(OUT / (name + "_comparisons.csv"), tests, list(tests[0]))
    print(f"\nCompleted and checked: {len(accepted)}/{len(rows)} runs")
    print(f"Results:     {result_path}")
    print(f"Mean / SD:   {OUT / (name + '_summary.csv')}")
    print(f"Comparisons: {OUT / (name + '_comparisons.csv')}")
    return len(accepted) == len(rows)


def run_experiment(family, argv=None):
    info = EXPERIMENTS[family]
    parser = argparse.ArgumentParser(description=info["title"] + "; all ten seeds by default.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="List the plan without generating graphs or training.")
    modes.add_argument("--graphs-only", action="store_true", help="Prepare graphs and their statistics CSV without training.")
    modes.add_argument("--summarize-only", action="store_true", help="Refresh CSVs from existing results without training.")
    args = parser.parse_args(argv)
    experiment_settings = settings()
    rows = [row for row in matrix(experiment_settings) if row["family"] == family]
    print(info["title"])
    print(f"Seeds: 42-51 | Runs: {len(rows)} | One GPU, sequential execution")
    print(f"CSV: {OUT / (info['name'] + '_results.csv')}")
    if args.dry_run:
        print("Dry run only. No graph generation or training.")
        return 0
    failures = []
    with WorkspaceLock():
        environment()  # Check the verified versions automatically.
        if args.summarize_only:
            return 0 if finish_reports(family, rows, experiment_settings) else 2
        accepted, initial_report = validate(rows, write=False)
        invalid = [row for row in initial_report["runs"] if row["status"] == "invalid"]
        require(not invalid, f"Existing results failed verification; nothing was overwritten: {invalid}")
        accepted = {row["run_id"]: row for row in accepted}
        export_results(family, rows, accepted)
        graph_rows, available = [], set()
        graph_path = OUT / (info["name"] + "_graph_statistics.csv")
        save_csv(graph_path, graph_rows, GRAPH_COLUMNS)
        unique_graphs = {row["graph_id"]: row for row in rows}
        for row in unique_graphs.values():
            print(f"\nPreparing {row['dataset']} / {row['construction']} graph...")
            try:
                generate(row["graph_spec"])
                manifest = read_graph_manifest(row["graph_spec"])
                stats = graph_csv_row(row, manifest)
                graph_rows.append(stats)
                available.add(row["graph_id"])
                save_csv(graph_path, graph_rows, GRAPH_COLUMNS)
                print(f"Edges: {stats['new_edges']:,} | Average degree: {stats['new_average_degree']:.6f} | Density: {stats['new_density']:.10g}")
                if stats["limited_randomization"]:
                    print("Limited rewiring: see successful swaps, retained edges and attempts in the graph CSV.")
            except Exception as exc:
                failures.append(dict(stage="graph", run=row["graph_id"], error=str(exc)))
                print(f"Graph preparation failed: {exc}")
        print(f"\nGraph statistics: {graph_path}")
        if not args.graphs_only:
            for number, row in enumerate(rows, 1):
                if row["graph_id"] not in available:
                    continue
                print(f"\n[{number}/{len(rows)}] {model_label(row)} | Seed {row['seed']}")
                try:
                    train_one(row, experiment_settings, retry_failed=True)
                    verified, report = validate([row], write=False)
                    require(len(verified) == 1, str(report["runs"][0]))
                    accepted[row["run_id"]] = verified[0]
                    csv_path = export_results(family, rows, accepted)
                    print(f"Saved CSV: {csv_path.name} ({len(accepted)} completed runs)")
                except Exception as exc:
                    accepted.pop(row["run_id"], None)
                    export_results(family, rows, accepted)
                    failures.append(dict(stage="training", run=row["run_id"], error=str(exc)))
                    print(f"Run failed: {exc}\nContinuing with the remaining runs.")
            complete = finish_reports(family, rows, experiment_settings)
        else:
            complete = len(available) == len(unique_graphs)
        save_csv(OUT / (info["name"] + "_failures.csv"), failures, ["stage", "run", "error"])
        if failures or not complete:
            print(f"To resume, repeat: python {info['script']}")
            return 1
    return 0
