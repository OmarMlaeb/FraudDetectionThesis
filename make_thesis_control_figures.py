"""Generate the connected/isolated and Elliptic control figures without training.

Run from the project root: python make_thesis_control_figures.py
Uses the audited predictions and completed committee experiments.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_thesis_figures import ROOT, SOURCE, MODELS, SEEDS, load_results, require, save, sha

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

CONTROLS = ROOT / "results/committee_fast_experiments"
CONSTRUCTIONS = ["main", "card", "address", "email_domain", "identifier_time_window-24h"]
TITLES = ["Main", "Card", "Address", "Email domain", "Identifier, 24 h"]
SUBGROUP_MODELS = ["MLP", "GCN", "GraphSAGE", "GAT"]
ARM_COLORS = ["#005F60", "#777777"]


class Inputs:
    """Check saved evidence and record precisely which files are used."""
    def __init__(self):
        self.files = {}
        self.predictions = {}

    def record(self, path, expected=None):
        path = Path(path).resolve()
        key = path.relative_to(ROOT).as_posix()
        if key not in self.files:
            self.files[key] = sha(path)
        require(expected is None or self.files[key] == expected, f"Hash mismatch: {key}")
        return path

    def csv(self, path):
        return pd.read_csv(self.record(path))

    def json(self, path):
        return json.loads(self.record(path).read_text(encoding="utf-8"))

    def reference_prediction(self, run_id):
        if run_id not in self.predictions:
            check = self.json(SOURCE / "checks" / f"{run_id}.json")
            require(check.get("accepted") is True, f"Unverified reference: {run_id}")
            path = self.record(SOURCE / check["prediction_file"], check["prediction_sha256"])
            with np.load(path, allow_pickle=False) as p:
                self.predictions[run_id] = {k: p[k].copy() for k in ["test_ids", "test_y", "test_probs"]}
        return self.predictions[run_id]


def ten_seeds(frame, keys):
    require(not frame.duplicated(keys + ["seed"]).any(), "Duplicate seed rows.")
    for key, group in frame.groupby(keys):
        require(len(group) == 10 and set(group.seed) == SEEDS, f"Expected seeds 42–51: {key}")


def subgroup_data(inputs):
    counts = inputs.csv(SOURCE / "subgroup_counts.csv")
    counts = counts[counts.dataset == "IEEE-CIS"].copy()
    data = inputs.csv(SOURCE / "subgroup_per_seed.csv")
    data = data[(data.dataset == "IEEE-CIS") & data.architecture.isin(SUBGROUP_MODELS)].copy()
    require(len(data) == 400 and len(counts) == 10, "Incomplete IEEE-CIS subgroup results.")
    require(data.status.eq("verified").all() and data.checkpoint_accepted.eq(True).all(), "Unverified subgroup rows.")
    ten_seeds(data, ["construction", "architecture", "group"])
    archived = inputs.csv(SOURCE / "subgroup_summary.csv").set_index(["dataset", "construction", "architecture", "group"])
    for construction in CONSTRUCTIONS:
        memberships = {}
        for group in ["connected", "isolated"]:
            count = counts[(counts.construction == construction) & (counts.group == group)].iloc[0]
            path = inputs.record(SOURCE / "subgroup_ids" / f"IEEE-CIS__{construction}__{group}.npz")
            with np.load(path, allow_pickle=False) as z:
                ids, labels = z["transaction_ids"], z["labels"]
            require(hashlib.sha256(ids.tobytes()).hexdigest() == count.id_sha256, "Group ID hash differs.")
            require(len(np.unique(ids)) == len(ids) == count.n and labels.sum() == count.positive, "Group counts differ.")
            require(np.isclose(labels.mean(), count.fraud_rate, atol=1e-14), "Group fraud rate differs.")
            memberships[group] = ids
            selected = data[(data.construction == construction) & (data.group == group)]
            for row in selected.itertuples():
                p = inputs.reference_prediction(row.run_id)
                index = pd.Index(p["test_ids"]).get_indexer(ids)
                require((index >= 0).all() and np.array_equal(p["test_y"][index], labels), "Group population differs.")
                require(row.n == count.n and row.positive == count.positive, "Model group sizes differ.")
                score = average_precision_score(labels, p["test_probs"][index])
                require(abs(score - row.pr_auc) < 1e-12, f"Subgroup AP mismatch: {row.run_id}")
            for model, values in selected.groupby("architecture"):
                old = archived.loc[("IEEE-CIS", construction, model, group)]
                require(abs(values.pr_auc.mean() - old.pr_auc_mean) < 1e-12 and
                        abs(values.pr_auc.std(ddof=1) - old.pr_auc_sample_sd) < 1e-12, "Subgroup summary mismatch.")
        require(len(np.intersect1d(*memberships.values())) == 0, "Subgroups overlap.")
        require(sum(map(len, memberships.values())) == 59054, "Subgroups do not cover the test population.")
    return data, counts




def control_data(inputs, references):
    original = references[references.dataset == "Elliptic"].copy()
    original["arm"] = "original"
    validation = inputs.json(CONTROLS / "analysis/validation.json")
    accepted = {r["run_id"] for r in validation["runs"] if r["status"] == "accepted"}
    stats = inputs.csv(CONTROLS / "random_graph_graph_statistics.csv")
    graph = stats[stats.dataset == "Elliptic"].iloc[0]
    require(bool(graph.exact_node_degrees_preserved) and graph.time_constraint == "within_time_step", "Unexpected topology control.")
    require(graph.original_edges == graph.new_edges and graph.graph_seed == 20261013, "Unexpected random graph settings.")
    frames = [original]
    for family, expected_models, dimension in [("local", MODELS, 93), ("random", MODELS[3:], 165)]:
        frame = inputs.csv(CONTROLS / "analysis" / f"{family}_per_seed.csv")
        frame = frame[frame.dataset == "elliptic"].copy()
        frame["architecture"] = frame.model.map({"mlp": "MLP", "lstm": "LSTM", "transformer": "Transformer",
                                                  "gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT"})
        require(set(frame.architecture) == set(expected_models) and len(frame) == 10*len(expected_models), "Missing control models.")
        ten_seeds(frame, ["architecture"])
        require(frame.status.eq("accepted").all() and frame.input_dimension.eq(dimension).all(), "Unaccepted control results.")
        archived = inputs.csv(CONTROLS / f"{'elliptic_local_features' if family == 'local' else 'random_graph'}_summary.csv")
        archived = archived[archived.dataset == "elliptic"].set_index("model")
        for row in frame.itertuples():
            require(row.run_id in accepted, f"Control is not accepted: {row.run_id}")
            status = inputs.json(row.execution_status_path)
            require(status["status"] == "complete" and status["run_id"] == row.run_id, "Control status mismatch.")
            for key in ["metrics", "predictions"]:
                a = status["artifacts"][key]
                inputs.record(a["path"], a["sha256"])
            a = status["config"]
            inputs.record(a["path"], a["sha256"])
            config = inputs.json(a["path"])["compatibility"]
            require(config["row"]["reference_run_id"] == row.reference_run_id, "Reference pairing mismatch.")
            if family == "random":
                require(row.resulting_topology_sha256 == graph.topology_sha256 and
                        row.effective_graph_sha256 == graph.effective_graph_sha256, "Different randomized topology.")
                inputs.record(config["topology"]["path"], graph.topology_sha256)
            old = inputs.reference_prediction(row.reference_run_id)
            with np.load(status["artifacts"]["predictions"]["path"], allow_pickle=False) as p:
                require(np.array_equal(p["test_ids"], old["test_ids"]) and np.array_equal(p["test_y"], old["test_y"]),
                        "Control/reference test populations differ.")
                ap = average_precision_score(p["test_y"], p["test_probs"])
                f1 = f1_score(p["test_y"], p["test_probs"] >= float(p["threshold"]))
                require(abs(ap-row.pr_auc) < 1e-12 and abs(f1-row.f1_score) < 1e-12, "Control metrics differ from predictions.")
        for model, group in frame.groupby("model"):
            for metric in ["pr_auc", "f1_score"]:
                require(abs(group[metric].mean()-archived.loc[model, metric+"_mean"]) < 1e-12 and
                        abs(group[metric].std(ddof=1)-archived.loc[model, metric+"_sample_sd"]) < 1e-12, "Control summary mismatch.")
        frame["arm"] = family
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    ten_seeds(data, ["arm", "architecture"])
    return data, graph.to_dict()




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "thesis_figures")
    parser.add_argument("--preset", choices=["thesis", "slides"], default="thesis")
    args = parser.parse_args()
    from thesis_plot_layouts import configure, output_path, plot_subgroups, plot_controls, finish
    configure(args.preset)
    out = output_path(args)
    out.mkdir(parents=True, exist_ok=True)
    inputs, figures = Inputs(), []
    inputs.record(Path(__file__))
    inputs.record(ROOT / "make_thesis_figures.py")
    inputs.record(ROOT / "thesis_plot_layouts.py")
    inputs.record(SOURCE / "selected_per_seed_results.csv")
    print("Checking subgroup populations and recomputing per-seed AP...", flush=True)
    data, counts = subgroup_data(inputs)
    plot_subgroups(data, counts, out, figures)
    print("Checking the 90 completed Elliptic control runs and paired test populations...", flush=True)
    controls, graph = control_data(inputs, load_results())
    plot_controls(controls, out, figures)
    data.to_csv(out / "connected_isolated_per_seed.csv", index=False)
    counts.to_csv(out / "connected_isolated_group_sizes.csv", index=False)
    subgroup_summary = data.groupby(["construction", "group", "architecture"]).pr_auc.agg(["mean", "std", "count"])
    subgroup_summary.to_csv(out / "connected_isolated_summary.csv")
    controls[["arm", "architecture", "seed", "pr_auc", "f1_score", "run_id"]].to_csv(out / "elliptic_controls_per_seed.csv", index=False)
    summary = controls.groupby(["arm", "architecture"])[["pr_auc", "f1_score"]].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(out / "elliptic_controls_summary.csv")
    (out / "control_figures_provenance.json").write_text(json.dumps({"sources_sha256": inputs.files,
        "seeds": sorted(SEEDS), "sd_ddof": 1, "randomized_graph": graph, "figures": figures}, indent=2), encoding="utf-8")
    latex = [r"% Requires \usepackage{graphicx}. Paths are relative to the project root.", ""]
    readme = ["# Additional thesis figures", "", "Run `python make_thesis_control_figures.py` from the project root.", "",
              "This script checks all ten seeds, verifies prediction hashes and paired test IDs, recomputes subgroup AP and "
              "control AP/F1, and checks the saved summaries. No training is performed. "
              "PDFs are vector figures; PNGs are 300 dpi. Numeric data are in the accompanying CSVs. "
              "`control_figures_provenance.json` records input hashes and graph settings. "
              "`control_figures.tex` contains insertion-ready LaTeX and captions.", ""]
    for item in figures:
        readme.extend([f"## {item['name']}", "", item["caption"], ""])
        caption = item["caption"].replace("%", r"\%").replace("–", "--").replace("±", r"$\pm$")
        latex.extend([r"\begin{figure}[htbp]", r"\centering",
                      rf"\includegraphics[width=\linewidth]{{{(out / (item['name']+'.pdf')).relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else (out / (item['name']+'.pdf')).as_posix()}}}",
                      rf"\caption{{{caption}}}", rf"\label{{fig:{item['name'].replace('_', '-')}}}", r"\end{figure}", ""])
    (out / "CONTROL_FIGURES.md").write_text("\n".join(readme), encoding="utf-8")
    (out / "control_figures.tex").write_text("\n".join(latex), encoding="utf-8")
    finish(out, "controls")
    print(f"Done: additional figures and their source data in {out}")


if __name__ == "__main__":
    main()
