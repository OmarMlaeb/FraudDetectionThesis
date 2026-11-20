"""Generate thesis figures from the existing verified experiments; no training.

Run: python make_thesis_figures.py
Requires numpy, pandas, matplotlib and scikit-learn (already in requirements.txt).
Outputs vector PDF, 300-dpi PNG, captions, LaTeX and source provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "analysis/supervisor_2026-10-13"
MODELS = ["MLP", "LSTM", "Transformer", "GCN", "GraphSAGE", "GAT"]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#332288"]
METRICS = {"pr_auc": "Average precision (PR-AUC)", "f1_score": "F1-score", "recall": "Recall"}
SEEDS = set(range(42, 52))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_results():
    path = SOURCE / "selected_per_seed_results.csv"
    df = pd.read_csv(path, keep_default_na=False)
    require(not df.run_id.duplicated().any(), "Duplicate result run IDs.")
    require(len(df) == 240, "Expected the 240 verified reference runs.")
    for key, group in df.groupby(["dataset", "architecture", "construction"]):
        require(len(group) == 10 and set(group.seed) == SEEDS, f"Incomplete seed group: {key}")
    require(np.isfinite(df[list(METRICS)].to_numpy()).all(), "Non-finite metrics.")
    require(df[list(METRICS)].apply(lambda s: s.between(0, 1).all()).all(), "Invalid metric range.")
    # The audit selects a consistent experiment set; do not combine old/new CSV rows.
    for row in df.itertuples():
        check = json.loads((SOURCE / "checks" / f"{row.run_id}.json").read_text())
        require(check.get("accepted") is True, f"Unverified run: {row.run_id}")
        for metric in METRICS:
            tolerance = 2e-5 if metric == "pr_auc" else 1e-10
            require(abs(getattr(row, metric) - check["whole_test_metrics"][metric]) <= tolerance,
                    f"Result differs from accepted reproduction: {row.run_id}, {metric}")
    return df


def save(fig, name, caption, out, figures):
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"{name}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    figures.append({"name": name, "caption": caption})
    print(f"Saved {name}.pdf and .png", flush=True)






def baseline(df, dataset):
    return df[(df.dataset == dataset) & df.construction.isin(["none", "main", "native"])]






def verify_construction_tests():
    path = SOURCE / "wilcoxon_holm_inference_verified.csv"
    tests = pd.read_csv(path)
    tests = tests[tests.family_id.str.startswith("IEEE-CIS__construction__")].copy()
    require(len(tests) == 180 and tests.metric.nunique() == 6, "Incomplete construction tests.")
    require(tests.planned_family_size.eq(30).all() and tests.n_pairs.eq(10).all() and
            tests.status.eq("complete").all(), "Construction test family changed.")
    require((tests.p_holm > .05).all() and tests.significant_holm_0_05.eq(False).all(),
            "Construction significance statement is no longer supported.")
    return tests, {"source": str(path.relative_to(ROOT)), "sha256": sha(path),
                   "comparisons": len(tests), "hypotheses_per_metric": 30,
                   "significant_after_holm": 0, "minimum_p_holm": float(tests.p_holm.min())}




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "thesis_figures")
    parser.add_argument("--seed", "--pr-seed", dest="pr_seed", type=int, choices=sorted(SEEDS), default=None, help="Optional single-run PR mode; default is ten-seed mean and SD")
    parser.add_argument("--preset", choices=["thesis", "slides"], default="thesis")
    args = parser.parse_args()
    from thesis_plot_layouts import (configure, output_path, pipeline, graph_example, results_plot, pr_plot, construction_results, finish)
    configure(args.preset)
    out = output_path(args)
    out.mkdir(parents=True, exist_ok=True)
    df = load_results()
    figures, provenance = [], []
    pipeline(out, figures)
    graph_example(out, figures)
    for dataset in ["IEEE-CIS", "Elliptic"]:
        results_plot(df, dataset, out, figures)
        pr_plot(df, dataset, args.pr_seed, out, figures, provenance)
    construction_evidence = construction_results(df, out, figures)
    pd.DataFrame(provenance).to_csv(out / "pr_curve_sources.csv", index=False)
    summary = df.groupby(["dataset", "architecture", "construction"])[list(METRICS)].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(out / "plotted_summary.csv")
    df.to_csv(out / "plotted_per_seed_results.csv", index=False)
    manifest = {"source_csv": str((SOURCE / "selected_per_seed_results.csv").relative_to(ROOT)),
                "source_sha256": sha(SOURCE / "selected_per_seed_results.csv"),
                "script_sha256": sha(Path(__file__)), "layout_script_sha256": sha(ROOT / "thesis_plot_layouts.py"), "preset": args.preset, "seeds": sorted(SEEDS), "pr_seed": args.pr_seed,
                "error_bars": "sample SD, ddof=1; not confidence intervals", "predictions": provenance,
                "construction_statistics": construction_evidence, "figures": figures}
    (out / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme = ["# Thesis figures", "", "Generated by `make_thesis_figures.py`; no models are trained or original results changed.", "",
              "Use PDF for LaTeX/vector quality, or PNG (300 dpi) for Word. `figures.tex` has ready-to-use figures and captions.", "",
              "## Reproduce", "", "From the project root:", "", "```powershell", "python make_thesis_figures.py", "```", "",
              "This project's existing portable Python also works:", "", "```powershell",
              r".\analysis\supervisor_2026-10-13\runtime\python310\python.exe .\make_thesis_figures.py", "```", "",
              "Dependencies: `numpy pandas matplotlib scikit-learn` (listed in the project requirements).", "",
              "## Data and interpretation", "",
              "The figures use the 240 verified reference runs in `analysis/supervisor_2026-10-13/selected_per_seed_results.csv`. "
              "They do not mix these with later committee experiments, partial runs or older duplicates. "
              "Main-model comparisons use the IEEE-CIS main graph and Elliptic native graph. "
              "All groups have ten seeds, 42–51. Error bars are sample SD, not standard error or confidence intervals. "
              "The project calls average precision PR-AUC; these figures follow that definition, not trapezoidal area. "
              "PR curves default to ten-seed means and sample-SD bands on 1001 recall points; AP legends summarize saved per-run AP. "
              "Use `--seed 51` for single-run mode. Use `--preset slides` for one 16:9 panel per file in the slides subfolder. "
              "The thesis preset preserves exactly 6.3-inch width with 8-pt base and 7-pt minimum text. "
              "Exact checkpoints, hashes, saved/recomputed AP and differences are recorded in `pr_curve_sources.csv`. "
              "Predictions are checked against audit SHA-256 hashes, labels/IDs and recorded average precision. "
              "The graph construction figure is explicitly synthetic. No model performance is simulated.", "",
              "The pipeline describes the historical experiments, including their validation selection rules. "
              "The August IEEE-CIS Transformer used validation F1 with a 20% alert-rate constraint; "
              "see `analysis/supervisor_2026-10-13/SEQUENCE_INPUTS.md` for details.", "",
              "`plotted_summary.csv` contains the numeric means/SDs/counts; `plotted_per_seed_results.csv` contains "
              "the selected source rows. `provenance.json` records input hashes and PR sources. "
              "The two illustrative CSVs document the toy graph. `construction_holm_checks.csv` includes all 180 "
              "construction comparisons (30 per metric); none survives Holm correction at alpha=0.05. "
              "Visual differences are descriptive and non-significance does not establish equivalence.", "", "## Suggested captions", ""]
    latex = [r"% Add \usepackage{graphicx} to your preamble.", r"% Paths assume thesis_figures/ is beside your main .tex file.", ""]
    for item in figures:
        readme.extend([f"### {item['name']}", "", item["caption"], ""])
        caption = item["caption"].replace("±", r"$\pm$").replace("–", "--").replace("%", r"\%").replace("_", r"\_")
        latex.extend([r"\begin{figure}[htbp]", r"\centering",
                      rf"\includegraphics[width=\linewidth]{{{(out / (item['name']+'.pdf')).relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else (out / (item['name']+'.pdf')).as_posix()}}}",
                      rf"\caption{{{caption}}}", rf"\label{{fig:{item['name'].replace('_', '-')}}}", r"\end{figure}", ""])
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    (out / "figures.tex").write_text("\n".join(latex), encoding="utf-8")
    finish(out, "base")
    print(f"Done: {len(figures)} figures in {out}")


if __name__ == "__main__":
    main()
