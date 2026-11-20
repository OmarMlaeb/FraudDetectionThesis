"""Publication/slide layouts only; experiment loaders and metric definitions stay unchanged."""
import hashlib
import json
import textwrap
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text
from sklearn.metrics import average_precision_score, precision_recall_curve

import make_thesis_figures as m

PRESET = "thesis"
BASE = 8
SMALL = 7
PAIR = ["#005F60", "#777777"]
STATS, TYPOGRAPHY = [], []
CONSTRUCTIONS = ["main", "card", "address", "email_domain", "identifier_time_window-24h"]
LABELS = ["Main", "Card", "Address", "Email domain", "Identifier, 24 h"]
OLD = {}
SD_NOTE = "Mean ± 1 sample SD (ddof=1); ten seeds, 42–51. Error bars are not confidence intervals."
GRAPH_NOTE = ("C, A, E, D = card signature, address signature, email domain, device (stand-in for the identifier fields). "
              "Identifier rule: pairs within 24 h, inclusive. Main: union of per-field consecutive links; "
              "the real main graph treats card1-card6 and addr1-addr2 as separate fields.")


def configure(preset):
    global PRESET, BASE, SMALL, OLD
    PRESET = preset
    BASE, SMALL = (8, 7) if preset == "thesis" else (16, 14)
    STATS.clear()
    TYPOGRAPHY.clear()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": BASE, "axes.labelsize": BASE,
                         "xtick.labelsize": BASE, "ytick.labelsize": BASE, "legend.fontsize": SMALL,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": None})
    # Preserve the old numeric outputs before either script regenerates any files.
    snapshot = m.ROOT / "thesis_figures/layout_before.json"
    if not snapshot.exists():
        names = ["plotted_summary.csv", "connected_isolated_summary.csv", "elliptic_controls_summary.csv"]
        content = {n: pd.read_csv(snapshot.parent/n).to_dict("records") for n in names}
        snapshot.write_text(json.dumps(content, indent=2), encoding="utf-8")
    OLD = json.loads(snapshot.read_text(encoding="utf-8"))


def output_path(args):
    out = args.output_dir.resolve()
    return out / "slides" if args.preset == "slides" else out


def stats(group, metric, figure, dataset, model, arm="main", source="reference"):
    values = group[metric].to_numpy(dtype=float)
    mean, sd = float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0
    # A selected single-run mode is intentionally not an aggregate comparison.
    if len(values) == 10:
        m.require(set(group.seed) == m.SEEDS, "Plotted seed set changed.")
        if source == "reference":
            match = [r for r in OLD["plotted_summary.csv"] if
                     (r["dataset"], r["architecture"], r["construction"]) == (dataset, model, arm)][0]
            old_mean, old_sd = match[metric+"_mean"], match[metric+"_std"]
        elif source == "subgroup":
            construction, subgroup = arm.split("/")
            match = [r for r in OLD["connected_isolated_summary.csv"] if
                     (r["construction"], r["group"], r["architecture"]) == (construction, subgroup, model)][0]
            old_mean, old_sd = match["mean"], match["std"]
        else:
            match = [r for r in OLD["elliptic_controls_summary.csv"] if
                     (r["arm"], r["architecture"]) == (arm, model)][0]
            old_mean, old_sd = match[metric+"_mean"], match[metric+"_std"]
        delta = max(abs(mean-old_mean), abs(sd-old_sd))
        m.require(delta < 1e-14, f"Aggregate changed: {figure}/{model}/{arm}/{metric}")
    else:
        delta = None
    STATS.append(dict(figure=figure, dataset=dataset, model=model, arm=arm, metric=metric,
                      n=len(values), mean=mean, sample_sd=sd, max_difference_from_old=delta))
    return mean, sd


def canvas(height=4, rows=1, cols=1, **kwargs):
    return plt.subplots(rows, cols, figsize=(6.3, height) if PRESET == "thesis" else (40/3, 7.5),
                        squeeze=False, **kwargs)


def batches(items, height, rows=1, cols=1):
    """A complete thesis layout, or one 16:9 file per subplot in slides mode."""
    if PRESET == "thesis":
        fig, axes = canvas(height, rows, cols)
        yield fig, list(zip(axes.flat, items)), ""
    else:
        for slug, label in items:
            fig, axes = canvas()
            axes[0, 0].set_title(label, fontsize=20, weight="bold", pad=16)
            yield fig, [(axes[0, 0], (slug, label))], slug


def footer(fig, text, y=.025, width=100):
    wrapped = "\n".join(textwrap.fill(line, width=width if PRESET == "thesis" else 120)
                        for line in text.splitlines())
    fig.text(.5, y, wrapped,
             ha="center", va="bottom", fontsize=SMALL, linespacing=1.35)


def save(fig, name, caption, out, figures, panel=""):
    destination = out / panel if panel else out
    destination.mkdir(parents=True, exist_ok=True)
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    for t in texts:
        m.require(t.get_fontsize() >= SMALL, f"Text too small: {name}: {t.get_text()}")
    fig.canvas.draw()
    for suffix in ("pdf", "png"):
        # No bbox_inches='tight': preserve the exact physical width and 16:9 canvas.
        fig.savefig(destination / f"{name}.{suffix}", dpi=300, facecolor="white", bbox_inches=None)
    width, height = fig.get_size_inches()
    TYPOGRAPHY.append(dict(figure=name, panel=panel, preset=PRESET, width_in=float(width), height_in=float(height),
                           base_pt=BASE, min_pt=min(t.get_fontsize() for t in texts),
                           max_pt=max(t.get_fontsize() for t in texts),
                           min_pt_at_6_3_in=min(t.get_fontsize() for t in texts)*6.3/width))
    plt.close(fig)
    figures.append(dict(name=f"{panel}/{name}" if panel else name, caption=caption))
    print(f"Saved {destination.name}/{name} ({width:.2f} x {height:.2f} in)", flush=True)


def pipeline(out, figures):
    fig, axes = canvas(8.4)
    ax = axes[0, 0]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    steps = [
        ("1. Input data", "IEEE-CIS: transactions and identities\nElliptic: features, classes and transaction edges"),
        ("2. Temporal splits", "IEEE-CIS: 80% / 10% / 10%\nElliptic time steps: 1–34 / 35–41 / 42–49"),
        ("3. Prepare inputs", "Train-fitted IEEE-CIS features; 165 Elliptic features\nTabular vectors / split-local windows (length 10) / graphs"),
        ("4. Train models", "MLP, LSTM, Transformer / GCN, GraphSAGE, GAT\nSeeds 42–51; weighted binary loss\nGNNs: full graph with training-label mask"),
        ("5. Validation-based selection", "Checkpoint / early stopping: maximize validation AP\nThreshold: maximize validation F1\nIEEE-CIS Transformer exception: checkpoint and threshold\nmaximize validation F1 with alert rate <= 20%"),
        ("6. Held-out test evaluation", "PR curves, AP, F1 and recall\nTen-seed means and sample SD; no test-based selection"),
        ("7. Statistical tests and controls", "Wilcoxon + Holm; connected/isolated, local-feature,\nsampled-complement, degree-preserving controls")]
    if PRESET == "thesis":
        fig.subplots_adjust(left=.04, right=.96, bottom=.02, top=.99)
        heights = [.105, .105, .105, .12, .16, .105, .105]
        top = .99
        centers = []
        for i, ((title, body), h) in enumerate(zip(steps, heights)):
            y = top-h
            ax.add_patch(FancyBboxPatch((.02, y), .96, h, boxstyle="round,pad=0.005",
                                       facecolor="#EAF2F8" if i < 3 else "#F2F4F3", edgecolor="#52616B", lw=.8))
            ax.text(.5, top-.015, title, ha="center", va="top", fontsize=BASE, weight="bold")
            ax.text(.5, y+.013, body, ha="center", va="bottom", fontsize=BASE, linespacing=1.25)
            if centers:
                ax.annotate("", xy=(.5, top+.004), xytext=(.5, centers[-1]-.004),
                            arrowprops=dict(arrowstyle="-|>", color="#52616B", lw=.8))
            centers.append(y)
            top = y-.025
    else:
        ax.set_title("Experimental pipeline", fontsize=22, weight="bold", pad=10)
        fig.subplots_adjust(left=.02, right=.98, bottom=.03, top=.91)
        for i, (title, body) in enumerate(steps):
            row, col = divmod(i, 4)
            x, y = .01+col*.25, .52-row*.49
            ax.add_patch(FancyBboxPatch((x, y), .225, .43, boxstyle="round,pad=.008",
                                       facecolor="#EAF2F8", edgecolor="#52616B"))
            ax.text(x+.1125, y+.395, textwrap.fill(title, 23), ha="center", va="top", fontsize=16, weight="bold")
            if i == 4:
                body = body.split("\nIEEE-CIS Transformer exception:")[0]
            ax.text(x+.1125, y+.28, textwrap.fill(body.replace("\n", "; ").replace(",; ", ", "), 25), ha="center", va="top", fontsize=14)
            if col < 3 and i < 6:
                ax.annotate("", xy=(x+.25, y+.22), xytext=(x+.23, y+.22), arrowprops=dict(arrowstyle="->"))
        ax.plot([.87, .87, .12], [.515, .49, .49], color="#52616B", lw=1)
        ax.annotate("", xy=(.12, .465), xytext=(.12, .49), arrowprops=dict(arrowstyle="->"))
        ax.add_patch(FancyBboxPatch((.76, .03), .225, .43, boxstyle="round,pad=.008", facecolor="#FFF3DF", edgecolor="#B58033"))
        ax.text(.8725, .425, "IEEE-CIS Transformer\nexception", ha="center", va="top", fontsize=16, weight="bold")
        ax.text(.8725, .31, "Checkpoint and threshold\nmaximize validation F1\nwith alert rate <= 20%", ha="center", va="top", fontsize=14)
    save(fig, "01_pipeline", "Experimental pipeline. Validation selects checkpoints and thresholds. "
         "The IEEE-CIS Transformer instead maximizes validation F1 subject to an alert rate of at most 20% "
         "for both selections. The final stage covers Wilcoxon/Holm tests and the four listed controls.", out, figures)


def graph_example(out, figures):
    # Same six illustrative nodes and same edge logic as the original plotting script.
    toy = pd.DataFrame({"Transaction": [f"T{i}" for i in range(1, 7)], "Hour": [0, 2, 4, 30, 32, 56],
                        "Card": ["C1", "C1", "C2", "C1", "C2", "C2"], "Address": ["A1", "A2", "A1", "A2", "A1", "A2"],
                        "Email": ["E1", "E2", "E1", "E3", "E2", "E3"], "Device": ["D1", "D1", "D1", "D1", "D2", "D2"]})
    def edges(column, window=False):
        result = set()
        for _, group in toy.groupby(column):
            ids = group.sort_values("Hour").index.tolist()
            if window:
                result.update((a, b) for i, a in enumerate(ids) for b in ids[i+1:] if toy.loc[b, "Hour"]-toy.loc[a, "Hour"] <= 24)
            else:
                result.update(zip(ids[:-1], ids[1:]))
        return result
    links = {"card": edges("Card"), "address": edges("Address"), "email": edges("Email"), "identifier": edges("Device", True)}
    links["main"] = set().union(*(edges(c) for c in ["Card", "Address", "Email", "Device"]))
    items = [("transactions", "Illustrative transactions"), ("card", "Card signature"), ("address", "Address signature"),
             ("email", "Email domain"), ("identifier", "Identifier, 24 h"), ("main", "Main")]
    angle = np.pi/2-np.arange(6)*np.pi/3
    pos = np.c_[np.cos(angle), np.sin(angle)]
    for fig, panels, panel in batches(items, 6.8, 3, 2):
        for ax, (slug, label) in panels:
            ax.axis("off")
            if slug == "transactions":
                table = ax.table(cellText=toy.values, colLabels=["Node", "Hour", "C", "A", "E", "D"], cellLoc="center", loc="center")
                table.auto_set_font_size(False)
                table.set_fontsize(SMALL if PRESET == "thesis" else BASE)
                table.scale(1, 1.25 if PRESET == "thesis" else 2)
                for (r, c), cell in table.get_celld().items():
                    cell.set_edgecolor("#CCD5DB")
                    if r == 0:
                        cell.set_facecolor("#EAF2F8")
            else:
                for a, b in sorted(links[slug]):
                    ax.plot(pos[[a, b], 0], pos[[a, b], 1], color="#68869B", lw=1, zorder=1)
                ax.scatter(pos[:, 0], pos[:, 1], s=240 if PRESET == "thesis" else 1000,
                           facecolor="#EAF2F8", edgecolor="#0072B2", zorder=2)
                for i, (x, y) in enumerate(pos):
                    ax.text(x, y, f"T{i+1}", ha="center", va="center", fontsize=BASE)
                ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal")
            if PRESET == "thesis":
                ax.text(.5, -.07, label, transform=ax.transAxes, ha="center", fontsize=BASE)
        fig.subplots_adjust(left=.06, right=.96, top=.97 if PRESET == "thesis" else .88,
                            bottom=.18 if PRESET == "thesis" else .18, hspace=.37, wspace=.25)
        footer(fig, GRAPH_NOTE, width=105)
        save(fig, "02_graph_constructions", GRAPH_NOTE + " Synthetic illustration; bidirectional edges are shown once and model self-loops are omitted.", out, figures, panel)
    toy.to_csv(out/"illustrative_transactions.csv", index=False)
    pd.DataFrame([dict(construction=dict(items)[key], source=f"T{a+1}", target=f"T{b+1}")
                  for key, value in links.items() for a, b in sorted(value)]).to_csv(out/"illustrative_edges.csv", index=False)


def results_plot(df, dataset, out, figures):
    data = m.baseline(df, dataset)
    name = f"03_results_{dataset.lower()}"
    items = [("pr_auc", "PR-AUC"), ("f1_score", "F1"), ("recall", "Recall")]
    for fig, panels, panel in batches(items, 3.5, 1, 3):
        for ax, (metric, label) in panels:
            for i, (model, color) in enumerate(zip(m.MODELS, m.COLORS)):
                group = data[data.architecture == model].sort_values("seed")
                mean, sd = stats(group, metric, name, dataset, model, group.construction.iloc[0])
                ax.scatter(i+np.linspace(-.12, .12, 10), group[metric], s=7 if PRESET == "thesis" else 25, color=color, alpha=.4)
                ax.errorbar(i, mean, yerr=sd, fmt="o", color=color, capsize=3, markersize=4 if PRESET == "thesis" else 7)
            labels = ["Transformer†" if model == "Transformer" and dataset == "IEEE-CIS" else model for model in m.MODELS]
            ax.set_xticks(range(6), labels, rotation=55 if PRESET == "thesis" else 25, ha="right")
            ax.tick_params(axis="x", labelsize=SMALL)
            ax.set(ylim=(0, .7), ylabel=label)
            ax.grid(axis="y", alpha=.2)
        fig.subplots_adjust(left=.08, right=.99, top=.96 if PRESET == "thesis" else .87, bottom=.35 if PRESET == "thesis" else .22, wspace=.5)
        note = SD_NOTE + " Small dots: individual seeds."
        if dataset == "IEEE-CIS":
            note += "\n† Different checkpoint/threshold procedure (Sec. 3.8)"
        footer(fig, note, width=108)
        save(fig, name, f"{dataset}: ten-seed means ± sample SD; GNNs use the {'main' if dataset == 'IEEE-CIS' else 'native'} graph. "
             "AP is average precision. F1 and recall use validation-selected thresholds. " +
             ("† Different checkpoint/threshold procedure (Sec. 3.8)." if dataset == "IEEE-CIS" else ""), out, figures, panel)


def pr_plot(df, dataset, seed, out, figures, provenance):
    name = f"04_pr_curves_{dataset.lower()}"
    data = m.baseline(df, dataset)
    fig, axes = canvas(4.8)
    ax = axes[0, 0]
    grid = np.linspace(0, 1, 1001)
    population, grid_rows = None, []
    for model, color in zip(m.MODELS, m.COLORS):
        group = data[data.architecture == model].sort_values("seed")
        if seed is not None:
            group = group[group.seed == seed]
        curves = []
        for row in group.itertuples():
            check = json.loads((m.SOURCE/"checks"/f"{row.run_id}.json").read_text())
            path = m.SOURCE/"predictions"/f"{row.run_id}.npz"
            digest = m.sha(path)
            m.require(digest == check["prediction_sha256"] and check["accepted"], "Prediction verification failed.")
            m.require(check["checkpoint_sha256"] == row.checkpoint_sha256, "Checkpoint identity changed.")
            with np.load(path, allow_pickle=False) as z:
                y, prob, ids = z["test_y"], z["test_probs"], z["test_ids"]
            if population is None:
                population = (ids, y)
            m.require(np.array_equal(ids, population[0]) and np.array_equal(y, population[1]), "PR populations differ.")
            ap = average_precision_score(y, prob)
            m.require(abs(ap-row.pr_auc) <= 2e-5, "Recomputed AP differs from saved run.")
            precision, recall, _ = precision_recall_curve(y, prob)
            # Remove sklearn's artificial endpoint before reversing/interpolating.
            recall, precision = recall[:-1][::-1], precision[:-1][::-1]
            # At repeated recall, keep the first point in ascending-recall order.
            # No across-recall precision envelope or smoothing is applied.
            unique, index = np.unique(recall, return_index=True)
            curve = np.interp(grid, unique, precision[index])
            curves.append(curve)
            provenance.append(dict(dataset=dataset, model=model, seed=int(row.seed), run_id=row.run_id,
                checkpoint_path=row.checkpoint_path, checkpoint_sha256=row.checkpoint_sha256,
                prediction_file=str(path.relative_to(m.ROOT)), sha256=digest, saved_run_ap=row.pr_auc,
                recomputed_ap=ap, absolute_ap_difference=abs(ap-row.pr_auc), test_count=len(y)))
        mean_ap, sd_ap = stats(group, "pr_auc", name, dataset, model, group.construction.iloc[0])
        curve_mean = np.mean(curves, axis=0)
        label = f"{model}: {mean_ap:.4f} ± {sd_ap:.4f}" if seed is None else f"{model}: {mean_ap:.4f}"
        ax.plot(grid, curve_mean, color=color, lw=1.2 if PRESET == "thesis" else 2, label=label)
        curve_sd = np.std(curves, axis=0, ddof=1) if seed is None else np.zeros_like(grid)
        if seed is None:
            ax.fill_between(grid, np.maximum(0, curve_mean-curve_sd), np.minimum(1, curve_mean+curve_sd), color=color, alpha=.12, linewidth=0)
        grid_rows.extend(dict(model=model, recall=float(r), precision_mean=float(p), precision_sample_sd=float(s))
                         for r, p, s in zip(grid, curve_mean, curve_sd))
    rate = float(population[1].mean())
    ax.axhline(rate, color="#555555", ls="--", lw=.9, label=f"Prevalence: {rate:.4f}")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Recall", ylabel="Precision")
    ax.grid(alpha=.15)
    ax.legend(loc="upper right", fontsize=SMALL, title="AP: mean ± sample SD" if seed is None else f"AP: seed {seed}", title_fontsize=SMALL)
    graph_text = "GNNs on the main graph" if dataset == "IEEE-CIS" else "GNNs on the native graph"
    if PRESET == "slides":
        ax.set_title(f"{dataset}: precision–recall\n{graph_text}", fontsize=20, weight="bold")
    else:
        fig.text(.5, .97, graph_text, ha="center", va="top", fontsize=BASE)
    footer(fig, "Mean precision ± 1 sample SD on a 1001-point recall grid; ten seeds (42–51).\nLegend AP summarizes the ten saved per-run AP scores, not the area under the mean curve."
           if seed is None else f"Seed {seed}: validation-selected checkpoints; legend AP uses the saved single-run scores.", width=108)
    fig.subplots_adjust(left=.10, right=.98, top=.91 if PRESET == "thesis" else .82, bottom=.19)
    pd.DataFrame(grid_rows).to_csv(out/f"{name}_grid.csv", index=False)
    save(fig, name, f"{dataset}; {graph_text}. " + ("Mean precision curve and ±1 sample-SD band across seeds 42–51, "
         "interpolated on 1001 common recall points. The synthetic (recall=0, precision=1) endpoint is removed first; "
         "at duplicate recall levels the first reversed point is retained, with constant endpoint extrapolation. "
         "Bands are clipped to [0,1]. Legend AP is the mean ± sample SD of the saved per-run AP scores, matching the "
         "ten-seed result tables; it is not the integrated mean curve. " if seed is None else f"Single-run mode, seed {seed}. ") +
         "Each curve uses its validation-selected checkpoint; checkpoint paths and hashes appear in pr_curve_sources.csv.", out, figures)


def construction_results(df, out, figures):
    tests, evidence = m.verify_construction_tests()
    tests.to_csv(out/"construction_holm_checks.csv", index=False)
    name = "05_graph_construction_results"
    for fig, panels, panel in batches([("pr_auc", "PR-AUC"), ("f1_score", "F1")], 3.8, 1, 2):
        for ax, (metric, label) in panels:
            for j, model in enumerate(m.MODELS[3:]):
                for i, construction in enumerate(CONSTRUCTIONS):
                    group = df[(df.dataset == "IEEE-CIS") & (df.architecture == model) & (df.construction == construction)]
                    mean, sd = stats(group, metric, name, "IEEE-CIS", model, construction)
                    ax.errorbar(i+(j-1)*.18, mean, yerr=sd, fmt="o", color=m.COLORS[j+3], capsize=3,
                                markersize=4 if PRESET == "thesis" else 7)
            mlp = df[(df.dataset == "IEEE-CIS") & (df.architecture == "MLP")]
            mean, sd = stats(mlp, metric, name, "IEEE-CIS", "MLP", "none")
            ax.axhline(mean, ls="--", color=m.COLORS[0], lw=1.2)
            ax.set_xticks(range(5), LABELS, rotation=35, ha="right")
            ax.tick_params(axis="x", labelsize=SMALL)
            ax.set(ylim=(0, .7), ylabel=label)
            ax.grid(axis="y", alpha=.2)
        handles = [Line2D([], [], marker="o", ls="", color=m.COLORS[i+3], label=model) for i, model in enumerate(m.MODELS[3:])]
        handles.append(Line2D([], [], ls="--", color=m.COLORS[0], label="MLP mean"))
        fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=SMALL,
                   bbox_to_anchor=(.5, .99 if PRESET == "thesis" else .95))
        footer(fig, SD_NOTE + "\nDescriptive differences: no construction comparison survives Holm correction (30 tests/metric; α=0.05).", width=105)
        fig.subplots_adjust(left=.09, right=.98, top=.87 if PRESET == "thesis" else .78, bottom=.36 if PRESET == "thesis" else .25, wspace=.3)
        save(fig, name, "IEEE-CIS construction comparison: ten-seed means ± sample SD. Dashed blue lines are the MLP "
             "means from the same saved result set. No construction comparison survives the prespecified Holm correction "
             "over 30 hypotheses per metric; descriptive improvements do not establish significance or equivalence.", out, figures, panel)
    return evidence


def plot_subgroups(data, counts, out, figures):
    name = "06_connected_vs_isolated"
    models = ["MLP", "GCN", "GraphSAGE", "GAT"]
    items = list(zip(CONSTRUCTIONS, LABELS))
    for fig, panels, panel in batches(items, 8.3, 3, 2):
        for ax, (construction, label) in panels:
            for arm, group_name in enumerate(["connected", "isolated"]):
                for i, model in enumerate(models):
                    group = data[(data.construction == construction) & (data.group == group_name) & (data.architecture == model)]
                    mean, sd = stats(group, "pr_auc", name, "IEEE-CIS", model, construction+"/"+group_name, "subgroup")
                    ax.errorbar(i+(arm-.5)*.24, mean, yerr=sd, fmt="o" if arm == 0 else "s", color=PAIR[arm],
                                markerfacecolor=PAIR[arm] if arm == 0 else "white", capsize=3, markersize=4 if PRESET == "thesis" else 7)
                c = counts[(counts.construction == construction) & (counts.group == group_name)].iloc[0]
                ax.text(.02, .97-arm*.1, f"{group_name.capitalize()}: n={int(c.n):,}, fraud={c.fraud_rate:.2%}",
                        transform=ax.transAxes, color=PAIR[arm], fontsize=SMALL, va="top")
            ax.set_xticks(range(4), models, rotation=25, ha="right")
            ax.tick_params(axis="x", labelsize=SMALL)
            ax.set(ylim=(0, 1), ylabel="PR-AUC", xlabel=label if PRESET == "thesis" else "")
            ax.grid(axis="y", alpha=.2)
        if PRESET == "thesis":
            ax = fig.axes[-1]
            ax.axis("off")
            ax.text(.02, .9, "Connected: degree > 0\nIsolated: degree = 0\nSelf-loops excluded.\n\nSame group IDs across models\nwithin each construction.\nMLP predictions reused per seed.",
                    transform=ax.transAxes, va="top", fontsize=BASE, linespacing=1.4)
        handles = [Line2D([], [], ls="", marker="o", color=PAIR[0], label="Connected"),
                   Line2D([], [], ls="", marker="s", color=PAIR[1], markerfacecolor="white", label="Isolated")]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, .995), fontsize=BASE)
        note = (SD_NOTE + "\nAddress and email connected groups contain only 37 and 65 fraud cases; "
                "PR-AUC is not comparable across groups because prevalence differs.")
        footer(fig, note, y=.02, width=105)
        fig.subplots_adjust(left=.09, right=.98, top=.94 if PRESET == "thesis" else .82,
                            bottom=.15 if PRESET == "thesis" else .24, hspace=.65, wspace=.3)
        save(fig, name, "IEEE-CIS connected/isolated AP on identical test IDs across models within each construction. "
             "Ten-seed means ± sample SD; group sizes and fraud rates are shown. Degree excludes self-loops. "
             "Address and email connected groups contain only 37 and 65 fraud cases; PR-AUC is not comparable across groups "
             "because prevalence differs. MLP predictions are reused across constructions.", out, figures, panel)


def plot_controls(data, out, figures):
    name = "07_elliptic_feature_topology_controls"
    for fig, panels, panel in batches([("local", "(a) Full versus local features"), ("random", "(b) Original versus degree-preserving rewired")], 6.6, 2, 1):
        for ax, (changed, title) in panels:
            if PRESET == "slides":
                ax.set_title("")
                fig.suptitle(title, fontsize=20, weight="bold", y=.97)
            models = m.MODELS if changed == "local" else m.MODELS[3:]
            labels = ["Full (165)", "Local only (93)"] if changed == "local" else ["Original", "Degree-preserving rewired"]
            for mi, metric in enumerate(["pr_auc", "f1_score"]):
                for ai, arm in enumerate(["original", changed]):
                    for i, model in enumerate(models):
                        group = data[(data.arm == arm) & (data.architecture == model)]
                        mean, sd = stats(group, metric, name, "Elliptic", model, arm, "control")
                        ax.errorbar(i+[-.27, -.09, .09, .27][mi*2+ai], mean, yerr=sd,
                                    fmt="o" if mi == 0 else "s", color=PAIR[ai], capsize=3,
                                    markerfacecolor=PAIR[ai] if mi == 0 else "white", markersize=4 if PRESET == "thesis" else 7)
            handles = [Line2D([], [], ls="", marker="o" if mi == 0 else "s", color=PAIR[ai],
                       markerfacecolor=PAIR[ai] if mi == 0 else "white", label=f"{labels[ai]} · {metric}")
                       for mi, metric in enumerate(["PR-AUC", "F1"]) for ai in range(2)]
            ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=SMALL, borderaxespad=0)
            ax.set_xticks(range(len(models)), models, rotation=20, ha="right")
            ax.set(ylim=(0, .7), ylabel="Test score", xlabel=title if PRESET == "thesis" else "")
            ax.grid(axis="y", alpha=.2)
        footer(fig, SD_NOTE + "\nFeature control: native topology. Topology control: full features; one graph rewired within time steps.", width=105)
        fig.subplots_adjust(left=.1, right=.98, top=.90 if PRESET == "thesis" else .73, bottom=.18 if PRESET == "thesis" else .23, hspace=.8)
        save(fig, name, "Elliptic controls: full (165) versus local-only (93) features for all six models; "
             "original versus degree-preserving rewired topology for the three GNNs. Circles show AP, open squares F1. "
             "Mean ± sample SD over seeds 42–51. Rewired edges remain within time steps; full feature values are retained. "
             "One graph seed, 20261013, is reused, so error bars measure training-seed variation, not graph-realization variation. "
             "Original historical runs and the validated sparse GNN control backend can differ in floating-point reduction order.", out, figures, panel)


def finish(out, group):
    """Print and retain every plotted statistic, plus precise physical typography."""
    summary = pd.DataFrame(STATS)
    summary.to_csv(out/f"{group}_plotted_statistics.csv", index=False)
    pd.DataFrame(TYPOGRAPHY).to_csv(out/f"{group}_text_sizes.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.10f}"))
    print("\nTYPOGRAPHY\n" + pd.DataFrame(TYPOGRAPHY).to_string(index=False))
    print("All ten-seed means and sample SDs match the pre-edit outputs to < 1e-14.")


def verify_thesis(thesis_pdf, out):
    """Read-only table/PDF audit; optional PyMuPDF dependency, no experiment writes."""
    import re
    import sys
    sys.path.insert(0, str(m.ROOT / "tmp/pdfs/deps"))
    import pymupdf
    from PIL import Image

    document = pymupdf.open(thesis_pdf)
    pages = [page.get_text() for page in document]
    reference = m.load_results()
    subgroups = pd.read_csv(m.SOURCE / "subgroup_per_seed.csv")
    local = pd.read_csv(m.ROOT / "results/committee_fast_experiments/analysis/local_per_seed.csv")
    model_names = {"mlp": "MLP", "lstm": "LSTM", "transformer": "Transformer", "gcn": "GCN",
                   "sage": "GraphSAGE", "graphsage": "GraphSAGE", "gat": "GAT"}
    local["architecture"] = local.model.map(model_names)
    pair = r"(\d+\.\d+)\s*±\s*(\d+\.\d+)"
    number = r"(\d+\.\d+)"
    checks = []

    def table(number):
        marker = f"Table {number}:"
        hits = [(i+1, text.split(marker, 1)[1].split("Table ", 1)[0])
                for i, text in enumerate(pages) if marker in text]
        m.require(len(hits) == 1, f"Cannot uniquely locate {marker}")
        return hits[0]

    def row_check(table_id, text, key, frame, metrics, with_sd=True):
        pattern = r"(?:^|\n)" + re.escape(key) + r"\s*" + r"\s*".join(
            [pair if with_sd else number] * len(metrics))
        match = re.search(pattern, text)
        m.require(match is not None, f"Missing table row: {table_id}/{key}")
        m.require(len(frame) == 10 and set(frame.seed) == m.SEEDS, "Unexpected verification seeds.")
        values = iter(match.groups())
        for metric in metrics:
            mean, sd = frame[metric].mean(), frame[metric].std(ddof=1)
            for statistic, actual in ([('mean', mean), ('sample_sd', sd)] if with_sd else [('mean', mean)]):
                printed = next(values)
                digits = len(printed.split('.')[1])
                agrees = f"{actual:.{digits}f}" == printed
                checks.append(dict(table=table_id, row=key.replace('\n', '/'), metric=metric,
                                   statistic=statistic, thesis_value=printed, calculated_value=actual, matches=agrees))
                m.require(agrees, f"Table mismatch: {table_id}/{key}/{metric}/{statistic}: {actual} != {printed}")

    _, text = table('4.5')
    for model in m.MODELS[3:]:
        for construction, label in zip(CONSTRUCTIONS, LABELS):
            frame = reference[(reference.dataset == 'IEEE-CIS') & (reference.architecture == model)
                              & (reference.construction == construction)]
            row_check('4.5', text, model+'\n'+label, frame, ['pr_auc', 'f1_score', 'recall'])
    for table_id, dataset in [('4.7', 'IEEE-CIS'), ('4.8', 'Elliptic')]:
        _, text = table(table_id)
        frame = m.baseline(reference, dataset)
        for model in m.MODELS:
            row_check(table_id, text, model, frame[frame.architecture == model],
                      ['recall', 'precision', 'f1_score', 'pr_auc', 'roc_auc'], with_sd=False)
    _, text = table('4.12')
    for model in ['MLP', 'GCN', 'GraphSAGE', 'GAT']:
        for group in ['connected', 'isolated']:
            frame = subgroups[(subgroups.dataset == 'IEEE-CIS') & (subgroups.construction == 'main')
                              & (subgroups.architecture == model) & (subgroups.group == group)]
            row_check('4.12', text, group+'\n'+model, frame, ['pr_auc', 'f1_score', 'recall', 'false_positive'])
    _, text = table('4.22')
    for model in m.MODELS:
        row_check('4.22', text, model, local[local.architecture == model],
                  ['pr_auc', 'f1_score', 'recall', 'false_positive'])
    audit = pd.DataFrame(checks)
    audit.to_csv(out / 'thesis_table_verification.csv', index=False)

    plotted = pd.concat([pd.read_csv(out / f'{name}_plotted_statistics.csv') for name in ['base', 'controls']], ignore_index=True)
    m.require(plotted.n.eq(10).all(), 'Default publication plots must use ten seeds.')
    m.require(plotted.max_difference_from_old.lt(1e-14).all(), 'A plotted aggregate changed.')
    plotted.to_csv(out / 'all_plotted_statistics.csv', index=False)
    sizes = pd.concat([pd.read_csv(out / f'{name}_text_sizes.csv') for name in ['base', 'controls']], ignore_index=True)

    pdf_checks = []
    for path in sorted(out.rglob('*.pdf')):
        if not re.match(r'0[1-7]_', path.name):
            continue
        pdf = pymupdf.open(path)
        m.require(len(pdf) == 1, f'Unexpected page count: {path}')
        page = pdf[0]
        slides = 'slides' in path.relative_to(out).parts
        width, height = page.rect.width/72, page.rect.height/72
        spans = [span for block in page.get_text('dict')['blocks'] if 'lines' in block
                 for line in block['lines'] for span in line['spans'] if span['text'].strip()]
        minimum = min(span['size'] for span in spans)
        m.require(all(page.rect.contains(pymupdf.Rect(span['bbox'])) for span in spans), f'Text outside PDF canvas: {path}')
        m.require(minimum >= (14 if slides else 7)-1e-4, f'PDF text too small: {path}')
        m.require(abs(width/height-16/9) < 1e-6 if slides else abs(width-6.3) < 1e-6, f'Wrong PDF dimensions: {path}')
        m.require(not page.get_images(), f'Raster image in vector figure: {path}')
        m.require(all(font[2] == 'Type0' for font in page.get_fonts()), f'Font is not embedded TrueType: {path}')
        with Image.open(path.with_suffix('.png')) as png:
            dpi = png.info.get('dpi', (0, 0))
            m.require(all(abs(d-300) < .1 for d in dpi), f'Incorrect PNG resolution: {path}')
        pdf_checks.append(dict(file=path.relative_to(out).as_posix(), width_in=width, height_in=height,
                               minimum_pt=minimum, vector=True, png_dpi=dpi[0]))
    pd.DataFrame(pdf_checks).to_csv(out / 'export_verification.csv', index=False)
    lines = ['# Figure verification', '', f'Source thesis: `{thesis_pdf}`',
             f'SHA-256: `{m.sha(Path(thesis_pdf))}`', '',
             f'All {len(audit)} checked numeric cells match the supplied thesis at its printed precision. '
             'Tables 4.7 and 4.8 contain means only; their SDs are verified against the unchanged pre-edit outputs.', '',
             '| Table | Numeric cells checked | Result |', '|---|---:|---|']
    for table_id, frame in audit.groupby('table', sort=False):
        lines.append(f'| {table_id} | {len(frame)} | All match |')
    lines += ['', f'All {len(plotted)} plotted mean/SD pairs agree with `layout_before.json` to less than 1e-14. '
              'The requested PR-curve rendering changes; the saved per-run AP and their aggregate statistics do not.', '',
              'Each default curve uses ten accepted validation-selected checkpoints (seeds 42–51). '
              'See `pr_curve_sources.csv` for their paths, hashes and prediction checks. '
              'Figure 04 uses 1001 common recall points, sample-SD bands, and excludes the synthetic endpoint. '
              'The first reversed point is retained at duplicate recall; endpoint extrapolation is constant. '
              'Legend AP is not the integral of the mean curve.', '',
              'Construction comparisons: 0/180 survive Holm correction (30 hypotheses per metric); '
              'the smallest adjusted p-value is 0.05859375.', '',
              '| Figure | Width (in) | Base text (pt) | Smallest text at 6.3 in (pt) |', '|---|---:|---:|---:|']
    for row in sizes.itertuples():
        lines.append(f'| {row.figure} | {row.width_in:.1f} | {row.base_pt:g} | {row.min_pt_at_6_3_in:g} |')
    lines += ['', f'Export checks passed for {len(pdf_checks)} PDFs and matching 300-dpi PNGs. '
              'Thesis PDFs are exactly 6.3 in wide; slide PDFs are 16:9 with one panel per file. '
              'PDFs contain embedded TrueType fonts and no raster images.', '',
              '## Plotted statistics', '', '| Figure | Dataset | Model | Group / arm | Metric | n | Mean | Sample SD |',
              '|---|---|---|---|---|---:|---:|---:|']
    for row in plotted.itertuples():
        lines.append(f'| {row.figure} | {row.dataset} | {row.model} | {row.arm} | {row.metric} | {row.n} | {row.mean:.10f} | {row.sample_sd:.10f} |')
    (out / 'VERIFICATION.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(audit.groupby('table', sort=False).agg(cells=('matches', 'size'), all_match=('matches', 'all')).to_string())
    print(f'All {len(plotted)} plotted mean/SD pairs unchanged; {len(pdf_checks)} PDF/PNG pairs verified.')
    print(sizes.to_string(index=False))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Verify plotted statistics against the supplied thesis and inspect exports.')
    parser.add_argument('--verify-thesis', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=m.ROOT / 'thesis_figures')
    args = parser.parse_args()
    verify_thesis(args.verify_thesis, args.output_dir)
