import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis_common import *
import collections
import itertools
import shutil
import subprocess
import zipfile

def fmt(x,digits=4):return 'undefined' if x is None else f'{x:.{digits}f}'

def main():
    selected=read('selected_runs.json')
    checks={p.stem:json.loads(p.read_text()) for p in (OUT/'checks').glob('*.json')}
    missing=[r['run_id'] for r in selected if r['run_id'] not in checks]
    failed=[r['run_id'] for r in selected if r['run_id'] in checks and not checks[r['run_id']]['accepted']]
    assert not missing and not failed,(missing,failed)
    assert len(checks)==240
    inventory=read('artifact_hashes_before.json');hashes={x['path']:x['sha256'] for x in inventory}
    differences=[]
    for item in inventory:
        now=sha(ROOT/item['path']) if (ROOT/item['path']).exists() else None
        if now!=item['sha256']:differences.append({'path':item['path'],'before':item['sha256'],'after':now})
    dump('original_artifact_preservation.json',{'checked_artifacts':len(inventory),'changed':differences,'all_unchanged':not differences})
    assert not differences
    extra_paths=[p for p in (ROOT/'results').glob('*') if p.is_file() and p.suffix in ['.pt','.csv'] and str(p.relative_to(ROOT)).replace('\\','/') not in hashes]
    extras=[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'absolute_path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in extra_paths]
    dump('additional_referenced_artifact_hashes.json',extras)
    flattened=[];manifest=[]
    for r in selected:
        c=checks[r['run_id']]
        flattened.append({'run_id':r['run_id'],'accepted':c['accepted'],'saved_threshold':r['threshold'],'reselected_validation_threshold':c['reselected_validation_threshold'],'validation_f1_at_saved_threshold':c['validation_f1_at_saved_threshold'],'saved_validation_f1':r['validation_f1_at_threshold'],'counts_exact':c['counts_exact'],'classification_metrics_match':c['classification_metrics_match'],'ranking_metrics_within_tolerance':c['ranking_metrics_within_tolerance'],'validation_f1_matches':c['validation_f1_matches'],'threshold_matches':c['threshold_matches'],'test_count':c['test_count'],**{'reproduced_'+k:v for k,v in c['whole_test_metrics'].items()},**{'delta_'+k:v for k,v in c['differences_from_saved'].items()}})
        manifest.append(dict(r,checkpoint_reproduction='accepted',reproduction_check=f"checks/{r['run_id']}.json",prediction_file=c['prediction_file'],prediction_sha256=c['prediction_sha256'],inference_feature_graph_path=c['feature_graph_path'],feature_graph_sha256=hashes[c['feature_graph_path']],node_mapping=f"node_mappings/{r['dataset']}.npz",test_ids_sha256=c['test_ids_sha256']))
    dump('run_manifest.json',manifest);write_csv('checkpoint_reproduction_checks.csv',flattened)
    max_ap=max(abs(c['differences_from_saved']['pr_auc']) for c in checks.values());max_roc=max(abs(c['differences_from_saved']['roc_auc']) for c in checks.values())
    dump('checkpoint_reproduction_summary.json',{'selected':240,'accepted':240,'required_gnn_mlp':200,'sequence_runs':40,'all_confusion_counts_exact':True,'max_absolute_ap_difference':max_ap,'max_absolute_roc_auc_difference':max_roc,'missing':missing,'mismatches':failed})
    counts=read('subgroup_counts.json');summary=read('subgroup_summary.json');per_seed=read('subgroup_per_seed.json')
    assert len(counts)==12 and len(per_seed)==480 and all(r['status']=='verified' for r in per_seed)
    assert len(summary)==48 and all(r['verified_seeds']==10 for r in summary)
    for a,b in itertools.combinations([r for r in per_seed if r['architecture']=='MLP' and r['group']=='connected'],2):
        if a['dataset']==b['dataset'] and a['seed']==b['seed']:assert a['run_id']==b['run_id']
    subgroup='# Connected versus isolated test-set analysis\n\n'
    subgroup+='All 200 required GNN/MLP checkpoints passed the predefined reproduction gate. These are post-hoc groups on the full saved inference graph, with self-loops ignored. MLP predictions are reused across constructions for identical IDs. Every run uses its saved validation-selected threshold; no test-group tuning occurred. PR-AUC means average precision. Mean ± sample SD across ten seeds.\n\n'
    subgroup+='## Group populations\n\n| Dataset | Construction | Group | N | Positive | Negative | Fraud rate |\n|---|---|---|---:|---:|---:|---:|\n'
    for r in counts:
        subgroup+=f"| {r['dataset']} | {r['construction']} | {r['group']} | {r['n']:,} | {r['positive']:,} | {r['negative']:,} | {fmt(100*r['fraud_rate'],4)+'%' if r['fraud_rate'] is not None else 'undefined'} |\n"
    subgroup+='\nElliptic has zero isolated labeled test targets. All 40 Elliptic isolated per-seed rows are explicitly empty; rates/AP/F1/recall are undefined and confusion counts are zero. This is a population limitation, not missing predictions. No connected-versus-isolated performance contrast can be estimated there.\n\n'
    for dataset,construction in dict.fromkeys((r['dataset'],r['construction']) for r in counts):
        subgroup+=f'## {dataset}: {construction}\n\n| Group | Model | Seeds | Average precision | F1 | Recall | False positives |\n|---|---|---:|---:|---:|---:|---:|\n'
        for r in summary:
            if (r['dataset'],r['construction'])!=(dataset,construction):continue
            vals=[('undefined' if r[m+'_mean'] is None else f"{r[m+'_mean']:.{1 if m=='false_positive' else 4}f} ± {r[m+'_sample_sd']:.{1 if m=='false_positive' else 4}f}") for m in ['pr_auc','f1_score','recall','false_positive']]
            subgroup+='| '+r['group']+' | '+r['architecture']+' | '+str(r['verified_seeds'])+' | '+' | '.join(vals)+' |\n'
    subgroup+='\n## Interpretation and checks\n\nOn the main IEEE-CIS graph, MLP mean AP is 0.5071 for connected and 0.2665 for isolated targets; GraphSAGE-main is 0.4547 and 0.1745 on exactly the same IDs. Connected and isolated fraud rates differ (4.6309% versus 2.5794%), so raw AP differences between groups do not by themselves measure the benefit of neighbors. The MLP control helps separate group difficulty from model behavior. The email-domain and address connected groups are small (1,488 and 481 targets), so retain their sizes and ten-seed SD when interpreting results. No subgroup significance tests or causal claims are made.\n\nAll 240 construction/model/seed evaluations reconcile their two subgroup confusion matrices exactly to the saved whole-test counts. See `subgroup_confusion_reconciliation.csv`. `subgroup_per_seed.csv` contains all 480 rows; `subgroup_summary.csv` also contains precision, ROC-AUC, counts, rates and the number of defined seed values. Undefined metrics are blank in CSV and null in JSON, with explicit status fields; they are not replaced by zero.\n'
    (OUT/'SUBGROUP_ANALYSIS.md').write_text(subgroup,encoding='utf-8')
    stats=read('wilcoxon_holm_inference_verified.json');source_stats=read('wilcoxon_holm_source_verified.json')
    assert all(r['n_pairs']==10 and r['status']=='complete' for r in stats)
    assert all(a['p_raw']==b['p_raw'] and a['p_holm']==b['p_holm'] for a,b in zip(stats,source_stats))
    statdoc='# Wilcoxon signed-rank tests with Holm correction\n\nAll 240 selected runs passed checkpoint reproduction. Tests use full-precision saved per-seed metrics from those accepted runs. The source-verified and inference-verified analyses now agree for every comparison. Exact sign permutations use average absolute ranks, drop exact zeros and retain all ten paired observations in the reported pair count. See METHODS.md for definitions fixed before results were interpreted.\n\n'
    statdoc+='## Complete family coverage\n\n| Dataset | Family | Hypotheses per metric | Metrics | Total | Significant after Holm |\n|---|---|---:|---:|---:|---:|\n'
    for dataset,family,size in [('IEEE-CIS','core',15),('Elliptic','core',15),('IEEE-CIS','construction',30),('IEEE-CIS','alternative_gnn_vs_mlp',12)]:
        rs=[r for r in stats if r['dataset']==dataset and r['family_id'].split('__')[1]==family]
        statdoc+=f"| {dataset} | {family} | {size} | 6 | {len(rs)} | {sum(r['significant_holm_0_05'] for r in rs)} |\n"
    statdoc+='\nThe metrics are average precision, F1, recall, precision, ROC-AUC and false positives. There are **432 tests in 24 families**, each with ten paired seeds. Holm is applied within each dataset/family/metric, never across a selected subset of printed findings. Construction comparisons pool all 30 within-architecture graph pairs; the alternative-GNN/MLP family includes all 12 non-main graph/architecture variants.\n\n## Selected changes to interpretation\n\n| Dataset/family | A minus B | Metric | Median paired difference | Raw p | Holm p | Significant after Holm |\n|---|---|---|---:|---:|---:|---|\n'
    wanted=[('IEEE-CIS','MLP','GraphSAGE-main'),('IEEE-CIS','GraphSAGE-main','GraphSAGE-email_domain'),('IEEE-CIS','GraphSAGE-email_domain','MLP'),('Elliptic','LSTM','Transformer')]
    for dataset,a,b in wanted:
        r=next(x for x in stats if x['dataset']==dataset and x['model_a']==a and x['model_b']==b and x['metric']=='pr_auc')
        statdoc+=f"| {r['family_id']} | {a} minus {b} | AP | {r['median_paired_difference_a_minus_b']:.8f} | {r['p_raw']:.8f} | {r['p_holm']:.8f} | {'Yes' if r['significant_holm_0_05'] else 'No'} |\n"
    statdoc+='\nNone of the construction comparisons survives the prespecified 30-hypothesis-per-metric family. With ten nonzero paired differences, the smallest possible two-sided exact signed-rank p-value is 2/1024 = 0.001953125; the first Holm step for 30 tests is at least 0.05859375. This is a limitation of the sample size and chosen family, not evidence of equivalence or absence of a construction effect. The family was not narrowed after observing this result.\n\nElliptic Transformer versus LSTM PR-AUC has raw p=0.02734375 but Holm p=0.08203125 in the complete core AP family. The median difference shown above is LSTM minus Transformer, so its negative sign favors Transformer descriptively. A non-significant adjusted comparison does not establish equal performance.\n\nEvery comparison, statistic, raw/adjusted p-value, total/nonzero/zero pair count, tie flag and median paired difference is in `wilcoxon_holm_inference_verified.csv`. `paired_differences_inference_verified.json` exposes the ten full-precision differences for every comparison. False-positive differences are counts; negative A-minus-B favors A for that metric.\n'
    (OUT/'WILCOXON_HOLM.md').write_text(statdoc,encoding='utf-8')
    representative=[];repdoc='# Proposed replacement Tables 4.9 and 4.10\n\nThese tables use **seed 51 from the same accepted August experiment sets as Tables 4.7 and 4.8**. IEEE-CIS GNNs are construction-tagged main runs. All twelve checkpoints passed reproduction. Original thesis/result files were not modified. Metrics below are formatted to four decimals; the companion CSV retains original full precision.\n\n'
    for dataset,number in [('IEEE-CIS','4.9'),('Elliptic','4.10')]:
        models=['MLP','LSTM','Transformer']+[a+('-main' if dataset=='IEEE-CIS' else '') for a in ['GCN','GraphSAGE','GAT']]
        rs=[r for r in selected if r['dataset']==dataset and r['seed']==51 and r['model'] in models]
        assert len(rs)==6
        rs.sort(key=lambda r:-float(r['f1_score']))
        repdoc+=f'## Proposed Table {number}: {dataset}, seed 51\n\n| Model | Recall | Precision | F1 | AP | ROC-AUC | TN | FP | FN | TP | Threshold |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        for r in rs:
            representative.append(dict(r,checkpoint_verified=True))
            repdoc+='| '+r['model']+' | '+' | '.join(f'{float(r[k]):.4f}' for k in ['recall','precision','f1_score','pr_auc','roc_auc'])+' | '+' | '.join(f'{int(r[k]):,}' for k in CM)+' | '+f"{float(r['threshold']):.6g}"+' |\n'
    repdoc+='\nElliptic-GAT seed 51 comes from `git:2be96ee:results/model_results.csv`, line 247, timestamp 2026-08-30T13:24:40, deleted by 5333c55. It is explicitly recovered, not substituted. All Elliptic models here evaluate 8,841 targets; all IEEE-CIS models evaluate 59,054. The older Table 4.10 warm-up exclusion does not apply.\n'
    write_csv('representative_seed51.csv',representative);(OUT/'REPRESENTATIVE_TABLES.md').write_text(repdoc,encoding='utf-8')
    versions=read('graph_diagnostics/graph_versions.json');comparisons=read('graph_diagnostics/edge_set_comparisons.json')
    gd='# Main-graph provenance discrepancy\n\nThe selected August `-main` GNN experiments are reproduced using the existing `results/ieee_cis_graph_trainfit_main_max1000.pt`, with **406,769 unique undirected non-self edges**. All 30 main GNN checkpoints pass the whole-test/validation reproduction gate on this cache. The older `ieee_cis_graph.pt` and `ieee_cis_graph_trainfit.pt` have exactly the same transaction-ID edge set (their features/serialization differ). None was rebuilt or replaced for inference.\n\n## Reproducing the reported 406,767\n\nThe inner graph-construction operation sorts each identifier group only by TransactionDT, with default quicksort. Equal timestamps have no explicit secondary key. Changing NumPy’s CPU sorting dispatch changes the order of ties and therefore which consecutive transactions are joined. The initial outer sort by TransactionID does not prevent an unstable inner sort from reordering those ties.\n\nControlled diagnostic results:\n\n| Diagnostic | NumPy / pandas | Inner sort | Edges | Fraud–fraud | Fraud–legitimate | Legitimate–legitimate |\n|---|---|---|---:|---:|---:|---:|\n'
    for name,label,runtime,sort in [('results/ieee_cis_graph_trainfit_main_max1000.pt','Saved experiment cache','existing artifact','existing'),('diagnostic_experiment_packages_quicksort.npz','Experiment packages, normal CPU dispatch','2.2.6 / 2.3.3','quicksort'),('diagnostic_experiment_scalar_sort_quicksort.npz','Same packages, AVX2/AVX512F disabled','2.2.6 / 2.3.3','quicksort'),('diagnostic_experiment_packages_mergesort.npz','Experiment packages, stable sort','2.2.6 / 2.3.3','mergesort'),('diagnostic_audit_runtime_quicksort.npz','Bundled audit packages','2.3.5 / 3.0.1','quicksort')]:
        r=next(x for x in versions if x['version']==name)
        gd+=f"| {label} | {runtime} | {sort} | {r['edges']:,} | {r['fraud_fraud_edges']:,} | {r['fraud_legitimate_edges']:,} | {r['legitimate_legitimate_edges']:,} |\n"
    gd+='\nSetting `NPY_DISABLE_CPU_FEATURES=AVX2,AVX512F` before starting the portable Python process reproduces **406,767 edges and exactly the thesis Table 4.4 class counts 8,475 / 18,644 / 379,648**. This identifies a concrete computational mechanism for the earlier reconstruction’s discrepancy. It does not establish which CPU dispatch was used historically: the original 406,767-edge file was not preserved. The reconstructed version is explicitly diagnostic, not a recovered training artifact.\n\n## Transaction-ID edge comparison\n\nThe normal-dispatch reconstruction matches the saved cache edge-for-edge. Compared with that cache, the scalar-sort 406,767-edge reconstruction removes **28** transaction-ID edges and adds **26**, for **54** differing edges and a net count difference of two. It is not simply deletion of two edges. `graph_diagnostics/transaction_id_edge_differences.csv` lists all differences, endpoint IDs, timestamps and labels. Stable-sort alternatives remove 24 and add 19 edges.\n\n'
    ref=np.load(OUT/'graph_diagnostics/saved_main_transaction_edges.npz')['transaction_id_pairs'];alt=np.load(OUT/'graph_diagnostics/diagnostic_experiment_scalar_sort_quicksort.npz')['transaction_id_pairs']
    same_active=np.array_equal(np.unique(ref),np.unique(alt));assert same_active
    gd+='The saved and scalar-reconstructed versions have the same set of nodes with a non-self neighbor, so connected/isolated membership is unchanged. Their neighborhoods differ, which can affect GNN predictions; this is why inference retains the saved experiment graph. No inference was performed on a diagnostic replacement.\n\n## Hashes\n\n| Version | Canonical transaction-edge SHA-256 | Artifact SHA-256 |\n|---|---|---|\n'
    for name in ['results/ieee_cis_graph_trainfit_main_max1000.pt','diagnostic_experiment_scalar_sort_quicksort.npz']:
        r=next(x for x in versions if x['version']==name);gd+=f"| `{name}` | `{r['transaction_edge_sha256']}` | `{r['file_sha256']}` |\n"
    gd+='\nCanonical edge hashes encode sorted, unique, undirected TransactionID pairs as little-endian int64 pairs. Full artifact versions, hashes and comparisons are in `graph_diagnostics/graph_versions.json` and `edge_set_comparisons.json`; all required experiment graphs/data/checkpoints have paths and SHA-256 hashes in `artifact_hashes_before.json`. The hardcoded 406,767 expectation in the later connectivity script is not a manifest of the August inference graph. No existing source or graph was changed to force either count.\n'
    (OUT/'GRAPH_PROVENANCE.md').write_text(gd,encoding='utf-8')
    blockers={'required_subgroup_analysis_blocked':False,'wilcoxon_holm_blocked':False,'missing_selected_artifacts':[],'checkpoint_mismatches':[],'limitations':[{'issue':'No isolated labeled Elliptic test targets','impact':'N=0 for isolated groups; AP/F1/recall/rates undefined. No subgroup performance contrast is estimable.','requires_retraining':False},{'issue':'Original 406767-edge reconstruction artifact absent','impact':'Its count and class counts are reproduced diagnostically via sorting dispatch; exact historical edge-set identity cannot be proven. Saved 406769 inference graph and every selected checkpoint reproduce.','requires_retraining':False},{'issue':'Floating-point ranking differences','impact':f'Largest absolute AP difference {max_ap:.12g}, ROC-AUC difference {max_roc:.12g}; all confusion counts, validation F1 and threshold checks pass. Full deltas recorded.','requires_retraining':False}]}
    dump('blockers_and_limitations.json',blockers)
    readme=f'''# Supervisor analyses, 2026-10-13

Both priority analyses are complete. **240/240 August checkpoints passed** the predefined whole-test and validation reproduction gate, including all 200 GNN/MLP runs needed for subgroups and all 40 sequence runs. No retraining or original-result/graph/checkpoint replacement occurred.

- [Connected-versus-isolated report](SUBGROUP_ANALYSIS.md): 12 groups, 480 per-seed metric rows, 48 mean/sample-SD summaries, and 240 exact confusion-count reconciliations. MLP uses identical group IDs and the same saved predictions across constructions.
- [Wilcoxon/Holm report](WILCOXON_HOLM.md): all 432 comparisons across 24 prespecified dataset/metric/family combinations, ten paired seeds each, raw/adjusted p-values and median paired differences.
- [Graph discrepancy](GRAPH_PROVENANCE.md): runtime-dependent tie ordering reproduces the 406,767 report; inference uses the verified 406,769 saved graph. Transaction-ID edge differences and hashes are supplied.
- [Representative tables](REPRESENTATIVE_TABLES.md): proposed Tables 4.9/4.10 use seed 51 from these same verified experiment sets, including explicitly recovered Elliptic-GAT.
- [Sequence inputs](SEQUENCE_INPUTS.md): selected August sequences retain all targets with split-local left-padding. The older final-run table is not used to infer their behavior.
- [Predeclared methods/families](METHODS.md), plus `comparison_families.json` with every planned comparison.

## Essential files

| Output | Content |
|---|---|
| `selected_per_seed_results.csv`, `selected_runs.json` | 240 selected rows with original full-precision strings, seed, source file/line and historical source snapshot. |
| `run_manifest.json` | Selected rows joined to verified checkpoints, graph hashes, test-ID mappings and predictions. |
| `recovered_result_provenance.json` | Exact Elliptic-GAT seed-51 row, Git commit/source line, historical CSV hash and deletion diff. |
| `artifact_hashes_before.json` | Paths and SHA-256 hashes for required data, graph caches, checkpoints and code. |
| `additional_referenced_artifact_hashes.json` | Hashes/paths of other saved graph/checkpoint/result files referenced by the archived audit. |
| `original_artifact_preservation.json` | All {len(inventory)} original input artifacts rehashed after analysis; no changes. |
| `feature_checks_*.json`, `reconstructed_ieee_preprocessor.json` | Exact feature/label/split reconstruction checks and explicitly reconstructed train-fitted preprocessing parameters. |
| `checkpoint_reproduction_checks.csv`, `checks/*.json` | Whole-test metric deltas, exact confusion checks, validation F1, reselected threshold and acceptance per run. |
| `predictions/*.npz` | New inference probabilities with validation/test transaction IDs and labels, saved threshold; these are not claimed to be originally saved predictions. |
| `node_mappings/*.npz`, `subgroup_ids/*.npz` | Full node/ID/label/split mappings and exact subgroup IDs. |
| `subgroup_counts.csv`, `subgroup_per_seed.csv`, `subgroup_summary.csv` | Counts/fraud rates, all individual-seed metrics, mean and sample SD, defined-value counts. JSON equivalents preserve nulls. |
| `subgroup_confusion_reconciliation.csv` | Every subgroup pair sums exactly to its saved whole-test confusion matrix. |
| `wilcoxon_holm_inference_verified.csv` | Final complete statistical analysis, including all comparison families. |
| `wilcoxon_holm_source_verified.csv` | Analysis prepared before inference; identical p-values/effects after all runs passed. |
| `paired_differences_*.json` | Every full-precision seed pair and A-minus-B difference. |
| `representative_seed51.csv` | Full-precision proposed representative rows, source references and verification status. |
| `graph_diagnostics/*` | Explicit diagnostic reconstructions, transaction-ID edge differences, counts and fingerprints. |
| `source_snapshots/*` | Relevant August source files from Git. |

## Unresolved issues and interpretation

No missing selected run or failed checkpoint blocks either priority analysis. Elliptic's isolated labeled test group is empty, so its performance metrics are undefined. The original 406,767-edge reconstruction file was not saved; the controlled reconstruction reproduces its reported total and class counts but cannot prove historical edge identity. All selected experiments instead reproduce on their saved caches. Ranking scores have small GPU floating-point/tie differences: maximum absolute AP difference {max_ap:.12g}, ROC-AUC {max_roc:.12g}; every confusion matrix matches exactly. See `blockers_and_limitations.json`.

The complete 30-comparison construction families yield no Holm-significant results at alpha .05 with ten seeds. Do not describe this as evidence of equivalence. The full report explains the discrete p-value limit and reports median effects.

## Reproduction

Run from the original repository root with the packages in `environment.json`: Python 3.10.0, torch 2.5.1+cu121, PyG 2.7.0, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2 and SciPy 1.15.3. Analysis used an NVIDIA RTX 3080 Ti. A portable official Python runtime was downloaded under this analysis folder because the venv's original executable was missing; it reuses the existing venv packages. The downloaded runtime and packages are excluded from the ZIP. Record of its URL and hash is in `environment.json`.

Use `python` below for that compatible interpreter. Scripts write only inside the separate analysis folder. Existing per-run check files are resumable; use a fresh copy of the analysis scripts and required inputs to rerun inference from scratch. Do not rerun blindly over a changed experiment set: verify the saved input hashes first.

```powershell
python analysis/supervisor_2026-10-13/prepare.py
python analysis/supervisor_2026-10-13/wilcoxon_holm.py --mode source_verified
python analysis/supervisor_2026-10-13/verify_features.py IEEE-CIS
python analysis/supervisor_2026-10-13/verify_features.py Elliptic
python analysis/supervisor_2026-10-13/infer.py IEEE-CIS
python analysis/supervisor_2026-10-13/infer.py Elliptic
python analysis/supervisor_2026-10-13/subgroups.py
python analysis/supervisor_2026-10-13/infer.py IEEE-CIS --kind sequence
python analysis/supervisor_2026-10-13/infer.py Elliptic --kind sequence
python analysis/supervisor_2026-10-13/verify_sequences.py
python analysis/supervisor_2026-10-13/wilcoxon_holm.py --mode inference_verified
python analysis/supervisor_2026-10-13/diagnose_graph.py experiment_packages
$env:NPY_DISABLE_CPU_FEATURES='AVX2,AVX512F'
python analysis/supervisor_2026-10-13/diagnose_graph.py experiment_scalar_sort
Remove-Item Env:NPY_DISABLE_CPU_FEATURES
```

The additional `audit_runtime` diagnostic uses bundled Python 3.12, NumPy 2.3.5 and pandas 3.0.1. Then run `compare_graphs.py`, `finalize.py` and `package.py` from this folder with the experiment-compatible interpreter. The scripts require the original raw datasets/checkpoints at their recorded paths; preparation also requires the original Git history for recovery. They never call model training routines.

## Package contents

The ZIP preserves the workspace-relative layout and includes **the entire earlier `audit/` tree**, not only `audit/thesis/`, this complete analysis folder (except portable runtime and Python caches), source code, and the small result CSVs. Large source datasets, full tensor graph caches and checkpoints are excluded and represented by their paths/hashes. Newly computed inference probabilities and diagnostic edge arrays are included. The original PDF is represented by its source path/hash and the prior audit's extracted text/tables/rendered previews.

The earlier audit is a historical snapshot. Since that audit, commits b464cb9/55a8d11 updated connectivity analysis and the general Wilcoxon CSV. The current manifest and reproduction outputs supersede the earlier audit's statements that inference was unavailable or that the then-current Wilcoxon file was necessarily still June-only. The older audit is retained as provenance, not silently rewritten.
'''
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    # PowerShell redirection used UTF-16; preserve logs in portable UTF-8.
    for p in OUT.glob('*.log'):
        b=p.read_bytes()
        if b.startswith(b'\xff\xfe'):p.write_text(b.decode('utf-16'),encoding='utf-8')
    print('Final reports/manifests written; 240 accepted runs; original inputs unchanged.',flush=True)

if __name__=='__main__':main()
