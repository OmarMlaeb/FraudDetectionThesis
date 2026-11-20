"""Validate predictions, summarize ten seeds, and test preregistered families."""
import csv
import io
import itertools
import zipfile
from datetime import datetime, timezone
import numpy as np
from scipy.stats import rankdata
from common import *
from plan import matrix, families, training_config
from graphs import read_graph_manifest
from runner import check_completed, metric_values


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    buf = io.StringIO(newline='')
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader(); w.writerows(rows)
    atomic_bytes(path, buf.getvalue().encode('utf-8'))


def signed_rank(differences):
    d = np.asarray(differences, dtype=np.float64)
    nz = d[d != 0]
    if not len(nz):
        return dict(p_raw=1.0, statistic=0.0, nonzero_pairs=0, zero_pairs=len(d), tied_absolute_differences=False)
    ranks = rankdata(np.abs(nz), method='average')
    positive = float(ranks[nz > 0].sum())
    statistic = min(positive, float(ranks.sum())-positive)
    sums = np.zeros(1)
    for r in ranks:
        sums = np.concatenate([sums, sums+r])
    p = min(1., 2*float(np.count_nonzero(sums <= statistic))/len(sums))
    return dict(p_raw=p, statistic=statistic, nonzero_pairs=len(nz), zero_pairs=len(d)-len(nz),
                tied_absolute_differences=len(np.unique(np.abs(nz))) != len(nz))


def holm(pvalues):
    effective = [1.0 if p is None else p for p in pvalues]
    order = np.argsort(effective, kind='stable')
    adjusted = [None]*len(effective)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(effective)-rank)*effective[i]))
        if pvalues[i] is not None:
            adjusted[i] = running
    return adjusted


def validate(rows, write=True):
    checked, accepted = [], []
    graph_cache = {}
    source_hash_cache = set()
    current_code = code_hashes()
    for row in rows:
        status_path = OUT / 'runs' / row['family'] / row['run_id'] / 'status.json'
        record = dict(run_id=row['run_id'], family=row['family'], status='missing')
        try:
            if not status_path.exists():
                checked.append(record)
                continue
            status = read(status_path)
            if status['status'] != 'complete':
                record.update(status=status['status'], rerun_command=status.get('rerun_command'))
                checked.append(record)
                continue
            config = check_completed(status)
            comp = config['compatibility']
            require(comp['row'] == row and comp['training'] == training_config(row['model']), 'Run does not match the requested matrix')
            require(comp['code_hashes'] == current_code, 'Implementation code changed since training')
            require(comp['packages'] == lock()['packages'], 'Run used unpinned package versions')
            if row['graph_id'] not in graph_cache:
                graph_cache[row['graph_id']] = read_graph_manifest(row['graph_spec'])
            graph = graph_cache[row['graph_id']]
            require(comp['topology'] == graph['topology'] and comp['graph_config'] == graph['config'] and comp['effective_graph_sha256'] == graph['effective_graph_sha256'], 'Graph/run provenance mismatch')
            source_path = graph['source_graph']['path']
            if source_path not in source_hash_cache:
                verify_artifact(graph['source_graph']); source_hash_cache.add(source_path)
            ref = reference(row)
            require(comp['reference'] == ref, 'Reference run provenance mismatch')
            metrics = read(status['artifacts']['metrics']['path'])
            require(all(metrics[k] == row[k] for k in ['run_id', 'seed', 'family', 'dataset', 'construction', 'model', 'graph_seed', 'feature_set', 'reference_run_id']), 'Metric row identity mismatch')
            require(metrics['checkpoint_sha256'] == status['artifacts']['checkpoint']['sha256'], 'Selected checkpoint mismatch')
            with np.load(status['artifacts']['predictions']['path']) as z, np.load(AUDIT / ref['prediction_file']) as refz:
                predictions = {k: z[k].copy() for k in z.files}
                for split in ['val', 'test']:
                    require(np.array_equal(z[split+'_ids'], refz[split+'_ids']), f'{split} IDs differ from verified reference')
                    require(np.array_equal(z[split+'_y'], refz[split+'_y']), f'{split} labels differ from verified reference')
                    probs = z[split+'_probs']
                    require(probs.ndim == 1 and len(probs) == len(z[split+'_ids']) and np.all(np.isfinite(probs)) and np.all((probs >= 0) & (probs <= 1)), 'Invalid predictions')
            require(float(predictions['threshold']) == metrics['threshold'], 'Prediction threshold mismatch')
            require(metrics['test_count'] == len(predictions['test_y']) and metrics['validation_count'] == len(predictions['val_y']), 'Saved population counts mismatch')
            expected_dim = 93 if row['feature_set'] == 'local93' else 358 if row['dataset'] == 'ieee' else 165
            require(metrics['input_dimension'] == expected_dim, 'Wrong input feature dimension')
            require(metrics['effective_graph_sha256'] == graph['effective_graph_sha256'], 'Wrong effective graph fingerprint')
            threshold, vf1 = frozen('metrics').find_best_threshold(predictions['val_y'], predictions['val_probs'], strategy='f1')
            require(threshold == metrics['threshold'] and abs(vf1-metrics['validation_f1_at_threshold']) <= 1e-12, 'Validation threshold mismatch')
            recomputed = metric_values(predictions['test_y'], predictions['test_probs'], threshold)
            require(all(abs(recomputed[k]-metrics[k]) <= 1e-12 for k in recomputed), 'Saved test metrics do not match prediction scores')
            require(sum(metrics[k] for k in COUNTS) == len(predictions['test_y']), 'Confusion counts do not sum to test size')
            require(1 <= metrics['best_epoch'] <= comp['training']['epochs'], 'Invalid selected epoch')
            vap = metric_values(predictions['val_y'], predictions['val_probs'], threshold)['pr_auc']
            require(abs(vap-metrics['best_validation_score']) <= 2e-5, 'Best checkpoint validation score mismatch')
            accepted.append(metrics)
            record.update(status='accepted', source=str(status_path), reference_run_id=ref['run_id'])
        except Exception as exc:
            record.update(status='invalid', error=repr(exc))
        checked.append(record)
    report = dict(expected=len(rows), accepted=len(accepted), complete=len(accepted) == len(rows), runs=checked)
    if write:
        atomic_json(OUT / 'analysis/validation.json', report)
    return accepted, report


def analyze(s):
    rows = matrix(s)
    accepted, validation = validate(rows)
    by_id = {r['run_id']: r for r in accepted}
    require(len(by_id) == len(accepted), 'Duplicate completed run rows')
    output = OUT / 'analysis'
    for family in ['complement', 'local', 'random']:
        per_seed, summaries = [], []
        planned = [r for r in rows if r['family'] == family]
        for row in planned:
            base = {k: row[k] for k in ['run_id', 'family', 'dataset', 'construction', 'model', 'feature_set', 'graph_seed', 'seed', 'reference_run_id']}
            per_seed.append(dict(base, status='accepted' if row['run_id'] in by_id else 'missing_or_invalid', **{
                k: v for k, v in by_id.get(row['run_id'], {}).items() if k not in base}))
        arms = sorted({(r['dataset'], r['construction'], r['model']) for r in planned})
        for dataset, construction, model in arms:
            group = [r for r in per_seed if (r['dataset'], r['construction'], r['model']) == (dataset, construction, model)]
            valid = [r for r in group if r['status'] == 'accepted']
            summary = dict(family=family, dataset=dataset, construction=construction, model=model,
                           n_seeds=len(valid), expected_seeds=10, complete=len(valid) == 10,
                           missing_seeds=','.join(str(r['seed']) for r in group if r['status'] != 'accepted'))
            for metric in list(dict.fromkeys(METRICS+COUNTS+['threshold'])):
                values = [r[metric] for r in valid]
                summary[metric+'_mean'] = float(np.mean(values)) if values else None
                summary[metric+'_sample_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summaries.append(summary)
        csv_write(output / f'{family}_per_seed.csv', per_seed)
        atomic_json(output / f'{family}_per_seed.json', per_seed)
        csv_write(output / f'{family}_summary.csv', summaries)
        atomic_json(output / f'{family}_summary.json', summaries)
    definitions = read(HERE / 'comparison_families.json')
    require(definitions == families(), 'Preregistered family definitions were changed')
    tests, pairs = [], []
    for definition in definitions:
        family_tests = []
        for comparison in definition['comparisons']:
            selected = [r for r in rows if r['family'] == definition['family'] and r['dataset'] == definition['dataset']
                        and r['model'] == comparison['model'] and r['construction'] == comparison['construction']]
            matched = []
            metric = definition['metric']
            for row in selected:
                if row['run_id'] not in by_id:
                    continue
                ref = reference(row)
                new, old = by_id[row['run_id']][metric], float(ref[metric])
                matched.append(dict(seed=row['seed'], new=new, original=old, difference=new-old,
                                    reference_run_id=ref['run_id'], new_run_id=row['run_id']))
            diff = [p['difference'] for p in matched]
            complete = len(matched) == 10
            test = dict(family_id=definition['id'], dataset=definition['dataset'], experiment_family=definition['family'],
                        metric=metric, **comparison, n_pairs=len(matched), expected_pairs=10, complete=complete,
                        planned_family_size=len(definition['comparisons']),
                        median_paired_difference=float(np.median(diff)) if diff else None,
                        mean_paired_difference=float(np.mean(diff)) if diff else None,
                        difference='new minus original', higher_is_better=metric != 'false_positive',
                        p_raw=None, statistic=None, nonzero_pairs=None, zero_pairs=None, tied_absolute_differences=None,
                        test_settings=definition['method'], alternative='two-sided', zero_method='wilcox', rounding='none')
            if complete:
                test.update(signed_rank(diff))
            family_tests.append(test)
            pairs.append(dict(family_id=definition['id'], **comparison, pairs=matched))
        for test, adjusted in zip(family_tests, holm([r['p_raw'] for r in family_tests])):
            test.update(p_holm=adjusted, significant_holm_0_05=adjusted <= .05 if adjusted is not None else None)
        tests.extend(family_tests)
    atomic_json(output / 'wilcoxon_holm.json', tests)
    csv_write(output / 'wilcoxon_holm.csv', tests)
    atomic_json(output / 'paired_differences.json', pairs)
    atomic_json(output / 'comparison_families.json', definitions)
    atomic_json(output / 'analysis_settings.json', dict(settings=s, expected_runs=300, planned_tests=180, planned_families=30,
                 original_metric_source='Full-precision verified August saved results, including recovered Elliptic GAT seed51',
                 reference_lock_sha256=sha(HERE / 'reference_lock.json'), complete=validation['complete']))
    require(len(tests) == 180 and len(definitions) == 30, 'Wrong complete family size')
    print(f"Accepted {len(accepted)}/300 runs. Missing/invalid runs and incomplete pairs are explicit. Statistical outputs: {output}")
    return validation


def package():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = OUT / ('committee_review_'+stamp+'.zip')
    OUT.mkdir(parents=True, exist_ok=True)
    included, excluded = {}, {}
    def consider(p):
        p = Path(p)
        if not p.is_file() or p.is_relative_to(AUDIT / 'runtime') or '__pycache__' in p.parts or p.suffix in ['.tmp', '.zip', '.sha256'] or p.name == '.active.lock':
            return
        rel = p.relative_to(ROOT).as_posix()
        # New and verified prediction NPZs are small enough to include. Graph NPZs are omitted.
        omit = p.suffix in ['.pt', '.pth'] or (p.suffix == '.npz' and 'predictions' not in rel and 'node_mappings' not in rel)
        (excluded if omit else included)[rel] = artifact(p)
    for base in [HERE, OUT, AUDIT, ROOT / 'audit']:
        for p in base.rglob('*'):
            consider(p)
    for name in ['run_sampled_complement.py', 'run_elliptic_local_features.py', 'run_random_graph.py']:
        consider(ROOT / name)
    for rel, expected_hash in lock()['files'].items():
        p = ROOT / rel
        if rel.startswith('analysis/'):
            require(sha(p) == expected_hash, f'Pinned review source changed: {p}')
            consider(p)
        elif rel.startswith(('data/', 'results/')):
            excluded[rel] = dict(path=str(p), sha256=expected_hash, hash_source='verified reference lock; original artifact')
    wanted = set(lock()['reference_runs'])
    for r in read(AUDIT / 'run_manifest.json'):
        if r['run_id'] in wanted:
            excluded[r['checkpoint_path']] = dict(path=str(ROOT / r['checkpoint_path']), sha256=r['checkpoint_sha256'],
                                                  hash_source='accepted supervisor checkpoint reproduction')
    manifest = dict(included=included, excluded=excluded, training_run_by_implementation=False,
                    contents_note='Presence/completeness of subsequent user runs is recorded in runs/*/status.json and analysis/validation.json, if available.')
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in sorted(included):
            z.write(ROOT / rel, rel)
        z.writestr('review_manifest.json', json.dumps(manifest, indent=2))
    atomic_bytes(path.with_suffix('.zip.sha256'), (sha(path)+'  '+path.name+'\n').encode())
    print(path)
    return path
