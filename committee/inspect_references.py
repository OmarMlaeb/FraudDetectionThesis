"""Read-only inspection of original artifacts; writes only NEW committee metadata.

--initialize is an implementation-time operation. It refuses to replace the lock.
No experiment topology is generated and no model is trained or evaluated.
"""
import argparse
import ast
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from plan import matrix, families, training_config
from topology import canonical, stats


def main():
    import numpy as np
    import pandas as pd
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initialize', action='store_true', required=True)
    args = p.parse_args()
    require(not (HERE / 'reference_lock.json').exists(), 'Reference lock already exists; refusing replacement.')
    expected = {r['path']: r['sha256'] for r in read(AUDIT / 'artifact_hashes_before.json') if r.get('sha256')}
    rows = read(AUDIT / 'run_manifest.json')
    wanted = {r['reference_run_id'] for r in matrix()}
    refs = [r for r in rows if r['run_id'] in wanted]
    require(len(refs) == 210, f'Expected 210 distinct verified references; found {len(refs)}')
    files = {}
    def pin(path):
        path = Path(path)
        rel = path.relative_to(ROOT).as_posix()
        h = sha(path)
        if rel in expected:
            require(h == expected[rel], f'Original audit hash mismatch: {rel}')
        files[rel] = h
    for name in ['run_manifest.json', 'selected_runs.json', 'artifact_hashes_before.json', 'environment.json',
                 'recovered_result_provenance.json', 'SEQUENCE_INPUTS.md', 'GRAPH_PROVENANCE.md',
                 'reconstructed_ieee_preprocessor.json', 'feature_checks_IEEE-CIS.json', 'feature_checks_Elliptic.json']:
        pin(AUDIT / name)
    for r in refs:
        require(r['checkpoint_reproduction'] == 'accepted', 'Reference not accepted')
        pin(AUDIT / r['prediction_file'])
        require(sha(AUDIT / r['prediction_file']) == r['prediction_sha256'], 'Reference prediction hash mismatch')
        pin(AUDIT / r['reproduction_check'])
    snapshot = AUDIT / 'source_snapshots/2be96ee/src'
    copies = {'metrics': 'common/metrics.py', 'elliptic_models': 'elliptic/models.py',
              'elliptic_gnn': 'elliptic/gnn_models.py', 'ieee_gnn': 'ieee_cis/gnn_models.py',
              'elliptic_sequence': 'elliptic/sequence.py'}
    (HERE / 'frozen').mkdir(exist_ok=True)
    for name, src in copies.items():
        target = HERE / 'frozen' / (name + '.py')
        require(not target.exists(), f'Frozen copy exists: {target}')
        shutil.copyfile(snapshot / src, target)
        pin(snapshot / src); pin(target)
    source_config = {}
    for ds, model in [('ieee_cis', 'gnn')] + [('elliptic', m) for m in ['gnn', 'mlp', 'lstm', 'transformer']]:
        train_snapshot = AUDIT / 'source_snapshots' / ('5333c55' if ds == 'ieee_cis' else '2be96ee') / 'src'
        path = train_snapshot / ds / f'train_{model}.py'
        tree = ast.parse(path.read_text())
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = node.value.value
        cfg = training_config('gcn' if model == 'gnn' else model)
        require(constants['EPOCHS'] == cfg['epochs'] and constants['EARLY_STOPPING_PATIENCE'] == cfg['patience'] and constants['LEARNING_RATE'] == cfg['learning_rate'], 'Frozen hyperparameter mismatch')
        if model != 'gnn':
            require(constants['BATCH_SIZE'] == cfg['batch_size'], 'Frozen batch mismatch')
        source_config[f'{ds}/{model}'] = dict(path=str(path), sha256=sha(path), constants=constants,
            criterion='validation average precision; strict improvement', threshold='unconstrained validation F1 grid .01 through .99',
            review='Training loops and loss/optimizer/selection reviewed against frozen source; IEEE Transformer is outside this 300-run design.')
        pin(path)
    for version in ['2be96ee', '5333c55']:
        for path in (AUDIT / 'source_snapshots' / version).rglob('*.py'):
            pin(path)
    feature_map = dict(raw_columns=167, txId_raw_index=0, time_step_raw_index=1,
        full_model_raw_indices=list(range(2, 167)), local_model_raw_indices=list(range(2, 95)),
        excluded_aggregate_raw_indices=list(range(95, 167)), local_loader_columns=[f'feature_{i}' for i in range(93)],
        excluded_loader_columns=[f'feature_{i}' for i in range(93, 165)],
        full_dimension=165, local_dimension=93, aggregate_dimension=72, indexing='zero-based raw CSV column positions',
        documentation='https://medium.com/elliptic/the-elliptic-data-set-opening-up-machine-learning-on-the-blockchain-e0a343d99a14',
        documentation_checked='2026-10-13',
        rationale='Publisher documents 166 features, first 94 local including time; loader excludes txId and time. Hence numeric local raw 2..94, aggregate 95..166.',
        limitation='Local columns can themselves contain address/transaction aggregates; removing 72 supplied neighbor aggregates does not eliminate all relational information.')
    graphs, inspection = {}, {}
    for dataset, ds in [('ieee', 'IEEE-CIS'), ('elliptic', 'Elliptic')]:
        mapping_path = AUDIT / 'node_mappings' / (ds + '.npz')
        pin(mapping_path)
        with np.load(mapping_path) as z:
            node_ids = z['node_ids'].copy()
            masks = {s: z[s+'_mask'].copy() for s in ['train', 'val', 'test']}
            labels = z['labels'].copy()
        if dataset == 'elliptic':
            raw = ROOT / 'data/elliptic_bitcoin_dataset/elliptic_txs_features.csv'
            require(pd.read_csv(raw, header=None, nrows=2).shape[1] == 167, 'Elliptic raw schema mismatch')
            meta = pd.read_csv(raw, header=None, usecols=[0, 1], names=['txId', 'time_step']).sort_values(['time_step', 'txId'])
            times = meta.time_step.to_numpy()
            require(np.array_equal(node_ids, meta.txId.to_numpy()), 'Elliptic node order mismatch')
            for raw in (ROOT / 'data/elliptic_bitcoin_dataset').glob('*.csv'):
                pin(raw)
        else:
            raw = ROOT / 'data/ieee-cis/train_transaction.csv'
            meta = pd.read_csv(raw, usecols=['TransactionID', 'TransactionDT']).sort_values(['TransactionDT', 'TransactionID'], kind='mergesort')
            times = meta.TransactionDT.to_numpy()
            require(np.array_equal(node_ids, meta.TransactionID.to_numpy()), 'IEEE node order mismatch')
            pin(raw); pin(ROOT / 'data/ieee-cis/train_identity.csv')
        for construction in CONSTRUCTIONS if dataset == 'ieee' else ['native']:
            r = next(r for r in refs if r['dataset'] == ds and r['architecture'] == 'GCN' and r['construction'] == construction)
            path = ROOT / r['graph_path']
            pin(path)
            print('Inspecting saved graph:', path.name, flush=True)
            g = torch.load(path, map_location='cpu', weights_only=False)
            edges = canonical(g.edge_index.numpy(), len(node_ids))
            require(g.x.shape == (len(node_ids), 358 if dataset == 'ieee' else 165), 'Unexpected feature shape')
            require(np.array_equal(g.y.numpy(), labels), 'Graph labels mismatch')
            for split, mask in masks.items():
                require(np.array_equal(getattr(g, split+'_mask').numpy(), mask), f'Mask mismatch {split}')
            cross = int(np.count_nonzero(times[edges[:, 0]] != times[edges[:, 1]]))
            max_gap = int(np.max(np.abs(times[edges[:, 0]]-times[edges[:, 1]])))
            if dataset == 'elliptic':
                require(cross == 0, 'Elliptic saved graph crosses time steps')
            if construction == 'identifier_time_window-24h':
                require(max_gap <= 86400, 'IEEE window violation')
            key = f'{dataset}__{construction}'
            graphs[key] = dict(path=r['graph_path'], sha256=sha(path), mapping_path=mapping_path.relative_to(ROOT).as_posix(),
                mapping_sha256=sha(mapping_path), feature_shape=list(g.x.shape), feature_sha256=array_sha(g.x.numpy()),
                label_sha256=array_sha(g.y.numpy()), node_ids_sha256=array_sha(node_ids),
                split_id_sha256={s: array_sha(node_ids[mask]) for s, mask in masks.items()},
                split_counts={s: int(mask.sum()) for s, mask in masks.items()}, time_sha256=array_sha(times))
            inspection[key] = dict(**stats(edges, len(node_ids)), symmetric=True, self_loops=0, duplicate_directed_edges=0,
                                  cross_time_edges=cross, maximum_time_gap=max_gap,
                                  actual_message_passing='symmetric undirected; original raw Elliptic DAG direction was removed by verified loader')
            del g, edges
    packages = {k: read(AUDIT / 'environment.json')[k] for k in ['numpy', 'pandas', 'scipy', 'sklearn', 'torch', 'torch_geometric']}
    output = dict(files=files, graphs=graphs, packages=packages, feature_mapping=feature_map,
                  frozen_training_configuration=source_config, reference_runs=sorted(wanted),
                  reference_comparability='Frozen architecture, inputs, splits, checkpoint and threshold rules match. Historical full-feature runs were not retrained; historical RNG/environment details beyond saved provenance cannot be established retroactively.')
    atomic_json(HERE / 'reference_lock.json', output)
    atomic_json(HERE / 'comparison_families.json', families())
    atomic_json(OUT / 'implementation_verification/reference_inspection.json', inspection)
    atomic_json(OUT / 'implementation_verification/environment.json', environment())
    atomic_json(OUT / 'implementation_verification/experiment_matrix.json', matrix())
    print('Pinned 210 verified references; planned 300 new runs. No training or graph generation performed.')


if __name__ == '__main__':
    main()
