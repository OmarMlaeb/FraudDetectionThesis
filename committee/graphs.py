"""Explicit graph preparation; never called implicitly by training."""
import os
import time
import numpy as np
from common import *
from plan import graph_id
from topology import canonical, symmetric, complement, rewire, degree, stats, overlap


def graph_config(spec):
    ref = lock()['graphs'][spec['dataset']+'__'+spec['construction']]
    return dict(spec=spec, reference=ref, feature_mapping=lock()['feature_mapping'] if spec['dataset'] == 'elliptic' else {
        'input_dimension': 358, 'preprocessor': str(AUDIT / 'reconstructed_ieee_preprocessor.json'),
        'preprocessing': 'Verified cached train-fit features, unchanged; TransactionID excluded; TransactionDT included'},
        implementation={n: sha(HERE / n) for n in ['graphs.py', 'topology.py', 'plan.py', 'common.py']},
        reference_lock_sha256=sha(HERE / 'reference_lock.json'))


def load_reference(spec):
    import torch
    cfg = graph_config(spec)
    r = cfg['reference']
    verify_lock_files([r['path'], r['mapping_path']])
    g = torch.load(ROOT / r['path'], weights_only=False, map_location='cpu')
    with np.load(ROOT / r['mapping_path']) as z:
        mapping = {k: z[k].copy() for k in z.files}
    require(array_sha(g.x.numpy()) == r['feature_sha256'], 'Reference features changed')
    require(array_sha(g.y.numpy()) == r['label_sha256'], 'Reference labels changed')
    require(array_sha(mapping['node_ids']) == r['node_ids_sha256'], 'Node IDs changed')
    require(np.array_equal(g.y.numpy(), mapping['labels']), 'Label mapping mismatch')
    for split in ['train', 'val', 'test']:
        mask = getattr(g, split+'_mask').numpy()
        require(np.array_equal(mask, mapping[split+'_mask']), 'Split mask mismatch')
        require(array_sha(mapping['node_ids'][mask]) == r['split_id_sha256'][split], 'Split IDs changed')
    return g, mapping, cfg


def time_metadata(spec, node_ids):
    import pandas as pd
    if spec['dataset'] == 'elliptic':
        p = 'data/elliptic_bitcoin_dataset/elliptic_txs_features.csv'
        verify_lock_files([p])
        require(pd.read_csv(ROOT / p, header=None, nrows=2).shape[1] == 167, 'Raw schema changed')
        df = pd.read_csv(ROOT / p, header=None, usecols=[0, 1], names=['id', 'time']).sort_values(['time', 'id'])
    else:
        p = 'data/ieee-cis/train_transaction.csv'
        verify_lock_files([p])
        df = pd.read_csv(ROOT / p, usecols=['TransactionID', 'TransactionDT']).rename(columns={'TransactionID': 'id', 'TransactionDT': 'time'}).sort_values(['time', 'id'], kind='mergesort')
    require(np.array_equal(node_ids, df.id.to_numpy()), 'Time metadata node order mismatch')
    result = df.time.to_numpy()
    require(array_sha(result) == lock()['graphs'][spec['dataset']+'__'+spec['construction']]['time_sha256'], 'Time metadata changed')
    return result


def read_graph_manifest(spec):
    path = OUT / 'graphs' / graph_id(spec) / 'manifest.json'
    require(path.exists(), f'Graph not prepared. Run the separate generate command first: {graph_id(spec)}')
    m = read(path)
    require(m['status'] == 'complete' and m['config'] == graph_config(spec), f'Graph config mismatch: {path}; do not overwrite it.')
    verify_artifact(m['topology'])
    verify_lock_files([m['config']['reference']['path'], m['config']['reference']['mapping_path']])
    require(m['source_graph']['sha256'] == m['config']['reference']['sha256'], 'Source graph manifest mismatch')
    require(digest(m['effective_graph']) == m['effective_graph_sha256'], 'Effective graph fingerprint mismatch')
    return m


def generate(spec):
    import torch
    directory = OUT / 'graphs' / graph_id(spec)
    if (directory / 'manifest.json').exists():
        read_graph_manifest(spec)
        print('Verified existing graph:', graph_id(spec), flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    attempt = directory / f'attempt_{len(list(directory.glob("attempt_*")))+1:03d}'
    attempt.mkdir(exist_ok=False)
    cfg = graph_config(spec)
    atomic_json(attempt / 'config.json', dict(config=cfg, environment=environment()))
    started = time.perf_counter()
    try:
        g, mapping, cfg = load_reference(spec)
        n = g.num_nodes
        original = canonical(g.edge_index.numpy(), n)
        diagnostics = {}
        if spec['family'] == 'complement':
            edges = complement(original, n, spec['graph_seed'], spec['average_degree'])
            require(overlap(original, edges, n)['shared_edges'] == 0, 'Complement contains original edges')
            require(2*len(edges) == n*20, 'Complement did not achieve average degree 20')
        elif spec['family'] == 'random':
            times = time_metadata(spec, mapping['node_ids']) if spec['time_constraint'] != 'none' else None
            edges, diagnostics = rewire(original, n, spec['graph_seed'], spec['swaps_per_edge'], spec['attempts_per_edge'],
                strata=times if spec['time_constraint'] == 'within_time_step' else None,
                times=times, max_delta=86400 if spec['time_constraint'] == 'max_86400_seconds' else None)
            require(np.array_equal(degree(original, n), degree(edges, n)), 'Degrees not exactly preserved')
            if spec['time_constraint'] == 'within_time_step':
                require(np.all(times[edges[:, 0]] == times[edges[:, 1]]), 'Rewired time-step violation')
            elif spec['time_constraint'] == 'max_86400_seconds':
                require(np.all(np.abs(times[edges[:, 0]]-times[edges[:, 1]]) <= 86400), 'Rewired 24-hour violation')
        else:
            # Native order is kept exactly, including floating-point accumulation order.
            time_metadata(spec, mapping['node_ids'])
            edges = original
        edge_index = g.edge_index.numpy() if spec['family'] in ['local', 'matched_full'] else symmetric(edges)
        require(np.array_equal(canonical(edge_index, n), edges), 'Invalid generated edge representation')
        topology_path = attempt / 'topology.npz'
        with (attempt / 'topology.tmp').open('wb') as f:
            np.savez_compressed(f, edge_index=edge_index)
        os.replace(attempt / 'topology.tmp', topology_path)
        deg_old, deg_new = degree(original, n), degree(edges, n)
        report = dict(original=stats(original, n), resulting=stats(edges, n),
            overlap=overlap(original, edges, n), exact_per_node_degree_equality=bool(np.array_equal(deg_old, deg_new)),
            isolated_node_ids_equal=bool(np.array_equal(deg_old == 0, deg_new == 0)),
            original_degree_sha256=array_sha(deg_old), resulting_degree_sha256=array_sha(deg_new),
            source_graph_sha256=cfg['reference']['sha256'], node_feature_label_split_preservation='Source tensors reused by pinned hash; local ablation selects recorded columns only.',
            density_change=2*(len(edges)-len(original))/(n*(n-1)), constraints=spec['time_constraint'],
            shared_identifier_rule_preserved=spec['family'] not in ['random', 'complement'],
            rewiring=diagnostics, runtime_seconds=time.perf_counter()-started)
        feature_hash = array_sha(g.x.numpy()[:, :93]) if spec['feature_set'] == 'local93' else cfg['reference']['feature_sha256']
        effective = dict(source_graph_sha256=cfg['reference']['sha256'], topology_file_sha256=sha(topology_path),
                         effective_feature_sha256=feature_hash, node_ids_sha256=cfg['reference']['node_ids_sha256'],
                         label_sha256=cfg['reference']['label_sha256'], split_id_sha256=cfg['reference']['split_id_sha256'])
        manifest = dict(status='complete', config=cfg, topology=artifact(topology_path), statistics=report,
                        effective_graph=effective, effective_graph_sha256=digest(effective),
                        source_graph=artifact(ROOT / cfg['reference']['path']), environment=environment())
        atomic_json(attempt / 'statistics.json', report)
        atomic_json(directory / 'manifest.json', manifest)
        print('Prepared:', graph_id(spec), flush=True)
    except BaseException as exc:
        atomic_json(attempt / 'failure.json', dict(status='failed', error=repr(exc), config=cfg))
        raise


def load_inputs(spec, manifest):
    import torch
    g, mapping, cfg = load_reference(spec)
    with np.load(verify_artifact(manifest['topology'])) as z:
        edge_index = z['edge_index'].copy()
    canonical(edge_index, g.num_nodes)
    g.edge_index = torch.from_numpy(edge_index)
    if spec['feature_set'] == 'local93':
        require(g.x.shape[1] == 165, 'Local ablation requires the verified 165-column matrix')
        included = np.array(lock()['feature_mapping']['local_model_raw_indices']) - 2
        g.x = g.x[:, torch.from_numpy(included)].contiguous()
        require(g.x.shape[1] == 93, 'Local feature dimension mismatch')
    require(array_sha(g.x.numpy()) == manifest['effective_graph']['effective_feature_sha256'], 'Effective input features mismatch')
    return g, mapping
