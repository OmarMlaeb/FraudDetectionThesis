"""The preregistered matrix and statistical families (no data loading)."""
from common import *


def settings(graph_seed=20261013, swaps_per_edge=10.0, attempts_per_edge=100.0):
    require(graph_seed >= 0 and swaps_per_edge > 0 and attempts_per_edge > 0, 'Positive swap/attempt budgets and nonnegative graph seed required.')
    return dict(graph_seed=graph_seed, swaps_per_edge=swaps_per_edge,
                attempts_per_edge=attempts_per_edge, complement_average_degree=20,
                algorithm_version=1)


def graph_spec(family, dataset, construction, s):
    local = family == 'local'
    return dict(family=family, dataset=dataset, construction=construction,
                feature_set='local93' if local else 'full',
                graph_variant={'local': 'native', 'complement': 'sampled_nonedge_replacement', 'random': 'degree_preserving_swaps', 'matched_full': 'native'}[family],
                graph_seed=None if family in ['local', 'matched_full'] else s['graph_seed'],
                average_degree=20 if family == 'complement' else None,
                swaps_per_edge=s['swaps_per_edge'] if family == 'random' else None,
                attempts_per_edge=s['attempts_per_edge'] if family == 'random' else None,
                time_constraint=('within_time_step' if dataset == 'elliptic' else 'max_86400_seconds' if construction == 'identifier_time_window-24h' else 'none') if family == 'random' else 'none',
                algorithm_version=s['algorithm_version'])


def graph_id(spec):
    gs = spec['graph_seed'] if spec['graph_seed'] is not None else 'none'
    return f"{spec['dataset']}__{spec['construction']}__{spec['feature_set']}__{spec['graph_variant']}__gseed{gs}__{digest(spec)[:12]}"


def matrix(s=None, optional=False):
    s = s or settings()
    rows = []
    groups = [('complement', d, 'main' if d == 'ieee' else 'native', GNN) for d in ['ieee', 'elliptic']]
    groups += [('local', 'elliptic', 'native', MODELS)]
    groups += [('random', 'ieee', c, GNN) for c in CONSTRUCTIONS] + [('random', 'elliptic', 'native', GNN)]
    if optional:
        groups += [('matched_full', 'elliptic', 'native', MODELS)]
    for family, dataset, construction, models in groups:
        spec = graph_spec(family, dataset, construction, s)
        for model in models:
            for seed in SEEDS:
                ds = 'IEEE-CIS' if dataset == 'ieee' else 'Elliptic'
                label = LABELS[model] + (f'-{construction}' if dataset == 'ieee' and model in GNN else '')
                row = dict(family=family, dataset=dataset, model=model, construction=construction,
                           feature_set=spec['feature_set'], graph_seed=spec['graph_seed'], seed=seed,
                           graph_id=graph_id(spec), graph_spec=spec, reference_run_id=f'{ds}__{label}__seed{seed}')
                row['run_id'] = row['graph_id'] + f'__{model}__trainseed{seed}'
                rows.append(row)
    return rows


def select(rows, args):
    return [r for r in rows if all(getattr(args, k, None) in [None, 'all', r[k]] for k in ['family', 'dataset', 'construction', 'model', 'seed'])]


def training_config(model):
    gnn = model in GNN
    return dict(epochs=100 if gnn else 50, patience=15 if gnn else 10,
                learning_rate=0.001, weight_decay=1e-5, optimizer='Adam',
                loss='BCEWithLogitsLoss', pos_weight='train_negative/train_positive',
                batch_size=None if gnn else 1024 if model == 'mlp' else 512,
                shuffle=False, num_workers=0, selection_metric='pr_auc', min_delta=0.0,
                threshold_strategy='f1', threshold_grid='numpy.arange(0.01,1.00,0.01)',
                threshold_tiebreak=['f1', 'recall', 'precision', 'first_lower_threshold'],
                sequence_length=10 if model in ['lstm', 'transformer'] else None,
                gradient_clip=1.0 if model == 'transformer' else None,
                deterministic_cudnn=True, cudnn_benchmark=False,
                mixed_precision=False, full_graph_transductive=gnn,
                model_defaults='frozen August architecture; only input dimension changes')


def families():
    definitions = []
    groups = [('complement', 'ieee', ['main'], GNN), ('complement', 'elliptic', ['native'], GNN),
              ('local', 'elliptic', ['native'], MODELS),
              ('random', 'ieee', CONSTRUCTIONS, GNN), ('random', 'elliptic', ['native'], GNN)]
    for family, dataset, constructions, models in groups:
        for metric in METRICS:
            definitions.append(dict(id=f'{family}__{dataset}__{metric}', family=family, dataset=dataset,
                metric=metric, comparisons=[dict(construction=c, model=m) for c in constructions for m in models],
                difference='new minus verified original', expected_seeds=SEEDS, alternative='two-sided',
                zero_method='wilcox: discard exact zeros', ties='average ranks of absolute differences',
                method='exhaustive sign permutation; 2*lower tail of min(W+,W-), capped at 1; all zero p=1',
                correction='Holm over entire planned family; incomplete comparisons use p=1 only for adjustment',
                incomplete='No inferential p-value until all ten pairs present', alpha=0.05))
    return definitions
