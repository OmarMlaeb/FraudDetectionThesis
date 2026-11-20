"""Simple undirected graph controls; no labels or model features are accepted."""
import math
import time
import numpy as np
from common import require, array_sha


def canonical(edge_index, n):
    e = np.asarray(edge_index, dtype=np.int64)
    require(e.ndim == 2 and e.shape[0] == 2, 'Expected edge_index shape (2,E)')
    require(not e.size or (e.min() >= 0 and e.max() < n), 'Invalid endpoint')
    require(not np.any(e[0] == e[1]), 'Stored self-loop found')
    keys = e[0] * n + e[1]
    require(len(np.unique(keys)) == len(keys), 'Duplicate directed edge found')
    require(np.array_equal(np.sort(keys), np.sort(e[1] * n + e[0])), 'Graph is not symmetric; refusing undirected control')
    edges = e[:, e[0] < e[1]].T
    return edges[np.lexsort((edges[:, 1], edges[:, 0]))]


def symmetric(edges):
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return np.concatenate([e.T, e[:, ::-1].T], axis=1)


def degree(edges, n):
    return np.bincount(np.asarray(edges).ravel(), minlength=n)


def stats(edges, n):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    d = degree(e, n)
    a = coo_matrix((np.ones(len(e), dtype=np.int8), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    return dict(nodes=n, unique_undirected_edges=len(e), stored_directed_columns=2*len(e),
                average_undirected_degree=2*len(e)/n if n else 0,
                density=2*len(e)/(n*(n-1)) if n > 1 else 0,
                isolated_nodes=int((d == 0).sum()), components=int(connected_components(a, directed=False, return_labels=False)),
                degree_sha256=array_sha(d), canonical_edge_sha256=array_sha(e))


def overlap(original, changed, n):
    a = original[:, 0]*n+original[:, 1]
    b = changed[:, 0]*n+changed[:, 1]
    intersection = len(np.intersect1d(a, b, assume_unique=True))
    union = len(a)+len(b)-intersection
    return dict(shared_edges=intersection, fraction_original_retained=intersection/len(a) if len(a) else None,
                jaccard=intersection/union if union else 1.0)


def complement(original, n, seed, average_degree=20):
    require(average_degree >= 0 and n >= 2, 'Invalid complement size')
    target = int(n * average_degree / 2)
    require(target * 2 == n * average_degree, 'Requested degree gives nonintegral edge count')
    forbidden = set((original[:, 0]*n+original[:, 1]).tolist())
    require(target <= n*(n-1)//2-len(forbidden), 'Not enough nonedges')
    rng = np.random.default_rng(seed)
    chosen = set()
    # Uniform unordered candidate pairs, rejection without replacement. No degree conditioning.
    while len(chosen) < target:
        batch = rng.integers(n, size=(min(262144, max(1024, target-len(chosen))), 2))
        for a, b in batch:
            if a == b:
                continue
            a, b = sorted((int(a), int(b)))
            k = a*n+b
            if k not in forbidden:
                chosen.add(k)
            if len(chosen) == target:
                break
    keys = np.array(sorted(chosen), dtype=np.int64)
    return np.stack([keys//n, keys % n], axis=1)


def rewire(original, n, seed, swaps_per_edge=10, attempts_per_edge=100, strata=None, times=None, max_delta=None):
    started = time.perf_counter()
    require(swaps_per_edge > 0 and attempts_per_edge > 0, 'Budgets must be positive')
    e = np.array(original, dtype=np.int64, copy=True)
    m = len(e)
    rng = np.random.default_rng(seed)
    keys = set((e[:, 0]*n+e[:, 1]).tolist())
    require(len(keys) == m and np.all(e[:, 0] < e[:, 1]), 'Input is not a simple canonical graph')
    groups = [np.arange(m)]
    if strata is not None:
        require(np.all(strata[e[:, 0]] == strata[e[:, 1]]), 'Reference crosses temporal strata')
        groups = [np.flatnonzero(strata[e[:, 0]] == s) for s in np.unique(strata[e[:, 0]])]
    if max_delta is not None:
        require(times is not None and np.all(np.abs(times[e[:, 0]]-times[e[:, 1]]) <= max_delta), 'Reference violates time limit')
    reports = []
    for ids in groups:
        target, limit = math.ceil(swaps_per_edge*len(ids)), math.ceil(attempts_per_edge*len(ids))
        accepted = attempts = 0
        while len(ids) >= 2 and accepted < target and attempts < limit:
            attempts += 1
            i, j = ids[rng.integers(len(ids), size=2)]
            if i == j:
                continue
            a, b = map(int, e[i]); c, d = map(int, e[j])
            if len({a, b, c, d}) != 4:
                continue
            if rng.integers(2):
                c, d = d, c
            p, q = tuple(sorted((a, c))), tuple(sorted((b, d)))
            pk, qk = p[0]*n+p[1], q[0]*n+q[1]
            if pk in keys or qk in keys or pk == qk:
                continue
            if max_delta is not None and (abs(times[p[0]]-times[p[1]]) > max_delta or abs(times[q[0]]-times[q[1]]) > max_delta):
                continue
            keys.remove(a*n+b)
            keys.remove(min(c, d)*n+max(c, d))
            keys.add(pk); keys.add(qk)
            e[i], e[j] = p, q
            accepted += 1
        reports.append(dict(stratum=None if strata is None else int(strata[e[ids[0], 0]]),
                            edges=len(ids), requested_swaps=target, max_attempts=limit,
                            successful_swaps=accepted, attempts=attempts, target_reached=accepted == target))
    e = e[np.lexsort((e[:, 1], e[:, 0]))]
    require(np.array_equal(degree(original, n), degree(e, n)), 'Degree sequence changed')
    require(np.array_equal(canonical(symmetric(e), n), e), 'Rewiring produced invalid topology')
    accepted = sum(r['successful_swaps'] for r in reports)
    attempts = sum(r['attempts'] for r in reports)
    ov = overlap(original, e, n)
    return e, dict(successful_swaps=accepted, attempts=attempts, acceptance_rate=accepted/attempts if attempts else 0,
                   requested_swaps=sum(r['requested_swaps'] for r in reports), max_attempts=sum(r['max_attempts'] for r in reports),
                   runtime_seconds=time.perf_counter()-started, strata=reports,
                   limited_randomization=any(not r['target_reached'] for r in reports) or (ov['fraction_original_retained'] is None or ov['fraction_original_retained'] >= .9),
                   limited_randomization_rule='Any stratum missed its requested swap budget, or at least 90% of original edges remain (predeclared descriptive flag).',
                   limitation='Finite constrained swaps; no guarantee of uniform sampling or mixing.', **ov)
