"""I/O and immutable provenance helpers; importing this module never runs experiments."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AUDIT = ROOT / 'analysis/supervisor_2026-10-13'
OUT = ROOT / 'results/committee_additional_experiments'
PYTHON = AUDIT / 'runtime/python310/python.exe'
SEEDS = list(range(42, 52))
GNN = ['gcn', 'sage', 'gat']
MODELS = ['mlp', 'lstm', 'transformer'] + GNN
CONSTRUCTIONS = ['main', 'card', 'address', 'email_domain', 'identifier_time_window-24h']
LABELS = dict(mlp='MLP', lstm='LSTM', transformer='Transformer', gcn='GCN', sage='GraphSAGE', gat='GAT')
METRICS = ['pr_auc', 'f1_score', 'recall', 'precision', 'roc_auc', 'false_positive']
COUNTS = ['true_negative', 'false_positive', 'false_negative', 'true_positive']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def array_sha(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, obj):
    atomic_bytes(path, (json.dumps(obj, indent=2, allow_nan=False) + '\n').encode())


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def artifact(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def verify_artifact(record):
    p = Path(record['path'])
    require(p.is_file(), f'Missing artifact: {p}')
    require(sha(p) == record['sha256'], f'Hash mismatch: {p}')
    return p


def lock():
    return read(HERE / 'reference_lock.json')


def verify_lock_files(paths):
    records = lock()['files']
    for p in paths:
        require(p in records, f'Unpinned reference: {p}')
        require(sha(ROOT / p) == records[p], f'Frozen reference changed: {p}')


def frozen(name):
    p = HERE / 'frozen' / (name + '.py')
    verify_lock_files([p.relative_to(ROOT).as_posix()])
    spec = importlib.util.spec_from_file_location('committee_frozen_' + name, p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def code_hashes():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(HERE.rglob('*'))
            if p.suffix in {'.py', '.json'} and '__pycache__' not in p.parts}


def environment(check=True):
    import numpy, pandas, scipy, sklearn, torch, torch_geometric
    packages = {m.__name__: m.__version__ for m in [numpy, pandas, scipy, sklearn, torch, torch_geometric]}
    expected = lock()['packages']
    if check:
        require(packages == expected, f'Package mismatch. Expected {expected}, got {packages}; do not silently change the experiment environment.')
        require(platform.python_version() == '3.10.0', 'Use the pinned Python 3.10.0 interpreter.')
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    return {'python': platform.python_version(), 'executable': sys.executable,
            'packages': packages, 'git_commit': git('rev-parse', 'HEAD'),
            'git_status': git('status', '--porcelain'), 'git_diff_sha256': hashlib.sha256(git('diff', 'HEAD').encode()).hexdigest(),
            'code_hashes': code_hashes(), 'platform': platform.platform(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()}


def reference(row):
    key = row['reference_run_id']
    path = AUDIT / 'run_manifest.json'
    verify_lock_files([path.relative_to(ROOT).as_posix()])
    r = next(r for r in read(path) if r['run_id'] == key)
    require(r['checkpoint_reproduction'] == 'accepted', f'Unverified reference {key}')
    paths = [str((AUDIT / r['reproduction_check']).relative_to(ROOT)).replace('\\', '/'),
             str((AUDIT / r['prediction_file']).relative_to(ROOT)).replace('\\', '/')]
    verify_lock_files(paths)
    require(read(AUDIT / r['reproduction_check'])['accepted'], f'Reference reproduction failed {key}')
    return r


def rerun_command(row, settings):
    return (f"& '{PYTHON}' committee/cli.py train --family {row['family']} --dataset {row['dataset']} "
            f"--construction {row['construction']} --model {row['model']} --seed {row['seed']} "
            f"--graph-seed {settings['graph_seed']} --swaps-per-edge {settings['swaps_per_edge']} "
            f"--attempts-per-edge {settings['attempts_per_edge']} --retry-failed")
