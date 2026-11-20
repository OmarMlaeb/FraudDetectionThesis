"""Separate sparse execution, preserving the original implementation and artifacts.

Only this process installs adapters. Graph generation, the training loop, seeds,
hyperparameters, checkpoint selection and statistical analyses remain frozen.
Completed original runs are validated using their ORIGINAL code hashes. New runs
are validated using BOTH the frozen code and this adapter's hashes.
"""
import contextlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'committee'), str(Path(__file__).resolve().parent)]
import common
import runner
import postprocess
import simple_runner
from sparse_backend import accelerate, BACKEND

ORIGINAL_OUT = common.OUT
FAST_OUT = ROOT / 'results/committee_fast_experiments'
CHECKS = ROOT / 'results/committee_runtime_checks'
REPORT_NAMES = ['synthetic_equivalence.json', 'saved_checkpoint_equivalence.json',
                'reference_checkpoint_equivalence.json']


def fast_hashes():
    hashes = common.code_hashes()
    files = list(Path(__file__).parent.glob('*.py')) + list(ROOT.glob('run_*_fast.py'))
    hashes.update({p.relative_to(ROOT).as_posix(): common.sha(p) for p in sorted(files)})
    return hashes


def verification():
    backend_hash = common.sha(Path(__file__).with_name('sparse_backend.py'))
    manifest = FAST_OUT / 'verification' / ('manifest_' + backend_hash + '.json')
    if manifest.exists():
        evidence = common.read(manifest)
        common.require(len(evidence) == len(REPORT_NAMES), 'Incomplete verification snapshot')
        for artifact in evidence:
            report = common.read(common.verify_artifact(artifact))
            common.require(report['passed'] and report['optimizer_steps'] == 0
                           and report['backend_sha256'] == backend_hash, 'Invalid verification snapshot')
        return evidence
    evidence = []
    for name in REPORT_NAMES:
        source = CHECKS / name
        report = common.read(source)
        common.require(report['passed'] and report['optimizer_steps'] == 0,
                       f'Sparse equivalence verification did not pass: {source}')
        common.require(report['backend_sha256'] == backend_hash,
                       f'Backend changed since verification: {source}')
        # Keep immutable evidence even if the user repeats a diagnostic later.
        destination = FAST_OUT / 'verification' / (common.sha(source) + '.json')
        if not destination.exists():
            common.atomic_bytes(destination, source.read_bytes())
        common.require(common.sha(destination) == common.sha(source), 'Verification snapshot changed')
        evidence.append(common.artifact(destination))
    common.atomic_json(manifest, evidence)
    return evidence


def source_for(row):
    relative = Path('runs') / row['family'] / row['run_id'] / 'status.json'
    old, new = ORIGINAL_OUT / relative, FAST_OUT / relative
    old_complete = old.exists() and common.read(old)['status'] == 'complete'
    new_complete = new.exists() and common.read(new)['status'] == 'complete'
    common.require(not (old_complete and new_complete),
                   f'Two completed versions of {row["run_id"]}; refusing to choose silently.')
    if old_complete or (old.exists() and not new.exists()):
        return ORIGINAL_OUT
    return FAST_OUT


@contextlib.contextmanager
def installed():
    """Scoped adapters allow ordinary validation/tests to retain original behavior."""
    changes = []
    old_validate = postprocess.validate
    old_train = runner.train_one
    old_factory = runner.model_factory
    old_compatibility = runner.compatibility

    def set_value(module, name, value):
        changes.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    def environment():
        env = common.environment()
        env.update(code_hashes=fast_hashes(), execution_backend=BACKEND,
                   sparse_verification=verification())
        return env

    def compatibility(row, graph, env, ref):
        result = old_compatibility(row, graph, env, ref)
        result.update(execution_backend=BACKEND if row['model'] in common.GNN else 'original_non_graph',
                      sparse_verification=env['sparse_verification'])
        return result

    def validate(rows, write=True):
        accepted, checked = [], []
        groups = {ORIGINAL_OUT: [], FAST_OUT: []}
        for row in rows:
            try:
                groups[source_for(row)].append(row)
            except Exception as exc:
                checked.append(dict(run_id=row['run_id'], family=row['family'], status='invalid', error=str(exc)))
        saved_out, saved_hashes = postprocess.OUT, postprocess.code_hashes
        try:
            for directory, selected in groups.items():
                if not selected:
                    continue
                postprocess.OUT = directory
                postprocess.code_hashes = common.code_hashes if directory == ORIGINAL_OUT else fast_hashes
                metrics, report = old_validate(selected, write=False)
                valid = {m['run_id']: m for m in metrics}
                for entry in report['runs']:
                    if entry['status'] != 'accepted':
                        checked.append(entry)
                        continue
                    metric = valid[entry['run_id']]
                    status_path = directory / 'runs' / metric['family'] / metric['run_id'] / 'status.json'
                    status = common.read(status_path)
                    comp = common.read(status['config']['path'])['compatibility']
                    backend = 'original_edge_index'
                    if directory == FAST_OUT:
                        try:
                            backend = BACKEND if metric['model'] in common.GNN else 'original_non_graph'
                            common.require(comp['execution_backend'] == backend, 'Wrong execution backend')
                            for evidence in comp['sparse_verification']:
                                path = common.verify_artifact(evidence)
                                evidence_report = common.read(path)
                                common.require(evidence_report['passed'] and evidence_report['backend_sha256'] == common.sha(Path(__file__).with_name('sparse_backend.py')), 'Invalid backend evidence')
                        except Exception as exc:
                            entry.update(status='invalid', error=str(exc))
                            checked.append(entry)
                            continue
                    metric.update(execution_backend=backend, execution_status_path=str(status_path))
                    entry['execution_backend'] = backend
                    accepted.append(metric)
                    checked.append(entry)
        finally:
            postprocess.OUT, postprocess.code_hashes = saved_out, saved_hashes
        report = dict(expected=len(rows), accepted=len(accepted), complete=len(accepted) == len(rows), runs=checked)
        if write:
            common.atomic_json(FAST_OUT / 'analysis/validation.json', report)
        return accepted, report

    def train_one(row, settings, retry_failed=False):
        if source_for(row) == ORIGINAL_OUT:
            status = common.read(ORIGINAL_OUT / 'runs' / row['family'] / row['run_id'] / 'status.json')
            if status['status'] == 'complete':
                accepted, report = validate([row], write=False)
                common.require(len(accepted) == 1, str(report))
                print('Verified original result; skipping:', row['run_id'], flush=True)
                return 'skipped'
        return old_train(row, settings, retry_failed=retry_failed)

    def export_results(family, rows, accepted):
        records, provenance = [], []
        for row in rows:
            if row['run_id'] not in accepted:
                continue
            metric = accepted[row['run_id']]
            status = common.read(metric['execution_status_path'])
            config = common.read(status['config']['path'])
            record = {key: metric[key] for key in simple_runner.RESULT_COLUMNS if key not in ['timestamp', 'model']}
            record.update(timestamp=config['created_utc'], model=simple_runner.model_label(row))
            records.append(record)
            provenance.append(dict(run_id=row['run_id'], seed=row['seed'], model=record['model'],
                                   execution_backend=metric['execution_backend'],
                                   status_path=metric['execution_status_path'],
                                   config_sha256=status['config']['sha256'],
                                   checkpoint_sha256=status['artifacts']['checkpoint']['sha256']))
        name = simple_runner.EXPERIMENTS[family]['name']
        path = FAST_OUT / (name + '_results.csv')
        simple_runner.save_csv(path, records, simple_runner.RESULT_COLUMNS)
        simple_runner.save_csv(FAST_OUT / (name + '_execution_provenance.csv'), provenance,
                               ['run_id', 'seed', 'model', 'execution_backend', 'status_path', 'config_sha256', 'checkpoint_sha256'])
        return path

    try:
        set_value(runner, 'OUT', FAST_OUT)
        set_value(runner, 'model_factory', lambda name, dataset, dimension: accelerate(old_factory(name, dataset, dimension), name))
        set_value(runner, 'environment', environment)
        set_value(runner, 'compatibility', compatibility)
        set_value(runner, 'rerun_command', lambda row, settings: 'python ' + simple_runner.EXPERIMENTS[row['family']]['script'])
        set_value(postprocess, 'OUT', FAST_OUT)
        set_value(postprocess, 'validate', validate)
        set_value(simple_runner, 'OUT', FAST_OUT)
        set_value(simple_runner, 'environment', environment)
        set_value(simple_runner, 'validate', validate)
        set_value(simple_runner, 'train_one', train_one)
        set_value(simple_runner, 'export_results', export_results)
        set_value(simple_runner, 'EXPERIMENTS', {k:dict(v, script=v['script'].replace('.py', '_fast.py')) for k,v in simple_runner.EXPERIMENTS.items()})
        # Graph artifacts and the exclusive lock remain in the ORIGINAL folder.
        # No model, checkpoint, result or graph file is overwritten there.
        yield
    finally:
        for module, name, value in reversed(changes):
            setattr(module, name, value)


def run(family, argv=None):
    with installed():
        print('Sparse execution | Same ten seeds, graphs and training settings')
        print('Completed original runs are verified and reused. New runs are saved separately.')
        return simple_runner.run_experiment(family, argv)
