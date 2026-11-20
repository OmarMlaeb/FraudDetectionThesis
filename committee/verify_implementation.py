"""Reproducible lightweight implementation verification; NEVER trains or generates full graphs."""
import ast
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from plan import matrix, families


def main():
    dest = OUT / 'implementation_verification'
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(HERE), '-p', 'test_*.py', '-v'], cwd=ROOT, text=True, capture_output=True)
    atomic_bytes(dest / 'synthetic_tests.txt', (result.stdout+result.stderr).encode())
    require(result.returncode == 0, 'Synthetic verification failed; see implementation_verification/synthetic_tests.txt')
    verify_lock_files(list(lock()['files']))
    preserved = []
    for r in read(AUDIT / 'artifact_hashes_before.json'):
        if r.get('sha256'):
            actual = sha(ROOT / r['path'])
            require(actual == r['sha256'], f'Original artifact changed: {r["path"]}')
            preserved.append(dict(path=r['path'], sha256=actual))
    # Compare completed supervisor/audit package members with their current files.
    package_preserved = []
    with zipfile.ZipFile(ROOT / 'analysis/supervisor_analyses_2026-10-13.zip') as z:
        for name in z.namelist():
            if name.startswith(('analysis/supervisor_2026-10-13/', 'audit/')) and not name.endswith('/'):
                path = ROOT / name
                require(path.is_file(), f'Packaged prior analysis is missing: {name}')
                expected = hashlib.sha256(z.read(name)).hexdigest()
                require(sha(path) == expected, f'Completed prior analysis changed: {name}')
                package_preserved.append(dict(path=name, sha256=expected))
    before = AUDIT / 'source_snapshots/2be96ee/src/ieee_cis/train_gnn.py'
    tagged = AUDIT / 'source_snapshots/5333c55/src/ieee_cis/train_gnn.py'
    def loop(path):
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
        return ast.dump(next(n for n in fn.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch'))
    require(loop(before) == loop(tagged), 'IEEE training epoch loop differs across snapshots')
    comparison = []
    for p in (AUDIT/'source_snapshots/2be96ee/src').rglob('*.py'):
        counterpart = AUDIT/'source_snapshots/5333c55/src'/p.relative_to(AUDIT/'source_snapshots/2be96ee/src')
        comparison.append(dict(file=p.relative_to(AUDIT/'source_snapshots/2be96ee/src').as_posix(),
                               earlier_sha256=sha(p), tagged_sha256=sha(counterpart), identical=sha(p) == sha(counterpart)))
    atomic_json(dest/'snapshot_comparison.json', dict(files=comparison, ieee_epoch_loop_ast_equal=True,
                difference_review='IEEE graph_data adds construction support. IEEE train_gnn adds graph selection, naming and output arguments; training epoch loop is identical. All architecture, Elliptic training, preprocessing, sequence and common metric files are byte-identical.'))
    atomic_json(dest/'preservation.json', dict(original_artifacts=preserved, completed_analysis_package_files=package_preserved,
                original_artifacts_unchanged=True, completed_analyses_unchanged=True))
    env = environment()
    atomic_json(dest/'environment.json', env)
    atomic_json(dest/'experiment_matrix.json', matrix())
    atomic_json(dest/'verification.json', dict(synthetic_test_exit_code=result.returncode,
                test_log='synthetic_tests.txt', expected_new_runs=len(matrix()), expected_graph_artifacts=9,
                expected_original_references=len(lock()['reference_runs']), planned_hypotheses=sum(len(f['comparisons']) for f in families()),
                planned_families=len(families()), original_artifacts_checked=len(preserved), prior_analysis_files_checked=len(package_preserved),
                training_executed=False, full_dataset_graph_generation_executed=False,
                verification_scope='Synthetic tests, CPU forward passes, mocked lifecycle/prediction fixtures, original file hash checks; no optimizer steps.',
                implementation_code_hashes=code_hashes()))
    print(f'Verification passed: {len(preserved)} original artifacts and {len(package_preserved)} prior-analysis files unchanged. No training or full graph generation.')


if __name__ == '__main__':
    main()
