"""Committee experiment CLI: planning, generation and training are separate commands."""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from plan import matrix, select, settings


class WorkspaceLock:
    def __init__(self):
        self.path = OUT / '.active.lock'
    def __enter__(self):
        OUT.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open('x', encoding='utf-8') as f:
                json.dump(dict(pid=os.getpid(), command=sys.argv), f)
        except FileExistsError:
            raise RuntimeError(f'Another command or a stale lock exists: {self.path}. Inspect the PID; use unlock only after the previous process has ended.')
        return self
    def __exit__(self, *args):
        self.path.unlink()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['list', 'generate', 'train', 'validate', 'analyze', 'package', 'preflight', 'unlock'])
    p.add_argument('--family', choices=['all', 'complement', 'local', 'random', 'matched_full'], default='all')
    p.add_argument('--dataset', choices=['ieee', 'elliptic'])
    p.add_argument('--construction', choices=CONSTRUCTIONS+['native'])
    p.add_argument('--model', choices=MODELS)
    p.add_argument('--seed', type=int, choices=SEEDS)
    p.add_argument('--graph-seed', type=int, default=20261013)
    p.add_argument('--swaps-per-edge', type=float, default=10.0)
    p.add_argument('--attempts-per-edge', type=float, default=100.0)
    p.add_argument('--retry-failed', action='store_true', help='Restart an incomplete run from its seed in a NEW attempt directory; completed runs are hash-checked.')
    p.add_argument('--continue-on-error', action='store_true', help='Record a failure and continue sequentially; overall exit code remains nonzero.')
    p.add_argument('--confirm-no-running-process', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    s = settings(args.graph_seed, args.swaps_per_edge, args.attempts_per_edge)
    rows = select(matrix(s, optional=args.family == 'matched_full'), args)
    require(rows, 'No runs match these filters')
    if args.action == 'list':
        for row in rows:
            print(row['run_id'])
        print(f'{len(rows)} runs; {len({r["graph_id"] for r in rows})} graph artifacts required. This command performs no generation or training.')
        return 0
    if args.action == 'preflight':
        env = environment()
        verify_lock_files(['analysis/supervisor_2026-10-13/run_manifest.json'])
        print(json.dumps(dict(environment=env, selected_runs=len(rows), training=False), indent=2))
        return 0
    if args.action == 'unlock':
        require(args.confirm_no_running_process, 'Inspect the PID in .active.lock, then pass --confirm-no-running-process only if it has ended.')
        p = OUT / '.active.lock'
        require(p.resolve().parent == OUT.resolve(), 'Invalid lock location')
        if p.exists():
            print('Removing explicitly acknowledged stale lock:', p.read_text())
            p.unlink()
        return 0
    with WorkspaceLock():
        if args.action == 'generate':
            from graphs import generate
            environment()
            seen = set()
            for row in rows:
                if row['graph_id'] not in seen:
                    print('Preparing:', row['graph_id'], flush=True)
                    generate(row['graph_spec']); seen.add(row['graph_id'])
        elif args.action == 'train':
            from runner import train_one
            failures = []
            for row in rows:
                try:
                    train_one(row, s, args.retry_failed)
                except Exception as exc:
                    failure = dict(run_id=row['run_id'], error=repr(exc), rerun_command=rerun_command(row, s))
                    print(json.dumps(failure), flush=True)
                    failures.append(failure)
                    atomic_json(OUT / 'last_runner_failures.json', failures)
                    if not args.continue_on_error:
                        raise
            return 1 if failures else 0
        elif args.action == 'validate':
            from postprocess import validate
            _, report = validate(rows)
            print(f"Validated {report['accepted']}/{report['expected']}; complete={report['complete']}")
            return 0 if report['complete'] else 2
        elif args.action == 'analyze':
            require(args.family == 'all' and not any([args.dataset, args.construction, args.model, args.seed]), 'Analyze always uses the full 300-run matrix and complete comparison families; omit selection filters.')
            from postprocess import analyze
            report = analyze(s)
            return 0 if report['complete'] else 2
        elif args.action == 'package':
            from postprocess import package
            package()
    return 0


if __name__ == '__main__':
    sys.exit(main())
