"""Synthetic verification of the simple commands and old-style CSVs; no training."""
import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simple_runner as simple
from common import ROOT, atomic_json
from plan import matrix


def fake_manifest():
    old = dict(nodes=40, unique_undirected_edges=20, average_undirected_degree=1,
               density=1/39, isolated_nodes=2, components=3)
    new = dict(nodes=40, unique_undirected_edges=400, stored_directed_columns=800,
               average_undirected_degree=20, density=20/39, isolated_nodes=0, components=1)
    return dict(statistics=dict(original=old, resulting=new, density_change=19/39,
        rewiring={}, exact_per_node_degree_equality=False, isolated_node_ids_equal=False,
        overlap=dict(shared_edges=0, jaccard=0), constraints="none", runtime_seconds=.1,
        source_graph_sha256="synthetic-source"), topology=dict(sha256="synthetic-topology"), effective_graph_sha256="synthetic-effective")


class SimpleScriptTests(unittest.TestCase):
    def test_exact_original_csv_header(self):
        with (ROOT / "results/model_results.csv").open(newline="", encoding="utf-8-sig") as f:
            self.assertEqual(simple.RESULT_COLUMNS, next(csv.reader(f)))

    def test_dry_run_has_no_work_or_writes(self):
        for family, count in [("complement", 60), ("local", 60), ("random", 180)]:
            output = io.StringIO()
            with patch.object(simple, "environment") as env, patch.object(simple, "generate") as generate, \
                 patch.object(simple, "train_one") as train, patch.object(simple, "export_results") as export, \
                 contextlib.redirect_stdout(output):
                self.assertEqual(simple.run_experiment(family, ["--dry-run"]), 0)
                self.assertIn(f"Runs: {count}", output.getvalue())
                env.assert_not_called(); generate.assert_not_called(); train.assert_not_called(); export.assert_not_called()

    def run_mocked(self, directory, fail_first=False, graph_only=False, repeat=False):
        rows = matrix()[:2]
        completed = {}

        def fake_train(row, settings, retry_failed):
            self.assertTrue(retry_failed)
            if fail_first and row["seed"] == 42:
                raise RuntimeError("Synthetic failure, no training")
            folder = directory / "runs/complement" / row["run_id"]
            atomic_json(folder / "config.json", {"created_utc": "synthetic-test"})
            atomic_json(folder / "status.json", {"config": {"path": str(folder / "config.json")}})
            completed[row["run_id"]] = dict({key: .5 for key in simple.RESULT_COLUMNS if key not in ["timestamp", "model"]},
                run_id=row["run_id"])
            completed[row["run_id"]]["seed"] = row["seed"]

        def fake_validate(selected, write=False):
            report = {"runs": [{"run_id": row["run_id"], "status": "accepted" if row["run_id"] in completed else "missing"} for row in selected]}
            return [completed[row["run_id"]] for row in selected if row["run_id"] in completed], report

        with patch.object(simple, "OUT", directory), patch.object(simple, "WorkspaceLock", return_value=contextlib.nullcontext()), \
             patch.object(simple, "matrix", return_value=rows), patch.object(simple, "environment"), \
             patch.object(simple, "generate") as generate, patch.object(simple, "read_graph_manifest", return_value=fake_manifest()), \
             patch.object(simple, "train_one", side_effect=fake_train) as train, \
             patch.object(simple, "validate", side_effect=fake_validate), \
             patch.object(simple, "finish_reports", return_value=not fail_first) as finish, contextlib.redirect_stdout(io.StringIO()):
            code = simple.run_experiment("complement", ["--graphs-only"] if graph_only else [])
            if repeat:
                self.assertEqual(simple.run_experiment("complement", []), 0)
            self.assertEqual(generate.call_count, 2 if repeat else 1)
            if graph_only:
                train.assert_not_called(); finish.assert_not_called()
            else:
                self.assertEqual(train.call_count, 4 if repeat else 2)
        return code

    def test_one_command_prepares_trains_and_saves_unique_csv_rows(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            self.assertEqual(self.run_mocked(directory, repeat=True), 0)
            with (directory / "sampled_complement_results.csv").open(newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, simple.RESULT_COLUMNS)
                rows = list(reader)
            self.assertEqual([r["seed"] for r in rows], ["42", "43"])
            self.assertEqual(rows[0]["model"], "GCN-main-sampled_complement")
            with (directory / "sampled_complement_graph_statistics.csv").open(newline="") as f:
                stats = list(csv.DictReader(f))
            self.assertEqual(len(stats), 1)
            self.assertEqual(float(stats[0]["new_average_degree"]), 20)
            self.assertEqual(float(stats[0]["new_density"]), 20/39)

    def test_failure_continues_and_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            self.assertEqual(self.run_mocked(directory, fail_first=True), 1)
            with (directory / "sampled_complement_results.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([row["seed"] for row in rows], ["43"])
            with (directory / "sampled_complement_failures.csv").open(newline="") as f:
                failures = list(csv.DictReader(f))
            self.assertEqual(len(failures), 1)
            self.assertIn("Synthetic failure", failures[0]["error"])

    def test_graph_only_never_trains(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            self.assertEqual(self.run_mocked(directory, graph_only=True), 0)
            with (directory / "sampled_complement_results.csv").open(newline="") as f:
                self.assertEqual(list(csv.DictReader(f)), [])
            self.assertTrue((directory / "sampled_complement_graph_statistics.csv").is_file())

    def test_local_model_label(self):
        row = next(row for row in matrix() if row["family"] == "local")
        self.assertEqual(simple.model_label(row), "Elliptic-MLP-local_features")


if __name__ == "__main__":
    unittest.main(verbosity=2)
