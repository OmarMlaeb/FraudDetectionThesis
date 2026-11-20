"""Run GCN, GraphSAGE and GAT on sampled complements, seeds 42 through 51."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "committee"))
from simple_runner import run_experiment


if __name__ == "__main__":
    sys.exit(run_experiment("complement"))
