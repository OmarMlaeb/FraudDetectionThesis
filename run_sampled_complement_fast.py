"""Run all ten seeds with memory-efficient sparse graph operations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'committee_fast'))
from runtime import run

if __name__ == '__main__':
    sys.exit(run('complement'))
