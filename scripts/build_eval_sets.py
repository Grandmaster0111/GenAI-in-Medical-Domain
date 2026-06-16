#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrag.data.load_eval_sets import build_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-dataset", type=int, default=50)
    args = parser.parse_args()
    build_all(args.n_per_dataset)
