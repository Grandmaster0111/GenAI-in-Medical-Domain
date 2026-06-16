#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrag.config import MODEL_REGISTRY
from medrag.data.load_eval_sets import LOADERS
from medrag.evaluation.evaluator import PIPELINES, run_eval
from medrag.generation.factory import get_backend

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, choices=list(PIPELINES))
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--dataset", required=True, choices=list(LOADERS))
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    backend = get_backend(args.model)
    run_eval(args.pipeline, args.model, args.dataset, backend, n=args.n)
