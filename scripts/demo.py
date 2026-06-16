#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrag.config import MODEL_REGISTRY
from medrag.data.schema import QAItem
from medrag.evaluation.evaluator import PIPELINES, get_retriever
from medrag.generation.factory import get_backend

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, choices=list(PIPELINES))
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--options",
        help='JSON dict of options, e.g. \'{"A": "...", "B": "..."}\'. Omit for free-form questions.',
        default=None,
    )
    args = parser.parse_args()

    options = json.loads(args.options) if args.options else None
    item = QAItem(id="demo", dataset="demo", question=args.question, options=options, answer="")

    backend = get_backend(args.model)
    retriever = get_retriever(args.pipeline)
    out = PIPELINES[args.pipeline].run(item, backend, retriever)

    print("\n=== Trajectory ===")
    print(json.dumps(out["trajectory"], indent=2))
    print("\n=== Answer ===")
    print(out["answer"])
