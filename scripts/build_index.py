#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrag.data.load_corpus import load_corpus
from medrag.retrieval import bm25_index, dense_index

if __name__ == "__main__":
    docs = load_corpus()
    print(f"Loaded {len(docs)} docs from corpus")
    dense_index.build(docs)
    bm25_index.build(docs)
