"""Streams a small sample of the MedRAG/pubmed corpus and saves it locally as jsonl."""
import json
import random

from datasets import load_dataset

from medrag.config import CORPUS_PATH
from medrag.data.schema import Doc


def build_corpus(n_samples: int, seed: int = 42) -> None:
    print(f"Streaming MedRAG/pubmed and sampling {n_samples} snippets (seed={seed})...")
    ds = load_dataset("MedRAG/pubmed", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=min(max(n_samples * 5, 2000), 20000))

    docs = []
    for row in ds:
        if len(docs) >= n_samples:
            break
        text = row.get("content") or row.get("contents") or ""
        if not text:
            continue
        docs.append(Doc(id=str(row.get("id") or row.get("PMID") or len(docs)), title=row.get("title", ""), text=text))

    with open(CORPUS_PATH, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc.to_dict()) + "\n")
    print(f"Wrote {len(docs)} docs -> {CORPUS_PATH}")


def load_corpus() -> list[Doc]:
    docs = []
    with open(CORPUS_PATH) as f:
        for line in f:
            docs.append(Doc.from_dict(json.loads(line)))
    return docs
