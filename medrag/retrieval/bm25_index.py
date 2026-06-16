import pickle

from rank_bm25 import BM25Okapi

from medrag.config import INDEX_DIR
from medrag.data.schema import Doc

BM25_PATH = INDEX_DIR / "bm25.pkl"


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def build(docs: list[Doc]) -> None:
    corpus = [_tokenize(f"{d.title} {d.text}") for d in docs]
    bm25 = BM25Okapi(corpus)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": docs}, f)
    print(f"Built BM25 index with {len(docs)} docs -> {BM25_PATH}")


class BM25Retriever:
    def __init__(self):
        with open(BM25_PATH, "rb") as f:
            data = pickle.load(f)
        self.bm25: BM25Okapi = data["bm25"]
        self.docs: list[Doc] = data["docs"]

    def search(self, query: str, top_k: int) -> list[tuple[Doc, float]]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.docs[i], float(scores[i])) for i in ranked]
