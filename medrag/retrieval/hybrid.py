from medrag.data.schema import Doc
from medrag.retrieval.bm25_index import BM25Retriever
from medrag.retrieval.dense_index import DenseRetriever


def reciprocal_rank_fusion(ranked_lists: list[list[Doc]], k: int = 60) -> list[Doc]:
    scores: dict[str, float] = {}
    doc_by_id: dict[str, Doc] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (k + rank + 1)
            doc_by_id[doc.id] = doc
    ordered_ids = sorted(scores, key=lambda i: scores[i], reverse=True)
    return [doc_by_id[i] for i in ordered_ids]


class HybridRetriever:
    def __init__(self):
        self.dense = DenseRetriever()
        self.bm25 = BM25Retriever()

    def search(self, query: str, top_k: int) -> list[Doc]:
        dense_docs = [d for d, _ in self.dense.search(query, top_k)]
        bm25_docs = [d for d, _ in self.bm25.search(query, top_k)]
        fused = reciprocal_rank_fusion([dense_docs, bm25_docs])
        return fused[:top_k]
