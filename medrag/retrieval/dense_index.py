import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from medrag.config import EMBEDDING_MODEL, INDEX_DIR
from medrag.data.schema import Doc

FAISS_PATH = INDEX_DIR / "dense.faiss"
DOCS_PATH = INDEX_DIR / "dense_docs.pkl"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def build(docs: list[Doc]) -> None:
    model = _get_model()
    texts = [f"{d.title} {d.text}".strip() for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(FAISS_PATH))
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)
    print(f"Built dense index with {len(docs)} docs -> {FAISS_PATH}")


class DenseRetriever:
    def __init__(self):
        self.index = faiss.read_index(str(FAISS_PATH))
        with open(DOCS_PATH, "rb") as f:
            self.docs: list[Doc] = pickle.load(f)
        self.model = _get_model()

    def search(self, query: str, top_k: int) -> list[tuple[Doc, float]]:
        emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores, idxs = self.index.search(emb, top_k)
        return [(self.docs[i], float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]
