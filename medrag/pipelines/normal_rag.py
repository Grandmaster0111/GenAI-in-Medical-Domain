from medrag.config import RETRIEVAL_TOP_K
from medrag.data.schema import QAItem
from medrag.generation.base import GenerationBackend
from medrag.pipelines.prompts import answer_prompt
from medrag.retrieval.dense_index import DenseRetriever


def run(item: QAItem, backend: GenerationBackend, retriever: DenseRetriever, top_k: int = RETRIEVAL_TOP_K) -> dict:
    docs = [d for d, _ in retriever.search(item.question, top_k)]
    prompt = answer_prompt(item, docs)
    answer = backend.generate(prompt)
    return {
        "answer": answer,
        "trajectory": {
            "pipeline": "normal_rag",
            "retrieved_doc_ids": [d.id for d in docs],
            "prompt": prompt,
        },
    }
