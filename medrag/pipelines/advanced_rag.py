from medrag.config import RETRIEVAL_TOP_K
from medrag.data.schema import QAItem
from medrag.generation.base import GenerationBackend
from medrag.pipelines.prompts import answer_prompt, format_question
from medrag.retrieval.hybrid import HybridRetriever


def run(item: QAItem, backend: GenerationBackend, retriever: HybridRetriever, top_k: int = RETRIEVAL_TOP_K) -> dict:
    query = format_question(item)
    docs = retriever.search(query, top_k)
    prompt = answer_prompt(item, docs)
    answer = backend.generate(prompt)
    return {
        "answer": answer,
        "trajectory": {
            "pipeline": "advanced_rag",
            "query": query,
            "retrieved_doc_ids": [d.id for d in docs],
            "prompt": prompt,
        },
    }
