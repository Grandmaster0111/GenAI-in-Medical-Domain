from medrag.config import RETRIEVAL_TOP_K
from medrag.data.schema import Doc, QAItem
from medrag.generation.base import GenerationBackend
from medrag.pipelines.prompts import answer_prompt, classify_prompt, format_question
from medrag.retrieval.hybrid import HybridRetriever

VALID_LABELS = ("agreement", "contradiction", "irrelevant")


def _classify(item: QAItem, doc: Doc, backend: GenerationBackend) -> str:
    raw = backend.generate(classify_prompt(item, doc), max_new_tokens=8).lower()
    for label in VALID_LABELS:
        if label in raw:
            return label
    return "irrelevant"


def _filter_context(classified: list[tuple[Doc, str]]) -> list[Doc]:
    agreement = [d for d, label in classified if label == "agreement"]
    if agreement:
        return agreement
    non_contradiction = [d for d, label in classified if label != "contradiction"]
    if non_contradiction:
        return non_contradiction
    return []


def run(item: QAItem, backend: GenerationBackend, retriever: HybridRetriever, top_k: int = RETRIEVAL_TOP_K) -> dict:
    query = format_question(item)
    docs = retriever.search(query, top_k)

    classified = [(d, _classify(item, d, backend)) for d in docs]
    filtered_docs = _filter_context(classified)

    prompt = answer_prompt(item, filtered_docs)
    answer = backend.generate(prompt)

    return {
        "answer": answer,
        "trajectory": {
            "pipeline": "crag",
            "query": query,
            "retrieved_doc_ids": [d.id for d in docs],
            "classifications": [{"doc_id": d.id, "label": label} for d, label in classified],
            "filtered_doc_ids": [d.id for d in filtered_docs],
            "prompt": prompt,
        },
    }
