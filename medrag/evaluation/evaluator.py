import json
import time

from medrag.config import EVAL_DIR, RESULTS_DIR
from medrag.data.schema import QAItem
from medrag.evaluation.metrics import accuracy, is_correct
from medrag.generation.base import GenerationBackend
from medrag.pipelines import advanced_rag, crag, normal_rag
from medrag.retrieval.dense_index import DenseRetriever
from medrag.retrieval.hybrid import HybridRetriever

PIPELINES = {
    "normal_rag": normal_rag,
    "advanced_rag": advanced_rag,
    "crag": crag,
}


def load_eval_set(dataset: str, n: int | None = None) -> list[QAItem]:
    path = EVAL_DIR / f"{dataset}.jsonl"
    items = []
    with open(path) as f:
        for line in f:
            items.append(QAItem.from_dict(json.loads(line)))
            if n is not None and len(items) >= n:
                break
    return items


def get_retriever(pipeline_name: str):
    if pipeline_name == "normal_rag":
        return DenseRetriever()
    return HybridRetriever()


def run_eval(pipeline_name: str, model_name: str, dataset: str, backend: GenerationBackend, n: int | None = None) -> dict:
    if pipeline_name not in PIPELINES:
        raise ValueError(f"Unknown pipeline '{pipeline_name}'. Available: {list(PIPELINES)}")
    pipeline = PIPELINES[pipeline_name]
    retriever = get_retriever(pipeline_name)
    items = load_eval_set(dataset, n)

    results = []
    for item in items:
        start = time.time()
        out = pipeline.run(item, backend, retriever)
        elapsed = time.time() - start
        correct = is_correct(item, out["answer"])
        results.append({
            "id": item.id,
            "question": item.question,
            "gold": item.answer,
            "predicted_raw": out["answer"],
            "correct": correct,
            "elapsed_sec": round(elapsed, 2),
            "trajectory": out["trajectory"],
        })
        print(f"  [{item.id}] correct={correct} ({elapsed:.1f}s)")

    acc = accuracy(results)
    summary = {
        "pipeline": pipeline_name,
        "model": model_name,
        "dataset": dataset,
        "n": len(results),
        "accuracy": acc,
        "results": results,
    }

    out_path = RESULTS_DIR / f"{model_name}_{pipeline_name}_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Accuracy: {acc:.2%} ({len(results)} items) -> {out_path}")
    return summary
