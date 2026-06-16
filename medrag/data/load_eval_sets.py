"""Loads small, normalized samples of the 5 MIRAGE-style eval datasets from public HF datasets."""
import json

from datasets import load_dataset

from medrag.config import EVAL_DIR, MMLU_MED_SUBJECTS
from medrag.data.schema import QAItem


def _save(items: list[QAItem], name: str) -> None:
    path = EVAL_DIR / f"{name}.jsonl"
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item.to_dict()) + "\n")
    print(f"  wrote {len(items)} items -> {path}")


def load_medqa(n: int) -> list[QAItem]:
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test", streaming=True)
    items = []
    for row in ds:
        if len(items) >= n:
            break
        options = row["options"]  # already a {"A": "...", ...} dict
        answer = row["answer_idx"]
        items.append(QAItem(id=f"medqa_{len(items)}", dataset="medqa", question=row["question"], options=options, answer=str(answer)))
    return items


def load_medmcqa(n: int) -> list[QAItem]:
    ds = load_dataset("openlifescienceai/medmcqa", split="validation", streaming=True)
    items = []
    letters = ["A", "B", "C", "D"]
    for row in ds:
        if len(items) >= n:
            break
        options = {"A": row["opa"], "B": row["opb"], "C": row["opc"], "D": row["opd"]}
        cop = row["cop"]
        # cop is 0-indexed in most releases; fall back to 1-indexed if out of range.
        idx = cop if 0 <= cop < 4 else cop - 1
        answer = letters[idx] if 0 <= idx < 4 else str(cop)
        items.append(QAItem(id=f"medmcqa_{len(items)}", dataset="medmcqa", question=row["question"], options=options, answer=answer))
    return items


def load_pubmedqa(n: int) -> list[QAItem]:
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", streaming=True)
    items = []
    for row in ds:
        if len(items) >= n:
            break
        options = {"yes": "yes", "no": "no", "maybe": "maybe"}
        items.append(QAItem(id=f"pubmedqa_{len(items)}", dataset="pubmedqa", question=row["question"], options=options, answer=row["final_decision"]))
    return items


def load_bioasq(n: int) -> list[QAItem]:
    ds = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages", split="test", streaming=True)
    items = []
    for row in ds:
        if len(items) >= n:
            break
        question = row.get("question") or row.get("query")
        answer = row.get("answer") or row.get("ideal_answer")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not question or not answer:
            continue
        items.append(QAItem(id=f"bioasq_{len(items)}", dataset="bioasq", question=question, options=None, answer=str(answer)))
    return items


def load_mmlu_med(n: int) -> list[QAItem]:
    items = []
    per_subject = max(1, n // len(MMLU_MED_SUBJECTS))
    for subject in MMLU_MED_SUBJECTS:
        ds = load_dataset("cais/mmlu", subject, split="test", streaming=True)
        count = 0
        for row in ds:
            if count >= per_subject:
                break
            letters = ["A", "B", "C", "D"]
            options = {letters[i]: c for i, c in enumerate(row["choices"])}
            answer = letters[row["answer"]]
            items.append(QAItem(id=f"mmlu_med_{len(items)}", dataset="mmlu_med", question=row["question"], options=options, answer=answer))
            count += 1
    return items[:n]


LOADERS = {
    "medqa": load_medqa,
    "medmcqa": load_medmcqa,
    "pubmedqa": load_pubmedqa,
    "bioasq": load_bioasq,
    "mmlu_med": load_mmlu_med,
}


def build_all(n_per_dataset: int) -> None:
    for name, loader in LOADERS.items():
        print(f"Loading {name} (n={n_per_dataset})...")
        try:
            items = loader(n_per_dataset)
            _save(items, name)
        except Exception as e:
            print(f"  FAILED to load {name}: {e}")
