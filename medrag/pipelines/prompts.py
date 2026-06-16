from medrag.data.schema import Doc, QAItem


def format_question(item: QAItem) -> str:
    if not item.options:
        return item.question
    opts = "\n".join(f"{k}. {v}" for k, v in item.options.items())
    return f"{item.question}\n{opts}"


def format_context(docs: list[Doc]) -> str:
    if not docs:
        return ""
    return "\n\n".join(f"[{i+1}] {d.title}\n{d.text}" for i, d in enumerate(docs))


def answer_prompt(item: QAItem, docs: list[Doc]) -> str:
    question_block = format_question(item)
    instruction = (
        "Answer the option letter only." if item.options and set(item.options) <= {"A", "B", "C", "D"}
        else "Answer with exactly one word: yes, no, or maybe." if item.options and set(item.options) == {"yes", "no", "maybe"}
        else "Answer concisely."
    )
    if docs:
        context_block = format_context(docs)
        return (
            "You are a medical question answering assistant. Use the context below if relevant.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question:\n{question_block}\n\n"
            f"{instruction}\nAnswer:"
        )
    return f"You are a medical question answering assistant.\n\nQuestion:\n{question_block}\n\n{instruction}\nAnswer:"


CLASSIFY_FEW_SHOT = """Classify the relationship between a document and a question as one of: agreement, contradiction, irrelevant.

Question: What is the first-line treatment for hypertension?
Document: ACE inhibitors are commonly recommended as first-line therapy for hypertension in many guidelines.
Label: agreement

Question: What is the first-line treatment for hypertension?
Document: Beta blockers are no longer recommended as first-line therapy for uncomplicated hypertension.
Label: contradiction

Question: What is the first-line treatment for hypertension?
Document: Photosynthesis occurs in the chloroplasts of plant cells.
Label: irrelevant
"""


def classify_prompt(item: QAItem, doc: Doc) -> str:
    question_block = format_question(item)
    return (
        f"{CLASSIFY_FEW_SHOT}\n"
        f"Question: {question_block}\n"
        f"Document: {doc.title} {doc.text}\n"
        "Label (agreement, contradiction, or irrelevant):"
    )


