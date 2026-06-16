import re

from medrag.data.schema import QAItem


def extract_choice(generated: str, options: dict) -> str | None:
    text = generated.strip().lower()
    keys = list(options.keys())

    # yes/no/maybe style
    if set(keys) == {"yes", "no", "maybe"}:
        for key in ("maybe", "yes", "no"):  # check "maybe" before "yes"/"no" substrings
            if key in text:
                return key
        return None

    # A/B/C/D style: look for a standalone letter, else first letter mentioned
    match = re.search(r"\b([a-d])\b", text)
    if match:
        return match.group(1).upper()
    for key in keys:
        if key.lower() in text:
            return key
    return None


def is_correct(item: QAItem, generated: str) -> bool:
    if item.options:
        predicted = extract_choice(generated, item.options)
        return predicted is not None and predicted.lower() == str(item.answer).lower()

    # free-form (e.g. BioASQ): soft match via token overlap with gold answer
    gold_tokens = set(re.findall(r"\w+", item.answer.lower()))
    pred_tokens = set(re.findall(r"\w+", generated.lower()))
    if not gold_tokens:
        return False
    overlap = len(gold_tokens & pred_tokens) / len(gold_tokens)
    return overlap >= 0.5


def accuracy(results: list[dict]) -> float:
    if not results:
        return 0.0
    correct = sum(1 for r in results if r["correct"])
    return correct / len(results)
