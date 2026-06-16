from dataclasses import dataclass, field


@dataclass
class QAItem:
    id: str
    dataset: str
    question: str
    options: dict | None  # e.g. {"A": "...", "B": "..."} or None for free-form yes/no/maybe
    answer: str  # option key, or raw label text (e.g. "yes")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "question": self.question,
            "options": self.options,
            "answer": self.answer,
        }

    @staticmethod
    def from_dict(d: dict) -> "QAItem":
        return QAItem(**d)


@dataclass
class Doc:
    id: str
    title: str
    text: str

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "text": self.text}

    @staticmethod
    def from_dict(d: dict) -> "Doc":
        return Doc(**d)
