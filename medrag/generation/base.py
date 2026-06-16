from abc import ABC, abstractmethod


class GenerationBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        ...
