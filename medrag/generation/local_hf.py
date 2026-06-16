from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from medrag.generation.base import GenerationBackend


class LocalHFBackend(GenerationBackend):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
