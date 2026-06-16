from huggingface_hub import InferenceClient

from medrag.config import HF_TOKEN
from medrag.generation.base import GenerationBackend


class HFInferenceAPIBackend(GenerationBackend):
    def __init__(self, model_id: str):
        if not HF_TOKEN:
            raise RuntimeError(
                f"HF_TOKEN is not set. Required to call {model_id} via the HF Inference API. "
                "Copy .env.example to .env and fill in HF_TOKEN."
            )
        self.model_id = model_id
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat_completion(messages, max_tokens=max_new_tokens)
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(
                f"HF Inference API call failed for model '{self.model_id}'. "
                f"It may not be available on serverless Inference API. Original error: {e}"
            ) from e
