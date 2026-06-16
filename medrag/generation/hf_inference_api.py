import time

from huggingface_hub import InferenceClient

from medrag.config import HF_TOKEN
from medrag.generation.base import GenerationBackend

MAX_RETRIES = 4
RETRY_BACKOFF_SEC = 5


class HFInferenceAPIBackend(GenerationBackend):
    def __init__(self, model_id: str, provider: str | None = None):
        if not HF_TOKEN:
            raise RuntimeError(
                f"HF_TOKEN is not set. Required to call {model_id} via the HF Inference API. "
                "Copy .env.example to .env and fill in HF_TOKEN."
            )
        self.model_id = model_id
        self.client = InferenceClient(model=model_id, provider=provider, api_key=HF_TOKEN)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        messages = [{"role": "user", "content": prompt}]
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat_completion(messages, max_tokens=max_new_tokens)
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF_SEC * (attempt + 1)
                    print(f"    [retry {attempt + 1}/{MAX_RETRIES - 1}] {self.model_id} call failed ({e}); retrying in {wait}s...")
                    time.sleep(wait)
        raise RuntimeError(
            f"HF Inference API call failed for model '{self.model_id}' after {MAX_RETRIES} attempts. "
            f"It may not be available on serverless Inference API. Original error: {last_error}"
        ) from last_error
