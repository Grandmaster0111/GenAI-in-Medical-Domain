from medrag.config import MODEL_REGISTRY
from medrag.generation.base import GenerationBackend
from medrag.generation.hf_inference_api import HFInferenceAPIBackend
from medrag.generation.local_hf import LocalHFBackend


def get_backend(model_name: str) -> GenerationBackend:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_name]
    if spec.backend == "local_hf":
        return LocalHFBackend(spec.model_id)
    if spec.backend == "hf_inference_api":
        return HFInferenceAPIBackend(spec.model_id)
    raise ValueError(f"Unknown backend type '{spec.backend}'")
