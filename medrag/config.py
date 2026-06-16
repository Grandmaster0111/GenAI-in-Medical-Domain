import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
EVAL_DIR = DATA_DIR / "eval"
INDEX_DIR = DATA_DIR / "indices"
RESULTS_DIR = ROOT_DIR / "results"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"

HF_TOKEN = os.environ.get("HF_TOKEN")

EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"

# HF dataset source for each eval set name -> (hf_path, hf_config, split)
EVAL_DATASET_SOURCES = {
    "medqa": ("bigbio/med_qa", "med_qa_en_4options_source", "test"),
    "medmcqa": ("openlifescienceai/medmcqa", None, "validation"),
    "pubmedqa": ("qiaojin/PubMedQA", "pqa_labeled", "train"),
    "bioasq": ("rag-datasets/rag-mini-bioasq", "question-answer-passages", "test"),
    "mmlu_med": ("cais/mmlu", None, "test"),
}

MMLU_MED_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "college_biology",
    "professional_medicine",
    "medical_genetics",
]

PUBMED_CORPUS_SOURCE = ("MedRAG/pubmed", None, "train")


@dataclass
class ModelSpec:
    name: str
    backend: str  # "local_hf" or "hf_inference_api"
    model_id: str
    provider: str | None = None  # HF Inference provider, e.g. "featherless-ai"; None = auto-route


MODEL_REGISTRY = {
    "flan-t5-small": ModelSpec("flan-t5-small", "local_hf", "google/flan-t5-small"),
    # BioMistral/BioMistral-7B has zero HF Inference providers (confirmed via model_info
    # inferenceProviderMapping) and cannot be called through the serverless API. Substituted
    # with Med42-8B, a real medical-domain instruction-tuned model available via featherless-ai.
    "biomistral-7b": ModelSpec("biomistral-7b", "hf_inference_api", "m42-health/Llama3-Med42-8B", provider="featherless-ai"),
    # google/medgemma-4b-it also has no HF Inference providers as of this writing; unconfirmed/unavailable.
    "medgemma-4b-it": ModelSpec("medgemma-4b-it", "hf_inference_api", "google/medgemma-4b-it"),
    # Niche models named in the thesis; unconfirmed on HF serverless Inference API.
    # Swap model_id below for an available checkpoint if these 404.
    "typhoon-si-med-thin": ModelSpec("typhoon-si-med-thin", "hf_inference_api", "scb10x/llama3.1-typhoon2-8b-instruct", provider="featherless-ai"),
    "ii-medica": ModelSpec("ii-medica", "hf_inference_api", "aaditya/Llama3-OpenBioLLM-8B", provider="featherless-ai"),
}

RETRIEVAL_TOP_K = 5

for d in (DATA_DIR, EVAL_DIR, INDEX_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
