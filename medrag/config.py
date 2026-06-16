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


MODEL_REGISTRY = {
    "flan-t5-small": ModelSpec("flan-t5-small", "local_hf", "google/flan-t5-small"),
    "biomistral-7b": ModelSpec("biomistral-7b", "hf_inference_api", "BioMistral/BioMistral-7B"),
    "medgemma-4b-it": ModelSpec("medgemma-4b-it", "hf_inference_api", "google/medgemma-4b-it"),
    # Niche models named in the thesis; unconfirmed on HF serverless Inference API.
    # Swap model_id below for an available checkpoint if these 404.
    "typhoon-si-med-thin": ModelSpec("typhoon-si-med-thin", "hf_inference_api", "scb10x/llama3.1-typhoon2-8b-instruct"),
    "ii-medica": ModelSpec("ii-medica", "hf_inference_api", "aaditya/Llama3-OpenBioLLM-8B"),
}

RETRIEVAL_TOP_K = 5

for d in (DATA_DIR, EVAL_DIR, INDEX_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
