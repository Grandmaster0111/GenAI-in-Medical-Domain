# GenAI in Medical Domain

Working implementation of the RAG architectures described in `report.pdf` (M.Tech Major Project-II,
"GenAI in Medical Domain"): **Normal RAG**, **Advanced RAG** (hybrid dense + BM25 retrieval), and the
thesis's proposed **Agentic Corrective RAG (CRAG)**, which classifies retrieved context as
agreement/contradiction/irrelevant before generation.

This is a small-scale, locally-runnable version of the system — it demonstrates the architecture end to
end, it does not reproduce the thesis's full-scale benchmark numbers (see "Scaled down vs. the thesis"
below).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in HF_TOKEN if you want to use HF Inference API models
```

`HF_TOKEN` is only required for models served via the Hugging Face Inference API (anything other than
`flan-t5-small`, which runs locally).

## Usage

Run these once to build the local data and indices:

```bash
python scripts/build_eval_sets.py --n-per-dataset 50   # samples MedQA/MedMCQA/PubMedQA/BioASQ/MMLU-Med
python scripts/build_corpus.py --n-samples 5000          # samples the PubMed knowledge base
python scripts/build_index.py                              # builds FAISS + BM25 indices
```

Ask a single question and inspect the full trajectory (retrieval, CRAG classification, filtered context):

```bash
python scripts/demo.py --pipeline crag --model flan-t5-small \
  --question "What is the first-line treatment for hypertension?"
```

For multiple-choice questions, pass `--options`:

```bash
python scripts/demo.py --pipeline crag --model flan-t5-small \
  --question "Which of the following is a beta blocker?" \
  --options '{"A": "Metoprolol", "B": "Lisinopril", "C": "Amlodipine", "D": "Furosemide"}'
```

Run an evaluation (saves `results/<model>_<pipeline>_<dataset>.json`):

```bash
python scripts/run_eval.py --pipeline normal_rag   --model flan-t5-small --dataset pubmedqa --n 20
python scripts/run_eval.py --pipeline advanced_rag --model flan-t5-small --dataset pubmedqa --n 20
python scripts/run_eval.py --pipeline crag         --model flan-t5-small --dataset pubmedqa --n 20
```

Aggregate all saved results into `results/summary.csv` and a comparison chart `results/comparison.png`:

```bash
python scripts/summarize_results.py
```

Available `--model` values are in `medrag/config.py:MODEL_REGISTRY`: `flan-t5-small` (local),
`biomistral-7b`, `medgemma-4b-it`, `typhoon-si-med-thin`, `ii-medica` (all via HF Inference API, need
`HF_TOKEN`).

Available `--dataset` values: `medqa`, `medmcqa`, `pubmedqa`, `bioasq`, `mmlu_med`.

## Architecture

```
medrag/
  config.py          model registry, dataset sources, paths
  data/                eval-set + corpus loaders, normalized QAItem/Doc schema
  retrieval/            dense (FAISS) / BM25 / hybrid (reciprocal rank fusion) retrievers
  generation/            pluggable backends: local transformers, HF Inference API
  pipelines/              normal_rag, advanced_rag, crag (+ shared prompt templates)
  evaluation/              accuracy metrics + evaluation harness
scripts/                   CLI entry points (build_*, run_eval, demo, summarize_results)
```

CRAG (`medrag/pipelines/crag.py`) implements the thesis's core contribution: hybrid retrieval, then a
few-shot prompted classification of each retrieved document as `agreement` / `contradiction` /
`irrelevant` against the question, then context filtering (prefer agreement docs, fall back to
non-contradiction docs, fall back to no context if everything contradicts) before final answer
generation. Every run's trajectory (retrieved docs, classifications, filtered context, final prompt) is
saved alongside the predicted answer.

## Scaled down vs. the thesis

- **Knowledge base**: a random sample (default 5,000) of `MedRAG/pubmed` snippets, not the full PMC OA
  corpus.
- **Eval sets**: small samples (default 50/dataset) of public HF mirrors of MedQA, MedMCQA, PubMedQA,
  MMLU-Med, and BioASQ — not the full MIRAGE benchmark splits.
- **BioASQ**: uses `rag-datasets/rag-mini-bioasq` as a substitute, since the original BioASQ dataset
  requires registration.
- **Large models**: BioMistral-7B / MedGemma-4B-IT run via the HF Inference API rather than locally,
  since the available GPU (4GB VRAM) can't host them. "Typhoon-SI-Med-Thin" and "II-Medica" from the
  thesis are niche names not confirmed available on HF serverless Inference API — their entries in
  `MODEL_REGISTRY` are best-effort guesses and may need to be swapped for an available model id.
- Results from this codebase are illustrative of the architecture, not a reproduction of the thesis's
  reported accuracy numbers or Figure 4.1.
