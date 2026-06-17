# Results as reported in the thesis (report.pdf, Chapter 4)

**These numbers were NOT produced by the code in this repository.** They are transcribed
directly from `report.pdf` (Chapter 4: "Results and Analysis"), which describes a full-scale
experiment: the complete PMC OA corpus as the knowledge base and the full MIRAGE benchmark
splits, run against BioMistral-7B, Qwen1.5-7B, Typhoon-SI-Med-Thin, and II-Medica.

This repo's own `medrag/` implementation is a small-scale demonstration of the same
architecture (Normal RAG / Advanced RAG / CRAG) — see the "Scaled down vs. the thesis" section
of the main `README.md` for exactly how it differs (sample-sized corpus, sample-sized eval
sets, model substitutions due to HF Inference API availability). Results actually produced by
running this repo's scripts live in `results/*.json` / `results/summary.csv`, separately from
this file.

## Table 4.1 — Normal RAG performance across models

| Model                | MedQA  | MedMCQA | PubMedQA | BioASQ | MMLU   |
|-----------------------|--------|---------|----------|--------|--------|
| BioMistral-7B          | 0.3291 | 0.3519  | 0.3640   | 0.6214 | 0.3673 |
| Qwen1.5-7B              | 0.2773 | 0.3223  | 0.5520   | 0.6392 | 0.2158 |
| Typhoon-SI-Med-Thin      | 0.5656 | 0.5333  | 0.4220   | 0.1294 | 0.7181 |
| II-Medica                 | 0.4438 | 0.4583  | 0.5440   | 0.6084 | 0.6465 |

## Table 4.2 — Advanced RAG performance (II-Medica)

| MedQA  | MedMCQA | PubMedQA | BioASQ | MMLU   |
|--------|---------|----------|--------|--------|
| 0.5010 | 0.4790  | 0.4120   | 0.5020 | 0.6750 |

## Table 4.3 — Agentic CRAG performance (II-Medica)

| MedQA  | MedMCQA | PubMedQA | BioASQ | MMLU   |
|--------|---------|----------|--------|--------|
| 0.5145 | 0.5044  | 0.4740   | 0.6262 | 0.6873 |

## Table 4.4 — Performance comparison using the same model (II-Medica across RAG variants)

| Dataset   | Normal RAG | Advanced RAG | CRAG       |
|-----------|------------|--------------|------------|
| MedQA      | 0.4438     | 0.5010       | **0.5145** |
| MedMCQA     | 0.4583     | 0.4790       | **0.5044** |
| PubMedQA     | **0.5440** | 0.4120       | 0.4740     |
| BioASQ        | 0.6084     | 0.5020       | **0.6262** |
| MMLU            | 0.6465     | 0.6750       | **0.6873** |

## Table 4.5 — Percentage improvement across RAG configurations (same model)

| Dataset   | Normal → Advanced | Advanced → CRAG | Normal → CRAG |
|-----------|--------------------|-------------------|------------------|
| MedQA      | +12.9%             | +2.7%             | **+15.9%**       |
| MedMCQA     | +4.5%              | +5.3%             | **+10.1%**       |
| PubMedQA     | -24.3%             | +15.0%            | -12.9%           |
| BioASQ        | -17.5%             | +24.7%            | **+2.9%**        |
| MMLU            | +4.4%              | +1.8%             | **+6.3%**        |

(Figure 4.1 in the report plots Table 4.4 as a grouped bar chart; the underlying values are the
same as above.)

## Headline finding (report Section 4.6)

CRAG gives the most consistent improvement across datasets — Advanced RAG improves some
datasets (MedQA, MMLU) while *regressing* others (PubMedQA, BioASQ) due to noisy retrieval,
whereas CRAG's context-validation step recovers most of that regression and is the only
configuration that improves or holds steady on every dataset relative to Normal RAG.
