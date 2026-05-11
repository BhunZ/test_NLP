# Benchmark Reports

This folder contains evaluation reports from running the RAG pipeline.

## Files

| File | Description |
|------|-------------|
| `answer_eval_FINAL.json` | Final answer evaluation results |
| `answer_eval_report.json` | Answer evaluation report |
| `ragas_report_*.json` | RAGAS benchmark evaluation results |

## Usage

These reports can be used to compare the performance of different pipeline configurations:
- With/without query rewrite
- Different retrieval methods
- Different LLM providers (Groq vs Mistral)

## Regenerate

Run evaluation scripts in `embed__data/` or `evaluation/` folders to regenerate these reports.