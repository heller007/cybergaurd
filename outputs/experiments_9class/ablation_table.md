# 9-Class Category Classification — Ablation Table

**Dataset:** `dataset_balanced_filtered` (9 categories)  
**Primary metric:** test category macro-F1  
**Generated:** 2026-07-06

| Rank | Method | Test Macro F1 | Val Macro F1 | Family | Notes |
|------|--------|---------------|--------------|--------|-------|
| 1 | LoRA fine-tune (Qwen2.5-0.5B-Instruct, r=64) | **0.5881** | — | llm | r=64, alpha=128; train n=18338; eval n=29978 |
| 2 | MuRIL-base + NER + calibrated | **0.5826** | 0.5848 | neural | Per-class threshold tuning on val set |
| 3 | MuRIL-base + NER + focal (best) | **0.5813** | 0.5755 | neural | 9-class filtered; weighted sampler |
| 4 | XLM-R-large + NER + focal | **0.5692** | 0.5876 | neural | Same 9-class setup, different encoder |
| 5 | MultinomialNB | **0.5548** | 0.4800 | classical | tfidf |
| 6 | LogisticRegression | **0.5488** | 0.5489 | classical | tfidf |
| 7 | LogisticRegression+calibrated | **0.5365** | 0.5297 | classical | tfidf_svd300+ner; per-class threshold calibration |
| 8 | MuRIL-base + weak_focal | **0.5258** | 0.5727 | neural | Weak-class targeted loss (underperformed) |
| 9 | LightGBM | **0.5013** | 0.4994 | classical | tfidf_svd300+ner |
| 10 | LightGBM | **0.4913** | 0.5120 | classical | tfidf_svd300 |
| 11 | LinearSVM | **0.4813** | 0.5590 | classical | tfidf+ner_sparse |
| 12 | LogisticRegression | **0.4714** | 0.4923 | classical | tfidf_svd300+ner |
| 13 | LinearSVM | **0.4713** | 0.5585 | classical | tfidf |
| 14 | LLM prompt (Qwen/Qwen2.5-0.5B-Instruct) | **0.0227** | — | llm | logits_fewshot1; n=1321; zero-shot digit scoring |
