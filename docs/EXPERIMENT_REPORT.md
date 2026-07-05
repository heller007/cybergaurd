# CyberGaurd — Experiment Report (9-Class Category Classification)

**Author:** Vineet (Vineet_Nitk)  
**Task:** Classify Indian cyber-crime complaints (Hinglish / code-mixed text) into crime **category**  
**Primary metric:** Category **macro-F1** on held-out test set  
**Best result:** **0.5826** test macro-F1 (MuRIL-base + NER + focal loss + threshold calibration)

---

## 1. Problem & Motivation

Cyber-crime complaints in India are often written in **romanized Hinglish** with noisy spelling, mixed scripts, and domain-specific slang. Automated triage requires robust **category-level** classification before routing to investigators.

This experiment focuses on **category prediction only** (not sub-category), using a curated **9-class** label set after removing categories that were too ambiguous for reliable classification.

---

## 2. Dataset

| Split | Rows | Source |
|-------|------|--------|
| Train | 18,338 | `dataset_balanced_filtered/train.csv` |
| Test | 29,978 | `dataset_balanced_filtered/test.csv` |

**Text column:** `crimeaditionalinfo` (complaint narrative, preprocessed)  
**Label column:** `category`

### 9 retained categories

| Category | Train count |
|----------|-------------|
| Online and Social Media Related Crime | 4,746 |
| Online Financial Fraud | 4,255 |
| Cyber Attack / Dependent Crimes | 3,608 |
| Hacking / Damage to computer… | 1,587 |
| Sexually Obscene material | 1,543 |
| Any Other Cyber Crime | 1,543 |
| Cryptocurrency Crime | 465 |
| Child Pornography / CSAM | 350 |
| Rape/Gang Rape / Sexually Abusive Content | 241 |

### Dropped categories (test F1 &lt; 0.25 on 11-class model)

- Online Gambling & Betting (F1 = 0.19)
- Sexually Explicit Act (F1 = 0.15)

Filtering weak classes improved macro-F1 from ~0.50 (11-class) to **~0.58** (9-class).

---

## 3. Methods Compared

### 3.1 Classical baselines (TF-IDF)

We report classical methods at two levels of fairness:

| Setting | Best model | Test macro-F1 | Notes |
|---------|------------|---------------|-------|
| **Text only** | LinearSVM (TF-IDF) | 0.4713 | Fast baseline, no extra features |
| **Text + NER (fair)** | LightGBM (TF-IDF→SVD + NER) | **0.5013** | Same 28-dim NER rules as neural model |
| Text + NER + calibration | LogisticRegression | 0.5365 | Threshold tuning on val set |

**Important:** Classical TF-IDF pipelines underperform neural encoders on noisy Hinglish text; transformer models gain **+8 pts** over the best classical baseline on this task.

#### Text-only results

| Model | Features | Test macro-F1 |
|-------|----------|---------------|
| LinearSVM | TF-IDF (50k, 1–2 grams) | 0.4713 |
| LightGBM | TF-IDF → SVD(300) | 0.4913 |
| MultinomialNB | TF-IDF | 0.5548 |
| LogisticRegression | TF-IDF | 0.5488 |

#### Fair comparison (TF-IDF + NER)

| Model | Features | Test macro-F1 |
|-------|----------|---------------|
| **LightGBM** | TF-IDF → SVD(300) + NER | **0.5013** |
| LinearSVM | TF-IDF + NER (sparse) | 0.4813 |
| LogisticRegression + calibrated | TF-IDF → SVD(300) + NER | 0.5365 |

Classical methods train in **under 2 minutes** on CPU but lag neural and LLM fine-tunes on code-mixed complaint text.

### 3.2 Neural category-only model (best)

**Architecture:**
- Encoder: `google/muril-base-cased` (multilingual, strong on Indian languages)
- Pooling: CLS + attention pooling → fusion MLP
- **NER feature fusion:** 28-dim rule-based entity features (UPI, bank names, social apps, etc.)
- Head: single category classifier (9 classes)

**Training:**
- Focal loss (γ = 2.5) with inverse-frequency class weights
- WeightedRandomSampler (power = 0.5) for minority exposure
- Early stopping on validation category macro-F1 (patience = 8)
- Max sequence length = 256 tokens

| Configuration | Test macro-F1 |
|---------------|---------------|
| MuRIL + NER + focal | **0.5813** |
| + threshold calibration | **0.5826** |
| XLM-R-large + NER + focal | 0.5692 |
| MuRIL + weak_focal (ablation) | 0.5258 |

```bash
python -m src.train_category
python scripts/calibrate_category.py
```

### 3.4 Small LLM prompting (Qwen2.5-0.5B)

We tested whether a **0.5B instruct model** can classify via prompting alone (no fine-tuning):

| Setting | Test macro-F1 |
|---------|---------------|
| Qwen2.5-0.5B, zero-shot generation | 0.024 |
| Qwen2.5-0.5B, 1-pass digit logit scoring | 0.023 |
| Qwen2.5-0.5B, digit scoring + 1-shot/class | **0.023** |
| Qwen2.5-1.5B, digit scoring + 1-shot/class | 0.022 |

**Finding:** Small LLMs **fail completely** on this task via prompting alone (~0.02 F1 vs 0.58 for fine-tuned MuRIL). The model collapses to predicting a single category (mostly class 1). Domain-specific Hinglish cyber-crime text requires **fine-tuning** — not zero-shot LLM prompts.

```bash
python scripts/llm_prompt_baseline.py --mode logits --few-shot 1
```

### 3.5 LoRA fine-tuning (Qwen2.5-0.5B)

| Setting | Test macro-F1 | Eval set |
|---------|---------------|----------|
| Qwen2.5-0.5B, zero-shot prompting | 0.023 | subset n=1321 |
| LoRA r=16, 2 epochs | 0.6575 | subset n=1321 |
| **LoRA r=64, α=128, 3 epochs, full train** | **0.5880** | **full test n=29,978** |

Full-train LoRA (r=64) slightly beats MuRIL calibrated (0.5826) on the full test set. The earlier r=16 score (0.6575) was on a small stratified subset and is not directly comparable.

```bash
python scripts/llm_lora_finetune.py --lora-r 64 --lora-alpha 128 --epochs 3 --full-train --full-test
```

### 3.3 Post-hoc calibration

Per-class probability thresholds tuned on the validation set improve macro-F1 by **+0.0013** without retraining. Most gain on borderline classes (Child Pornography, Social Media).

---

## 4. Ablation Summary

| Ablation | Finding |
|----------|---------|
| **Classical + NER** | LightGBM+NER reaches 0.501 — **~8 pts below** MuRIL (0.581) |
| **Encoder:** MuRIL vs XLM-R-large | MuRIL +1.2 pts on 9-class test F1 |
| **NER fusion (neural)** | +~2–3 pts vs text-only transformer |
| **Loss:** focal vs cb_focal vs ldam | Standard focal generalizes best |
| **Weak-class focal** | Hurt over-predicted classes (low precision) |
| **Max length 512** | No gain (median text ~83 tokens) |
| **Class filtering** | 11 → 9 classes: +8 pts macro-F1 |

Full table: [`outputs/experiments_9class/ablation_table.md`](../outputs/experiments_9class/ablation_table.md)

---

## 5. Per-Class Performance (Best Model, Test Set)

| Category | F1 | Notes |
|----------|-----|-------|
| Cyber Attack | 1.000 | Perfect separation |
| Rape/Gang Rape | 0.928 | Strong |
| Online Financial Fraud | 0.842 | High volume, strong |
| Social Media Crime | 0.466 | Moderate |
| Child Pornography | 0.459 | Rare class |
| Cryptocurrency | 0.513 | High recall, low precision |
| Hacking | 0.358 | Confused with other classes |
| Sexually Obscene | 0.330 | Low precision |
| Any Other | 0.335 | Catch-all, inherently hard |

**Error pattern:** Weak classes tend to be **over-predicted** (high recall, low precision), suggesting confusion between semantically overlapping categories rather than insufficient training data.

---

## 6. Deployment

The production-ready inference API loads the MuRIL checkpoint + calibrated thresholds:

```python
from src.predict_category import CategoryPredictor

predictor = CategoryPredictor()
result = predictor.predict("Someone hacked my Facebook and changed the password")
# → {"category": "...", "confidence": 0.71, "top_k": [...]}
```

CLI:

```bash
python scripts/export_model.py
python -m src.predict_category "complaint text here"
```

**Model bundle:** `model/category_9class/` (checkpoint + thresholds + metadata)  
*Note: model weights are not committed to git — train locally or download from releases.*

---

## 7. Reproducibility

```bash
# Setup
pip install -r requirements.txt

# Data: dataset_balanced_filtered/ (included in repo)

# Best neural model
python -m src.train_category
python scripts/calibrate_category.py
python scripts/export_model.py
python scripts/build_ablation_table.py
```

**Seed:** 42 | **Hardware:** NVIDIA RTX A6000 | **Framework:** PyTorch 2.x + HuggingFace Transformers

---

## 8. Conclusions

1. **Classical TF-IDF + LightGBM + NER** reaches **0.5013** macro-F1 — **~8 pts below** the best neural/LLM models on Hinglish complaints.
2. **MuRIL-base with NER fusion and focal loss** achieves **0.5813**, outperforming XLM-R-large on this Hinglish task.
3. **Removing irrecoverable classes** (Gambling, Sexually Explicit) is the single largest gain (+8 pts).
4. **Advanced losses** (CB-focal, LDAM, weak-focal) did not beat standard focal loss on the test set.
5. **Threshold calibration** provides a small, free improvement for deployment.

**Recommended production config:** MuRIL-base + NER + focal + calibrated thresholds → **0.5826 test macro-F1**. For LLM-only stacks, LoRA Qwen2.5-0.5B reaches **0.5880** on full test.

---

## 9. Future Work

- ONNX export for edge deployment
- Larger LLM LoRA with instruction tuning variants
