# Dataset: 9-class cyber-crime category classification

This project uses **`dataset_balanced_filtered/`** — a balanced train/test split with **9 crime categories** of Indian cyber-crime complaints (Hinglish / code-mixed English).

## Files

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 18,338 | Training complaints |
| `test.csv` | 29,978 | Test complaints |
| `filter_stats.json` | — | How the 9-class set was derived |

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `category` | string | Crime category label (one of 9) |
| `sub_category` | string | Original fine-grained label (kept for provenance; **not used** in this project) |
| `crimeaditionalinfo` | string | Complaint narrative text |
| `is_short` | bool | Flag for very short texts |

## Categories (9)

1. Online Financial Fraud  
2. Online and Social Media Related Crime  
3. Cyber Attack / Dependent Crimes  
4. Hacking / Damage to computer  
5. Sexually Obscene material  
6. Any Other Cyber Crime  
7. Cryptocurrency Crime  
8. Child Pornography / CSAM  
9. Rape/Gang Rape / Sexually Abusive Content  

## Filtering from 11 → 9 classes

Two categories were dropped because models could not learn them reliably (test macro-F1 &lt; 0.25):

- Online Gambling & Betting (F1 ≈ 0.19)  
- Sexually Explicit Act (F1 ≈ 0.15)  

See `filter_stats.json` for exact counts and dropped-class F1 scores.

## Train class distribution (approximate)

| Category | Train count |
|----------|-------------|
| Online and Social Media Related Crime | 4,746 |
| Online Financial Fraud | 4,255 |
| Cyber Attack / Dependent Crimes | 3,608 |
| Hacking / Damage to computer | 1,587 |
| Sexually Obscene material | 1,543 |
| Any Other Cyber Crime | 1,543 |
| Cryptocurrency Crime | 465 |
| Child Pornography / CSAM | 350 |
| Rape/Gang Rape / Sexually Abusive Content | 241 |

## Usage

```python
import pandas as pd

train = pd.read_csv("dataset_balanced_filtered/train.csv")
test = pd.read_csv("dataset_balanced_filtered/test.csv")
```

## Data note

Complaint text may contain sensitive content. Use only under appropriate data-handling policies. Do not redistribute raw data beyond your license terms.
