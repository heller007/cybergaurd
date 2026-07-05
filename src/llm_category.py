"""Shared helpers for LLM category classification (prompt + LoRA)."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

TRAIN_PATH = "dataset_balanced_filtered/train.csv"
TEST_PATH = "dataset_balanced_filtered/test.csv"
MAX_TEXT_CHARS = 1200


def load_data(train_path: str = TRAIN_PATH, test_path: str = TEST_PATH):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    for df in (train, test):
        df["text"] = df["crimeaditionalinfo"].fillna("").astype(str)
        df.drop(df[df["text"].str.len() == 0].index, inplace=True)
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    cat_enc = LabelEncoder().fit(
        pd.concat([train["category"], test["category"]]).unique()
    )
    train["cat_label"] = cat_enc.transform(train["category"])
    test["cat_label"] = cat_enc.transform(test["category"])
    return train, test, cat_enc


def stratified_subset(df, max_per_class: int, seed: int = 42) -> pd.DataFrame:
    parts = []
    for label in df["cat_label"].unique():
        chunk = df[df["cat_label"] == label]
        n = min(len(chunk), max_per_class)
        parts.append(chunk.sample(n=n, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def short_name(category: str) -> str:
    mapping = {
        "Any Other Cyber Crime": "Any Other Cyber Crime",
        "Child Pornography CPChild Sexual Abuse Material CSAM": "Child Pornography / CSAM",
        "Cryptocurrency Crime": "Cryptocurrency Crime",
        "Cyber Attack/ Dependent Crimes": "Cyber Attack",
        "Hacking  Damage to computercomputer system etc": "Hacking / Computer Damage",
        "Online Financial Fraud": "Online Financial Fraud",
        "Online and Social Media Related Crime": "Social Media Crime",
        "RapeGang Rape RGRSexually Abusive Content": "Rape / Sexual Abuse Content",
        "Sexually Obscene material": "Sexually Obscene Material",
    }
    return mapping.get(category, category)


def build_category_block(classes) -> str:
    lines = []
    for i, c in enumerate(classes, 1):
        lines.append(f"{i}. {short_name(c)}")
    return "\n".join(lines)


def build_mc_prompt(text: str, classes, few_shot_block: str = "") -> str:
    text = text[:MAX_TEXT_CHARS].replace("\n", " ")
    cats = build_category_block(classes)
    return (
        "Classify this Indian cyber-crime complaint (Hinglish/English).\n"
        "Reply with ONLY one digit (1-9). No other text.\n\n"
        f"{cats}\n\n"
        f"{few_shot_block}"
        f'Complaint: "{text}"\n'
        "Answer:"
    )


def label_to_digit(cat_label: int) -> str:
    return str(cat_label + 1)


def parse_prediction(raw: str, classes) -> int:
    raw = raw.strip().split("\n")[0].strip()
    m = re.search(r"\b([1-9])\b", raw)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(classes):
            return idx

    raw = re.sub(r"^[\d\.\)\-\*]+\s*", "", raw).strip().strip('"').strip("'")
    short_to_idx = {short_name(c).lower(): i for i, c in enumerate(classes)}
    full_to_idx = {c.lower(): i for i, c in enumerate(classes)}
    low = raw.lower()
    if low in full_to_idx:
        return full_to_idx[low]
    if low in short_to_idx:
        return short_to_idx[low]

    best_idx, best_score = 0, -1.0
    for i, c in enumerate(classes):
        for candidate in (c, short_name(c)):
            score = SequenceMatcher(None, low, candidate.lower()).ratio()
            if score > best_score:
                best_score = score
                best_idx = i
    return best_idx


def evaluate_predictions(preds, labels, classes):
    macro = f1_score(labels, preds, average="macro", zero_division=0)
    weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    report = classification_report(
        labels,
        preds,
        labels=list(range(len(classes))),
        target_names=list(classes),
        zero_division=0,
        output_dict=True,
    )
    return macro, weighted, report
