#!/usr/bin/env python3
"""
Small LLM prompting baseline for 9-class category classification.

Uses a ~0.5B instruct model with zero-shot (and optional few-shot) prompting.
Evaluates on a stratified test subset by default for reasonable runtime.

  python scripts/llm_prompt_baseline.py
  python scripts/llm_prompt_baseline.py --max-per-class 200
  python scripts/llm_prompt_baseline.py --few-shot 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

TRAIN_PATH = "dataset_balanced_filtered/train.csv"
TEST_PATH = "dataset_balanced_filtered/test.csv"
OUT_DIR = "outputs/logs_9class_llm"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 42
MAX_TEXT_CHARS = 1200
MAX_NEW_TOKENS = 40

os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
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


def stratified_subset(df, max_per_class: int, seed: int = SEED) -> pd.DataFrame:
    parts = []
    for label in df["cat_label"].unique():
        chunk = df[df["cat_label"] == label]
        n = min(len(chunk), max_per_class)
        parts.append(chunk.sample(n=n, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def short_name(category: str) -> str:
    """Compact label for prompting."""
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


def build_few_shot_examples(train_df, classes, shots_per_class: int) -> str:
    if shots_per_class <= 0:
        return ""
    blocks = ["Here are example classifications:\n"]
    for cls in classes:
        rows = train_df[train_df["category"] == cls]
        if rows.empty:
            continue
        sample = rows.sample(
            n=min(shots_per_class, len(rows)), random_state=SEED
        )
        for _, row in sample.iterrows():
            text = row["text"][:400].replace("\n", " ")
            blocks.append(f'Complaint: "{text}"')
            blocks.append(f"Category: {short_name(cls)}\n")
    return "\n".join(blocks) + "\n"


def build_mc_prompt(text: str, classes, few_shot_block: str = "") -> str:
    """Multiple-choice prompt — model answers with digit 1-9 only."""
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


def build_prompt(text: str, classes, few_shot_block: str = "") -> str:
    return build_mc_prompt(text, classes, few_shot_block)


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


@torch.no_grad()
def classify_by_answer_logits(
    model,
    tokenizer,
    text: str,
    classes,
    few_shot_block: str,
    device: torch.device,
) -> int:
    """One forward pass: pick digit 1-9 with highest logit at 'Answer:' position."""
    prefix = build_mc_prompt(text, classes, few_shot_block)
    enc = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=2048)
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits[0, -1, :]

    best_idx, best_score = 0, -1e9
    for i in range(len(classes)):
        for piece in (str(i + 1), " " + str(i + 1)):
            tok_ids = tokenizer.encode(piece, add_special_tokens=False)
            if not tok_ids:
                continue
            score = logits[tok_ids[0]].item()
            if score > best_score:
                best_score = score
                best_idx = i
    return best_idx


@torch.no_grad()
def classify_by_digit_likelihood(
    model,
    tokenizer,
    text: str,
    classes,
    few_shot_block: str,
    device: torch.device,
) -> int:
    """Score digits 1-9 after 'Answer:' — robust for small LMs."""
    prefix = build_mc_prompt(text, classes, few_shot_block)
    best_idx, best_score = 0, -1e9

    for i in range(len(classes)):
        digit = str(i + 1)
        full = prefix + " " + digit
        enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=2048)
        enc = {k: v.to(device) for k, v in enc.items()}
        digit_ids = tokenizer(" " + digit, add_special_tokens=False)["input_ids"]
        if not digit_ids:
            digit_ids = tokenizer(digit, add_special_tokens=False)["input_ids"]
        n_tok = len(digit_ids)

        out = model(**enc)
        log_probs = torch.log_softmax(out.logits, dim=-1)
        seq_len = enc["input_ids"].shape[1]
        score = 0.0
        for t in range(n_tok):
            pos = seq_len - n_tok + t
            if pos < 1:
                continue
            tok_id = enc["input_ids"][0, pos].item()
            score += log_probs[0, pos - 1, tok_id].item()
        score /= max(n_tok, 1)

        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    device: torch.device,
    batch_size: int = 8,
    use_chat_template: bool = False,
) -> list[str]:
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            messages_batch = [
                [{"role": "user", "content": p}] for p in batch
            ]
            texts = [
                tokenizer.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in messages_batch
            ]
        else:
            texts = batch

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        input_len = enc["input_ids"].shape[1]
        gen = model.generate(
            **enc,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        for j in range(len(batch)):
            new_tokens = gen[j, input_len:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(decoded)
    return outputs


@torch.no_grad()
def classify_by_likelihood(
    model,
    tokenizer,
    text: str,
    classes,
    few_shot_block: str,
    device: torch.device,
) -> int:
    """Delegate to digit-based scoring (plain prompt, no chat template)."""
    return classify_by_digit_likelihood(
        model, tokenizer, text, classes, few_shot_block, device
    )


def evaluate(preds, labels, classes):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-per-class", type=int, default=150)
    parser.add_argument("--few-shot", type=int, default=0, help="Examples per class")
    parser.add_argument(
        "--mode",
        choices=["generate", "likelihood", "logits"],
        default="logits",
        help="logits=1-pass digit scoring; generate=autoregressive; likelihood=9-pass",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--full-test", action="store_true")
    args = parser.parse_args()

    train_df, test_df, cat_enc = load_data()
    classes = list(cat_enc.classes_)

    if args.full_test:
        eval_df = test_df.reset_index(drop=True)
    else:
        eval_df = stratified_subset(test_df, args.max_per_class)

    few_shot = build_few_shot_examples(train_df, classes, args.few_shot)
    print(f"Model: {args.model}")
    print(f"Eval samples: {len(eval_df):,} ({'full test' if args.full_test else 'stratified subset'})")
    print(f"Few-shot: {args.few_shot} per class")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    load_s = time.time() - t0
    print(f"Model loaded in {load_s:.1f}s")

    prompts = [build_prompt(t, classes, few_shot) for t in eval_df["text"]]
    labels = eval_df["cat_label"].values

    preds = []
    raw_outputs = []
    infer_t0 = time.time()

    if args.mode == "likelihood":
        for text in tqdm(eval_df["text"], desc="LLM likelihood"):
            idx = classify_by_digit_likelihood(
                model, tokenizer, text, classes, few_shot, device
            )
            preds.append(idx)
            raw_outputs.append(str(idx + 1))
    elif args.mode == "logits":
        for text in tqdm(eval_df["text"], desc="LLM logits"):
            idx = classify_by_answer_logits(
                model, tokenizer, text, classes, few_shot, device
            )
            preds.append(idx)
            raw_outputs.append(str(idx + 1))
    else:
        for start in tqdm(range(0, len(prompts), args.batch_size), desc="LLM generate"):
            batch_prompts = prompts[start : start + args.batch_size]
            batch_raw = generate_batch(
                model, tokenizer, batch_prompts, device, args.batch_size,
                use_chat_template=False,
            )
            raw_outputs.extend(batch_raw)
            preds.extend(parse_prediction(r, classes) for r in batch_raw)

    infer_s = time.time() - infer_t0
    preds = np.array(preds)
    macro, weighted, report = evaluate(preds, labels, classes)

    print(f"\n{'=' * 60}")
    print(f"Inference: {infer_s:.1f}s ({len(eval_df) / max(infer_s, 1):.1f} samples/s)")
    print(f"Category macro-F1:    {macro:.4f}")
    print(f"Category weighted-F1: {weighted:.4f}")

    # Save predictions sample for debugging
    sample_rows = []
    for i in range(min(20, len(eval_df))):
        sample_rows.append(
            {
                "text_preview": eval_df.iloc[i]["text"][:120],
                "true": classes[labels[i]],
                "pred": classes[preds[i]],
                "raw": raw_outputs[i][:120],
            }
        )

    tag = args.model.split("/")[-1]
    suffix = args.mode + (f"_fewshot{args.few_shot}" if args.few_shot else "")
    out = {
        "model": args.model,
        "family": "llm_prompt",
        "prompting": (
            f"{args.mode}"
            + ("" if args.few_shot == 0 else f"_fewshot{args.few_shot}")
        ),
        "eval_samples": len(eval_df),
        "full_test": args.full_test,
        "max_per_class": None if args.full_test else args.max_per_class,
        "load_seconds": round(load_s, 2),
        "infer_seconds": round(infer_s, 2),
        "samples_per_second": round(len(eval_df) / max(infer_s, 1), 2),
        "test": {
            "cat_macro_f1": round(macro, 4),
            "cat_weighted_f1": round(weighted, 4),
        },
        "categories": classes,
        "sample_predictions": sample_rows,
    }

    out_path = os.path.join(OUT_DIR, f"{tag}_{suffix}_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    report_path = os.path.join(OUT_DIR, f"{tag}_{suffix}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Copy to experiments folder for ablation table
    exp_path = "outputs/experiments_9class/llm_prompt_results.json"
    os.makedirs(os.path.dirname(exp_path), exist_ok=True)
    with open(exp_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved → {out_path}")
    return macro


if __name__ == "__main__":
    main()
