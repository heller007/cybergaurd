#!/usr/bin/env python3
"""
LoRA fine-tune Qwen2.5-0.5B-Instruct for 9-class category classification.

Trains on digit-answer completion (same prompt format as llm_prompt_baseline),
then evaluates with 1-pass digit logit scoring.

  python scripts/llm_lora_finetune.py
  python scripts/llm_lora_finetune.py --epochs 3 --full-test
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.llm_category import (
    build_mc_prompt,
    evaluate_predictions,
    label_to_digit,
    load_data,
    stratified_subset,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_DIR = "outputs/checkpoints_llm_lora_qwen05b"
LOG_DIR = "outputs/logs_9class_llm"
SEED = 42
MAX_LEN = 1024


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_hf_dataset(df: pd.DataFrame, classes) -> Dataset:
    prompts, completions = [], []
    for _, row in df.iterrows():
        prompts.append(build_mc_prompt(row["text"], classes))
        completions.append(label_to_digit(row["cat_label"]))
    return Dataset.from_dict({"prompt": prompts, "completion": completions})


@torch.no_grad()
def classify_by_answer_logits(model, tokenizer, text, classes, device):
    prefix = build_mc_prompt(text, classes)
    enc = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits[0, -1, :]
    return _pick_digit_from_logits(logits, tokenizer, len(classes))


def _pick_digit_from_logits(logits, tokenizer, n_classes: int) -> int:
    best_idx, best_score = 0, -1e9
    for i in range(n_classes):
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
def run_eval(model, tokenizer, eval_df, classes, device, batch_size: int = 8):
    preds = []
    texts = list(eval_df["text"])
    model.eval()
    for start in tqdm(range(0, len(texts), batch_size), desc="Eval"):
        batch_texts = texts[start : start + batch_size]
        prefixes = [build_mc_prompt(t, classes) for t in batch_texts]
        enc = tokenizer(
            prefixes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        attn = enc["attention_mask"]
        for j in range(len(batch_texts)):
            last_pos = int(attn[j].sum().item()) - 1
            preds.append(
                _pick_digit_from_logits(out.logits[j, last_pos, :], tokenizer, len(classes))
            )
        del enc, out, attn
        if device.type == "cuda":
            torch.cuda.empty_cache()
    labels = eval_df["cat_label"].values
    return np.array(preds), labels


def load_eval_model(model_name: str, adapter_dir: str, device: torch.device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    model.to(device)
    return model, tokenizer


def release_gpu(*objs):
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--full-train", action="store_true", help="Use all train rows (no val split)")
    parser.add_argument("--max-per-class", type=int, default=150)
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-only", action="store_true", help="Skip training; load adapter and eval")
    parser.add_argument("--adapter-dir", default="", help="Adapter path for --eval-only")
    parser.add_argument("--max-train", type=int, default=0, help="Cap train rows (0=all)")
    parser.add_argument("--tag", default="", help="Suffix for result filenames")
    parser.add_argument("--output-dir", default=OUT_DIR)
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    train_df, test_df, cat_enc = load_data()
    classes = list(cat_enc.classes_)

    if args.max_train > 0:
        per_class = max(1, args.max_train // len(classes))
        train_df = stratified_subset(train_df, per_class)
        print(f"Capped train set to {len(train_df):,} samples")

    if args.full_train:
        train_split = train_df.reset_index(drop=True)
        val_split = train_df.iloc[:0].copy()
    else:
        train_split, val_split = train_test_split(
            train_df,
            test_size=args.val_size,
            random_state=SEED,
            stratify=train_df["cat_label"],
        )
        train_split = train_split.reset_index(drop=True)
        val_split = val_split.reset_index(drop=True)

    if args.full_test:
        eval_df = test_df.reset_index(drop=True)
    else:
        eval_df = stratified_subset(test_df, args.max_per_class)

    print(f"Model: {args.model}")
    print(f"Train: {len(train_split):,} | Val: {len(val_split):,} | Test eval: {len(eval_df):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    adapter_dir = args.adapter_dir or os.path.join(args.output_dir, "best_adapter")
    train_s = 0.0

    if args.eval_only:
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(f"Adapter not found: {adapter_dir}")
        print(f"Eval-only from {adapter_dir}")
        eval_model, tokenizer = load_eval_model(args.model, adapter_dir, device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        train_ds = make_hf_dataset(train_split, classes)
        val_ds = make_hf_dataset(val_split, classes) if len(val_split) else None

        sft_args = SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            logging_steps=50,
            eval_strategy="epoch" if val_ds is not None else "no",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=val_ds is not None,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=device.type == "cuda",
            report_to="none",
            seed=SEED,
            dataloader_num_workers=4,
            max_length=MAX_LEN,
            completion_only_loss=True,
            dataset_kwargs={"skip_prepare_dataset": False},
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
        )

        train_t0 = time.time()
        trainer.train()
        train_s = time.time() - train_t0
        print(f"Training finished in {train_s / 60:.1f} min")

        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        release_adapter = os.path.join(ROOT, "releases/llm_lora_qwen05b_r64/best_adapter")
        if args.output_dir.startswith(os.path.join(ROOT, "releases/")):
            os.makedirs(os.path.dirname(release_adapter), exist_ok=True)
            if os.path.exists(release_adapter):
                shutil.rmtree(release_adapter)
            shutil.copytree(adapter_dir, release_adapter)
            print(f"Release copy → {release_adapter}")

        release_gpu(trainer, model, train_ds, val_ds)
        eval_model, tokenizer = load_eval_model(args.model, adapter_dir, device)

    infer_t0 = time.time()
    preds, labels = run_eval(
        eval_model, tokenizer, eval_df, classes, device, batch_size=args.eval_batch_size
    )
    infer_s = time.time() - infer_t0

    macro, weighted, report = evaluate_predictions(preds, labels, classes)

    val_macro, val_weighted = None, None
    if len(val_split):
        val_preds, val_labels = run_eval(
            eval_model, tokenizer, val_split, classes, device,
            batch_size=args.eval_batch_size,
        )
        val_macro, val_weighted, _ = evaluate_predictions(val_preds, val_labels, classes)

    print(f"\n{'=' * 60}")
    if val_macro is not None:
        print(f"Val  macro-F1: {val_macro:.4f}")
    print(f"Test macro-F1: {macro:.4f}")
    print(f"Test weighted-F1: {weighted:.4f}")
    print(f"Inference: {infer_s:.1f}s ({len(eval_df) / max(infer_s, 1):.1f} samples/s)")

    tag = args.model.split("/")[-1]
    suffix = args.tag or f"r{args.lora_r}_full"
    out = {
        "model": args.model,
        "family": "llm_lora",
        "method": f"LoRA fine-tune ({tag}, r={args.lora_r})",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "full_train": args.full_train,
        "train_samples": len(train_split),
        "val_samples": len(val_split),
        "eval_samples": len(eval_df),
        "full_test": args.full_test,
        "train_seconds": round(train_s, 2) if train_s else None,
        "infer_seconds": round(infer_s, 2),
        "adapter_dir": adapter_dir,
        "val": {
            "cat_macro_f1": round(val_macro, 4) if val_macro is not None else None,
            "cat_weighted_f1": round(val_weighted, 4) if val_weighted is not None else None,
        },
        "test": {
            "cat_macro_f1": round(macro, 4),
            "cat_weighted_f1": round(weighted, 4),
        },
        "categories": classes,
    }

    out_path = os.path.join(LOG_DIR, f"{tag}_lora_{suffix}_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    report_path = os.path.join(LOG_DIR, f"{tag}_lora_{suffix}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    exp_path = "outputs/experiments_9class/llm_lora_results.json"
    os.makedirs(os.path.dirname(exp_path), exist_ok=True)
    with open(exp_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved → {out_path}")
    return macro


if __name__ == "__main__":
    main()
