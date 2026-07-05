"""
Per-class threshold calibration for category-only model.

Usage:
    python scripts/calibrate_category.py
    python scripts/calibrate_category.py --checkpoint outputs/checkpoints_category/best_model.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibrate import calibrate_thresholds, threshold_predict
from src.train_category import (
    BATCH_SIZE,
    CKPT_DIR,
    CategoryClassifier,
    CategoryDataset,
    FP16,
    LOG_DIR,
    NUM_WORKERS,
    USE_NER,
    get_ner_matrix,
    prepare_data,
)
from transformers import AutoTokenizer


def get_probabilities(model, loader, device):
    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting probabilities"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            cl = batch["cat_label"].to(device)
            ner = batch.get("entity_features")
            ner = ner.to(device) if ner is not None else None
            with autocast(enabled=FP16):
                logits = model(ids, mask, entity_features=ner)
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            all_true.extend(cl.cpu().numpy())
    return np.vstack(all_probs), np.array(all_true)


def evaluate_with_thresholds(probs, true_labels, thresholds):
    preds = threshold_predict(probs, thresholds)
    return {
        "macro_f1": f1_score(true_labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(true_labels, preds, average="weighted", zero_division=0),
        "preds": preds,
    }


def run_calibration(checkpoint=None, output_dir=None):
    checkpoint = checkpoint or os.path.join(CKPT_DIR, "best_model.pt")
    output_dir = output_dir or CKPT_DIR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    num_cat = ckpt["num_cat"]
    use_ner = ckpt.get("use_ner", USE_NER)

    model = CategoryClassifier(num_cat, use_ner=use_ner).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    train_df, val_df, test_df, cat_enc = prepare_data()
    tokenizer = AutoTokenizer.from_pretrained(ckpt.get("model_name", "xlm-roberta-large"))
    val_ner = get_ner_matrix(val_df, "val") if use_ner else None
    test_ner = get_ner_matrix(test_df, "test") if use_ner else None

    val_loader = DataLoader(
        CategoryDataset(val_df, tokenizer, val_ner),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        CategoryDataset(test_df, tokenizer, test_ner),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print("\nCollecting validation probabilities...")
    val_probs, val_true = get_probabilities(model, val_loader, device)
    baseline_val = f1_score(val_true, val_probs.argmax(1), average="macro", zero_division=0)
    print(f"  Val macro F1 (argmax): {baseline_val:.4f}")

    print("\nCalibrating per-class thresholds on validation set...")
    thresholds = calibrate_thresholds(val_probs, val_true, num_cat)

    cal_val = evaluate_with_thresholds(val_probs, val_true, thresholds)
    print(
        f"  Val macro F1 (calibrated): {cal_val['macro_f1']:.4f}  "
        f"(delta {cal_val['macro_f1'] - baseline_val:+.4f})"
    )

    print("\nCollecting test probabilities...")
    test_probs, test_true = get_probabilities(model, test_loader, device)
    baseline_test = f1_score(test_true, test_probs.argmax(1), average="macro", zero_division=0)
    cal_test = evaluate_with_thresholds(test_probs, test_true, thresholds)

    print(f"\n{'=' * 60}")
    print("TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"  Argmax macro F1:      {baseline_test:.4f}")
    print(f"  Calibrated macro F1:  {cal_test['macro_f1']:.4f}  "
          f"(delta {cal_test['macro_f1'] - baseline_test:+.4f})")

    report = classification_report(
        test_true,
        cal_test["preds"],
        labels=list(range(num_cat)),
        target_names=list(cat_enc.classes_),
        zero_division=0,
        output_dict=True,
    )
    print("\nPer-category F1 after calibration:")
    for cls in cat_enc.classes_:
        r = report.get(cls, {})
        print(f"  {cls[:55]:<55} F1={r.get('f1-score', 0):.3f}  n={int(r.get('support', 0))}")

    thresh_path = os.path.join(output_dir, "cat_thresholds.npy")
    np.save(thresh_path, thresholds)

    results = {
        "checkpoint": checkpoint,
        "val": {
            "argmax_macro_f1": round(baseline_val, 4),
            "calibrated_macro_f1": round(cal_val["macro_f1"], 4),
        },
        "test": {
            "argmax_macro_f1": round(baseline_test, 4),
            "calibrated_macro_f1": round(cal_test["macro_f1"], 4),
            "calibrated_weighted_f1": round(cal_test["weighted_f1"], 4),
        },
        "thresholds": thresholds.tolist(),
        "cat_classes": list(cat_enc.classes_),
    }
    out_json = os.path.join(LOG_DIR, "calibration_results.json")
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Thresholds saved → {thresh_path}")
    print(f"✅ Results saved → {out_json}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(CKPT_DIR, "best_model.pt"),
    )
    parser.add_argument("--output-dir", default=CKPT_DIR)
    args = parser.parse_args()
    return run_calibration(checkpoint=args.checkpoint, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
