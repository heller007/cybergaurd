"""
Deployable category-only inference for the 9-class MuRIL model.

Usage:
    from src.predict_category import CategoryPredictor

    predictor = CategoryPredictor()
    result = predictor.predict("Someone hacked my Facebook account...")
    print(result["category"], result["confidence"])

CLI:
    python -m src.predict_category "complaint text here"
    python -m src.predict_category --file complaints.txt
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import torch
from torch.cuda.amp import autocast
from transformers import AutoTokenizer

from src.calibrate import threshold_predict
from src.ner import CyberCrimeNER
from src.preprocess import clean_text
from src.train_category import FP16, MAX_LEN, CategoryClassifier

DEPLOY_DIR = "model/category_9class"
DEFAULT_CHECKPOINT = os.path.join(DEPLOY_DIR, "best_model.pt")
DEFAULT_THRESHOLDS = os.path.join(DEPLOY_DIR, "cat_thresholds.npy")


class CategoryPredictor:
    """Load once, predict many — category-only 9-class classifier."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        thresholds_path: Optional[str] = DEFAULT_THRESHOLDS,
        device: Optional[str] = None,
        use_calibration: bool = True,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.cat_classes = list(ckpt["cat_classes"])
        self.use_ner = ckpt.get("use_ner", True)
        model_name = ckpt.get("model_name", "google/muril-base-cased")

        self.model = CategoryClassifier(
            num_cat=ckpt["num_cat"],
            use_ner=self.use_ner,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ner = CyberCrimeNER(annotate=False) if self.use_ner else None

        self.thresholds = None
        if use_calibration and thresholds_path and os.path.exists(thresholds_path):
            self.thresholds = np.load(thresholds_path)

        meta_path = os.path.join(os.path.dirname(checkpoint), "model_meta.json")
        self.meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)

    def _ner_features(self, text: str) -> Optional[torch.Tensor]:
        if not self.use_ner:
            return None
        feats = self.ner.process(text).feature_vector
        return torch.tensor(feats, dtype=torch.float32).unsqueeze(0)

    @torch.no_grad()
    def predict(
        self,
        text: str,
        top_k: int = 3,
        return_probs: bool = False,
    ) -> dict:
        cleaned = clean_text(text)
        enc = self.tokenizer(
            cleaned,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        ner = self._ner_features(cleaned)
        if ner is not None:
            ner = ner.to(self.device)

        with autocast(enabled=FP16 and self.device.type == "cuda"):
            logits = self.model(input_ids, attention_mask, entity_features=ner)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        if self.thresholds is not None:
            pred_idx = int(threshold_predict(probs.reshape(1, -1), self.thresholds)[0])
        else:
            pred_idx = int(probs.argmax())

        top_idx = probs.argsort()[::-1][:top_k]
        result = {
            "category": self.cat_classes[pred_idx],
            "label_id": pred_idx,
            "confidence": float(probs[pred_idx]),
            "top_k": [
                {"category": self.cat_classes[i], "confidence": float(probs[i])}
                for i in top_idx
            ],
            "calibrated": self.thresholds is not None,
        }
        if return_probs:
            result["probabilities"] = {
                self.cat_classes[i]: float(probs[i]) for i in range(len(self.cat_classes))
            }
        return result

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]


def main():
    parser = argparse.ArgumentParser(description="Predict cyber-crime category")
    parser.add_argument("text", nargs="?", help="Complaint text")
    parser.add_argument("--file", "-f", help="File with one complaint per line")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--no-calibration", action="store_true")
    args = parser.parse_args()

    predictor = CategoryPredictor(
        checkpoint=args.checkpoint,
        use_calibration=not args.no_calibration,
    )

    if args.file:
        with open(args.file) as f:
            texts = [line.strip() for line in f if line.strip()]
        for text, result in zip(texts, predictor.predict_batch(texts)):
            print(f"\n{text[:80]}...")
            print(f"  → {result['category']} ({result['confidence']:.3f})")
    elif args.text:
        print(json.dumps(predictor.predict(args.text), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
