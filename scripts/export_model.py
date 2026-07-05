#!/usr/bin/env python3
"""
Export deployment metadata next to the trained category checkpoint.

  python scripts/export_model.py
"""

import json
import os
import shutil

CKPT_DIR = "outputs/checkpoints_category_filtered_muril"
BEST_MODEL = os.path.join(CKPT_DIR, "best_model.pt")
META_PATH = os.path.join(CKPT_DIR, "model_meta.json")
DEPLOY_DIR = "model/category_9class"


def main():
    import torch

    if not os.path.exists(BEST_MODEL):
        raise FileNotFoundError(
            f"Checkpoint not found: {BEST_MODEL}\n"
            "Train first: python -m src.train_category"
        )

    ckpt = torch.load(BEST_MODEL, map_location="cpu")
    meta = {
        "task": "category_classification",
        "num_classes": ckpt["num_cat"],
        "classes": list(ckpt["cat_classes"]),
        "model_name": ckpt.get("model_name", "google/muril-base-cased"),
        "use_ner": ckpt.get("use_ner", True),
        "max_len": 256,
        "metric": "cat_macro_f1",
        "best_val_cat_macro_f1": ckpt.get("cat_macro_f1"),
        "epoch": ckpt.get("epoch"),
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {META_PATH}")

    os.makedirs(DEPLOY_DIR, exist_ok=True)
    for name in ("best_model.pt", "cat_thresholds.npy", "model_meta.json"):
        src = os.path.join(CKPT_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(DEPLOY_DIR, name))
            print(f"  copied → {DEPLOY_DIR}/{name}")

    print(f"\nDeploy bundle ready in {DEPLOY_DIR}/")
    print("Usage: python -m src.predict_category --checkpoint model/category_9class/best_model.pt")


if __name__ == "__main__":
    main()
