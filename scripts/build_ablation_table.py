#!/usr/bin/env python3
"""
Aggregate all 9-class category experiment results into a single ablation table.

  python scripts/build_ablation_table.py
"""

import json
import os
from datetime import date

OUT_DIR = "outputs/experiments_9class"
RESULTS_PATH = os.path.join(OUT_DIR, "ablation_table.json")
REPORT_PATH = os.path.join(OUT_DIR, "ablation_table.md")


def load_json(path):
    full = os.path.join(OUT_DIR, path) if not path.startswith("outputs/") else path
    if os.path.exists(full):
        with open(full) as f:
            return json.load(f)
    return None


def neural_row(name, path, *, notes=""):
    data = load_json(path)
    if not data:
        return None
    test = data.get("test", {})
    return {
        "method": name,
        "family": "neural",
        "test_cat_macro_f1": test.get("cat_macro_f1"),
        "test_cat_weighted_f1": test.get("cat_weighted_f1"),
        "val_cat_macro_f1": data.get("best_val_cat_macro_f1"),
        "best_epoch": data.get("best_epoch"),
        "model": data.get("model"),
        "loss_type": data.get("loss_type"),
        "use_ner": data.get("use_ner"),
        "notes": notes,
        "source": f"outputs/experiments_9class/{path}",
    }


def classical_rows(path):
    data = load_json(path)
    if not data:
        return []
    rows = []
    for m in data.get("methods", []):
        features = m.get("features", "tfidf")
        calibrated = m.get("calibrated", False)
        notes = features
        if calibrated:
            notes += "; per-class threshold calibration"
        rows.append(
            {
                "method": m["model"],
                "family": "classical",
                "test_cat_macro_f1": m["test"]["macro_f1"],
                "test_cat_weighted_f1": m["test"]["weighted_f1"],
                "val_cat_macro_f1": m["val"]["macro_f1"],
                "best_epoch": None,
                "model": m["model"],
                "loss_type": None,
                "use_ner": "ner" in features,
                "notes": notes,
                "source": f"outputs/experiments_9class/{path}",
            }
        )
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    rows.extend(classical_rows("classical_baselines.json"))

    llm = load_json("llm_prompt_results.json")
    if llm:
        rows.append(
            {
                "method": f"LLM prompt ({llm.get('model', 'Qwen2.5-0.5B')})",
                "family": "llm",
                "test_cat_macro_f1": llm["test"]["cat_macro_f1"],
                "test_cat_weighted_f1": llm["test"]["cat_weighted_f1"],
                "val_cat_macro_f1": None,
                "best_epoch": None,
                "model": llm.get("model"),
                "loss_type": None,
                "use_ner": False,
                "notes": (
                    f"{llm.get('prompting')}; n={llm.get('eval_samples')}; "
                    "zero-shot digit scoring"
                ),
                "source": "outputs/experiments_9class/llm_prompt_results.json",
            }
        )

    llm_lora = load_json("llm_lora_results.json")
    if llm_lora:
        rows.append(
            {
                "method": llm_lora.get("method", "LoRA Qwen2.5-0.5B"),
                "family": "llm",
                "test_cat_macro_f1": llm_lora["test"]["cat_macro_f1"],
                "test_cat_weighted_f1": llm_lora["test"]["cat_weighted_f1"],
                "val_cat_macro_f1": llm_lora.get("val", {}).get("cat_macro_f1"),
                "best_epoch": llm_lora.get("epochs"),
                "model": llm_lora.get("model"),
                "loss_type": "LoRA SFT",
                "use_ner": False,
                "notes": (
                    f"r={llm_lora.get('lora_r')}, alpha={llm_lora.get('lora_alpha')}; "
                    f"train n={llm_lora.get('train_samples')}; "
                    f"eval n={llm_lora.get('eval_samples')}"
                ),
                "source": "outputs/experiments_9class/llm_lora_results.json",
            }
        )

    neural_experiments = [
        (
            "MuRIL-base + NER + focal (best)",
            "muril_focal_results.json",
            "9-class filtered; weighted sampler",
        ),
        (
            "MuRIL-base + NER + calibrated",
            "muril_calibrated_results.json",
            "Per-class threshold tuning on val set",
        ),
        (
            "XLM-R-large + NER + focal",
            "xlmr_focal_results.json",
            "Same 9-class setup, different encoder",
        ),
        (
            "MuRIL-base + weak_focal",
            "muril_weak_focal_results.json",
            "Weak-class targeted loss (underperformed)",
        ),
    ]

    for name, path, notes in neural_experiments:
        if "calibrated" in path:
            data = load_json(path)
            if data:
                rows.append(
                    {
                        "method": name,
                        "family": "neural",
                        "test_cat_macro_f1": data.get("test", {}).get(
                            "calibrated_macro_f1"
                        ),
                        "test_cat_weighted_f1": data.get("test", {}).get(
                            "calibrated_weighted_f1"
                        ),
                        "val_cat_macro_f1": data.get("val", {}).get(
                            "calibrated_macro_f1"
                        ),
                        "best_epoch": None,
                        "model": "google/muril-base-cased",
                        "loss_type": "focal + calibration",
                        "use_ner": True,
                        "notes": notes,
                        "source": f"outputs/experiments_9class/{path}",
                    }
                )
        else:
            row = neural_row(name, path, notes=notes)
            if row:
                rows.append(row)

    rows = [r for r in rows if r.get("test_cat_macro_f1") is not None]
    rows.sort(key=lambda r: -r["test_cat_macro_f1"])

    best = rows[0] if rows else None
    payload = {
        "task": "category_classification",
        "dataset": "dataset_balanced_filtered",
        "num_classes": 9,
        "metric": "test_cat_macro_f1",
        "generated": str(date.today()),
        "best_method": best["method"] if best else None,
        "best_test_cat_macro_f1": best["test_cat_macro_f1"] if best else None,
        "experiments": rows,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# 9-Class Category Classification — Ablation Table",
        "",
        f"**Dataset:** `dataset_balanced_filtered` (9 categories)  ",
        f"**Primary metric:** test category macro-F1  ",
        f"**Generated:** {date.today()}",
        "",
        "| Rank | Method | Test Macro F1 | Val Macro F1 | Family | Notes |",
        "|------|--------|---------------|--------------|--------|-------|",
    ]
    for i, r in enumerate(rows, 1):
        val = (
            f"{r['val_cat_macro_f1']:.4f}"
            if r.get("val_cat_macro_f1") is not None
            else "—"
        )
        test = f"{r['test_cat_macro_f1']:.4f}"
        lines.append(
            f"| {i} | {r['method']} | **{test}** | {val} | {r['family']} | {r.get('notes', '')} |"
        )

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(rows)} experiments → {RESULTS_PATH}")
    print(f"Markdown table → {REPORT_PATH}")
    if best:
        print(f"Best: {best['method']} (F1={best['test_cat_macro_f1']:.4f})")


if __name__ == "__main__":
    main()
