"""
Category-only training on dataset_balanced_filtered/ (9 classes).

Run: python -m src.train_category
"""

import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from src.losses import (
    build_category_loss,
    compute_weak_class_sampler_multipliers,
)
from src.ner import N_FEATURES, precompute_ner_features
from src.wandb_metrics import (
    finish_run,
    log_checkpoint,
    log_early_stop,
    log_epoch,
    log_test,
    log_train_step,
)

try:
    import wandb
except ImportError:
    wandb = None

# ── Config ────────────────────────────────────────────────────────────────
MODEL_NAME = "google/muril-base-cased"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 80
PATIENCE = 8
MIN_DELTA = 0.001
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRAD_ACCUM = 4
MAX_GRAD_NORM = 1.0
FP16 = True
SEED = 42
VAL_SIZE = 0.1
NUM_WORKERS = 8
DROPOUT = 0.2
FOCAL_GAMMA = 2.5
LOSS_TYPE = "focal"            # focal | weak_focal | cb_focal | ldam
CB_BETA = 0.999
LABEL_SMOOTHING = 0.05
LDAM_MAX_M = 0.5
LDAM_S = 30.0
LOG_EVERY_STEPS = 50
USE_WEIGHTED_SAMPLER = True
SAMPLER_POWER = 0.5
# Weak-class targeting (uses prior validation/test F1 report)
USE_WEAK_CLASS_BOOST = False
WEAK_CLASS_REPORT = "outputs/logs_category_filtered_muril/test_cat_report.json"
WEAK_F1_THRESHOLD = 0.45
WEAK_GAMMA_BOOST = 1.0
WEAK_SAMPLER_BOOST = 1.8
USE_NER = True

TRAIN_PATH = "dataset_balanced_filtered/train.csv"
TEST_PATH = "dataset_balanced_filtered/test.csv"
PRED_DIR = "outputs/predictions"
LOG_DIR = "outputs/logs_category_filtered_muril"
CKPT_DIR = "outputs/checkpoints_category_filtered_muril"
NER_CACHE_DIR = "outputs/cache/ner_category_filtered"

WANDB_ENABLED = True
WANDB_ENTITY = "Vineet_Nitk"
WANDB_PROJECT = "cybergaurd"
WANDB_RUN_NAME = "category-9class-focal-muril-base"

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(NER_CACHE_DIR, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_wandb(num_cat, n_train, n_val, n_test):
    if not WANDB_ENABLED or wandb is None:
        if WANDB_ENABLED and wandb is None:
            print("⚠️  wandb not installed")
        return None
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY or None,
        name=WANDB_RUN_NAME,
        tags=["category_only", "ner", "weak_focal", "muril-base", "filtered_9class"],
        config={
            "run_type": "category_only",
            "optimize_metric": "cat_macro_f1",
            "architecture": "cls_attnpool_ner_fusion",
            "model_name": MODEL_NAME,
            "max_len": MAX_LEN,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "lr": LR,
            "encoder_lr": LR,
            "head_lr": LR * 10,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "grad_accum_steps": GRAD_ACCUM,
            "dropout": DROPOUT,
            "focal_gamma": FOCAL_GAMMA,
            "loss_type": LOSS_TYPE,
            "cb_beta": CB_BETA,
            "label_smoothing": LABEL_SMOOTHING,
            "use_weak_class_boost": USE_WEAK_CLASS_BOOST,
            "weak_f1_threshold": WEAK_F1_THRESHOLD,
            "weak_gamma_boost": WEAK_GAMMA_BOOST,
            "weak_sampler_boost": WEAK_SAMPLER_BOOST,
            "weak_class_report": WEAK_CLASS_REPORT,
            "use_ner": USE_NER,
            "use_weighted_sampler": USE_WEIGHTED_SAMPLER,
            "sampler_power": SAMPLER_POWER,
            "num_categories": num_cat,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "data": "dataset_balanced_filtered",
        },
    )
    print(f"W&B run: {run.url}")
    return run


def prepare_data():
    print("=" * 60)
    print("Loading dataset_balanced — category-only training")
    print("=" * 60)
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)

    train_raw["text"] = train_raw["crimeaditionalinfo"].fillna("").astype(str)
    test_raw["text"] = test_raw["crimeaditionalinfo"].fillna("").astype(str)
    train_raw = train_raw[train_raw["text"].str.len() > 0].reset_index(drop=True)
    test_raw = test_raw[test_raw["text"].str.len() > 0].reset_index(drop=True)

    print(f"  Train: {len(train_raw):,} | Test: {len(test_raw):,}")
    print(f"  Categories: {train_raw['category'].nunique()}")

    all_cats = pd.concat([train_raw["category"], test_raw["category"]]).unique()
    cat_enc = LabelEncoder().fit(all_cats)

    train_raw["cat_label"] = cat_enc.transform(train_raw["category"])
    test_raw["cat_label"] = cat_enc.transform(test_raw["category"])

    train_df, val_df = train_test_split(
        train_raw,
        test_size=VAL_SIZE,
        stratify=train_raw["cat_label"],
        random_state=SEED,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"  Split → train={len(train_df):,} val={len(val_df):,} test={len(test_raw):,}")
    print("\n  Category distribution (train):")
    for cls, cnt in train_df["category"].value_counts().items():
        print(f"    {cls[:50]:<50} {cnt:5d}")

    return train_df, val_df, test_raw, cat_enc


def get_ner_matrix(df, split_name):
    cache = os.path.join(NER_CACHE_DIR, f"{split_name}.npy")
    return precompute_ner_features(
        df["text"].tolist(),
        cache_path=cache,
        n_workers=NUM_WORKERS,
    )


class CategoryDataset(Dataset):
    def __init__(self, df, tokenizer, ner_features=None):
        self.texts = df["text"].tolist()
        self.cat_labels = df["cat_label"].tolist()
        self.tokenizer = tokenizer
        self.ner_features = ner_features

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "cat_label": torch.tensor(self.cat_labels[idx], dtype=torch.long),
        }
        if self.ner_features is not None:
            item["entity_features"] = torch.from_numpy(self.ner_features[idx])
        return item


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        scores = self.attn(hidden_states).squeeze(-1)
        fill = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(attention_mask == 0, fill)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)


class CategoryClassifier(nn.Module):
    """XLM-R + dual pooling + optional NER fusion → category head only."""

    def __init__(self, num_cat: int, use_ner: bool = True):
        super().__init__()
        self.use_ner = use_ner
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size

        self.attn_pool = AttentionPooling(hidden)
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        fusion_in = hidden
        if use_ner:
            ner_hidden = 64
            self.entity_encoder = nn.Sequential(
                nn.Linear(N_FEATURES, ner_hidden * 2),
                nn.LayerNorm(ner_hidden * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(ner_hidden * 2, ner_hidden),
                nn.GELU(),
            )
            self.fusion_proj = nn.Sequential(
                nn.Linear(hidden + ner_hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(DROPOUT),
            )
            fusion_in = hidden

        self.cat_head = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Dropout(DROPOUT),
            nn.Linear(fusion_in, hidden // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden // 2, num_cat),
        )

        for module in (self.pool_proj, self.cat_head):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        if use_ner:
            for layer in self.entity_encoder.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_mask, entity_features=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = out.last_hidden_state
        cls_embed = hidden_states[:, 0, :]
        attn_embed = self.attn_pool(hidden_states, attention_mask)
        pooled = self.pool_proj(torch.cat([cls_embed, attn_embed], dim=-1))

        if self.use_ner and entity_features is not None:
            ner_embed = self.entity_encoder(entity_features)
            pooled = self.fusion_proj(torch.cat([pooled, ner_embed], dim=-1))

        return self.cat_head(pooled)


def compute_class_weights(labels, num_classes, device):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    min_w = weights[weights > 0].min()
    weights = np.clip(weights, min_w, min_w * 20)
    return torch.FloatTensor(weights).to(device)


def load_prior_class_f1(report_path: str) -> dict | None:
    if not report_path or not os.path.exists(report_path):
        return None
    with open(report_path) as f:
        report = json.load(f)
    return {
        k: v for k, v in report.items()
        if isinstance(v, dict) and "f1-score" in v
    }


def build_weighted_sampler(
    labels,
    power=SAMPLER_POWER,
    class_multipliers: np.ndarray | None = None,
):
    """Inverse-frequency weights raised to `power` (0.5 = sqrt, gentler than 1.0)."""
    counts = np.bincount(labels)
    counts = np.where(counts == 0, 1, counts)
    inv_freq = 1.0 / counts[labels]
    sample_weights = inv_freq ** power
    if class_multipliers is not None:
        sample_weights *= class_multipliers[labels]
    sample_weights = sample_weights / sample_weights.mean()
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(labels),
        replacement=True,
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    cat_preds, cat_trues = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            cl = batch["cat_label"].to(device)
            ner = batch.get("entity_features")
            ner = ner.to(device) if ner is not None else None

            with autocast(enabled=FP16):
                logits = model(ids, mask, entity_features=ner)
                loss = criterion(logits, cl)

            total_loss += loss.item()
            cat_preds.extend(logits.argmax(1).cpu().numpy())
            cat_trues.extend(cl.cpu().numpy())

    n = max(len(loader), 1)
    return (
        total_loss / n,
        f1_score(cat_trues, cat_preds, average="macro", zero_division=0),
        f1_score(cat_trues, cat_preds, average="weighted", zero_division=0),
        cat_preds,
        cat_trues,
    )


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_df, val_df, test_df, cat_enc = prepare_data()
    NUM_CAT = len(cat_enc.classes_)

    prior_f1 = None
    if USE_WEAK_CLASS_BOOST and LOSS_TYPE == "weak_focal":
        prior_f1 = load_prior_class_f1(WEAK_CLASS_REPORT)
        if prior_f1:
            print(f"\n  Weak-class targeting from: {WEAK_CLASS_REPORT}")
            for cls in cat_enc.classes_:
                stats = prior_f1.get(cls)
                if stats and stats.get("f1-score", 1.0) < WEAK_F1_THRESHOLD:
                    print(
                        f"    weak: {cls[:45]:<45} "
                        f"F1={stats['f1-score']:.3f} "
                        f"P={stats['precision']:.3f} R={stats['recall']:.3f}"
                    )
        else:
            print(f"\n  ⚠️  No prior F1 report at {WEAK_CLASS_REPORT}; using uniform focal")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ner = val_ner = test_ner = None
    if USE_NER:
        print("\nPrecomputing NER features...")
        train_ner = get_ner_matrix(train_df, "train")
        val_ner = get_ner_matrix(val_df, "val")
        test_ner = get_ner_matrix(test_df, "test")

    train_ds = CategoryDataset(train_df, tokenizer, train_ner)
    val_ds = CategoryDataset(val_df, tokenizer, val_ner)
    test_ds = CategoryDataset(test_df, tokenizer, test_ner)

    sampler = None
    shuffle = True
    if USE_WEIGHTED_SAMPLER:
        class_mult = None
        if USE_WEAK_CLASS_BOOST and prior_f1:
            class_mult = compute_weak_class_sampler_multipliers(
                list(cat_enc.classes_),
                prior_f1=prior_f1,
                f1_threshold=WEAK_F1_THRESHOLD,
                weak_boost=WEAK_SAMPLER_BOOST,
            )
            boosted = [
                cat_enc.classes_[i]
                for i, m in enumerate(class_mult)
                if m > 1.01
            ]
            if boosted:
                print(f"  Sampler boost for under-predicted weak: {len(boosted)} classes")
        sampler = build_weighted_sampler(
            train_df["cat_label"].values,
            power=SAMPLER_POWER,
            class_multipliers=class_mult,
        )
        shuffle = False
        print(f"  Using WeightedRandomSampler (power={SAMPLER_POWER})")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    cat_criterion = build_category_loss(
        LOSS_TYPE,
        train_df["cat_label"].values,
        NUM_CAT,
        device,
        focal_gamma=FOCAL_GAMMA,
        cb_beta=CB_BETA,
        label_smoothing=LABEL_SMOOTHING,
        ldam_max_m=LDAM_MAX_M,
        ldam_s=LDAM_S,
        class_names=list(cat_enc.classes_),
        prior_f1=prior_f1,
        weak_f1_threshold=WEAK_F1_THRESHOLD,
        weak_gamma_boost=WEAK_GAMMA_BOOST,
    )

    print(f"\nBuilding category-only model ({MODEL_NAME})")
    print(
        f"  Loss: {LOSS_TYPE}  γ={FOCAL_GAMMA}  "
        f"weak_thr={WEAK_F1_THRESHOLD}  γ_boost={WEAK_GAMMA_BOOST}"
    )
    model = CategoryClassifier(NUM_CAT, use_ner=USE_NER).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    head_params = (
        list(model.attn_pool.parameters())
        + list(model.pool_proj.parameters())
        + list(model.cat_head.parameters())
    )
    if USE_NER:
        head_params += list(model.entity_encoder.parameters()) + list(
            model.fusion_proj.parameters()
        )

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": LR, "weight_decay": WEIGHT_DECAY},
            {"params": head_params, "lr": LR * 10, "weight_decay": 0.0},
        ]
    )
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = GradScaler(enabled=FP16)

    wb_run = init_wandb(NUM_CAT, len(train_df), len(val_df), len(test_df))

    best_cat_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience_ctr = 0
    history = []
    global_step = 0
    window_loss, window_cat_acc, window_cat_loss = [], [], []

    print(f"\nTraining up to {EPOCHS} epochs (patience={PATIENCE}, metric=cat_macro_f1)")
    print(f"  Batches: train={len(train_loader)} val={len(val_loader)}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_cat_correct = 0
        epoch_n = 0
        optimizer.zero_grad()
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for step, batch in enumerate(pbar):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            cl = batch["cat_label"].to(device)
            ner = batch.get("entity_features")
            ner = ner.to(device) if ner is not None else None

            with autocast(enabled=FP16):
                logits = model(ids, mask, entity_features=ner)
                cat_loss = cat_criterion(logits, cl)
                loss = cat_loss / GRAD_ACCUM

            scaler.scale(loss).backward()
            batch_loss = loss.item() * GRAD_ACCUM
            epoch_loss += batch_loss
            window_loss.append(batch_loss)
            window_cat_loss.append(cat_loss.item())

            with torch.no_grad():
                cat_acc_b = (logits.argmax(1) == cl).float().mean().item()
                window_cat_acc.append(cat_acc_b)
                epoch_cat_correct += (logits.argmax(1) == cl).sum().item()
                epoch_n += cl.size(0)

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % LOG_EVERY_STEPS == 0:
                    avg_loss = float(np.mean(window_loss))
                    avg_cat_acc = float(np.mean(window_cat_acc))
                    enc_lr = optimizer.param_groups[0]["lr"]
                    head_lr = optimizer.param_groups[1]["lr"]
                    print(
                        f"\n  step={global_step} | loss={avg_loss:.4f} "
                        f"cat_acc={avg_cat_acc:.3f}",
                        flush=True,
                    )
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", cat_acc=f"{avg_cat_acc:.3f}")
                    log_train_step(
                        loss=avg_loss,
                        cat_loss=float(np.mean(window_cat_loss)),
                        cat_acc=avg_cat_acc,
                        lr_encoder=enc_lr,
                        lr_head=head_lr,
                        epoch=epoch,
                        step=global_step,
                    )
                    window_loss.clear()
                    window_cat_acc.clear()
                    window_cat_loss.clear()

        val_loss, cat_mf1, cat_wf1, cat_preds, cat_trues = evaluate(
            model, val_loader, cat_criterion, device
        )

        avg_train_loss = epoch_loss / len(train_loader)
        train_cat_acc = epoch_cat_correct / max(epoch_n, 1)
        val_cat_acc = float((np.array(cat_preds) == np.array(cat_trues)).mean())
        elapsed = time.time() - t0
        is_best = cat_mf1 > best_cat_f1 + MIN_DELTA

        print(f"\nEpoch {epoch:02d}/{EPOCHS} | {elapsed:.0f}s")
        print(f"  train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}")
        print(f"  train cat_acc={train_cat_acc:.4f}  val cat_acc={val_cat_acc:.4f}")
        print(f"  cat → macro={cat_mf1:.4f}  weighted={cat_wf1:.4f}")
        print(
            f"  cat_macro_f1 = {cat_mf1:.4f}  "
            f"{'⭐ NEW BEST' if is_best else f'(no improve {patience_ctr + 1}/{PATIENCE})'}"
        )

        cat_report = classification_report(
            cat_trues,
            cat_preds,
            labels=list(range(NUM_CAT)),
            target_names=list(cat_enc.classes_),
            zero_division=0,
            output_dict=True,
        )
        per_class_f1 = {
            cls: cat_report.get(cls, {}).get("f1-score", 0) for cls in cat_enc.classes_
        }

        # Log combined_f1 = cat_macro_f1 for W&B chart compatibility
        log_epoch(
            epoch=epoch,
            train_loss=avg_train_loss,
            train_cat_acc=train_cat_acc,
            train_sub_acc=0.0,
            val_loss=val_loss,
            val_cat_acc=val_cat_acc,
            val_sub_acc=0.0,
            val_cat_macro_f1=cat_mf1,
            val_sub_macro_f1=0.0,
            val_cat_weighted_f1=cat_wf1,
            val_sub_weighted_f1=0.0,
            val_combined_f1=cat_mf1,
            best_combined_f1=max(best_cat_f1, cat_mf1),
            patience=0 if is_best else patience_ctr + 1,
            step=global_step,
            per_class_cat_f1=per_class_f1,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "cat_macro_f1": cat_mf1,
                "cat_weighted_f1": cat_wf1,
            }
        )

        if is_best:
            best_cat_f1 = cat_mf1
            best_epoch = epoch
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": best_state,
                    "epoch": epoch,
                    "cat_macro_f1": cat_mf1,
                    "num_cat": NUM_CAT,
                    "cat_classes": list(cat_enc.classes_),
                    "use_ner": USE_NER,
                    "model_name": MODEL_NAME,
                },
                os.path.join(CKPT_DIR, "best_model.pt"),
            )
            log_checkpoint(cat_mf1, step=global_step)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(
                    f"\n  Early stopping at epoch {epoch} "
                    f"(best epoch {best_epoch}, cat_macro_f1={best_cat_f1:.4f})"
                )
                log_early_stop(
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_combined_f1=best_cat_f1,
                    step=global_step,
                )
                break

    if best_state is None:
        raise RuntimeError("No checkpoint saved")

    print(f"\n{'=' * 60}\nTest evaluation (best epoch {best_epoch})\n{'=' * 60}")
    model.load_state_dict(best_state)
    model.to(device)
    _, cat_mf1_t, cat_wf1_t, cat_preds_t, cat_trues_t = evaluate(
        model, test_loader, cat_criterion, device
    )

    print(f"\n  Category macro F1:    {cat_mf1_t:.4f}  ← primary metric")
    print(f"  Category weighted F1: {cat_wf1_t:.4f}")

    cat_report_t = classification_report(
        cat_trues_t,
        cat_preds_t,
        labels=list(range(NUM_CAT)),
        target_names=list(cat_enc.classes_),
        zero_division=0,
    )
    print("\nPer-category test report:")
    print(cat_report_t)

    log_test(
        cat_macro_f1=cat_mf1_t,
        cat_weighted_f1=cat_wf1_t,
        sub_macro_f1=0.0,
        sub_weighted_f1=0.0,
        combined_f1=cat_mf1_t,
        best_val_combined_f1=best_cat_f1,
        best_epoch=best_epoch,
        step=global_step,
    )

    pred_df = test_df.copy()
    pred_df["pred_category"] = cat_enc.inverse_transform(cat_preds_t)
    pred_path = os.path.join(PRED_DIR, "category_only_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved → {pred_path}")

    results = {
        "run_type": "category_only",
        "model": MODEL_NAME,
        "use_ner": USE_NER,
        "sampler_power": SAMPLER_POWER,
        "loss_type": LOSS_TYPE,
        "use_weak_class_boost": USE_WEAK_CLASS_BOOST,
        "weak_f1_threshold": WEAK_F1_THRESHOLD,
        "cb_beta": CB_BETA,
        "label_smoothing": LABEL_SMOOTHING,
        "best_val_cat_macro_f1": round(best_cat_f1, 4),
        "best_epoch": best_epoch,
        "test": {
            "cat_macro_f1": round(cat_mf1_t, 4),
            "cat_weighted_f1": round(cat_wf1_t, 4),
        },
        "history": history,
    }
    results_path = os.path.join(LOG_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    report_path = os.path.join(LOG_DIR, "test_cat_report.json")
    with open(report_path, "w") as f:
        json.dump(
            classification_report(
                cat_trues_t,
                cat_preds_t,
                labels=list(range(NUM_CAT)),
                target_names=list(cat_enc.classes_),
                zero_division=0,
                output_dict=True,
            ),
            f,
            indent=2,
        )

    print(f"\n✅ Done. Test category macro F1 = {cat_mf1_t:.4f}")
    finish_run(
        test_combined_f1=cat_mf1_t,
        best_val_combined_f1=best_cat_f1,
        best_epoch=best_epoch,
    )
    return cat_mf1_t


if __name__ == "__main__":
    main()
