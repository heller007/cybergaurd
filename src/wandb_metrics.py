"""
Shared Weights & Biases metric keys for all training scripts.

Use these helpers so baseline, full NER pipeline, and future models
plot on the same W&B charts for fair comparison.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import wandb
except ImportError:
    wandb = None


# ── Canonical metric keys (do not rename without updating all trainers) ───
# Step (optimizer steps):
#   train/loss, train/cat_loss, train/sub_loss
#   train/cat_acc, train/sub_acc
#   train/lr_encoder, train/lr_head, train/epoch
# Epoch:
#   epoch
#   train/epoch_loss, train/epoch_cat_acc, train/epoch_sub_acc
#   val/loss, val/cat_acc, val/sub_acc
#   val/cat_macro_f1, val/sub_macro_f1
#   val/cat_weighted_f1, val/sub_weighted_f1
#   val/combined_f1, val/best_combined_f1, val/patience
# Checkpoint / early stop:
#   checkpoint/best_combined_f1
#   early_stop/epoch, early_stop/best_epoch, early_stop/best_combined_f1
# Test (held-out):
#   test/cat_macro_f1, test/cat_weighted_f1
#   test/sub_macro_f1, test/sub_weighted_f1
#   test/combined_f1, test/best_val_combined_f1, test/best_epoch
# Summary:
#   test_combined_f1, best_val_combined_f1, best_epoch


def wandb_log(payload: Dict[str, Any], step: Optional[int] = None) -> None:
    if wandb is None or wandb.run is None:
        return
    if step is not None:
        wandb.log(payload, step=step)
    else:
        wandb.log(payload)


def log_train_step(
    *,
    loss: float,
    lr_encoder: float,
    lr_head: float,
    epoch: int,
    step: int,
    cat_loss: Optional[float] = None,
    sub_loss: Optional[float] = None,
    cat_acc: Optional[float] = None,
    sub_acc: Optional[float] = None,
) -> None:
    payload: Dict[str, Any] = {
        "train/loss": loss,
        "train/lr_encoder": lr_encoder,
        "train/lr_head": lr_head,
        "train/epoch": epoch,
    }
    if cat_loss is not None:
        payload["train/cat_loss"] = cat_loss
    if sub_loss is not None:
        payload["train/sub_loss"] = sub_loss
    if cat_acc is not None:
        payload["train/cat_acc"] = cat_acc
    if sub_acc is not None:
        payload["train/sub_acc"] = sub_acc
    wandb_log(payload, step=step)


def log_epoch(
    *,
    epoch: int,
    train_loss: float,
    train_cat_acc: float,
    train_sub_acc: float,
    val_loss: float,
    val_cat_acc: float,
    val_sub_acc: float,
    val_cat_macro_f1: float,
    val_sub_macro_f1: float,
    val_cat_weighted_f1: float,
    val_sub_weighted_f1: float,
    val_combined_f1: float,
    best_combined_f1: float,
    patience: int,
    step: int,
    per_class_cat_f1: Optional[Dict[str, float]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "train/epoch_loss": train_loss,
        "train/epoch_cat_acc": train_cat_acc,
        "train/epoch_sub_acc": train_sub_acc,
        "val/loss": val_loss,
        "val/cat_acc": val_cat_acc,
        "val/sub_acc": val_sub_acc,
        "val/cat_macro_f1": val_cat_macro_f1,
        "val/sub_macro_f1": val_sub_macro_f1,
        "val/cat_weighted_f1": val_cat_weighted_f1,
        "val/sub_weighted_f1": val_sub_weighted_f1,
        "val/combined_f1": val_combined_f1,
        "val/best_combined_f1": best_combined_f1,
        "val/patience": patience,
    }
    if per_class_cat_f1:
        for name, f1 in per_class_cat_f1.items():
            key = name[:40].replace("/", "_").replace(" ", "_")
            payload[f"val/cat_f1/{key}"] = f1
    wandb_log(payload, step=step)


def log_checkpoint(best_combined_f1: float, step: int) -> None:
    wandb_log({"checkpoint/best_combined_f1": best_combined_f1}, step=step)


def log_early_stop(
    *,
    epoch: int,
    best_epoch: int,
    best_combined_f1: float,
    step: int,
) -> None:
    wandb_log(
        {
            "early_stop/epoch": epoch,
            "early_stop/best_epoch": best_epoch,
            "early_stop/best_combined_f1": best_combined_f1,
        },
        step=step,
    )


def log_test(
    *,
    cat_macro_f1: float,
    cat_weighted_f1: float,
    sub_macro_f1: float,
    sub_weighted_f1: float,
    combined_f1: float,
    best_val_combined_f1: float,
    best_epoch: int,
    step: Optional[int] = None,
) -> None:
    wandb_log(
        {
            "test/cat_macro_f1": cat_macro_f1,
            "test/cat_weighted_f1": cat_weighted_f1,
            "test/sub_macro_f1": sub_macro_f1,
            "test/sub_weighted_f1": sub_weighted_f1,
            "test/combined_f1": combined_f1,
            "test/best_val_combined_f1": best_val_combined_f1,
            "test/best_epoch": best_epoch,
        },
        step=step,
    )


def finish_run(
    *,
    test_combined_f1: Optional[float] = None,
    best_val_combined_f1: Optional[float] = None,
    best_epoch: Optional[int] = None,
) -> None:
    if wandb is None or wandb.run is None:
        return
    if test_combined_f1 is not None:
        wandb.run.summary["test_combined_f1"] = test_combined_f1
    if best_val_combined_f1 is not None:
        wandb.run.summary["best_val_combined_f1"] = best_val_combined_f1
    if best_epoch is not None:
        wandb.run.summary["best_epoch"] = best_epoch
    wandb.run.finish()
