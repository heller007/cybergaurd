# Model weights (Google Drive)

**Weights are not on GitHub.** Download from Google Drive, then extract into the project root.

| Bundle | Size | Extract to | Macro-F1 |
|--------|------|------------|----------|
| `muril_category_9class/` | ~915 MB | `model/category_9class/` | 0.5826 |
| `llm_lora_qwen05b_r64/best_adapter/` | ~146 MB | `releases/llm_lora_qwen05b_r64/checkpoints/best_adapter/` | 0.5880 |

## Download link

**MuRIL `best_model.pt` (~915 MB):**  
https://drive.google.com/open?id=1xvHaKqB-kY8GB3cekO1CDGGCpZDYeJCK

Place the file at `model/category_9class/best_model.pt`. For calibrated inference, also use `cat_thresholds.npy` and `model_meta.json` from the local `releases/muril_category_9class/` bundle (or re-run `scripts/calibrate_category.py` after training).

**LoRA adapter:** upload separately when available.

## After download

**MuRIL (deploy / UI):**
```bash
mkdir -p model/category_9class
# unzip muril_category_9class/* → model/category_9class/

python -m src.predict_category "UPI fraud OTP scam"
uvicorn app.server:app --port 8000
```

**LoRA adapter (optional):**
```bash
mkdir -p releases/llm_lora_qwen05b_r64/checkpoints/best_adapter
# unzip adapter files into that folder

python scripts/llm_lora_finetune.py --eval-only \
  --adapter-dir releases/llm_lora_qwen05b_r64/checkpoints/best_adapter \
  --full-test --full-train --lora-r 64
```

## Upload to Drive (this machine)

rclone is installed (`~/.config/rclone/rclone.conf`, encrypted). From the project root:

```bash
export RCLONE_CONFIG_PASS='your-rclone-config-password'
export RCLONE_REMOTE='YOUR_REMOTE:cybergaurd'   # e.g. gdrive:cybergaurd

./scripts/upload_weights.sh
```

List your remotes (will prompt for config password):

```bash
rclone listremotes
```

The script uploads only deploy bundles (~1.06 GB), not training checkpoints.

## Local package (manual upload)

```bash
python scripts/package_releases.py
# → releases/cybergaurd_models_YYYY-MM-DD.tar.gz
```

Upload that tarball to Drive via browser or `rclone copy`.
