# Model releases

Deploy weights live on **Google Drive**, not GitHub.

**MuRIL:** https://drive.google.com/open?id=1xvHaKqB-kY8GB3cekO1CDGGCpZDYeJCK

See [`WEIGHTS.md`](../WEIGHTS.md) for setup.

Local folders (gitignored):

| Folder | ~Size | Contents |
|--------|-------|----------|
| `muril_category_9class/` | 915 MB | `best_model.pt`, thresholds, meta |
| `llm_lora_qwen05b_r64/checkpoints/best_adapter/` | 146 MB | LoRA adapter only |

```bash
python scripts/package_releases.py      # tarball for manual upload
./scripts/upload_weights.sh             # rclone → Drive (needs RCLONE_CONFIG_PASS)
```
