#!/usr/bin/env bash
# Upload deploy weights to Google Drive via rclone.
#
#   export RCLONE_CONFIG_PASS='...'
#   export RCLONE_REMOTE='gdrive:cybergaurd'   # run: rclone listremotes
#   ./scripts/upload_weights.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="${ROOT}/releases/_gdrive_upload"
REMOTE="${RCLONE_REMOTE:?Set RCLONE_REMOTE, e.g. gdrive:cybergaurd}"

if [[ -z "${RCLONE_CONFIG_PASS:-}" ]]; then
  echo "Set RCLONE_CONFIG_PASS (rclone config is encrypted on this machine)."
  exit 1
fi

export RCLONE_CONFIG_PASS

MURIL_SRC="${ROOT}/releases/muril_category_9class"
[[ -f "${MURIL_SRC}/best_model.pt" ]] || MURIL_SRC="${ROOT}/model/category_9class"
LORA_SRC="${ROOT}/releases/llm_lora_qwen05b_r64/checkpoints/best_adapter"

[[ -f "${MURIL_SRC}/best_model.pt" ]] || { echo "Missing MuRIL bundle"; exit 1; }
[[ -f "${LORA_SRC}/adapter_model.safetensors" ]] || { echo "Missing LoRA adapter"; exit 1; }

rm -rf "${STAGE}"
mkdir -p "${STAGE}/muril_category_9class" "${STAGE}/llm_lora_qwen05b_r64/best_adapter"
cp -a "${MURIL_SRC}/." "${STAGE}/muril_category_9class/"
cp -a "${LORA_SRC}/." "${STAGE}/llm_lora_qwen05b_r64/best_adapter/"

echo "Staging $(du -sh "${STAGE}" | cut -f1) → ${REMOTE}"
rclone copy "${STAGE}" "${REMOTE}" --progress -v

echo ""
echo "Done. Get a share link in Drive UI, or try:"
echo "  rclone link '${REMOTE}/muril_category_9class/best_model.pt'"
